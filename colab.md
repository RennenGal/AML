# Colab FT-Transformer — Problems & Fixes

## Problem Summary

Running FT-Transformer on 500 features creates a sequence of 501 tokens (500 features + CLS).
Standard `nn.MultiheadAttention` computes a full 501×501 attention matrix per head, which is:
- Memory-heavy: caused OOM at batch_size=512, pushed 13.6/15.0 GB even at batch_size=256
- Slow: ~2 hours per Optuna trial on T4, making 15 trials = ~30 hours (session limit ~5 hours)

---

## Fix 1 — Flash Attention (biggest win, 2–3× speedup)

Replace `nn.MultiheadAttention` in `TransformerBlock` with `F.scaled_dot_product_attention`.
PyTorch 2.0+ uses Flash Attention under the hood on Turing (T4) GPUs — O(n) memory vs O(n²),
faster CUDA kernels, and frees enough VRAM to increase batch size.

**Before:**
```python
self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=attn_dropout, batch_first=True)

def forward(self, x):
    h = self.norm1(x)
    h, _ = self.attn(h, h, h)
    ...
```

**After:**
```python
self.q_proj = nn.Linear(d_token, d_token)
self.k_proj = nn.Linear(d_token, d_token)
self.v_proj = nn.Linear(d_token, d_token)
self.out_proj = nn.Linear(d_token, d_token)
self.n_heads = n_heads
self.attn_drop = attn_dropout

def forward(self, x):
    B, T, D = x.shape
    h = self.norm1(x)
    Q = self.q_proj(h).view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
    K = self.k_proj(h).view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
    V = self.v_proj(h).view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
    drop = self.attn_drop if self.training else 0.0
    h = F.scaled_dot_product_attention(Q, K, V, dropout_p=drop)
    h = h.transpose(1, 2).contiguous().view(B, T, D)
    h = self.out_proj(h)
    x = x + self.drop(h)
    ...
```

---

## Fix 2 — TF32 matmuls (free, one line)

T4 (Turing) supports TF32 for matrix multiplications — ~1.5× faster with negligible accuracy loss.
Add at the top of Cell 3:

```python
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
```

---

## Fix 3 — torch.compile (10–30% speedup)

PyTorch 2.0+ compiles the model to optimized CUDA kernels. Add after model creation in both
the objective and Cell 5:

```python
model = torch.compile(model)
```

Note: first forward pass takes ~30s to compile — worth it for long training runs.

---

## Fix 4 — Larger batch size (with Flash Attention freeing VRAM)

With Flash Attention, VRAM usage drops significantly. Increase `BATCH_SIZE` from 256 to 1024.
Fewer batches per epoch = less kernel launch overhead = faster epochs.

```python
BATCH_SIZE = 1024
```

---

## Fix 5 — autocast in eval (faster inference)

Currently `eval_bal_acc` runs in fp32. Wrapping it in `autocast` speeds up inference:

```python
@torch.no_grad()
def eval_bal_acc(model, loader):
    model.eval()
    preds, labels = [], []
    for X_batch, y_batch in loader:
        with torch.amp.autocast('cuda'):
            preds.append(model(X_batch.to(DEVICE)).argmax(1).cpu().numpy())
        labels.append(y_batch.numpy())
    return balanced_accuracy_score(np.concatenate(labels), np.concatenate(preds))
```

---

## Fix 6 — DataLoader improvements

```python
def make_loader(X_t, y_t, batch_size, shuffle=True):
    ds = TensorDataset(X_t, torch.tensor(y_t, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=2, persistent_workers=True)
```

`persistent_workers=True` avoids recreating worker processes between epochs.

---

## Fix 7 — Smaller tune subsample + fewer epochs

Reduce from 100K to 50K for tuning — each trial is 2× faster, accuracy estimate is still reliable.
With faster convergence expected from better LR range, reduce epochs too:

```python
TUNE_N      = 50_000
TUNE_EPOCHS = 20
PATIENCE    = 5
```

---

## Fix 8 — Eval every 2 epochs

Halves eval overhead during tuning:

```python
for epoch in range(TUNE_EPOCHS):
    train_epoch(model, optimizer, tune_train_loader)
    if epoch % 2 == 1 or epoch == TUNE_EPOCHS - 1:
        acc = eval_bal_acc(model, tune_val_loader)
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if acc > best_acc:
            best_acc, no_improve = acc, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
```

---

## Fix 9 — Updated autocast syntax (remove FutureWarning)

Replace:
```python
from torch.cuda.amp import autocast, GradScaler
with autocast():
```

With:
```python
from torch.amp import autocast, GradScaler
with autocast('cuda'):
```

---

## Expected Result

| Optimization | Speedup |
|---|---|
| Flash Attention | 2–3× |
| TF32 + cudnn.benchmark | 1.5× |
| torch.compile | 1.1–1.3× |
| Larger batch size (256→1024) | 1.5–2× |
| Smaller tune subsample (100K→50K) | 2× |
| Eval every 2 epochs | 1.3× |

Combined estimate: trial time drops from **~2 hours → ~15–25 minutes**.
15 trials would complete in **~4–6 hours**, fitting within a single Colab session.
