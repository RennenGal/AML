# %% [markdown]
# # FT-Transformer (Pure PyTorch) — Competition
# Run on Colab with a T4 GPU (Runtime → Change runtime type → T4 GPU)

# %% [markdown]
# ## Cell 1 — Install dependencies

# %%
!pip install optuna --quiet

# %% [markdown]
# ## Cell 2 — Mount Google Drive and load data

# %%
from google.colab import drive
drive.mount('/content/drive')

import numpy as np

DATA_PATH = '/content/drive/MyDrive/competition_train.npz'

data = np.load(DATA_PATH)
X_train_full = data['X_train'].astype(np.float32)   # (500000, 500)
y_train_full = data['y_train'].astype(np.int64)     # (500000,)
X_test       = data['X_test'].astype(np.float32)    # (200000, 500)

print('X_train:', X_train_full.shape, '  X_test:', X_test.shape)
print('Classes:', np.unique(y_train_full))

# %% [markdown]
# ## Cell 3 — FT-Transformer (pure PyTorch) + utilities

# %%
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score
import optuna

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = '/content/drive/MyDrive/'
print('Device:', DEVICE)

# Free TF32 speedup on Turing (T4) — negligible accuracy impact
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# --- FT-Transformer architecture ---

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias  # [B, n_features, d_token]


class TransformerBlock(nn.Module):
    """Pre-norm block using F.scaled_dot_product_attention (Flash Attention on PyTorch 2.x).
    Avoids materialising the full [B, heads, 501, 501] attention matrix — saves ~4 GB at
    batch=1024 vs nn.MultiheadAttention."""
    def __init__(self, d_token, n_heads, ffn_d_hidden, attn_dropout, ffn_dropout, residual_dropout):
        super().__init__()
        self.n_heads   = n_heads
        self.attn_drop = attn_dropout
        self.norm1     = nn.LayerNorm(d_token)
        self.q_proj    = nn.Linear(d_token, d_token)
        self.k_proj    = nn.Linear(d_token, d_token)
        self.v_proj    = nn.Linear(d_token, d_token)
        self.out_proj  = nn.Linear(d_token, d_token)
        self.norm2     = nn.LayerNorm(d_token)
        self.ffn       = nn.Sequential(
            nn.Linear(d_token, ffn_d_hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden, d_token),
        )
        self.drop = nn.Dropout(residual_dropout)

    def forward(self, x):
        B, T, D = x.shape
        head_dim = D // self.n_heads
        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, head_dim).transpose(1, 2)
        drop = self.attn_drop if self.training else 0.0
        h = F.scaled_dot_product_attention(Q, K, V, dropout_p=drop)
        h = h.transpose(1, 2).contiguous().view(B, T, D)
        h = self.out_proj(h)
        x = x + self.drop(h)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_num_features, d_token, n_blocks, n_heads,
                 ffn_d_hidden, attn_dropout, ffn_dropout, residual_dropout, n_classes):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_token, n_heads, ffn_d_hidden, attn_dropout, ffn_dropout, residual_dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls    = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([tokens, cls], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens[:, -1]))


def make_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if any(nd in name for nd in ['bias', 'norm', 'cls_token']):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': wd},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr,
    )


# --- Preprocessing ---
scaler = StandardScaler()
scaler.fit(X_train_full)

def preprocess(X):
    return torch.tensor(scaler.transform(X), dtype=torch.float32)

def make_loader(X_t, y_t, batch_size, shuffle=True):
    ds = TensorDataset(X_t, torch.tensor(y_t, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=2, persistent_workers=True)

def make_infer_loader(X_t, batch_size=1024):
    return DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False,
                      pin_memory=True, num_workers=2, persistent_workers=True)

# --- Train / eval helpers ---
def train_epoch(model, optimizer, loader, grad_scaler):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        with autocast('cuda'):
            loss = F.cross_entropy(model(X_batch), y_batch)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_bal_acc(model, loader):
    model.eval()
    preds, labels = [], []
    for X_batch, y_batch in loader:
        with autocast('cuda'):
            preds.append(model(X_batch.to(DEVICE)).argmax(1).cpu().numpy())
        labels.append(y_batch.numpy())
    return balanced_accuracy_score(np.concatenate(labels), np.concatenate(preds))

@torch.no_grad()
def predict_proba(model, X_np, batch_size=1024):
    model.eval()
    loader = make_infer_loader(preprocess(X_np), batch_size)
    probs  = []
    for (X_batch,) in loader:
        with autocast('cuda'):
            probs.append(F.softmax(model(X_batch.to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)

# %% [markdown]
# ## Cell 4 — Optuna tuning (50K subsample, 30 trials)

# %%
TUNE_N      = 50_000   # 2× faster than 100K; enough signal for HPO
TUNE_EPOCHS = 35       # enough for mid/high LRs to converge; low-LR configs failing here is valid signal
PATIENCE    = 4        # in eval intervals (eval every 2 epochs → effective patience = 8 epochs)
BATCH_SIZE  = 512      # reduced from 1024 — d_token=256/n_blocks=4 OOM'd at batch=1024 on T4

rng = np.random.default_rng(42)
idx = rng.choice(len(X_train_full), TUNE_N, replace=False)
X_sub, y_sub = X_train_full[idx], y_train_full[idx]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, val_idx = next(sss.split(X_sub, y_sub))

X_tr_t  = preprocess(X_sub[tr_idx]);  y_tr  = y_sub[tr_idx]
X_val_t = preprocess(X_sub[val_idx]); y_val = y_sub[val_idx]

tune_train_loader = make_loader(X_tr_t, y_tr, BATCH_SIZE)
tune_val_loader   = make_loader(X_val_t, y_val, BATCH_SIZE * 2, shuffle=False)

def objective(trial):
    torch.cuda.empty_cache()
    d_token   = trial.suggest_categorical('d_token', [128, 192, 256])
    n_blocks  = trial.suggest_int('n_blocks', 2, 4)
    n_heads   = trial.suggest_categorical('n_heads', [4, 8])
    attn_drop = trial.suggest_float('attn_drop', 0.0, 0.3)
    ffn_drop  = trial.suggest_float('ffn_drop',  0.0, 0.3)
    ffn_mult  = trial.suggest_categorical('ffn_mult', [4/3, 2, 8/3])
    lr        = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    wd        = trial.suggest_float('wd', 1e-6, 1e-3, log=True)

    model = FTTransformer(
        n_num_features=X_tr_t.shape[1],
        d_token=d_token,
        n_blocks=n_blocks,
        n_heads=n_heads,
        ffn_d_hidden=int(d_token * ffn_mult),
        attn_dropout=attn_drop,
        ffn_dropout=ffn_drop,
        residual_dropout=0.0,
        n_classes=7,
    ).to(DEVICE)

    optimizer  = make_optimizer(model, lr, wd)
    grad_scaler = GradScaler('cuda')   # fresh scaler per trial — avoids state bleed on NaN

    best_acc, no_improve = 0.0, 0
    for epoch in range(TUNE_EPOCHS):
        train_epoch(model, optimizer, tune_train_loader, grad_scaler)
        if epoch % 2 == 1 or epoch == TUNE_EPOCHS - 1:   # eval every 2 epochs
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

    return best_acc

def checkpoint_callback(study, trial):
    """Save best params to Drive after every completed trial — survives session crashes."""
    if trial.value is None:   # pruned trial, no value to save
        return
    checkpoint = {
        'completed_trials': len([t for t in study.trials if t.value is not None]),
        'best_trial':  study.best_trial.number,
        'best_value':  study.best_value,
        'best_params': study.best_params,
    }
    path = os.path.join(SAVE_DIR, 'ft_optuna_checkpoint.json')
    with open(path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    if study.best_trial.number == trial.number:
        print(f'  ★ New best {trial.value:.4f} — checkpoint saved to Drive')

# Persistent SQLite storage — saves full study state to Drive so sessions can be resumed.
# On first run: creates a new study. On subsequent runs: loads existing trials and continues.
STUDY_DB     = f'sqlite:///{SAVE_DIR}ft_optuna_study.db'
TOTAL_TRIALS = 30

study = optuna.create_study(
    storage=STUDY_DB,
    study_name='ft_transformer',
    load_if_exists=True,
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    sampler=optuna.samplers.TPESampler(seed=42),
)

n_done = len([t for t in study.trials if t.value is not None])
n_remaining = max(0, TOTAL_TRIALS - n_done)
print(f'Completed trials so far: {n_done} / {TOTAL_TRIALS}  →  running {n_remaining} more')

study.optimize(objective, n_trials=n_remaining, show_progress_bar=True,
               callbacks=[checkpoint_callback])

print('\nBest val balanced accuracy:', study.best_value)
print('Best params:', study.best_params)

# %% [markdown]
# ## Cell 5 — Train final model on full 500K

# %%
FINAL_EPOCHS   = 200
PATIENCE_FINAL = 20
p = study.best_params

X_full_t = preprocess(X_train_full)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
tr_f, val_f = next(sss2.split(X_full_t.numpy(), y_train_full))

final_train_loader = make_loader(X_full_t[tr_f], y_train_full[tr_f], BATCH_SIZE)
final_val_loader   = make_loader(X_full_t[val_f], y_train_full[val_f], BATCH_SIZE * 2, shuffle=False)

model = FTTransformer(
    n_num_features=X_full_t.shape[1],
    d_token=p['d_token'],
    n_blocks=p['n_blocks'],
    n_heads=p['n_heads'],
    ffn_d_hidden=int(p['d_token'] * p['ffn_mult']),
    attn_dropout=p['attn_drop'],
    ffn_dropout=p['ffn_drop'],
    residual_dropout=0.0,
    n_classes=7,
).to(DEVICE)

model = torch.compile(model)

optimizer   = make_optimizer(model, p['lr'], p['wd'])
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)
grad_scaler = GradScaler('cuda')

best_val, no_improve, best_state = 0.0, 0, None
training_log = []

for epoch in range(FINAL_EPOCHS):
    loss = train_epoch(model, optimizer, final_train_loader, grad_scaler)
    acc  = eval_bal_acc(model, final_val_loader)
    scheduler.step()

    is_best = acc > best_val
    training_log.append({'epoch': epoch + 1, 'loss': round(loss, 6), 'val_bal_acc': round(acc, 6), 'is_best': is_best})
    with open(os.path.join(SAVE_DIR, 'ft_training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f'Epoch {epoch+1:3d}  loss={loss:.4f}  val_bal_acc={acc:.4f}')

    if is_best:
        best_val, no_improve = acc, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, os.path.join(SAVE_DIR, 'ft_best_model.pt'))
        print(f'  ★ New best {best_val:.4f} — weights saved to Drive')
    else:
        no_improve += 1
        if no_improve >= PATIENCE_FINAL:
            print('Early stopping')
            break

model.load_state_dict(best_state)
print(f'\nFinal val balanced accuracy: {best_val:.4f}')

# %% [markdown]
# ## Cell 6 — Save soft probabilities to Drive (for ensembling)

# %%
ft_probs = predict_proba(model, X_test)
print('FT-Transformer probs shape:', ft_probs.shape)

np.save(os.path.join(SAVE_DIR, 'ft_transformer_probs.npy'), ft_probs)
print('Saved ft_transformer_probs.npy to Drive')

# %% [markdown]
# ## Cell 7 — (Optional) Standalone predictions without ensembling

# %%
hard_preds = ft_probs.argmax(axis=1)
np.savez(
    os.path.join(SAVE_DIR, '1385_ft_only_predictions.npz'),
    test_predictions=hard_preds,
)
print('FT-only val balanced acc:', best_val)
print('Standalone predictions saved.')
