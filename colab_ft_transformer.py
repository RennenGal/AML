# %% [markdown]
# # FT-Transformer Tuning — Competition
# Run on Colab with a T4 GPU (Runtime → Change runtime type → T4 GPU)

# %% [markdown]
# ## Cell 1 — Install dependencies

# %%
# !pip install rtdl optuna --quiet

# %% [markdown]
# ## Cell 2 — Mount Google Drive and load data

# %%
from google.colab import drive
drive.mount('/content/drive')

import numpy as np

# Adjust path to wherever you uploaded competition_train.npz in your Drive
DATA_PATH = '/content/drive/MyDrive/competition_train.npz'

data = np.load(DATA_PATH)
X_train_full = data['X_train'].astype(np.float32)   # (500000, 500)
y_train_full = data['y_train'].astype(np.int64)     # (500000,)
X_test       = data['X_test'].astype(np.float32)    # (200000, 500)

print('X_train:', X_train_full.shape, '  X_test:', X_test.shape)
print('Classes:', np.unique(y_train_full))

# %% [markdown]
# ## Cell 3 — Preprocessing + helper utilities

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score
import rtdl
import optuna
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# Fit scaler on full training set
scaler = StandardScaler()
scaler.fit(X_train_full)

def preprocess(X):
    return torch.tensor(scaler.transform(X), dtype=torch.float32)

def make_loader(X_t, y_t, batch_size, shuffle=True):
    ds = TensorDataset(X_t, torch.tensor(y_t, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=2)

def train_epoch(model, optimizer, loader):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch, None)   # None = no categorical features
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_bal_acc(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(DEVICE), None)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y_batch.numpy())
    return balanced_accuracy_score(
        np.concatenate(all_labels),
        np.concatenate(all_preds)
    )

# %% [markdown]
# ## Cell 4 — Optuna tuning (100K subsample, 15 trials)

# %%
# --- subsample for fast tuning ---
TUNE_N      = 100_000
TUNE_EPOCHS = 40
PATIENCE    = 8
BATCH_SIZE  = 1024

rng = np.random.default_rng(42)
idx = rng.choice(len(X_train_full), TUNE_N, replace=False)
X_sub = X_train_full[idx]
y_sub = y_train_full[idx]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, val_idx = next(sss.split(X_sub, y_sub))

X_tr_t  = preprocess(X_sub[tr_idx])
X_val_t = preprocess(X_sub[val_idx])
y_tr    = y_sub[tr_idx]
y_val   = y_sub[val_idx]

train_loader = make_loader(X_tr_t, y_tr, BATCH_SIZE)
val_loader   = make_loader(X_val_t, y_val, BATCH_SIZE * 2, shuffle=False)

def objective(trial):
    d_token   = trial.suggest_categorical('d_token', [128, 192, 256])
    n_blocks  = trial.suggest_int('n_blocks', 2, 4)
    n_heads   = trial.suggest_categorical('n_heads', [4, 8])
    attn_drop = trial.suggest_float('attn_drop', 0.0, 0.3)
    ffn_drop  = trial.suggest_float('ffn_drop', 0.0, 0.3)
    ffn_mult  = trial.suggest_categorical('ffn_mult', [4/3, 2, 8/3])
    lr        = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    wd        = trial.suggest_float('wd', 1e-6, 1e-3, log=True)

    model = rtdl.FTTransformer.make_baseline(
        n_num_features=X_tr_t.shape[1],
        cat_cardinalities=None,
        d_token=d_token,
        n_blocks=n_blocks,
        attention_n_heads=n_heads,
        attention_dropout=attn_drop,
        ffn_d_hidden=int(d_token * ffn_mult),
        ffn_dropout=ffn_drop,
        residual_dropout=0.0,
        last_layer_query_idx=[-1],
        d_out=7,
    ).to(DEVICE)

    optimizer = model.make_default_optimizer()
    # Override lr and wd
    for g in optimizer.param_groups:
        g['lr'] = lr
        if g['weight_decay'] > 0:
            g['weight_decay'] = wd

    best_acc, no_improve = 0.0, 0
    for epoch in range(TUNE_EPOCHS):
        train_epoch(model, optimizer, train_loader)
        acc = eval_bal_acc(model, val_loader)
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

study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=15, show_progress_bar=True)

print('\nBest val balanced accuracy:', study.best_value)
print('Best params:', study.best_params)

# %% [markdown]
# ## Cell 5 — Train final model on full 500K

# %%
FINAL_EPOCHS = 100
PATIENCE_FINAL = 10
p = study.best_params

X_full_t = preprocess(X_train_full)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
tr_f, val_f = next(sss2.split(X_full_t.numpy(), y_train_full))

final_train_loader = make_loader(X_full_t[tr_f], y_train_full[tr_f], BATCH_SIZE)
final_val_loader   = make_loader(X_full_t[val_f], y_train_full[val_f], BATCH_SIZE * 2, shuffle=False)

model = rtdl.FTTransformer.make_baseline(
    n_num_features=X_full_t.shape[1],
    cat_cardinalities=None,
    d_token=p['d_token'],
    n_blocks=p['n_blocks'],
    attention_n_heads=p['n_heads'],
    attention_dropout=p['attn_drop'],
    ffn_d_hidden=int(p['d_token'] * p['ffn_mult']),
    ffn_dropout=p['ffn_drop'],
    residual_dropout=0.0,
    last_layer_query_idx=[-1],
    d_out=7,
).to(DEVICE)

optimizer = model.make_default_optimizer()
for g in optimizer.param_groups:
    g['lr'] = p['lr']
    if g['weight_decay'] > 0:
        g['weight_decay'] = p['wd']

best_val, no_improve, best_state = 0.0, 0, None
for epoch in range(FINAL_EPOCHS):
    loss = train_epoch(model, optimizer, final_train_loader)
    acc  = eval_bal_acc(model, final_val_loader)
    print(f'Epoch {epoch+1:3d}  loss={loss:.4f}  val_bal_acc={acc:.4f}')
    if acc > best_val:
        best_val, no_improve = acc, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= PATIENCE_FINAL:
            print('Early stopping')
            break

model.load_state_dict(best_state)
print(f'\nFinal val balanced accuracy: {best_val:.4f}')

# %% [markdown]
# ## Cell 6 — Get soft probabilities on X_test (for ensembling with CatBoost)

# %%
@torch.no_grad()
def predict_proba(model, X_np, batch_size=2048):
    model.eval()
    X_t = preprocess(X_np)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    probs = []
    for (X_batch,) in loader:
        logits = model(X_batch.to(DEVICE), None)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)  # (N, 7)

ft_probs = predict_proba(model, X_test)   # shape (200000, 7)
print('FT-Transformer probs shape:', ft_probs.shape)

# Save to Drive so you can download and ensemble on your Mac
SAVE_DIR = '/content/drive/MyDrive/'
np.save(os.path.join(SAVE_DIR, 'ft_transformer_probs.npy'), ft_probs)
np.save(os.path.join(SAVE_DIR, 'ft_transformer_best_params.npy'), study.best_params)
print('Saved ft_transformer_probs.npy to Drive')

# %% [markdown]
# ## Cell 7 — (Optional) Standalone predictions if not ensembling

# %%
hard_preds = ft_probs.argmax(axis=1)
np.savez(
    os.path.join(SAVE_DIR, '1385_ft_only_predictions.npz'),
    test_predictions=hard_preds
)
print('FT-only predictions saved. Balanced acc on val:', best_val)
