"""
Ensemble CatBoost soft probabilities with FT-Transformer probabilities.
Run locally after downloading ft_transformer_probs.npy from Colab/Drive.

Usage:
    conda run -n AML python ensemble_catboost_ft.py
"""

import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
import json

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'competition_train.npz'
FT_PROBS_PATH = BASE_DIR / 'ft_transformer_probs.npy'   # downloaded from Colab
OUTPUT_DIR = BASE_DIR / 'competition_output'
STUDENT_ID = '1385'

data = np.load(DATA_PATH)
X_train = data['X_train']
y_train = data['y_train']
X_test  = data['X_test']

# --- CatBoost soft probabilities ---
print('Training CatBoost for soft probabilities...')
with open(BASE_DIR / 'best_params.json') as f:
    best = json.load(f)['best_params']

cb = CatBoostClassifier(
    iterations=3000,
    learning_rate=best['learning_rate'],
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    task_type='CPU',
    verbose=200,
)
cb.fit(X_train, y_train)
cb_probs = cb.predict_proba(X_test)   # (200000, 7)
print('CatBoost probs shape:', cb_probs.shape)

# --- FT-Transformer soft probabilities ---
ft_probs = np.load(FT_PROBS_PATH)     # (200000, 7)
print('FT-Transformer probs shape:', ft_probs.shape)

# --- Sweep ensemble weights and pick best on a val split ---
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
tr_idx, val_idx = next(sss.split(X_train, y_train))

cb_val  = cb.predict_proba(X_train[val_idx])
# FT probs on val — you'd need to also save those from Colab; if not available, use equal weight
best_acc, best_w = 0.0, 0.5
for w in np.arange(0.3, 0.8, 0.05):
    # w = weight on CatBoost; (1-w) on FT-Transformer
    # For val we only have CB probs here; skip sweep if FT val probs not saved
    pass

# Default: equal weighting (tune manually if you save FT val probs too)
CB_WEIGHT = 0.5
FT_WEIGHT = 1.0 - CB_WEIGHT

ensemble_probs = CB_WEIGHT * cb_probs + FT_WEIGHT * ft_probs
final_preds = ensemble_probs.argmax(axis=1)

OUTPUT_DIR.mkdir(exist_ok=True)
out_path = OUTPUT_DIR / f'{STUDENT_ID}_competition_predictions.npz'
np.savez(out_path, test_predictions=final_preds)
print(f'\nEnsemble predictions saved → {out_path}')
print(f'Weights: CatBoost={CB_WEIGHT}, FT-Transformer={FT_WEIGHT}')
