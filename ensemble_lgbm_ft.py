"""
Ensemble LightGBM soft probabilities with FT-Transformer probabilities.
Run locally after downloading ft_transformer_probs.npy from Colab/Drive.

Usage:
    conda run -n AML python ensemble_lgbm_ft.py
"""

import numpy as np
from pathlib import Path
import lightgbm as lgb
import json
import time

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'competition_train.npz'
FT_PROBS_PATH = BASE_DIR / 'ft_transformer_probs.npy'
OUTPUT_DIR = BASE_DIR / 'competition_output'
STUDENT_ID = '1385'

data = np.load(DATA_PATH)
X_train = data['X_train'].astype(np.float32)
y_train = data['y_train']
X_test  = data['X_test'].astype(np.float32)
N_CLASSES = len(np.unique(y_train))

with open(BASE_DIR / 'best_params.json') as f:
    saved = json.load(f)
best = saved['best_params']
print(f"Loaded best params (trial {saved['best_trial']}, CV score {saved['best_value']:.4f})")

params = {
    **best,
    'objective': 'multiclass',
    'num_class':  N_CLASSES,
    'metric':     'multi_logloss',
    'verbose':    -1,
    'n_jobs':     -1,
}

print("\nPre-building full training dataset (500K)...")
t0 = time.time()
dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False, params={'feature_pre_filter': False})
dtrain.construct()
print(f"Dataset ready in {(time.time()-t0)/60:.1f} min\n")

print("Training LightGBM (5 seeds, 3000 trees)...")
all_lgbm_probs = []
for seed in range(5):
    t_seed = time.time()
    booster = lgb.train(
        {**params, 'seed': seed},
        dtrain,
        num_boost_round=3000,
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    all_lgbm_probs.append(booster.predict(X_test))
    print(f"  seed {seed} done in {(time.time()-t_seed)/60:.1f} min")

lgbm_probs = np.mean(np.stack(all_lgbm_probs, axis=0), axis=0)  # (200000, 7)
print(f"LightGBM probs shape: {lgbm_probs.shape}")

ft_probs = np.load(FT_PROBS_PATH)
print(f"FT-Transformer probs shape: {ft_probs.shape}")

# Adjust LGBM_WEIGHT based on relative val scores after running Colab
# e.g. if FT val acc << 0.946, increase LGBM_WEIGHT to 0.7
LGBM_WEIGHT = 0.5
FT_WEIGHT   = 1.0 - LGBM_WEIGHT

ensemble_probs = LGBM_WEIGHT * lgbm_probs + FT_WEIGHT * ft_probs
final_preds = ensemble_probs.argmax(axis=1).astype(int)

OUTPUT_DIR.mkdir(exist_ok=True)
out_path = OUTPUT_DIR / f'{STUDENT_ID}_competition_predictions.npz'
np.savez(out_path, test_predictions=final_preds)
print(f"\nEnsemble predictions saved → {out_path}")
print(f"Weights: LightGBM={LGBM_WEIGHT}, FT-Transformer={FT_WEIGHT}")
print(f"Total runtime: {(time.time()-t0)/3600:.2f} hours")
