import os
import json
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import lightgbm as lgb

print("Loading data...")
data = np.load('competition_train.npz', mmap_mode='r')
X_full = np.array(data['X_train'])
y_full = np.array(data['y_train'])
X_test = np.array(data['X_test'])
N_CLASSES = len(np.unique(y_full))
print(f"Full train: {X_full.shape}  |  Test: {X_test.shape}  |  Classes: {N_CLASSES}")

with open('best_params.json') as f:
    saved = json.load(f)
best_params = saved['best_params']
print(f"Loaded best params (trial {saved['best_trial']}, CV score {saved['best_value']:.4f})")

print("\nPre-building full training dataset (500K)...")
t0 = time.time()
dtrain_full = lgb.Dataset(X_full, label=y_full, free_raw_data=False, params={'feature_pre_filter': False})
dtrain_full.construct()
print(f"Dataset ready in {(time.time()-t0)/60:.1f} min\n")

params = {
    **best_params,
    'objective': 'multiclass',
    'num_class':  N_CLASSES,
    'metric':     'multi_logloss',
    'verbose':    -1,
    'n_jobs':     -1,
}

N_SEEDS = 5
print(f"Training ensemble ({N_SEEDS} seeds, 3000 trees)...")
all_probs = []
for seed in range(N_SEEDS):
    t_seed = time.time()
    booster = lgb.train(
        {**params, 'seed': seed},
        dtrain_full,
        num_boost_round=3000,
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    all_probs.append(booster.predict(X_test))
    print(f"  seed {seed} done in {(time.time()-t_seed)/60:.1f} min")

avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
final_preds = avg_probs.argmax(axis=1).astype(int)

student_id = '1385'
os.makedirs('competition_output', exist_ok=True)

# Soft probabilities — used by ensemble_lgbm_ft.py when blending with FT-Transformer
np.save('lgbm_probs.npy', avg_probs)
print(f"Saved: lgbm_probs.npy  (shape: {avg_probs.shape})")

# Standalone submission — valid on its own without FT-Transformer
out_path = f'competition_output/{student_id}_competition_predictions.npz'
np.savez(out_path, test_predictions=final_preds)

total = (time.time() - t0) / 3600
print(f"Saved: {out_path}  (shape: {final_preds.shape})")
print(f"Total runtime: {total:.2f} hours")
