"""
Optuna tuning for LightGBM on a 100K subsample with early stopping.
After tuning, trains final ensemble on full 500K and saves predictions.

Speed optimisation: LightGBM Dataset objects are pre-built once so histogram
construction (bin boundaries + bin indices) is not repeated every trial.
"""
import os
import json
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
data = np.load('competition_train.npz', mmap_mode='r')
X_full = np.array(data['X_train'])
y_full = np.array(data['y_train'])
X_test = np.array(data['X_test'])
N_CLASSES = len(np.unique(y_full))
print(f"Full train: {X_full.shape}  |  Test: {X_test.shape}  |  Classes: {N_CLASSES}")

# 100K subsample for tuning
rng = np.random.default_rng(42)
idx = rng.choice(len(X_full), size=100_000, replace=False)
X_tune = X_full[idx]
y_tune = y_full[idx]
print(f"Tune subsample: {X_tune.shape}")

# ── Pre-build fold datasets (histogram construction done once, not per trial) ─
print("\nPre-building fold datasets...")
t_prep = time.time()
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_data = []
for tr_idx, val_idx in skf.split(X_tune, y_tune):
    X_tr, y_tr = X_tune[tr_idx], y_tune[tr_idx]
    X_val, y_val = X_tune[val_idx], y_tune[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False, params={'feature_pre_filter': False})
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False, params={'feature_pre_filter': False})
    dtrain.construct()
    dval.construct()
    fold_data.append((dtrain, dval, X_val, y_val))
print(f"Fold datasets ready in {time.time()-t_prep:.1f}s\n")

# ── Pre-build full training dataset for final model ───────────────────────────
print("Pre-building full training dataset (500K)...")
t_prep2 = time.time()
dtrain_full = lgb.Dataset(X_full, label=y_full, free_raw_data=False, params={'feature_pre_filter': False})
dtrain_full.construct()
print(f"Full dataset ready in {(time.time()-t_prep2)/60:.1f} min\n")

# ── Objective ─────────────────────────────────────────────────────────────────
def objective(trial):
    params = dict(
        objective         = 'multiclass',
        num_class         = N_CLASSES,
        metric            = 'multi_logloss',
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        num_leaves        = trial.suggest_int('num_leaves', 31, 511),
        min_child_samples = trial.suggest_int('min_child_samples', 5, 150),
        feature_fraction  = trial.suggest_float('feature_fraction', 0.4, 1.0),
        bagging_fraction  = trial.suggest_float('bagging_fraction', 0.4, 1.0),
        bagging_freq      = 1,
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        verbose           = -1,
        n_jobs            = -1,
        seed              = 42,
    )

    fold_scores = []
    for fold, (dtrain, dval, X_val, y_val) in enumerate(fold_data):
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        y_pred = booster.predict(X_val).argmax(axis=1)
        score  = balanced_accuracy_score(y_val, y_pred)
        fold_scores.append(score)

        trial.report(float(np.mean(fold_scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


# ── Study ─────────────────────────────────────────────────────────────────────
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)

def save_best_callback(study, trial):
    try:
        if study.best_trial.number == trial.number:
            with open('best_params.json', 'w') as f:
                json.dump({
                    'best_trial':  trial.number,
                    'best_value':  study.best_value,
                    'best_params': study.best_params,
                }, f, indent=2)
    except ValueError:
        pass  # no completed trials yet

N_TRIALS = 100
print(f"Starting Optuna tuning ({N_TRIALS} trials)...")
t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, callbacks=[save_best_callback])
elapsed = time.time() - t0

print(f"\nTuning done in {elapsed/3600:.2f} hours")
print(f"Best CV score : {study.best_value:.4f}")
print(f"Best params   : {study.best_params}")

# ── Retrain on full 500K (5 seeds, 3000 trees, shared pre-built dataset) ──────
print("\nRetraining on full 500K (5 seeds, 3000 trees)...")
best = study.best_params.copy()
best.update({
    'objective':          'multiclass',
    'num_class':           N_CLASSES,
    'metric':             'multi_logloss',
    'verbose':            -1,
    'n_jobs':             -1,
})

all_probs = []
for seed in range(5):
    t_seed = time.time()
    booster = lgb.train(
        {**best, 'seed': seed},
        dtrain_full,
        num_boost_round=3000,
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    all_probs.append(booster.predict(X_test))  # (n_test, n_classes)
    print(f"  seed {seed} done in {(time.time()-t_seed)/60:.1f} min")

# Soft voting: average probabilities across seeds
avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)  # (n_test, n_classes)
final_preds = avg_probs.argmax(axis=1).astype(int)

# ── Save submission ────────────────────────────────────────────────────────────
student_id = '1385'
os.makedirs('competition_output', exist_ok=True)
out_path = f'competition_output/{student_id}_competition_predictions.npz'
np.savez(out_path, test_predictions=final_preds)

total = (time.time() - t0) / 3600
print(f"\nSaved: {out_path}  (shape: {final_preds.shape})")
print(f"Submit file: {student_id}_competition_predictions.npz")
print(f"Total runtime: {total:.2f} hours")
