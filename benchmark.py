"""Quick model benchmark on 50K subsample to pick the best model family."""
import os
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
data = np.load('competition_train.npz', mmap_mode='r')
X_train_full = data['X_train']
y_train_full = data['y_train']

# Subsample 50K for speed
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train_full), size=50_000, replace=False)
X = np.array(X_train_full[idx])
y = np.array(y_train_full[idx])

print(f"Subsample shape: {X.shape}, classes: {np.unique(y)}")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

models = {
    'LightGBM': LGBMClassifier(
        n_estimators=500, learning_rate=0.1, num_leaves=127,
        n_jobs=-1, random_state=42, verbose=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=6,
        n_jobs=-1, random_state=42, verbosity=0,
        tree_method='hist', device='cpu'
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500, learning_rate=0.1, depth=6,
        random_seed=42, verbose=0, thread_count=-1
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_features='sqrt',
        n_jobs=-1, random_state=42
    ),
}

results = {}
for name, model in models.items():
    print(f"\n--- {name} ---")
    t0 = time.time()
    scores = cross_val_score(model, X, y, cv=skf, scoring='balanced_accuracy', n_jobs=1)
    elapsed = time.time() - t0
    mean, std = scores.mean(), scores.std()
    results[name] = mean
    print(f"  Balanced acc: {mean:.4f} ± {std:.4f}  ({elapsed:.1f}s)")

print("\n=== RANKING ===")
for name, score in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name}: {score:.4f}")

best = max(results, key=results.get)
print(f"\nBest model family: {best} ({results[best]:.4f})")
