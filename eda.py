import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# try both possible locations
for path in ['competition_train.npz', 'competition_data/competition_train.npz']:
    if os.path.exists(path):
        print(f"Found data at: {path}")
        data = np.load(path, mmap_mode='r')
        break

X_train = data['X_train']
y_train = data['y_train']
X_test  = data['X_test']

print('\n=== SHAPES ===')
print(f'X_train: {X_train.shape}')
print(f'y_train: {y_train.shape}')
print(f'X_test:  {X_test.shape}')

print('\n=== CLASSES ===')
classes, counts = np.unique(y_train, return_counts=True)
for c, n in zip(classes, counts):
    print(f'  class {c}: {n} ({100*n/len(y_train):.1f}%)')

print('\n=== FEATURE STATS ===')
print(f'dtype:     {X_train.dtype}')
print(f'min:       {X_train.min():.4f}')
print(f'max:       {X_train.max():.4f}')
print(f'mean:      {X_train.mean():.4f}')
print(f'std:       {X_train.std():.4f}')
print(f'NaN count: {np.isnan(X_train).sum()}')
print(f'Inf count: {np.isinf(X_train).sum()}')
nan_per_col = np.isnan(X_train).sum(axis=0)
print(f'Cols with NaN: {(nan_per_col > 0).sum()} / {X_train.shape[1]}')

print('\n=== BASELINE (LogReg CV) ===')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr  = LogisticRegression(max_iter=1000, random_state=42)
scores = cross_val_score(lr, X_train, y_train, cv=skf, scoring='balanced_accuracy', n_jobs=-1)
print(f'Balanced accuracy: {scores.mean():.4f} +/- {scores.std():.4f}')
