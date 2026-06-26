import numpy as np
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score

# realistic size — similar to what Optuna will use
X = np.random.rand(50_000, 500).astype(np.float32)
y = np.random.randint(0, 7, 50_000)

X_tr, X_val = X[:40000], X[40000:]
y_tr, y_val = y[:40000], y[40000:]

print("Testing device='cuda' on 50K x 500 dataset...")
try:
    m = lgb.LGBMClassifier(device='cuda', n_estimators=50, num_leaves=127,
                            verbose=-1, n_jobs=-1, random_state=42)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.log_evaluation(period=10)])
    score = balanced_accuracy_score(y_val, m.predict(X_val))
    print(f'LightGBM CUDA: OK  — balanced_acc={score:.4f}')
except Exception as e:
    print(f'LightGBM CUDA: FAILED — {e}')
