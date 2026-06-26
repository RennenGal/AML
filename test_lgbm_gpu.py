import numpy as np
import lightgbm as lgb

X = np.random.rand(1000, 20).astype(np.float32)
y = np.random.randint(0, 2, 1000)

for device in ['cuda', 'gpu']:
    try:
        m = lgb.LGBMClassifier(device=device, n_estimators=10, verbose=-1)
        m.fit(X, y)
        print(f'LightGBM {device}: OK')
    except Exception as e:
        print(f'LightGBM {device}: FAILED — {e}')
