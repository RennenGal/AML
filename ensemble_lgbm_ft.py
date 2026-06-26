"""
Blend LightGBM and FT-Transformer soft probabilities into a final submission.

Requires (download both from Colab/Drive before running):
  - lgbm_probs.npy          produced by retrain_ensemble.py
  - ft_transformer_probs.npy  produced by colab_ft_transformer.ipynb

Usage:
    conda run -n AML python ensemble_lgbm_ft.py
"""

import numpy as np
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'competition_output'
STUDENT_ID = '1385'

# Adjust this weight based on each model's val balanced accuracy:
#   LGBM_WEIGHT = 0.5  →  equal blend (good default)
#   LGBM_WEIGHT = 0.7  →  trust LightGBM more if FT val acc is much lower
LGBM_WEIGHT = 0.5

lgbm_probs = np.load(BASE_DIR / 'lgbm_probs.npy')
ft_probs   = np.load(BASE_DIR / 'ft_transformer_probs.npy')

assert lgbm_probs.shape == ft_probs.shape, \
    f"Shape mismatch: lgbm {lgbm_probs.shape} vs ft {ft_probs.shape}"

print(f'LightGBM probs:      {lgbm_probs.shape}')
print(f'FT-Transformer probs:{ft_probs.shape}')
print(f'Weights: LightGBM={LGBM_WEIGHT}, FT-Transformer={1-LGBM_WEIGHT}')

ensemble_probs = LGBM_WEIGHT * lgbm_probs + (1 - LGBM_WEIGHT) * ft_probs
final_preds    = ensemble_probs.argmax(axis=1).astype(int)

OUTPUT_DIR.mkdir(exist_ok=True)
out_path = OUTPUT_DIR / f'{STUDENT_ID}_competition_predictions.npz'
np.savez(out_path, test_predictions=final_preds)
print(f'\nEnsemble predictions saved → {out_path}')
