# Competition Progress Log

## Environment

- **Conda env:** `AML` (Anaconda at `%USERPROFILE%\anaconda3`)
- **Run scripts:** `conda run -n AML python <script.py>`
- **Key packages:** lightgbm 4.6.0, xgboost 3.2.0, catboost 1.2.8, optuna 4.6.0, sklearn 1.7.1

---

## Overview
- **Metric:** Balanced accuracy (= regular accuracy here since classes are perfectly balanced)
- **Grading:** Rank-based — `score = max(60, 100 * (1 - (rank-1)/(n-1)))`. Rank 1 = 100.
- **Deadline:** 07.03.2026
- **Submission file:** `competition_output/{student_id}_competition_predictions.npz`
- **Student ID:** set `student_id = '1385'` in competition.py (or group member's last 4 digits)

---

## Dataset
| Property | Value |
|---|---|
| X_train shape | (500,000, 500) |
| X_test shape | (200,000, 500) |
| Classes | 7, perfectly balanced (~14.3% each) |
| dtype | float32 |
| NaN / Inf | None |
| Feature range | ~[-470, +479], std ≈ 27.5 |

**Key insight:** Classes are perfectly balanced → balanced accuracy = regular accuracy. SMOTE and class_weight tricks are irrelevant.

---

## Baseline
| Model | Balanced Accuracy |
|---|---|
| LogisticRegression (5-fold CV) | 0.4867 ± 0.0011 |

---

## Experiments

### Round 1 — EDA (2026-06-15)
- Ran EDA: shapes, class distribution, feature stats, baseline LogReg
- Confirmed perfectly balanced classes, high-dimensional (500 features), large scale (500K samples)
- Low LogReg baseline (0.487) suggests highly non-linear structure

### Round 2 — Model Benchmark on 50K subsample (2026-06-15)
- Tested LightGBM, XGBoost, CatBoost, RandomForest (3-fold CV, 50K subsample)

| Model | Balanced Accuracy | Time |
|---|---|---|
| **LightGBM** | **0.8452 ± 0.0030** | 1005s |
| XGBoost | 0.7758 ± 0.0033 | 986s |
| CatBoost | 0.6841 ± 0.0038 | 664s |
| RandomForest | 0.4729 ± 0.0001 | 253s |

- **LightGBM is the clear winner** — 7 points ahead of XGBoost
- RandomForest essentially failed (≈ LogReg baseline) — dataset requires deep trees
- All tuning effort going into LightGBM

---

## TODO / Next Steps
- [x] Quick model comparison: LightGBM, XGBoost, CatBoost, RandomForest
- [x] Optuna tuning of LightGBM → 0.9464
- [ ] Run FT-Transformer on Colab T4 (see Round 3 section)
- [ ] Ensemble LightGBM + FT-Transformer soft probabilities
- [ ] Tune ensemble weight based on FT val score
- [ ] Final submission

---

## Best Result So Far
| Model | CV Score | Notes |
|---|---|---|
| LogReg baseline | 0.4867 | Starting point |
| LightGBM (default params, 50K) | 0.8452 | 50K subsample, 3-fold CV |
| **LightGBM (Optuna, 100 trials)** | **0.9464** | Full 500K, best so far |

**Current best submission:** LightGBM 0.9464

---

## Round 3 — FT-Transformer via Google Colab

### Plan
Beat 0.946 by ensembling LightGBM with an FT-Transformer neural network.
The two models make different errors, so blending soft probabilities is the main lever.

### Why FT-Transformer?
- State-of-the-art tabular NN (Gorishniy et al. 2021, "Revisiting Deep Learning Models for Tabular Data")
- Each of the 500 features becomes a token; attention captures complex feature interactions
- 500K samples is large enough for the Transformer layers to learn without overfitting
- Low LogReg baseline (0.487) confirms deep non-linear structure → good fit

### Hardware decision
| Machine | Estimate for full tuning sweep | Decision |
|---|---|---|
| M1 Max (MPS, no Tensor Cores) | ~15–30 hours | Too slow |
| Quadro P1000 (CUDA, 4 GB VRAM) | ~20–40 hours | Too slow + VRAM tight |
| **Colab T4 (free)** | **~3–6 hours** | ✅ Use this |

### Files
| File | Purpose |
|---|---|
| `colab_ft_transformer.ipynb` | Upload directly to Colab, run all cells |
| `colab_ft_transformer.py` | Source version (same content) |
| `ensemble_lgbm_ft.py` | Blend LightGBM + FT-Transformer probs locally |

### Colab Notebook — Cell Summary
| Cell | What it does |
|---|---|
| 1 | `pip install rtdl optuna` |
| 2 | Mount Google Drive, load `competition_train.npz` |
| 3 | `StandardScaler`, DataLoader helpers, train/eval functions |
| 4 | Optuna tuning — 15 trials on 100K subsample, 40 epochs + early stopping |
| 5 | Final model — best params, full 500K, 100 epochs + early stopping |
| 6 | `predict_proba()` on `X_test` → saves `ft_transformer_probs.npy` to Drive |
| 7 | (Optional) standalone hard predictions without ensemble |

### Key Optuna search space
| Hyperparameter | Range |
|---|---|
| `d_token` | 128 / 192 / 256 |
| `n_blocks` | 2 – 4 |
| `n_heads` | 4 / 8 |
| `attn_drop`, `ffn_drop` | 0.0 – 0.3 |
| `lr` | 1e-5 – 1e-3 (log) |
| `wd` | 1e-6 – 1e-3 (log) |

### Ensemble strategy
1. Colab outputs `ft_transformer_probs.npy` — shape `(200000, 7)` soft probabilities
2. LightGBM outputs soft probabilities on the same `X_test` (5-seed ensemble)
3. Blend: `ensemble = w * lgbm_probs + (1-w) * ft_probs`, default `w=0.5`
4. If FT val acc << 0.946, increase LightGBM weight (e.g. `w=0.7`)
5. Final hard predictions: `argmax(ensemble_probs)`

### Steps to complete on other machine
- [ ] Upload `competition_train.npz` to Google Drive
- [ ] Open `colab_ft_transformer.ipynb` in Colab → T4 GPU → Run all
- [ ] Download `ft_transformer_probs.npy` to project root
- [ ] `conda run -n AML python ensemble_lgbm_ft.py`
- [ ] Check val score printout; tune `LGBM_WEIGHT` in `ensemble_lgbm_ft.py` if needed
- [ ] Submit `competition_output/1385_competition_predictions.npz`
