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
- [ ] Test LightGBM GPU support (Quadro P1000, CUDA 12.8)
- [ ] Optuna tuning of LightGBM (100K subsample, early stopping, ~50 trials)
- [ ] Train final model on full 500K with best params
- [ ] Multi-seed ensemble of tuned models
- [ ] Final submission

---

## Best Result So Far
| Model | CV Score | Notes |
|---|---|---|
| LogReg baseline | 0.4867 | Starting point |
| LightGBM (default params, 50K) | 0.8452 | 50K subsample, 3-fold CV |

**Current best submission:** None yet
