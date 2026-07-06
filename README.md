# Project 2: Supervised Learning — Fraud Detection Pipeline

**DecodeLabs Data Science Industrial Training Kit — Batch 2026**

## Overview

This project builds a leak-free, production-style fraud detection pipeline on a highly imbalanced transaction dataset (0.5% fraud rate). The focus is **algorithmic precision**, not just prediction: correctly handling class imbalance with SMOTE, avoiding data leakage, and evaluating with metrics that actually matter for fraud detection (Precision, Recall, ROC-AUC) instead of misleading Accuracy.

## Files

| File | Description |
|---|---|
| `transactions.csv` | Simulated dataset (10,000 transactions, 50 fraudulent) |
| `fraud_detection_pipeline.py` | Full pipeline — split, SMOTE, model training, tuning, evaluation |
| `model_evaluation.png` | Confusion matrices & ROC curves for both models |
| `README.md` | This file |

## Dataset

Simulated transaction data with overlapping (realistic, non-trivial) fraud/legitimate distributions:

- `transaction_amount` — fraud transactions trend higher and more variable
- `time_of_day` — fraud skews slightly toward late-night/early-morning hours
- `distance_from_home_km` — fraud transactions occur farther from the cardholder's usual location
- `num_transactions_24h` — fraud accounts show a higher transaction velocity
- `merchant_risk_score` — a composite risk indicator, higher on average for fraud
- `is_fraud` — target (1 = fraud, 0 = legitimate)

Class distribution: **9,950 legitimate (99.5%) vs. 50 fraudulent (0.5%)** — deliberately extreme to mirror real-world financial datasets (e.g., the classic ULB Credit Card Fraud dataset's 0.17% fraud rate).

## Why Accuracy Fails Here

A model that predicts "Legitimate" for every single transaction achieves **99.50% accuracy** while catching **zero fraud** — a textbook illustration of the accuracy trap in imbalanced classification. This project deliberately discards accuracy and instead reports:

- **Precision** = TP / (TP + FP) — when we flag fraud, are we right?
- **Recall** = TP / (TP + FN) — did we catch all the fraud?
- **ROC-AUC** — the model's overall ability to separate the two classes

## The Zero-Leakage Protocol

The pipeline strictly follows five rules to avoid the two most common traps in imbalanced classification:

1. **Train/test split happens first**, before any scaling or resampling touches the data.
2. **SMOTE is never applied to the full dataset up front.** Doing so would let synthetic samples derived from test-set neighbors leak into training, giving artificially perfect validation scores.
3. **`imblearn.pipeline.Pipeline` is used instead of `sklearn.pipeline.Pipeline`**, since scikit-learn's pipeline doesn't support resampling steps that modify both `X` and `y`.
4. **SMOTE and scaling live *inside* the pipeline**, so `GridSearchCV`'s cross-validation re-applies them fresh within every fold — meaning the validation fold is always evaluated in its original imbalanced state.
5. **Hyperparameters for both the resampler and the model are tuned together** inside `GridSearchCV`, since the right SMOTE neighbor count interacts with the model's decision boundary.

## Models Trained

| Model | Needs Scaling? | Why |
|---|---|---|
| Logistic Regression | Yes (`StandardScaler` inside pipeline) | Regularization penalties are distorted by unscaled features with large variance (e.g., transaction amount) |
| Random Forest | No | Tree splits partition feature space ordinally — scale-invariant by construction |

Both models were tuned via `GridSearchCV` with 5-fold stratified cross-validation, optimizing for ROC-AUC.

## Results (this run)

| Model | Best Params | Test ROC-AUC | Fraud Recall | Fraud Precision |
|---|---|---|---|---|
| Logistic Regression | C=0.01, k_neighbors=5 | 0.9625 | 0.90 | 0.05 |
| Random Forest | max_depth=None, n_estimators=200, k_neighbors=5 | 0.9455 | 0.40 | 0.11 |

**Interpretation:** Logistic Regression achieves higher recall (catches 9/10 fraud cases) but very low precision — it over-flags legitimate transactions as fraud. Random Forest is more conservative (fewer false alarms) but misses more fraud. This precision/recall trade-off is the central real-world decision fraud teams must make: aggressive fraud-catching costs customer friction from false declines, while conservative flagging risks financial loss from missed fraud. Neither model is "better" in the abstract — the right choice depends on the cost of a false positive vs. a false negative for the specific business.

## How to Run

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
python fraud_detection_pipeline.py
```

Output: console log of tuning results, classification reports for both models, and `model_evaluation.png` with confusion matrices and ROC curves.

## Key Takeaway

Fraud detection is not a "predict the label" problem — it's a cost-sensitive decision problem, hidden inside a severely imbalanced classification task. SMOTE lets the model learn from a balanced view of the minority class, and a leak-free pipeline ensures every reported metric reflects genuine generalization, not a shortcut. Precision, Recall, and ROC-AUC — not Accuracy — are the only metrics that tell the truth about a fraud model's real-world value.

---

