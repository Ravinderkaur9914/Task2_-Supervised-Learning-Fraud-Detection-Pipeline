"""
Project 2: Supervised Learning (Fraud Detection Pipeline)
DecodeLabs Data Science Industrial Training Kit - Batch 2026

Dataset: Simulated transaction data (10,000 rows, 0.5% fraud rate)
Author: [Your Name]

Pipeline principles (The Zero-Leakage Protocol):
  1. Ditch Accuracy -> optimize/report Precision, Recall, F1, ROC-AUC
  2. SMOTE interpolates minority class -> never simple duplication
  3. NEVER apply SMOTE or scalers before the train/test split
  4. ALWAYS use imblearn.pipeline.Pipeline so resampling is isolated inside CV folds
  5. Tune preprocessing + model hyperparameters together inside GridSearchCV
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

# =============================================================
# LOAD DATA
# =============================================================
df = pd.read_csv("transactions.csv")
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["is_fraud"].value_counts())
print((df["is_fraud"].value_counts(normalize=True) * 100).round(3), "\n")

X = df.drop(columns=["transaction_id", "is_fraud"])
y = df["is_fraud"]

# =============================================================
# TRAIN/TEST SPLIT -- BEFORE any scaling or resampling (Trap #2 avoidance)
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Train fraud rate: {y_train.mean()*100:.3f}% | Test fraud rate: {y_test.mean()*100:.3f}%\n")

# =============================================================
# PIPELINE 1: Logistic Regression (needs scaling -> Fatal without it)
# =============================================================
lr_pipeline = ImbPipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])

lr_param_grid = {
    "smote__k_neighbors": [3, 5],
    "classifier__C": [0.01, 0.1, 1.0, 10],
}

# =============================================================
# PIPELINE 2: Random Forest (scale-invariant, no scaler needed)
# =============================================================
rf_pipeline = ImbPipeline(steps=[
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
])

rf_param_grid = {
    "smote__k_neighbors": [3, 5],
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 20, None],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# =============================================================
# GRIDSEARCHCV -- SMOTE is re-applied inside every fold, zero leakage
# =============================================================
results = {}

for name, pipeline, grid in [
    ("Logistic Regression", lr_pipeline, lr_param_grid),
    ("Random Forest", rf_pipeline, rf_param_grid),
]:
    print(f"\n{'='*60}\nTuning {name}\n{'='*60}")
    search = GridSearchCV(
        pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}\n")
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], digits=4))

    results[name] = {
        "model": best_model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "roc_auc": roc_auc,
        "best_params": search.best_params_,
    }

# =============================================================
# WHY NOT ACCURACY -- explicit demonstration
# =============================================================
naive_accuracy = (y_test == 0).mean()
print(f"\n{'='*60}")
print("The Illusion of Accuracy")
print(f"{'='*60}")
print(f"A model predicting 'Legitimate' for every transaction achieves "
      f"{naive_accuracy*100:.2f}% accuracy while catching ZERO fraud.")
print("This confirms why Precision, Recall, and ROC-AUC are used instead.\n")

# =============================================================
# CONFUSION MATRICES + ROC CURVES
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=axes[0, i], colorbar=False, cmap="Blues")
    axes[0, i].set_title(f"{name}\nConfusion Matrix")

    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[1, i].plot(fpr, tpr, label=f'ROC-AUC = {res["roc_auc"]:.4f}', color="darkorange")
    axes[1, i].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1, i].set_xlabel("False Positive Rate")
    axes[1, i].set_ylabel("True Positive Rate")
    axes[1, i].set_title(f"{name}\nROC Curve")
    axes[1, i].legend(loc="lower right")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
print("Saved evaluation plots to model_evaluation.png")

# =============================================================
# FINAL MODEL SELECTION
# =============================================================
best_name = max(results, key=lambda k: results[k]["roc_auc"])
print(f"\n{'='*60}")
print(f"Best performing model: {best_name} (Test ROC-AUC = {results[best_name]['roc_auc']:.4f})")
print(f"{'='*60}")
