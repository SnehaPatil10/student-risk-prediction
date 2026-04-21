"""
baseline_model.py
-----------------
Baseline: Logistic Regression on the Student Risk dataset.
Teams should use this as a starting point and improve upon it.

Run:
    python baseline/baseline_model.py

Output:
    baseline/baseline_predictions.csv
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score

# ── Paths ───────────────────────────────────────────────────────────────────
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
OUTPUT_PATH = "baseline/baseline_predictions.csv"

# ── Load data ───────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"\nTrain columns: {list(train.columns)}")

# ── Feature engineering ─────────────────────────────────────────────────────
LABEL_COL = "risk"
ID_COL = None

feature_cols = [c for c in train.columns if c not in [LABEL_COL, ID_COL]]

X_train = train[feature_cols].copy()
y_train = train[LABEL_COL].copy()
X_test  = test[feature_cols].copy()

# Encode categoricals (e.g. gender)
le = LabelEncoder()
for col in X_train.select_dtypes(include="object").columns:
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col]  = le.transform(X_test[col].astype(str))

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train ────────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Training performance (for your reference)
train_preds = model.predict(X_train_scaled)
print("\n── Training Performance ──────────────────────────")
print(classification_report(y_train, train_preds))
print(f"Macro F1: {f1_score(y_train, train_preds, average='macro'):.4f}")

# ── Predict on test ──────────────────────────────────────────────────────────
# Ensure test has same features as train
X_test = test.copy()

# Predict
y_pred = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    "id": range(1, len(y_pred) + 1),
    "prediction": y_pred
})

# Save file
submission.to_csv(OUTPUT_PATH, index=False)

print("✅ Baseline predictions saved!")