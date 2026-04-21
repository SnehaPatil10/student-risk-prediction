import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# Paths
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
OUTPUT_PATH = "baseline/baseline_predictions.csv"

# Load data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"\nTrain columns: {list(train.columns)}")

# Target column
LABEL_COL = "risk"

# Split features and target
X_train = train.drop(LABEL_COL, axis=1)
y_train = train[LABEL_COL]

# Model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)

# Evaluate on training data (for display)
y_train_pred = model.predict(X_train)

print("\n── Training Performance ──────────────────────────")
print(classification_report(y_train, y_train_pred))

print("Macro F1:", round(f1_score(y_train, y_train_pred, average="macro"), 4))

# Ensure test has same features
X_test = test.copy()

# Predict
y_pred = model.predict(X_test)

# Create submission (with generated ID)
submission = pd.DataFrame({
    "id": range(1, len(y_pred) + 1),
    "prediction": y_pred
})

# Save predictions
submission.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Baseline predictions saved at:", OUTPUT_PATH)