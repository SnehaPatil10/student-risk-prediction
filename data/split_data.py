import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("student_data.csv")

X = df.drop("Risk", axis=1)
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train = X_train.copy()
train["risk"] = y_train
train.to_csv("train.csv", index=False)

X_test.to_csv("test.csv", index=False)

pd.DataFrame({"risk": y_test}).to_csv("test_labels.csv", index=False)

print("✅ Dataset split done")