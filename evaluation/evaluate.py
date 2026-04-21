import pandas as pd
import os
from sklearn.metrics import accuracy_score

GROUND_TRUTH = "data/test_labels.csv"
SUBMISSION_FOLDER = "submissions/"
LEADERBOARD_FILE = "evaluation/leaderboard.csv"

gt = pd.read_csv(GROUND_TRUTH)

results = []

for file in os.listdir(SUBMISSION_FOLDER):
    if file.endswith(".csv"):
        path = os.path.join(SUBMISSION_FOLDER, file)

        try:
            sub = pd.read_csv(path)

            if "id" not in sub.columns or "prediction" not in sub.columns:
                print(f"{file} skipped (wrong format)")
                continue

            score = accuracy_score(gt["risk"], sub["prediction"])

            results.append({
                "team": file.replace(".csv", ""),
                "accuracy": round(score, 4)
            })

        except Exception as e:
            print(f"Error in {file}: {e}")

leaderboard = pd.DataFrame(results)

if not leaderboard.empty:
    leaderboard = leaderboard.sort_values(by="accuracy", ascending=False)

leaderboard.to_csv(LEADERBOARD_FILE, index=False)

print("✅ Leaderboard updated!")