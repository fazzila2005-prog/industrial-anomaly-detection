import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "results" / "anomalyclip_external"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SCORES WE ALREADY OBTAINED
# ============================================================

# NORMAL = 0
normal_scores = {
    "30ml-clear-flat-shoulder-glass-20260202155049057.webp": 0.9428,
    "a-bottle-of-soda-free-photo.jpg": 0.8830,
    "empty-beer-bottle-23348207.webp": 0.6069,
    "good bottle 2.jpg": 0.8437,
    "good bottle 3.jpg": 0.7989,
    "good bottle external image.jpg": 0.9681,
}


# ANOMALY = 1
anomaly_scores = {
    "broken bottle 2 external.jpg": 0.9501,
    "broken bottle external.webp": 0.9897,
    "images (1).jpg": 0.9778,
    "images (2).jpg": 0.9641,
    "images (3).jpg": 0.9701,
    "images (4).jpg": 0.9888,
    "images (5).jpg": 0.9653,
    "images (6).jpg": 0.9315,
    "images (7).jpg": 0.9747,
    "images.jpg": 0.9724,
    "xzdpzqmug5051.jpg": 0.9603,
}


# ============================================================
# 3. CREATE COMBINED DATASET
# ============================================================

records = []

for filename, score in normal_scores.items():
    records.append({
        "Image_Name": filename,
        "True_Label": 0,
        "True_Class": "NORMAL",
        "AnomalyCLIP_Score": score
    })


for filename, score in anomaly_scores.items():
    records.append({
        "Image_Name": filename,
        "True_Label": 1,
        "True_Class": "ANOMALY",
        "AnomalyCLIP_Score": score
    })


df = pd.DataFrame(records)

# Sort by score
df = df.sort_values("AnomalyCLIP_Score").reset_index(drop=True)


# ============================================================
# 4. SAVE COMBINED CSV
# ============================================================

combined_csv = OUTPUT_DIR / "anomalyclip_external_combined.csv"

df.to_csv(combined_csv, index=False)


# ============================================================
# 5. AUROC
# ============================================================

y_true = df["True_Label"]
y_scores = df["AnomalyCLIP_Score"]

auroc = roc_auc_score(y_true, y_scores)


# ============================================================
# 6. AVERAGE PRECISION
# ============================================================

average_precision = average_precision_score(
    y_true,
    y_scores
)


# ============================================================
# 7. EXPLORATORY THRESHOLD = 0.95
# ============================================================

# IMPORTANT:
# This threshold is NOT our final deployment threshold.
# It is only used to show how the current scores behave.

exploratory_threshold = 0.95

y_pred = (y_scores >= exploratory_threshold).astype(int)


accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(y_true, y_pred)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print()
print("=" * 65)
print("ANOMALYCLIP EXTERNAL BOTTLE EVALUATION")
print("=" * 65)

print()

print("Images tested:", len(df))
print("Normal images:", len(normal_scores))
print("Anomaly images:", len(anomaly_scores))

print()

print("NORMAL SCORE RANGE:")
print(
    f"{min(normal_scores.values()):.4f}"
    f" - "
    f"{max(normal_scores.values()):.4f}"
)

print()

print("ANOMALY SCORE RANGE:")
print(
    f"{min(anomaly_scores.values()):.4f}"
    f" - "
    f"{max(anomaly_scores.values()):.4f}"
)

print()
print("-" * 65)

print(f"AUROC:              {auroc:.4f}")
print(f"Average Precision:  {average_precision:.4f}")

print()
print("-" * 65)

print(f"Exploratory threshold: {exploratory_threshold}")

print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1 Score:           {f1:.4f}")

print()

print("CONFUSION MATRIX")
print()
print("                 Predicted")
print("              NORMAL  ANOMALY")
print(
    f"Actual NORMAL    {cm[0,0]:3d}      {cm[0,1]:3d}"
)
print(
    f"Actual ANOMALY   {cm[1,0]:3d}      {cm[1,1]:3d}"
)

print()
print("-" * 65)

print("Combined CSV saved to:")
print(combined_csv)

print("=" * 65)
print()


# ============================================================
# 9. SHOW ALL SCORES
# ============================================================

print("ALL EXTERNAL TEST SCORES")
print()

print(
    df[
        [
            "Image_Name",
            "True_Class",
            "AnomalyCLIP_Score"
        ]
    ].to_string(index=False)
)

print()