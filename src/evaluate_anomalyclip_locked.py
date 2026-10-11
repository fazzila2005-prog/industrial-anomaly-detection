import os
import csv
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "anomalyclip_adaptation"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "anomalyclip_locked_external_evaluation.csv"
)


# ============================================================
# 2. LOCKED THRESHOLD
# ============================================================

# This threshold was calculated ONLY from the
# normal adaptation images.

LOCKED_THRESHOLD = 0.8392


# ============================================================
# 3. HELD-OUT EXTERNAL TEST SCORES
# ============================================================

# These are the AnomalyCLIP scores obtained earlier.
#
# IMPORTANT:
# These images were NOT used to calculate the threshold.

test_data = [

    # --------------------------------------------------------
    # NORMAL EXTERNAL TEST IMAGES
    # Ground truth:
    # 0 = NORMAL
    # --------------------------------------------------------

    {
        "Image_Name": "30ml-clear-flat-shoulder-glass-20260202155049057.webp",
        "Ground_Truth": 0,
        "Score": 0.9428
    },

    {
        "Image_Name": "a-bottle-of-soda-free-photo.jpg",
        "Ground_Truth": 0,
        "Score": 0.8830
    },

    {
        "Image_Name": "empty-beer-bottle-23348207.webp",
        "Ground_Truth": 0,
        "Score": 0.6069
    },

    {
        "Image_Name": "good bottle 2.jpg",
        "Ground_Truth": 0,
        "Score": 0.8437
    },

    {
        "Image_Name": "good bottle 3.jpg",
        "Ground_Truth": 0,
        "Score": 0.7989
    },

    {
        "Image_Name": "good bottle external image.jpg",
        "Ground_Truth": 0,
        "Score": 0.9681
    },


    # --------------------------------------------------------
    # ANOMALOUS EXTERNAL TEST IMAGES
    # Ground truth:
    # 1 = ANOMALY
    # --------------------------------------------------------

    {
        "Image_Name": "broken bottle 2 external.jpg",
        "Ground_Truth": 1,
        "Score": 0.9501
    },

    {
        "Image_Name": "broken bottle external.webp",
        "Ground_Truth": 1,
        "Score": 0.9897
    },

    {
        "Image_Name": "images (1).jpg",
        "Ground_Truth": 1,
        "Score": 0.9778
    },

    {
        "Image_Name": "images (2).jpg",
        "Ground_Truth": 1,
        "Score": 0.9641
    },

    {
        "Image_Name": "images (3).jpg",
        "Ground_Truth": 1,
        "Score": 0.9701
    },

    {
        "Image_Name": "images (4).jpg",
        "Ground_Truth": 1,
        "Score": 0.9888
    },

    {
        "Image_Name": "images (5).jpg",
        "Ground_Truth": 1,
        "Score": 0.9653
    },

    {
        "Image_Name": "images (6).jpg",
        "Ground_Truth": 1,
        "Score": 0.9315
    },

    {
        "Image_Name": "images (7).jpg",
        "Ground_Truth": 1,
        "Score": 0.9747
    },

    {
        "Image_Name": "images.jpg",
        "Ground_Truth": 1,
        "Score": 0.9724
    },

    {
        "Image_Name": "xzdpzqmug5051.jpg",
        "Ground_Truth": 1,
        "Score": 0.9603
    }
]


# ============================================================
# 4. EXTRACT SCORES AND LABELS
# ============================================================

y_true = np.array(
    [
        item["Ground_Truth"]
        for item in test_data
    ]
)

scores = np.array(
    [
        item["Score"]
        for item in test_data
    ]
)


# ============================================================
# 5. APPLY LOCKED THRESHOLD
# ============================================================

y_pred = (
    scores >= LOCKED_THRESHOLD
).astype(int)


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

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

auroc = roc_auc_score(
    y_true,
    scores
)

average_precision = average_precision_score(
    y_true,
    scores
)

cm = confusion_matrix(
    y_true,
    y_pred
)


# ============================================================
# 7. SAVE IMAGE-LEVEL RESULTS
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Image_Name",
            "Ground_Truth",
            "AnomalyCLIP_Score",
            "Locked_Threshold",
            "Predicted_Label",
            "Predicted_Status"
        ]
    )

    writer.writeheader()

    for item, prediction in zip(
        test_data,
        y_pred
    ):

        writer.writerow(
            {
                "Image_Name": item["Image_Name"],
                "Ground_Truth": (
                    "ANOMALY"
                    if item["Ground_Truth"] == 1
                    else "NORMAL"
                ),
                "AnomalyCLIP_Score": item["Score"],
                "Locked_Threshold": LOCKED_THRESHOLD,
                "Predicted_Label": prediction,
                "Predicted_Status": (
                    "ANOMALY"
                    if prediction == 1
                    else "NORMAL"
                )
            }
        )


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("ANOMALYCLIP LOCKED-THRESHOLD EXTERNAL EVALUATION")
print("=" * 70)

print()

print(
    f"Locked threshold: {LOCKED_THRESHOLD:.4f}"
)

print(
    "Threshold source: NORMAL adaptation images ONLY"
)

print()

print(
    f"Total test images: {len(test_data)}"
)

print(
    f"Normal test images: "
    f"{sum(y_true == 0)}"
)

print(
    f"Anomaly test images: "
    f"{sum(y_true == 1)}"
)

print()

print("=" * 70)
print("CLASSIFICATION METRICS")
print("=" * 70)

print()

print(
    f"Accuracy:           {accuracy:.4f}"
)

print(
    f"Precision:          {precision:.4f}"
)

print(
    f"Recall:             {recall:.4f}"
)

print(
    f"F1 Score:           {f1:.4f}"
)

print()

print("=" * 70)
print("THRESHOLD-INDEPENDENT METRICS")
print("=" * 70)

print()

print(
    f"AUROC:              {auroc:.4f}"
)

print(
    f"Average Precision:  {average_precision:.4f}"
)

print()

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(
    "                 Predicted"
)

print(
    "              NORMAL  ANOMALY"
)

print(
    f"Actual NORMAL    {cm[0,0]:>3}      {cm[0,1]:>3}"
)

print(
    f"Actual ANOMALY   {cm[1,0]:>3}      {cm[1,1]:>3}"
)

print()


# ============================================================
# 9. SHOW EVERY PREDICTION
# ============================================================

print("=" * 70)
print("INDIVIDUAL PREDICTIONS")
print("=" * 70)

print()

for item, prediction in zip(
    test_data,
    y_pred
):

    actual = (
        "ANOMALY"
        if item["Ground_Truth"] == 1
        else "NORMAL"
    )

    predicted = (
        "ANOMALY"
        if prediction == 1
        else "NORMAL"
    )

    correct = (
        "CORRECT"
        if item["Ground_Truth"] == prediction
        else "WRONG"
    )

    print(
        f"{item['Image_Name']:<55} "
        f"Score={item['Score']:.4f}  "
        f"Actual={actual:<7}  "
        f"Predicted={predicted:<7}  "
        f"{correct}"
    )


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print()

print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print()

print(
    "The threshold was NOT changed after seeing the test results."
)

print(
    "The test images were used ONLY for final evaluation."
)

print()

print("Results saved to:")

print(
    OUTPUT_CSV
)

print()