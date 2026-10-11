import sys
import os

# ============================================================
# 0. FIX: make the AnomalyCLIP repo importable
# ============================================================
ANOMALYCLIP_ROOT = r"C:\Users\Fazzila\OneDrive\Desktop\AnomalyCLIP"

sys.path.insert(0, ANOMALYCLIP_ROOT)
os.chdir(ANOMALYCLIP_ROOT)

import csv
import random

import numpy as np
import torch
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform


# ============================================================
# 1. PATHS
# ============================================================
PROJECT_ROOT = r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"

# Held-out normal Cable images
NORMAL_DIR = os.path.join(
    PROJECT_ROOT,
    "external images",
    "cable",
    "test_normal"
)

# Held-out anomalous Cable images
ANOMALY_DIR = os.path.join(
    PROJECT_ROOT,
    "external images",
    "cable",
    "test_anomaly"
)

# AnomalyCLIP checkpoint
CHECKPOINT_PATH = os.path.join(
    ANOMALYCLIP_ROOT,
    "checkpoints",
    "9_12_4_multiscale_visa",
    "epoch_15.pth"
)

# Results folder
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "anomalyclip_external_cable"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "anomalyclip_external_cable_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "anomalyclip_external_cable_summary.txt"
)


# ============================================================
# 2. LOCKED THRESHOLD
# ============================================================
# IMPORTANT:
#
# This threshold was calculated BEFORE looking at the
# held-out test results.
#
# It came ONLY from the 11 normal Cable adaptation images.
# ============================================================

LOCKED_THRESHOLD = 0.9346


# ============================================================
# 3. ANOMALYCLIP SETTINGS
# ============================================================
IMAGE_SIZE = 518
FEATURES_LIST = [6, 12, 18, 24]

DEPTH = 9
N_CTX = 12
T_N_CTX = 4

SEED = 111


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 5. DEVICE
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("ANOMALYCLIP EXTERNAL CABLE EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Locked threshold: {LOCKED_THRESHOLD:.4f}")
print()


# ============================================================
# 6. CHECK PATHS
# ============================================================
if not os.path.exists(NORMAL_DIR):
    raise FileNotFoundError(
        f"\nCable normal test folder not found:\n{NORMAL_DIR}"
    )

if not os.path.exists(ANOMALY_DIR):
    raise FileNotFoundError(
        f"\nCable anomaly test folder not found:\n{ANOMALY_DIR}"
    )

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"\nAnomalyCLIP checkpoint not found:\n{CHECKPOINT_PATH}"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 7. SUPPORTED IMAGE FORMATS
# ============================================================
VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".avif"
)


# ============================================================
# 8. FIND NORMAL TEST IMAGES
# ============================================================
normal_files = sorted(
    os.path.join(NORMAL_DIR, f)
    for f in os.listdir(NORMAL_DIR)
    if f.lower().endswith(VALID_EXTENSIONS)
)


# ============================================================
# 9. FIND ANOMALY TEST IMAGES
# ============================================================
anomaly_files = sorted(
    os.path.join(ANOMALY_DIR, f)
    for f in os.listdir(ANOMALY_DIR)
    if f.lower().endswith(VALID_EXTENSIONS)
)


# ============================================================
# 10. CHECK IMAGE COUNTS
# ============================================================
if len(normal_files) == 0:
    raise RuntimeError(
        f"\nNo normal Cable test images found in:\n{NORMAL_DIR}"
    )

if len(anomaly_files) == 0:
    raise RuntimeError(
        f"\nNo anomalous Cable test images found in:\n{ANOMALY_DIR}"
    )


print("TEST DATA")
print("-" * 70)

print(f"Normal test images : {len(normal_files)}")
print(f"Anomaly test images: {len(anomaly_files)}")
print(f"Total test images  : {len(normal_files) + len(anomaly_files)}")

print()


print("NORMAL TEST IMAGES")
for path in normal_files:
    print(" -", os.path.basename(path))

print()

print("ANOMALY TEST IMAGES")
for path in anomaly_files:
    print(" -", os.path.basename(path))

print()


# ============================================================
# 11. CREATE ANOMALYCLIP MODEL
# ============================================================
print("Loading AnomalyCLIP model...")

AnomalyCLIP_parameters = {
    "Prompt_length": N_CTX,
    "learnabel_text_embedding_depth": DEPTH,
    "learnabel_text_embedding_length": T_N_CTX,
}

model, _ = AnomalyCLIP_lib.load(
    "ViT-L/14@336px",
    device=DEVICE,
    design_details=AnomalyCLIP_parameters,
)

model.eval()


# ============================================================
# 12. PREPROCESSING
# ============================================================
class TransformArgs:
    image_size = IMAGE_SIZE


preprocess, _ = get_transform(TransformArgs())


# ============================================================
# 13. LOAD LEARNED PROMPT MODEL
# ============================================================
print("Loading learned anomaly prompts...")

prompt_learner = AnomalyCLIP_PromptLearner(
    model.to("cpu"),
    AnomalyCLIP_parameters
)


# ============================================================
# 14. LOAD CHECKPOINT
# ============================================================
print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if "prompt_learner" not in checkpoint:
    raise RuntimeError(
        "\nThe checkpoint does not contain 'prompt_learner'.\n"
        "This does not look like the expected AnomalyCLIP checkpoint."
    )

prompt_learner.load_state_dict(
    checkpoint["prompt_learner"]
)

prompt_learner.to(DEVICE)


# ============================================================
# 15. ENABLE VISUAL ADAPTATION
# ============================================================
model.to(DEVICE)

model.visual.DAPM_replace(
    DPAM_layer=20
)


# ============================================================
# 16. CREATE LEARNED TEXT FEATURES
# ============================================================
print("Creating learned text features...")

with torch.no_grad():

    prompts, tokenized_prompts, compound_prompts_text = (
        prompt_learner(cls_id=None)
    )

    text_features = model.encode_text_learn(
        prompts,
        tokenized_prompts,
        compound_prompts_text
    ).float()

    # Separate normal and abnormal text features
    text_features = torch.stack(
        torch.chunk(
            text_features,
            dim=0,
            chunks=2
        ),
        dim=1
    )

    # Normalize text features
    text_features = (
        text_features /
        text_features.norm(
            dim=-1,
            keepdim=True
        )
    )

print("Text features ready.")
print()


# ============================================================
# 17. FUNCTION TO SCORE ONE IMAGE
# ============================================================
def get_anomaly_score(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = (
        preprocess(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        image_features, patch_features = model.encode_image(
            image_tensor,
            FEATURES_LIST,
            DPAM_layer=20
        )

        # Normalize image features
        image_features = (
            image_features /
            image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        # Compare image and text features
        text_probs = (
            image_features @
            text_features.permute(0, 2, 1)
        )

        # Convert to probabilities
        text_probs = (
            text_probs / 0.07
        ).softmax(dim=-1)

        # Second class = anomaly
        anomaly_score = text_probs[
            :, 0, 1
        ].item()

    return anomaly_score


# ============================================================
# 18. SCORE ALL TEST IMAGES
# ============================================================
results = []

print("=" * 70)
print("SCORING HELD-OUT TEST IMAGES")
print("=" * 70)
print()


# ------------------------------------------------------------
# NORMAL IMAGES
# ------------------------------------------------------------
print("NORMAL TEST IMAGES")
print("-" * 70)

for index, image_path in enumerate(
    normal_files,
    start=1
):

    filename = os.path.basename(image_path)

    print(
        f"[NORMAL {index}/{len(normal_files)}] "
        f"{filename}"
    )

    try:

        anomaly_score = get_anomaly_score(
            image_path
        )

        # Apply LOCKED threshold
        predicted_label = (
            "ANOMALY"
            if anomaly_score >= LOCKED_THRESHOLD
            else "NORMAL"
        )

        correct = (
            predicted_label == "NORMAL"
        )

        results.append({
            "Image_Name": filename,
            "Image_Path": image_path,
            "Actual_Label": "NORMAL",
            "AnomalyCLIP_Anomaly_Score": anomaly_score,
            "Predicted_Label": predicted_label,
            "Correct": correct,
        })

        print(
            f"    Score: {anomaly_score:.4f}"
        )

        print(
            f"    Prediction: {predicted_label}"
        )

        print(
            f"    Correct: {correct}"
        )

    except Exception as error:

        print(
            f"    ERROR: {error}"
        )


print()


# ------------------------------------------------------------
# ANOMALY IMAGES
# ------------------------------------------------------------
print("ANOMALY TEST IMAGES")
print("-" * 70)

for index, image_path in enumerate(
    anomaly_files,
    start=1
):

    filename = os.path.basename(image_path)

    print(
        f"[ANOMALY {index}/{len(anomaly_files)}] "
        f"{filename}"
    )

    try:

        anomaly_score = get_anomaly_score(
            image_path
        )

        # Apply LOCKED threshold
        predicted_label = (
            "ANOMALY"
            if anomaly_score >= LOCKED_THRESHOLD
            else "NORMAL"
        )

        correct = (
            predicted_label == "ANOMALY"
        )

        results.append({
            "Image_Name": filename,
            "Image_Path": image_path,
            "Actual_Label": "ANOMALY",
            "AnomalyCLIP_Anomaly_Score": anomaly_score,
            "Predicted_Label": predicted_label,
            "Correct": correct,
        })

        print(
            f"    Score: {anomaly_score:.4f}"
        )

        print(
            f"    Prediction: {predicted_label}"
        )

        print(
            f"    Correct: {correct}"
        )

    except Exception as error:

        print(
            f"    ERROR: {error}"
        )


print()


# ============================================================
# 19. KEEP ONLY SUCCESSFUL RESULTS
# ============================================================
successful_results = [
    r for r in results
    if isinstance(
        r["AnomalyCLIP_Anomaly_Score"],
        (float, int)
    )
]

if len(successful_results) == 0:
    raise RuntimeError(
        "\nNo test images were successfully scored."
    )


# ============================================================
# 20. CREATE TRUE LABELS AND SCORES
# ============================================================
y_true = []

y_scores = []

y_pred = []


for result in successful_results:

    # NORMAL = 0
    # ANOMALY = 1

    if result["Actual_Label"] == "NORMAL":
        y_true.append(0)
    else:
        y_true.append(1)

    y_scores.append(
        result["AnomalyCLIP_Anomaly_Score"]
    )

    # Prediction from locked threshold
    if (
        result["Predicted_Label"] == "ANOMALY"
    ):
        y_pred.append(1)
    else:
        y_pred.append(0)


y_true = np.array(y_true)

y_scores = np.array(y_scores)

y_pred = np.array(y_pred)


# ============================================================
# 21. CALCULATE METRICS
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

# AUROC evaluates ranking quality across thresholds.
auroc = roc_auc_score(
    y_true,
    y_scores
)

# Average Precision evaluates precision-recall performance.
average_precision = average_precision_score(
    y_true,
    y_scores
)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# 23. SAVE INDIVIDUAL RESULTS
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
            "Image_Path",
            "Actual_Label",
            "AnomalyCLIP_Anomaly_Score",
            "Predicted_Label",
            "Correct",
        ]
    )

    writer.writeheader()

    writer.writerows(
        successful_results
    )


# ============================================================
# 24. SAVE SUMMARY
# ============================================================
with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ANOMALYCLIP EXTERNAL CABLE EVALUATION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Normal test images: {len(normal_files)}\n"
    )

    file.write(
        f"Anomaly test images: {len(anomaly_files)}\n"
    )

    file.write(
        f"Successfully evaluated: "
        f"{len(successful_results)}\n\n"
    )

    file.write(
        f"LOCKED THRESHOLD: "
        f"{LOCKED_THRESHOLD:.6f}\n\n"
    )

    file.write(
        "METRICS\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1 Score: {f1:.4f}\n"
    )

    file.write(
        f"AUROC: {auroc:.4f}\n"
    )

    file.write(
        f"Average Precision: "
        f"{average_precision:.4f}\n\n"
    )

    file.write(
        "CONFUSION MATRIX\n"
    )

    file.write(
        "                 Predicted\n"
    )

    file.write(
        "              NORMAL  ANOMALY\n"
    )

    file.write(
        f"Actual NORMAL     {tn:3d}      {fp:3d}\n"
    )

    file.write(
        f"Actual ANOMALY    {fn:3d}      {tp:3d}\n\n"
    )

    file.write(
        "IMPORTANT:\n"
    )

    file.write(
        "The threshold was calculated exclusively from "
        "normal Cable adaptation images.\n"
    )

    file.write(
        "No Cable anomaly images were used to select the threshold.\n"
    )

    file.write(
        "The held-out Cable test images were evaluated only "
        "after the threshold was locked.\n"
    )


# ============================================================
# 25. FINAL RESULTS
# ============================================================
print("=" * 70)
print("ANOMALYCLIP EXTERNAL CABLE EVALUATION COMPLETE")
print("=" * 70)

print()

print(
    f"Images tested: "
    f"{len(successful_results)}"
)

print(
    f"Normal images: "
    f"{len(normal_files)}"
)

print(
    f"Anomaly images: "
    f"{len(anomaly_files)}"
)

print()

print(
    f"LOCKED THRESHOLD: "
    f"{LOCKED_THRESHOLD:.4f}"
)

print()

print("METRICS")
print("-" * 70)

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

print(
    f"AUROC:              {auroc:.4f}"
)

print(
    f"Average Precision:  {average_precision:.4f}"
)

print()

print("CONFUSION MATRIX")
print()

print(
    "                 Predicted"
)

print(
    "              NORMAL  ANOMALY"
)

print(
    f"Actual NORMAL      {tn:3d}      {fp:3d}"
)

print(
    f"Actual ANOMALY     {fn:3d}      {tp:3d}"
)

print()

print("Results saved to:")
print(OUTPUT_CSV)

print()

print("Summary saved to:")
print(OUTPUT_SUMMARY)

print()

print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "The threshold was LOCKED at 0.9346 before "
    "evaluating the test images."
)

print(
    "Do not change the threshold based on these test results."
)

print("=" * 70)