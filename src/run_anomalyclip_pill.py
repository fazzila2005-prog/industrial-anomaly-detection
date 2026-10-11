# ============================================================
# ANOMALYCLIP - COMPLETE EXTERNAL EVALUATION (one category)
#
# 1. Scores normal adaptation images
# 2. Calculates 95th and 99th percentile thresholds
# 3. Locks the 99th percentile threshold
# 4. Scores held-out normal + anomaly test images
# 5. Reports Accuracy, Precision, Recall, F1, AUROC, AP
# 6. Saves CSV + summary
#
# External anomalies are NEVER used for the threshold.
# AnomalyCLIP itself is NOT retrained.
# ============================================================

import sys
import os
import csv
import random
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Optional: lets PIL open .avif files (pip install pillow-avif-plugin)
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)


# ============================================================
# 1. PATHS
# ============================================================
PROJECT_ROOT = Path(r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project")
ANOMALYCLIP_ROOT = Path(r"C:\Users\Fazzila\OneDrive\Desktop\AnomalyCLIP")

CATEGORY = "pill"

BASE_EXTERNAL_DIR = PROJECT_ROOT / "external images" / CATEGORY
ADAPTATION_DIR = BASE_EXTERNAL_DIR / "adaptation_normal"
NORMAL_TEST_DIR = BASE_EXTERNAL_DIR / "test_normal"
ANOMALY_TEST_DIR = BASE_EXTERNAL_DIR / "test_anomaly"

CHECKPOINT = ANOMALYCLIP_ROOT / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth"

OUTPUT_DIR = PROJECT_ROOT / "results" / "anomalyclip_external" / CATEGORY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================
IMAGE_SIZE = 518
FEATURES_LIST = [6, 12, 18, 24]
DEPTH = 9
N_CTX = 12
T_N_CTX = 4
SEED = 111

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif")

# Decide this BEFORE looking at test results. Do not change afterwards.
LOCKED_PERCENTILE = 99


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 70)
print(f"ANOMALYCLIP - {CATEGORY.upper()} COMPLETE EXTERNAL EVALUATION")
print("=" * 70)
print()
print("Category :", CATEGORY)
print("Device   :", DEVICE)
print()

for folder in (ADAPTATION_DIR, NORMAL_TEST_DIR, ANOMALY_TEST_DIR):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found:\n{folder}")
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found:\n{CHECKPOINT}")


# ============================================================
# 4. IMPORT ANOMALYCLIP
# ============================================================
sys.path.insert(0, str(ANOMALYCLIP_ROOT))
os.chdir(ANOMALYCLIP_ROOT)

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform


# ============================================================
# 5. MODEL PARAMETERS (FIXED KEY NAMES)
# ============================================================
AnomalyCLIP_parameters = {
    "Prompt_length": N_CTX,
    "learnabel_text_embedding_depth": DEPTH,
    "learnabel_text_embedding_length": T_N_CTX,
}


# ============================================================
# 6. LOAD MODEL
# ============================================================
print("Loading AnomalyCLIP model...")
print("This can take some time on CPU.")
print()

model, _ = AnomalyCLIP_lib.load(
    "ViT-L/14@336px",
    device=DEVICE,
    design_details=AnomalyCLIP_parameters,
)
model.eval()


# ============================================================
# 7. LOAD CHECKPOINT
# ============================================================
print("Loading checkpoint...")

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

if "prompt_learner" not in checkpoint:
    raise RuntimeError("Checkpoint does not contain 'prompt_learner'.")

prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
prompt_learner.load_state_dict(checkpoint["prompt_learner"])
prompt_learner.to(DEVICE)

model.to(DEVICE)
model.visual.DAPM_replace(DPAM_layer=20)

print("Checkpoint loaded.")
print()


# ============================================================
# 8. PREPROCESSING (FIXED)
# ============================================================
class TransformArgs:
    image_size = IMAGE_SIZE


preprocess, _ = get_transform(TransformArgs())


# ============================================================
# 9. LEARNED TEXT FEATURES
# ============================================================
print("Creating text features...")

with torch.no_grad():
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)

    text_features = model.encode_text_learn(
        prompts, tokenized_prompts, compound_prompts_text
    ).float()

    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print("Text features ready.")
print()


# ============================================================
# 10. FIND IMAGES + LEAKAGE CHECK
# ============================================================
def get_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def file_hash(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


adaptation_images = get_images(ADAPTATION_DIR)
normal_test_images = get_images(NORMAL_TEST_DIR)
anomaly_test_images = get_images(ANOMALY_TEST_DIR)

print("=" * 70)
print("IMAGE COUNTS")
print("=" * 70)
print("Normal adaptation images :", len(adaptation_images))
print("Normal test images       :", len(normal_test_images))
print("Anomaly test images      :", len(anomaly_test_images))
print()

if not adaptation_images or not normal_test_images or not anomaly_test_images:
    raise RuntimeError("One of the folders has no images.")

# Identical files appearing in more than one folder = leakage
groups = {
    "adaptation_normal": adaptation_images,
    "test_normal": normal_test_images,
    "test_anomaly": anomaly_test_images,
}
seen = {}
leaks = []
for group, files in groups.items():
    for p in files:
        h = file_hash(p)
        if h in seen and seen[h][0] != group:
            leaks.append((p.name, seen[h][0], group))
        else:
            seen.setdefault(h, (group, p.name))

if leaks:
    print("WARNING: identical images found in more than one folder:")
    for name, g1, g2 in leaks:
        print(f"  - {name}: {g1} and {g2}")
    print()


# ============================================================
# 11. SCORING FUNCTION
# ============================================================
def calculate_anomaly_score(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features, patch_features = model.encode_image(
            image_tensor, FEATURES_LIST, DPAM_layer=20
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_probs = image_features @ text_features.permute(0, 2, 1)
        text_probs = (text_probs / 0.07).softmax(dim=-1)

        return text_probs[:, 0, 1].item()


# ============================================================
# 12. STEP 1 - NORMAL-ONLY ADAPTATION
# ============================================================
print("=" * 70)
print("STEP 1 - NORMAL-ONLY ADAPTATION")
print("=" * 70)
print()
print("Only normal adaptation images are used here.")
print()

adaptation_scores = []
for i, p in enumerate(adaptation_images, start=1):
    print(f"[{i}/{len(adaptation_images)}] {p.name}")
    try:
        s = calculate_anomaly_score(p)
        adaptation_scores.append(s)
        print(f"    Score: {s:.4f}")
    except Exception as e:
        print(f"    ERROR: {e}")

if not adaptation_scores:
    raise RuntimeError("No adaptation images were scored successfully.")

adaptation_scores = np.array(adaptation_scores)

threshold_95 = np.percentile(adaptation_scores, 95)
threshold_99 = np.percentile(adaptation_scores, 99)

print()
print("-" * 70)
print(f"Minimum score     : {adaptation_scores.min():.4f}")
print(f"Maximum score     : {adaptation_scores.max():.4f}")
print(f"Mean score        : {adaptation_scores.mean():.4f}")
print(f"Median score      : {np.median(adaptation_scores):.4f}")
print(f"95th percentile   : {threshold_95:.4f}")
print(f"99th percentile   : {threshold_99:.4f}")
print("-" * 70)

LOCKED_THRESHOLD = float(np.percentile(adaptation_scores, LOCKED_PERCENTILE))

print()
print(f"LOCKED THRESHOLD ({LOCKED_PERCENTILE}th percentile) = {LOCKED_THRESHOLD:.4f}")
print("Calculated ONLY from normal adaptation images.")
print()


# ============================================================
# 13. STEP 2 - HELD-OUT TEST
# ============================================================
print("=" * 70)
print("STEP 2 - HELD-OUT EXTERNAL TEST")
print("=" * 70)
print()
print("Threshold is LOCKED. No tuning after seeing test results.")
print()

all_results = []


def run_group(files, actual_label, title):
    print("-" * 70)
    print(title)
    print("-" * 70)
    for i, p in enumerate(files, start=1):
        try:
            s = calculate_anomaly_score(p)
        except Exception as e:
            print(f"[{i}/{len(files)}] {p.name}\n    ERROR: {e}")
            continue

        prediction = "ANOMALY" if s >= LOCKED_THRESHOLD else "NORMAL"
        correct = prediction == actual_label

        print(f"[{i}/{len(files)}] {p.name}")
        print(f"    Score: {s:.4f} | Prediction: {prediction} | Correct: {correct}")

        all_results.append({
            "Category": CATEGORY,
            "Image_Name": p.name,
            "Image_Path": str(p),
            "Actual_Label": actual_label,
            "Anomaly_Score": s,
            "Locked_Threshold": LOCKED_THRESHOLD,
            "Predicted_Label": prediction,
            "Correct": correct,
        })


run_group(normal_test_images, "NORMAL", "NORMAL TEST IMAGES")
print()
run_group(anomaly_test_images, "ANOMALY", "ANOMALY TEST IMAGES")


# ============================================================
# 14. METRICS
# ============================================================
y_true = np.array([1 if r["Actual_Label"] == "ANOMALY" else 0 for r in all_results])
y_pred = np.array([1 if r["Predicted_Label"] == "ANOMALY" else 0 for r in all_results])
y_scores = np.array([r["Anomaly_Score"] for r in all_results])

if len(set(y_true)) < 2:
    raise RuntimeError("Need both normal and anomaly images scored to compute AUROC.")

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
auroc = roc_auc_score(y_true, y_scores)
average_precision = average_precision_score(y_true, y_scores)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

n_normal = int((y_true == 0).sum())
fpr = cm[0, 1] / n_normal if n_normal else float("nan")


# ============================================================
# 15. SAVE CSV
# ============================================================
csv_path = OUTPUT_DIR / f"{CATEGORY}_external_results.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Category", "Image_Name", "Image_Path", "Actual_Label",
            "Anomaly_Score", "Locked_Threshold", "Predicted_Label", "Correct",
        ],
    )
    writer.writeheader()
    writer.writerows(all_results)


# ============================================================
# 16. SAVE SUMMARY
# ============================================================
summary_path = OUTPUT_DIR / f"{CATEGORY}_external_summary.txt"

summary_text = f"""
ANOMALYCLIP EXTERNAL {CATEGORY.upper()} EVALUATION
========================================

Category: {CATEGORY}
Device:   {DEVICE}

DATASET COUNTS
--------------
Normal adaptation images: {len(adaptation_images)}
Normal test images:       {len(normal_test_images)}
Anomaly test images:      {len(anomaly_test_images)}

ADAPTATION
----------
Minimum score:     {adaptation_scores.min():.4f}
Maximum score:     {adaptation_scores.max():.4f}
Mean score:        {adaptation_scores.mean():.4f}
Median score:      {np.median(adaptation_scores):.4f}
95th percentile:   {threshold_95:.4f}
99th percentile:   {threshold_99:.4f}

LOCKED THRESHOLD ({LOCKED_PERCENTILE}th percentile)
----------------
{LOCKED_THRESHOLD:.4f}

Calculated exclusively from normal adaptation images.
No anomaly test images were used for threshold calibration.

EXTERNAL TEST RESULTS
---------------------
Accuracy:           {accuracy:.4f}
Precision:          {precision:.4f}
Recall:             {recall:.4f}
F1 Score:           {f1:.4f}
False-positive rate:{fpr:.4f}
AUROC:              {auroc:.4f}
Average Precision:  {average_precision:.4f}

CONFUSION MATRIX
----------------
                 Predicted
              NORMAL  ANOMALY
Actual NORMAL     {cm[0, 0]:>3}     {cm[0, 1]:>3}
Actual ANOMALY    {cm[1, 0]:>3}     {cm[1, 1]:>3}

METHODOLOGY
-----------
AnomalyCLIP (pretrained VisA checkpoint) was used without retraining.
External normal adaptation images were used only to calibrate the threshold.
Held-out normal and anomaly images were used only for final evaluation.

INTERPRETATION
--------------
Small external pilot for the {CATEGORY} category. Not proof of broad
industrial generalization. AUROC and AP are independent of the threshold.
Check that normal and anomaly images are the same kind of photo, otherwise
the model may be separating photo style rather than damage.
"""

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)


# ============================================================
# 17. FINAL PRINT
# ============================================================
print()
print("=" * 70)
print(f"                 FINAL {CATEGORY.upper()} RESULT")
print("=" * 70)
print()
print(f"Normal adaptation images : {len(adaptation_images)}")
print(f"Normal test images       : {len(normal_test_images)}")
print(f"Anomaly test images      : {len(anomaly_test_images)}")
print()
print(f"95% threshold            : {threshold_95:.4f}")
print(f"99% threshold            : {threshold_99:.4f}")
print(f"LOCKED THRESHOLD         : {LOCKED_THRESHOLD:.4f}")
print()
print("-" * 70)
print(f"Accuracy                 : {accuracy:.4f}")
print(f"Precision                : {precision:.4f}")
print(f"Recall                   : {recall:.4f}")
print(f"F1 Score                 : {f1:.4f}")
print(f"False-positive rate      : {fpr:.4f}")
print(f"AUROC                    : {auroc:.4f}")
print(f"Average Precision        : {average_precision:.4f}")
print("-" * 70)
print()
print("CONFUSION MATRIX")
print()
print("                 Predicted")
print("              NORMAL  ANOMALY")
print(f"Actual NORMAL    {cm[0, 0]:>3}      {cm[0, 1]:>3}")
print(f"Actual ANOMALY   {cm[1, 0]:>3}      {cm[1, 1]:>3}")
print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)
print()
print(csv_path)
print(summary_path)
print()
print("=" * 70)
print(f"{CATEGORY.upper()} EVALUATION COMPLETE")
print("=" * 70)