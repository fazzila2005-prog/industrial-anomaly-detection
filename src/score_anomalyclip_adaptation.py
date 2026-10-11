import sys
import os

# ============================================================
# 0. FIX: make the AnomalyCLIP repo importable
# ============================================================
ANOMALYCLIP_ROOT = r"C:\Users\Fazzila\OneDrive\Desktop\AnomalyCLIP"
sys.path.insert(0, ANOMALYCLIP_ROOT)   # insert(0) so the repo's utils.py wins
os.chdir(ANOMALYCLIP_ROOT)             # in case the repo uses relative paths

import csv
import random

import numpy as np
import torch
from PIL import Image

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform


# ============================================================
# 1. PATHS
# ============================================================
PROJECT_ROOT = r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"

# ONLY NORMAL adaptation images are used here.
ADAPTATION_DIR = os.path.join(PROJECT_ROOT, "external images", "bottle", "adaptation_normal")

CHECKPOINT_PATH = os.path.join(
    ANOMALYCLIP_ROOT, "checkpoints", "9_12_4_multiscale_visa", "epoch_15.pth"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "anomalyclip_adaptation")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "anomalyclip_adaptation_scores.csv")
OUTPUT_THRESHOLD_FILE = os.path.join(OUTPUT_DIR, "anomalyclip_adaptation_thresholds.txt")


# ============================================================
# 2. ANOMALYCLIP SETTINGS
# ============================================================
IMAGE_SIZE = 518
FEATURES_LIST = [6, 12, 18, 24]
DEPTH = 9
N_CTX = 12
T_N_CTX = 4
SEED = 111


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. DEVICE
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("ANOMALYCLIP NORMAL-ONLY ADAPTATION")
print("=" * 70)
print(f"Device: {DEVICE}\n")


# ============================================================
# 5. CHECK PATHS
# ============================================================
if not os.path.exists(ADAPTATION_DIR):
    raise FileNotFoundError(f"\nAdaptation folder not found:\n{ADAPTATION_DIR}")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"\nAnomalyCLIP checkpoint not found:\n{CHECKPOINT_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 6. FIND ADAPTATION IMAGES
# ============================================================
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

image_files = sorted(
    os.path.join(ADAPTATION_DIR, f)
    for f in os.listdir(ADAPTATION_DIR)
    if f.lower().endswith(VALID_EXTENSIONS)
)

if len(image_files) == 0:
    raise RuntimeError(f"\nNo images found in:\n{ADAPTATION_DIR}")

print(f"Normal adaptation images found: {len(image_files)}\n")
for path in image_files:
    print(" -", os.path.basename(path))
print()


# ============================================================
# 7. CREATE ANOMALYCLIP MODEL
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
# 8. PREPROCESSING
# ============================================================
class TransformArgs:
    image_size = IMAGE_SIZE


preprocess, _ = get_transform(TransformArgs())


# ============================================================
# 9. LEARNED PROMPT MODEL
# ============================================================
print("Loading learned anomaly prompts...")

prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)


# ============================================================
# 10. LOAD CHECKPOINT
# ============================================================
print("Loading checkpoint...")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

if "prompt_learner" not in checkpoint:
    raise RuntimeError(
        "\nThe checkpoint does not contain 'prompt_learner'.\n"
        "This does not look like the expected AnomalyCLIP checkpoint."
    )

prompt_learner.load_state_dict(checkpoint["prompt_learner"])
prompt_learner.to(DEVICE)


# ============================================================
# 11. ENABLE ANOMALYCLIP VISUAL ADAPTATION
# ============================================================
model.to(DEVICE)
model.visual.DAPM_replace(DPAM_layer=20)


# ============================================================
# 12. LEARNED TEXT FEATURES
# ============================================================
print("Creating learned text features...")

with torch.no_grad():
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)

    text_features = model.encode_text_learn(
        prompts, tokenized_prompts, compound_prompts_text
    ).float()

    # Separate normal and abnormal text features
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print("Text features ready.\n")


# ============================================================
# 13. SCORE EACH NORMAL ADAPTATION IMAGE
# ============================================================
results = []

for index, image_path in enumerate(image_files, start=1):
    filename = os.path.basename(image_path)
    print(f"[{index}/{len(image_files)}] {filename}")

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features, patch_features = model.encode_image(
                image_tensor, FEATURES_LIST, DPAM_layer=20
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(dim=-1)

            # Second value = anomaly probability
            anomaly_score = text_probs[:, 0, 1].item()

        results.append({
            "Image_Name": filename,
            "Image_Path": image_path,
            "AnomalyCLIP_Anomaly_Score": anomaly_score,
            "Status": "ADAPTATION_NORMAL",
        })
        print(f"    Anomaly score: {anomaly_score:.4f}")

    except Exception as error:
        print(f"    ERROR: {error}")
        results.append({
            "Image_Name": filename,
            "Image_Path": image_path,
            "AnomalyCLIP_Anomaly_Score": "",
            "Status": f"ERROR: {error}",
        })


# ============================================================
# 14. KEEP ONLY SUCCESSFUL SCORES
# ============================================================
successful_scores = [
    float(r["AnomalyCLIP_Anomaly_Score"])
    for r in results
    if r["Status"] == "ADAPTATION_NORMAL"
]

if len(successful_scores) == 0:
    raise RuntimeError("\nNo adaptation images were successfully scored.")


# ============================================================
# 15. THRESHOLDS
# ============================================================
scores = np.array(successful_scores, dtype=float)

threshold_95 = np.percentile(scores, 95)
threshold_99 = np.percentile(scores, 99)
minimum_score = np.min(scores)
maximum_score = np.max(scores)
mean_score = np.mean(scores)
median_score = np.median(scores)


# ============================================================
# 16. SAVE SCORES CSV
# ============================================================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["Image_Name", "Image_Path", "AnomalyCLIP_Anomaly_Score", "Status"],
    )
    writer.writeheader()
    writer.writerows(results)


# ============================================================
# 17. SAVE THRESHOLDS
# ============================================================
with open(OUTPUT_THRESHOLD_FILE, "w", encoding="utf-8") as file:
    file.write("ANOMALYCLIP NORMAL-ONLY ADAPTATION THRESHOLDS\n")
    file.write("=" * 60 + "\n\n")
    file.write(f"Number of successful normal images: {len(successful_scores)}\n\n")
    file.write(f"Minimum normal score: {minimum_score:.6f}\n")
    file.write(f"Maximum normal score: {maximum_score:.6f}\n")
    file.write(f"Mean normal score: {mean_score:.6f}\n")
    file.write(f"Median normal score: {median_score:.6f}\n")
    file.write(f"95th percentile threshold: {threshold_95:.6f}\n")
    file.write(f"99th percentile threshold: {threshold_99:.6f}\n\n")
    file.write("IMPORTANT:\n")
    file.write("Only external NORMAL adaptation images were used to calculate these thresholds.\n")
    file.write("External anomaly images were NOT used.\n")
    file.write("The held-out external test images were NOT used.\n")
    file.write("These thresholds are intended to be locked before evaluating the held-out external test set.\n")


# ============================================================
# 18. FINAL SUMMARY
# ============================================================
print()
print("=" * 70)
print("ANOMALYCLIP NORMAL-ONLY ADAPTATION COMPLETE")
print("=" * 70)
print()
print(f"Normal adaptation images found: {len(image_files)}")
print(f"Successfully scored: {len(successful_scores)}")
print()
print("NORMAL SCORE STATISTICS")
print(f"Minimum : {minimum_score:.4f}")
print(f"Maximum : {maximum_score:.4f}")
print(f"Mean    : {mean_score:.4f}")
print(f"Median  : {median_score:.4f}")
print()
print("THRESHOLDS")
print(f"95th percentile : {threshold_95:.4f}")
print(f"99th percentile : {threshold_99:.4f}")
print()
print("Results saved to:")
print(OUTPUT_CSV)
print()
print("Thresholds saved to:")
print(OUTPUT_THRESHOLD_FILE)
print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)
print("These thresholds were calculated ONLY from normal adaptation images.")
print("No anomaly images were used.")
print("Do NOT evaluate the held-out test images yet.")
print("First inspect the thresholds above.")
print("=" * 70)