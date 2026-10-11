import csv
import os

PROJECT_ROOT = r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "external_comparison_raw",
    "bottle"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# EXISTING ANOMALYCLIP SCORES
# These were already obtained earlier.
# We are NOT running AnomalyCLIP again.
# ============================================================

scores = {
    # NORMAL
    "30ml-clear-flat-shoulder-glass-20260202155049057.webp": 0.9428,
    "a-bottle-of-soda-free-photo.jpg": 0.8830,
    "empty-beer-bottle-23348207.webp": 0.6069,
    "good bottle 2.jpg": 0.8437,
    "good bottle 3.jpg": 0.7989,
    "good bottle external image.jpg": 0.9681,

    # ANOMALY
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
# ACTUAL TEST FOLDERS
# ============================================================

ext = os.path.join(PROJECT_ROOT, "external images", "bottle")

normal_dir = os.path.join(ext, "test_normal")
anomaly_dir = os.path.join(ext, "test_anomaly")

valid_ext = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".avif",
)

rows = []

# Normal images
for name in sorted(os.listdir(normal_dir)):
    if name.lower().endswith(valid_ext):
        if name not in scores:
            print("WARNING: Missing AnomalyCLIP score for normal:", name)
            continue

        rows.append([
            name,
            0,
            scores[name]
        ])

# Anomaly images
for name in sorted(os.listdir(anomaly_dir)):
    if name.lower().endswith(valid_ext):
        if name not in scores:
            print("WARNING: Missing AnomalyCLIP score for anomaly:", name)
            continue

        rows.append([
            name,
            1,
            scores[name]
        ])

# ============================================================
# SAVE
# ============================================================

out_path = os.path.join(
    OUT_DIR,
    "bottle_anomalyclip_scores.csv"
)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "image",
        "true_label",
        "AnomalyCLIP_raw"
    ])

    writer.writerows(rows)

print("=" * 60)
print("BOTTLE ANOMALYCLIP COMPARISON FILE CREATED")
print("=" * 60)

print("Total images:", len(rows))
print("Normal:", sum(r[1] == 0 for r in rows))
print("Anomaly:", sum(r[1] == 1 for r in rows))

print()
print("Saved:")
print(out_path)

print()
print("Expected:")
print("10 normal + 11 anomaly = 21 images")