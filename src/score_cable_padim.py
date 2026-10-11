import os
import csv
import torch
import anomalib

from torch.serialization import add_safe_globals

add_safe_globals([
    anomalib.PrecisionType
])

from anomalib.engine import Engine
from anomalib.models import Padim


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"

EXTERNAL_ROOT = os.path.join(
    PROJECT_ROOT,
    "external images",
    "cable"
)

NORMAL_DIR = os.path.join(
    EXTERNAL_ROOT,
    "test_normal"
)

ANOMALY_DIR = os.path.join(
    EXTERNAL_ROOT,
    "test_anomaly"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "external_comparison",
    "cable"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIND CHECKPOINT
# ============================================================

def find_checkpoint():

    root = os.path.join(
        PROJECT_ROOT,
        "results",
        "padim_cable"
    )

    for dirpath, _, filenames in os.walk(root):

        if "model.ckpt" in filenames:

            return os.path.join(
                dirpath,
                "model.ckpt"
            )

    raise FileNotFoundError(
        "PaDiM checkpoint not found."
    )


CHECKPOINT = find_checkpoint()


# ============================================================
# IMAGES
# ============================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".avif"
)


def get_images(folder):

    files = []

    for filename in sorted(
        os.listdir(folder)
    ):

        if filename.lower().endswith(
            VALID_EXTENSIONS
        ):

            files.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    return files


normal_images = get_images(
    NORMAL_DIR
)

anomaly_images = get_images(
    ANOMALY_DIR
)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("PADIM - EXTERNAL CABLE SCORING")
print("=" * 70)

print(
    "\nCheckpoint:"
)

print(CHECKPOINT)


model = Padim(
    backbone="resnet18",
    layers=[
        "layer1",
        "layer2",
        "layer3"
    ]
)

engine = Engine(
    accelerator="cpu",
    devices=1
)


# ============================================================
# RUN
# ============================================================

results = []

all_images = []

for path in normal_images:

    all_images.append(
        (path, 0)
    )

for path in anomaly_images:

    all_images.append(
        (path, 1)
    )


print(
    f"\nImages: {len(all_images)}"
)


for i, (image_path, true_label) in enumerate(
    all_images,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    print(
        f"\n[{i}/{len(all_images)}] {filename}"
    )

    predictions = engine.predict(
        model=model,
        data_path=image_path,
        ckpt_path=CHECKPOINT
    )

    prediction = predictions[0]

    score = prediction.pred_score

    if torch.is_tensor(score):

        score = score.item()

    score = float(score)

    print(
        f"    PaDiM score: {score:.6f}"
    )

    results.append({
        "image": filename,
        "path": image_path,
        "true_label": true_label,
        "PaDiM_score": score
    })


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "cable_padim_scores.csv"
)

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image",
            "path",
            "true_label",
            "PaDiM_score"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print("\n" + "=" * 70)
print("PADIM SCORING COMPLETE")
print("=" * 70)

print(output_file)