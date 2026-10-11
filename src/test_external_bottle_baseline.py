from pathlib import Path
import csv
import torch

from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim
from torch.serialization import add_safe_globals
import anomalib


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_DIR = PROJECT_ROOT / "external images" / "bottle" / "test_normal"

PATCHCORE_DIR = PROJECT_ROOT / "results" / "patchcore_bottle"
PADIM_DIR = PROJECT_ROOT / "results" / "padim_bottle"

OUTPUT_DIR = PROJECT_ROOT / "results" / "external_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FIND CHECKPOINT
# --------------------------------------------------

def find_checkpoint(folder):
    checkpoints = list(folder.rglob("model.ckpt"))

    if not checkpoints:
        raise FileNotFoundError(
            f"No model.ckpt found inside:\n{folder}"
        )

    return checkpoints[0]


patchcore_ckpt = find_checkpoint(PATCHCORE_DIR)
padim_ckpt = find_checkpoint(PADIM_DIR)

print("\nCHECKPOINTS")
print("PatchCore:", patchcore_ckpt)
print("PaDiM:", padim_ckpt)


# --------------------------------------------------
# PYTORCH 2.6+ CHECKPOINT COMPATIBILITY
# --------------------------------------------------

add_safe_globals([anomalib.PrecisionType])


# --------------------------------------------------
# FIND EXTERNAL IMAGES
# --------------------------------------------------

image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

images = sorted(
    [
        p for p in EXTERNAL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
)

if not images:
    raise FileNotFoundError(
        f"No images found in:\n{EXTERNAL_DIR}"
    )

print(f"\nExternal normal images found: {len(images)}")


# --------------------------------------------------
# MODELS
# --------------------------------------------------

patchcore = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.01,
    num_neighbors=9,
)

padim = Padim(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
)


# --------------------------------------------------
# INFERENCE FUNCTION
# --------------------------------------------------

def run_model(model, checkpoint, image_path):

    engine = Engine(
        accelerator="cpu",
        devices=1,
        
    )

    predictions = engine.predict(
        model=model,
        data_path=str(image_path),
        ckpt_path=str(checkpoint),
    )

    if not predictions:
        raise RuntimeError(
            f"No prediction returned for {image_path}"
        )

    prediction = predictions[0]

    score = float(prediction.pred_score.item())

    label = "ANOMALY" if score >= 0.5 else "NORMAL"

    return score, label


# --------------------------------------------------
# RUN PATCHCORE + PADIM
# --------------------------------------------------

results = []

for i, image_path in enumerate(images, start=1):

    print(
        f"\n[{i}/{len(images)}] "
        f"{image_path.name}"
    )

    patchcore_score, patchcore_label = run_model(
        patchcore,
        patchcore_ckpt,
        image_path,
    )

    padim_score, padim_label = run_model(
        padim,
        padim_ckpt,
        image_path,
    )

    print(
        f"PatchCore: {patchcore_score:.4f} "
        f"→ {patchcore_label}"
    )

    print(
        f"PaDiM:     {padim_score:.4f} "
        f"→ {padim_label}"
    )

    results.append({
        "image": image_path.name,
        "image_path": str(image_path),
        "PatchCore_score": patchcore_score,
        "PatchCore_label": patchcore_label,
        "PaDiM_score": padim_score,
        "PaDiM_label": padim_label,
    })


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output_file = OUTPUT_DIR / "external_bottle_baseline.csv"

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

patchcore_anomalies = sum(
    r["PatchCore_label"] == "ANOMALY"
    for r in results
)

padim_anomalies = sum(
    r["PaDiM_label"] == "ANOMALY"
    for r in results
)

print("\n" + "=" * 60)
print("EXTERNAL BOTTLE BASELINE COMPLETE")
print("=" * 60)

print(f"Images tested: {len(results)}")

print(
    f"PatchCore predicted ANOMALY: "
    f"{patchcore_anomalies}/{len(results)}"
)

print(
    f"PaDiM predicted ANOMALY: "
    f"{padim_anomalies}/{len(results)}"
)

print("\nResults saved to:")
print(output_file)