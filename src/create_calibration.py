import os
import json
import numpy as np
import torch
import anomalib

from pathlib import Path
from torch.serialization import add_safe_globals

from anomalib.data import PredictDataset
from anomalib.models import Patchcore, Padim
from anomalib.engine import Engine


# Allow Anomalib checkpoint metadata to load with PyTorch 2.6+
add_safe_globals([anomalib.PrecisionType])


# ============================================================
# SETTINGS
# ============================================================

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper"
]

DATA_ROOT = Path("data")

OUTPUT_DIRECTORY = Path("results/calibration")
OUTPUT_FILE = OUTPUT_DIRECTORY / "calibration_stats.json"


# ============================================================
# FIND CHECKPOINT AUTOMATICALLY
# ============================================================

def find_checkpoint(model_name, category):

    if model_name == "Patchcore":
        base_folder = Path("results") / f"patchcore_{category}" / "Patchcore"

    elif model_name == "Padim":
        base_folder = Path("results") / f"padim_{category}" / "Padim"

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if not base_folder.exists():
        return None

    checkpoints = list(base_folder.rglob("model.ckpt"))

    if len(checkpoints) == 0:
        return None

    # If multiple checkpoints exist, use the newest one
    checkpoints = sorted(
        checkpoints,
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    return checkpoints[0]


# ============================================================
# RUN MODEL ON TRAINING IMAGES
# ============================================================

def run_model_on_training_images(
    model,
    image_folder,
    checkpoint_path
):

    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir="results/calibration_temp"
    )

    dataset = PredictDataset(
        path=image_folder,
        image_size=(256, 256)
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(checkpoint_path),
        return_predictions=True
    )

    return predictions


# ============================================================
# FLATTEN PREDICTIONS
# ============================================================

def flatten_predictions(predictions):

    flattened = []

    if predictions is None:
        return flattened

    for batch in predictions:

        try:

            items = list(batch)

            if items:
                flattened.extend(items)
                continue

        except Exception:
            pass

        flattened.append(batch)

    return flattened


# ============================================================
# EXTRACT ANOMALY SCORE
# ============================================================

def extract_score(prediction):

    score = getattr(
        prediction,
        "pred_score",
        None
    )

    if score is None and isinstance(prediction, dict):

        score = prediction.get(
            "pred_score"
        )

    if score is None:

        raise ValueError(
            "Could not find pred_score in prediction."
        )

    # PyTorch tensor
    if torch.is_tensor(score):

        score = score.detach().cpu().squeeze()

        if score.ndim > 0:

            score = score.flatten()[0]

        return float(score.item())

    # NumPy array
    if isinstance(score, np.ndarray):

        score = np.squeeze(score)

        if score.ndim > 0:

            score = score.flatten()[0]

        return float(score)

    return float(score)


# ============================================================
# CALCULATE CALIBRATION STATISTICS
# ============================================================

def calculate_statistics(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(scores) == 0:

        raise ValueError(
            "No scores available for calibration."
        )

    lower = float(
        np.percentile(scores, 1)
    )

    upper = float(
        np.percentile(scores, 99)
    )

    # Safety fallback
    if upper <= lower:

        minimum = float(
            np.min(scores)
        )

        maximum = float(
            np.max(scores)
        )

        lower = minimum
        upper = maximum

    return {

        "count": int(
            len(scores)
        ),

        "min": float(
            np.min(scores)
        ),

        "max": float(
            np.max(scores)
        ),

        "mean": float(
            np.mean(scores)
        ),

        "median": float(
            np.median(scores)
        ),

        "percentile_1": lower,

        "percentile_99": upper
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("LEAKAGE-SAFE SCORE CALIBRATION")
    print("=" * 80)

    print("\nCalibration source:")
    print("MVTec train/good images ONLY")

    print("\nMVTec test images are NOT used.")
    print("External images are NOT used.")

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    all_calibration = {}

    # ========================================================
    # LOOP THROUGH ALL CATEGORIES
    # ========================================================

    for category in CATEGORIES:

        print("\n" + "-" * 80)
        print(
            f"CALIBRATING: {category.upper()}"
        )
        print("-" * 80)

        # ----------------------------------------------------
        # Training folder
        # ----------------------------------------------------

        train_folder = (
            DATA_ROOT
            / category
            / "train"
            / "good"
        )

        if not train_folder.exists():

            print(
                f"⚠️ Training folder missing:\n"
                f"{train_folder}"
            )

            continue

        # ----------------------------------------------------
        # Automatically find checkpoints
        # ----------------------------------------------------

        patchcore_ckpt = find_checkpoint(
            "Patchcore",
            category
        )

        padim_ckpt = find_checkpoint(
            "Padim",
            category
        )

        if patchcore_ckpt is None:

            print(
                "⚠️ PatchCore checkpoint not found."
            )

            continue

        if padim_ckpt is None:

            print(
                "⚠️ PaDiM checkpoint not found."
            )

            continue

        print(
            "\nPatchCore checkpoint:"
        )

        print(
            patchcore_ckpt
        )

        print(
            "\nPaDiM checkpoint:"
        )

        print(
            padim_ckpt
        )

        # ----------------------------------------------------
        # Find training images
        # ----------------------------------------------------

        image_files = []

        for extension in [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp"
        ]:

            image_files.extend(
                train_folder.glob(extension)
            )

        image_files = sorted(
            image_files
        )

        print(
            f"\nTraining images found: "
            f"{len(image_files)}"
        )

        if len(image_files) == 0:

            print(
                "⚠️ No training images found."
            )

            continue

        # ====================================================
        # PATCHCORE
        # ====================================================

        print(
            "\nRunning PatchCore on "
            "training/good images..."
        )

        patchcore_model = Patchcore(

            backbone="wide_resnet50_2",

            layers=[
                "layer2",
                "layer3"
            ],

            coreset_sampling_ratio=0.01,

            num_neighbors=9
        )

        patchcore_predictions = (
            run_model_on_training_images(

                patchcore_model,

                train_folder,

                patchcore_ckpt
            )
        )

        patchcore_predictions = (
            flatten_predictions(
                patchcore_predictions
            )
        )

        patchcore_scores = []

        for prediction in patchcore_predictions:

            try:

                score = extract_score(
                    prediction
                )

                patchcore_scores.append(
                    score
                )

            except Exception as e:

                print(
                    "⚠️ Could not extract "
                    f"PatchCore score: {e}"
                )

        print(
            f"PatchCore predictions: "
            f"{len(patchcore_scores)}"
        )

        # ====================================================
        # PADIM
        # ====================================================

        print(
            "\nRunning PaDiM on "
            "training/good images..."
        )

        padim_model = Padim(

            backbone="resnet18",

            layers=[
                "layer1",
                "layer2",
                "layer3"
            ]
        )

        padim_predictions = (
            run_model_on_training_images(

                padim_model,

                train_folder,

                padim_ckpt
            )
        )

        padim_predictions = (
            flatten_predictions(
                padim_predictions
            )
        )

        padim_scores = []

        for prediction in padim_predictions:

            try:

                score = extract_score(
                    prediction
                )

                padim_scores.append(
                    score
                )

            except Exception as e:

                print(
                    "⚠️ Could not extract "
                    f"PaDiM score: {e}"
                )

        print(
            f"PaDiM predictions: "
            f"{len(padim_scores)}"
        )

        # ====================================================
        # CHECK RESULTS
        # ====================================================

        if len(patchcore_scores) == 0:

            print(
                "❌ No PatchCore scores."
                " Skipping category."
            )

            continue

        if len(padim_scores) == 0:

            print(
                "❌ No PaDiM scores."
                " Skipping category."
            )

            continue

        # ====================================================
        # CALCULATE STATISTICS
        # ====================================================

        patchcore_stats = (
            calculate_statistics(
                patchcore_scores
            )
        )

        padim_stats = (
            calculate_statistics(
                padim_scores
            )
        )

        # ====================================================
        # SAVE CATEGORY
        # ====================================================

        all_calibration[category] = {

            "reference_split":
                "train/good",

            "reference_count":
                len(image_files),

            "PatchCore":
                patchcore_stats,

            "PaDiM":
                padim_stats
        }

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            "\nPatchCore calibration:"
        )

        print(
            "  1st percentile  : "
            f"{patchcore_stats['percentile_1']:.6f}"
        )

        print(
            "  99th percentile : "
            f"{patchcore_stats['percentile_99']:.6f}"
        )

        print(
            "\nPaDiM calibration:"
        )

        print(
            "  1st percentile  : "
            f"{padim_stats['percentile_1']:.6f}"
        )

        print(
            "  99th percentile : "
            f"{padim_stats['percentile_99']:.6f}"
        )

        print(
            f"\n✓ {category.upper()} "
            "calibration complete."
        )

    # ========================================================
    # FINAL CHECK
    # ========================================================

    if not all_calibration:

        print(
            "\n❌ No calibration data generated."
        )

        raise SystemExit(1)

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_calibration,
            file,
            indent=4
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)

    print(
        f"\nCategories calibrated: "
        f"{len(all_calibration)} / "
        f"{len(CATEGORIES)}"
    )

    print(
        "\nCalibration file:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nIMPORTANT:"
        "\nThe calibration statistics were calculated "
        "only from MVTec train/good images."
        "\nMVTec test images and external images "
        "were not used to create these ranges."
    )

    print(
        "\nThese fixed category-specific ranges will "
        "later be used to normalize PatchCore and "
        "PaDiM scores for the app."
    )