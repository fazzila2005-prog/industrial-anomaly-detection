# ================================================================
# CREATE AUGMENTED MODEL CALIBRATION
#
# Purpose:
# Create leakage-safe calibration statistics specifically for
# the AUGMENTED PatchCore and AUGMENTED PaDiM bottle models.
#
# IMPORTANT:
# - Uses ONLY MVTec bottle/train/good images
# - Does NOT use MVTec test images
# - Does NOT use external images
# - Does NOT modify the original calibration file
# ================================================================

from pathlib import Path
import json

import numpy as np
import torch

# ================================================================
# PyTorch 2.6+ checkpoint compatibility
# ================================================================

import anomalib

from torch.serialization import add_safe_globals

add_safe_globals(
    [anomalib.PrecisionType]
)

# ================================================================
# Anomalib
# ================================================================

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results"
)


# ================================================================
# MVTec TRAIN/GOOD
# ================================================================

TRAIN_GOOD_DIR = (
    PROJECT_ROOT
    / "data"
    / "bottle"
    / "train"
    / "good"
)


# ================================================================
# AUGMENTED CHECKPOINT DIRECTORIES
# ================================================================

PATCHCORE_DIR = (
    RESULTS_DIR
    / "patchcore_bottle_augmented"
)

PADIM_DIR = (
    RESULTS_DIR
    / "padim_bottle_augmented"
)


# ================================================================
# OUTPUT
# ================================================================

CALIBRATION_DIR = (
    RESULTS_DIR
    / "calibration"
)

CALIBRATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    CALIBRATION_DIR
    / "bottle_augmented_calibration_stats.json"
)


# ================================================================
# FIND CHECKPOINT
# ================================================================

def find_checkpoint(
    model_directory
):

    model_directory = Path(
        model_directory
    )

    if not model_directory.exists():

        raise FileNotFoundError(
            f"\nModel directory does not exist:\n"
            f"{model_directory}"
        )

    checkpoints = list(
        model_directory.rglob(
            "model.ckpt"
        )
    )

    if not checkpoints:

        raise FileNotFoundError(
            f"\nNo model.ckpt found inside:\n"
            f"{model_directory}"
        )

    checkpoints.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    checkpoint = checkpoints[0]

    print("\nCheckpoint found:")
    print(checkpoint)

    return checkpoint


# ================================================================
# FIND SCORE
# ================================================================

def extract_score(
    prediction
):

    possible_names = [
        "pred_score",
        "pred_scores",
        "image_score",
        "image_scores",
        "anomaly_score",
        "anomaly_scores"
    ]

    value = None

    # ------------------------------------------------------------
    # Dictionary
    # ------------------------------------------------------------

    if isinstance(
        prediction,
        dict
    ):

        for name in possible_names:

            if name in prediction:

                value = prediction[name]

                break

    # ------------------------------------------------------------
    # Object attributes
    # ------------------------------------------------------------

    if value is None:

        for name in possible_names:

            if hasattr(
                prediction,
                name
            ):

                value = getattr(
                    prediction,
                    name
                )

                break

    if value is None:

        raise RuntimeError(
            "\nCould not find image anomaly "
            "score in prediction."
        )

    # ------------------------------------------------------------
    # Convert to numpy
    # ------------------------------------------------------------

    if torch.is_tensor(value):

        value = (
            value
            .detach()
            .cpu()
            .numpy()
        )

    value = np.asarray(
        value
    ).squeeze()

    return float(
        value
    )


# ================================================================
# RUN PATCHCORE
# ================================================================

def collect_patchcore_scores(
    images,
    checkpoint
):

    print("\n")
    print("=" * 70)
    print("COLLECTING AUGMENTED PATCHCORE SCORES")
    print("=" * 70)

    model = Patchcore(

        backbone="wide_resnet50_2",

        layers=[
            "layer2",
            "layer3"
        ],

        pre_trained=True,

        coreset_sampling_ratio=0.01,

        num_neighbors=9
    )

    engine = Engine(

        accelerator="cpu",

        devices=1,

        default_root_dir=(
            RESULTS_DIR
            / "augmented_calibration_temp"
        )
    )

    scores = []

    total = len(
        images
    )

    for index, image_path in enumerate(
        images,
        start=1
    ):

        print(
            f"\nPatchCore "
            f"{index}/{total}: "
            f"{image_path.name}"
        )

        dataset = PredictDataset(

            path=str(
                image_path
            ),

            image_size=(
                256,
                256
            )
        )

        predictions = engine.predict(

            model=model,

            dataset=dataset,

            ckpt_path=str(
                checkpoint
            ),

            return_predictions=True
        )

        if (
            predictions is None
            or len(predictions) == 0
        ):

            raise RuntimeError(
                f"No prediction returned for:\n"
                f"{image_path}"
            )

        score = extract_score(
            predictions[0]
        )

        scores.append(
            score
        )

    return scores


# ================================================================
# RUN PADIM
# ================================================================

def collect_padim_scores(
    images,
    checkpoint
):

    print("\n")
    print("=" * 70)
    print("COLLECTING AUGMENTED PADIM SCORES")
    print("=" * 70)

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

        devices=1,

        default_root_dir=(
            RESULTS_DIR
            / "augmented_calibration_temp"
        )
    )

    scores = []

    total = len(
        images
    )

    for index, image_path in enumerate(
        images,
        start=1
    ):

        print(
            f"\nPaDiM "
            f"{index}/{total}: "
            f"{image_path.name}"
        )

        dataset = PredictDataset(

            path=str(
                image_path
            ),

            image_size=(
                256,
                256
            )
        )

        predictions = engine.predict(

            model=model,

            dataset=dataset,

            ckpt_path=str(
                checkpoint
            ),

            return_predictions=True
        )

        if (
            predictions is None
            or len(predictions) == 0
        ):

            raise RuntimeError(
                f"No prediction returned for:\n"
                f"{image_path}"
            )

        score = extract_score(
            predictions[0]
        )

        scores.append(
            score
        )

    return scores


# ================================================================
# STATISTICS
# ================================================================

def calculate_statistics(
    scores
):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

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

        "percentile_1": float(
            np.percentile(
                scores,
                1
            )
        ),

        "percentile_99": float(
            np.percentile(
                scores,
                99
            )
        )
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print("\n")
    print("=" * 70)
    print("AUGMENTED BOTTLE CALIBRATION")
    print("=" * 70)

    # ============================================================
    # CHECK TRAIN/GOOD
    # ============================================================

    if not TRAIN_GOOD_DIR.exists():

        raise FileNotFoundError(
            f"\nMVTec train/good directory not found:\n"
            f"{TRAIN_GOOD_DIR}"
        )

    # ============================================================
    # GET TRAIN/GOOD IMAGES
    # ============================================================

    images = sorted(
        list(
            TRAIN_GOOD_DIR.glob(
                "*.png"
            )
        )
    )

    if len(images) == 0:

        raise RuntimeError(
            "\nNo training images found."
        )

    print("\nCalibration images:")
    print(
        TRAIN_GOOD_DIR
    )

    print(
        f"\nNumber of train/good images: "
        f"{len(images)}"
    )

    # ============================================================
    # SAFETY CHECK
    # ============================================================

    print("\n")
    print(
        "LEAKAGE SAFETY CHECK"
    )

    print(
        "Using ONLY:"
    )

    print(
        "data/bottle/train/good/"
    )

    print(
        "\nNOT using:"
    )

    print(
        "data/bottle/test/"
    )

    print(
        "external images/"
    )

    print(
        "ground_truth/"
    )

    # ============================================================
    # CHECKPOINTS
    # ============================================================

    patchcore_checkpoint = (
        find_checkpoint(
            PATCHCORE_DIR
        )
    )

    padim_checkpoint = (
        find_checkpoint(
            PADIM_DIR
        )
    )

    # ============================================================
    # PATCHCORE SCORES
    # ============================================================

    patchcore_scores = (
        collect_patchcore_scores(
            images,
            patchcore_checkpoint
        )
    )

    # ============================================================
    # PADIM SCORES
    # ============================================================

    padim_scores = (
        collect_padim_scores(
            images,
            padim_checkpoint
        )
    )

    # ============================================================
    # STATISTICS
    # ============================================================

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

    # ============================================================
    # FINAL CALIBRATION OBJECT
    # ============================================================

    calibration = {

        "category": "bottle",

        "model_version": "augmented",

        "calibration_source":
            "MVTec bottle train/good only",

        "num_images": len(
            images
        ),

        "leakage_safe": True,

        "test_images_used": False,

        "external_images_used": False,

        "ground_truth_used": False,

        "PatchCore":
            patchcore_stats,

        "PaDiM":
            padim_stats
    }

    # ============================================================
    # SAVE
    # ============================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            calibration,
            f,
            indent=4
        )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print("\n")
    print("=" * 70)
    print("AUGMENTED CALIBRATION COMPLETE")
    print("=" * 70)

    print("\nPatchCore:")
    print(
        f"Count:       "
        f"{patchcore_stats['count']}"
    )
    print(
        f"Min:         "
        f"{patchcore_stats['min']:.6f}"
    )
    print(
        f"Max:         "
        f"{patchcore_stats['max']:.6f}"
    )
    print(
        f"Mean:        "
        f"{patchcore_stats['mean']:.6f}"
    )
    print(
        f"Median:      "
        f"{patchcore_stats['median']:.6f}"
    )
    print(
        f"Percentile 1:"
        f" {patchcore_stats['percentile_1']:.6f}"
    )
    print(
        f"Percentile 99:"
        f" {patchcore_stats['percentile_99']:.6f}"
    )

    print("\nPaDiM:")
    print(
        f"Count:       "
        f"{padim_stats['count']}"
    )
    print(
        f"Min:         "
        f"{padim_stats['min']:.6f}"
    )
    print(
        f"Max:         "
        f"{padim_stats['max']:.6f}"
    )
    print(
        f"Mean:        "
        f"{padim_stats['mean']:.6f}"
    )
    print(
        f"Median:      "
        f"{padim_stats['median']:.6f}"
    )
    print(
        f"Percentile 1:"
        f" {padim_stats['percentile_1']:.6f}"
    )
    print(
        f"Percentile 99:"
        f" {padim_stats['percentile_99']:.6f}"
    )

    print("\nSaved calibration to:")

    print(
        OUTPUT_FILE
    )

    print("\n")
    print("=" * 70)

    print(
        "IMPORTANT:"
    )

    print(
        "Calibration was calculated ONLY from "
        "MVTec bottle/train/good."
    )

    print(
        "MVTec test images were NOT used."
    )

    print(
        "External images were NOT used."
    )

    print(
        "This calibration is specifically "
        "for the AUGMENTED bottle models."
    )

    print("=" * 70)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()