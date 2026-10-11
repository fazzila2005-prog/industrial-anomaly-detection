# ================================================================
# EXPERIMENT 2
# CONTROLLED PREPROCESSING FOR EXTERNAL NORMAL BOTTLE
#
# Uses:
#   - AUGMENTED PatchCore
#   - AUGMENTED PaDiM
#   - AUGMENTED calibration
#
# Tests the SAME external normal bottle with different
# preprocessing strategies.
#
# No retraining.
# No test images used for calibration.
# No external images used for calibration.
# ================================================================

from pathlib import Path
import json

import cv2
import numpy as np
import torch
from PIL import Image

import anomalib
from torch.serialization import add_safe_globals

add_safe_globals(
    [anomalib.PrecisionType]
)

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim


# ================================================================
# PROJECT
# ================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"
)

RESULTS_DIR = PROJECT_ROOT / "results"


# ================================================================
# INPUT IMAGE
# ================================================================

INPUT_IMAGE = (
    PROJECT_ROOT
    / "external images"
    / "good bottle external image.jpg"
)


# ================================================================
# MODELS
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
# AUGMENTED CALIBRATION
# ================================================================

CALIBRATION_FILE = (
    RESULTS_DIR
    / "calibration"
    / "bottle_augmented_calibration_stats.json"
)


# ================================================================
# OUTPUT
# ================================================================

OUTPUT_DIR = (
    RESULTS_DIR
    / "preprocessing_experiment"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# FIND CHECKPOINT
# ================================================================

def find_checkpoint(directory):

    checkpoints = list(
        Path(directory).rglob("model.ckpt")
    )

    if not checkpoints:

        raise FileNotFoundError(
            f"No model.ckpt found in:\n{directory}"
        )

    checkpoints.sort(
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    return checkpoints[0]


# ================================================================
# LOAD CALIBRATION
# ================================================================

def load_calibration():

    with open(
        CALIBRATION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    pc = data["PatchCore"]
    pd = data["PaDiM"]

    return {

        "patchcore_p1":
            pc["percentile_1"],

        "patchcore_p99":
            pc["percentile_99"],

        "padim_p1":
            pd["percentile_1"],

        "padim_p99":
            pd["percentile_99"]
    }


# ================================================================
# NORMALIZE
# ================================================================

def normalize_score(
    score,
    p1,
    p99
):

    if p99 <= p1:

        return 0.0

    value = (
        score - p1
    ) / (
        p99 - p1
    )

    return float(
        np.clip(
            value,
            0.0,
            1.0
        )
    )


# ================================================================
# EXTRACT SCORE
# ================================================================

def extract_score(prediction):

    names = [
        "pred_score",
        "pred_scores",
        "image_score",
        "image_scores",
        "anomaly_score",
        "anomaly_scores"
    ]

    value = None

    if isinstance(
        prediction,
        dict
    ):

        for name in names:

            if name in prediction:

                value = prediction[name]
                break

    if value is None:

        for name in names:

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
            "Could not find anomaly score."
        )

    if torch.is_tensor(value):

        value = (
            value
            .detach()
            .cpu()
            .numpy()
        )

    return float(
        np.asarray(
            value
        ).squeeze()
    )


# ================================================================
# CREATE VARIANTS
# ================================================================

def create_variants():

    print("\n")
    print("=" * 70)
    print("CREATING PREPROCESSING VARIANTS")
    print("=" * 70)

    if not INPUT_IMAGE.exists():

        raise FileNotFoundError(
            f"\nInput image not found:\n"
            f"{INPUT_IMAGE}"
        )

    image = cv2.imread(
        str(INPUT_IMAGE)
    )

    if image is None:

        raise RuntimeError(
            "OpenCV could not read the image."
        )

    height, width = image.shape[:2]

    print(
        f"\nOriginal size: "
        f"{width} x {height}"
    )

    variants = {}

    # ============================================================
    # 1. ORIGINAL
    # ============================================================

    variants["01_original"] = image.copy()

    # ============================================================
    # 2. SQUARE PADDED
    # ============================================================

    size = max(
        height,
        width
    )

    square = np.zeros(
        (
            size,
            size,
            3
        ),
        dtype=np.uint8
    )

    y_offset = (
        size - height
    ) // 2

    x_offset = (
        size - width
    ) // 2

    square[
        y_offset:
        y_offset + height,

        x_offset:
        x_offset + width
    ] = image

    variants[
        "02_square_padded"
    ] = square

    # ============================================================
    # 3. CENTER CROP
    # ============================================================

    crop_size = min(
        height,
        width
    )

    y1 = (
        height - crop_size
    ) // 2

    x1 = (
        width - crop_size
    ) // 2

    center_crop = image[
        y1:
        y1 + crop_size,

        x1:
        x1 + crop_size
    ]

    variants[
        "03_center_crop"
    ] = center_crop

    # ============================================================
    # 4. LOWER-RIGHT OBJECT CROP
    #
    # Your external bottle is positioned toward the lower-right
    # portion of the image.
    #
    # We deliberately keep a generous region so that we don't
    # accidentally crop the bottle itself.
    # ============================================================

    crop_x1 = int(
        width * 0.25
    )

    crop_y1 = int(
        height * 0.25
    )

    object_crop = image[
        crop_y1:height,
        crop_x1:width
    ]

    variants[
        "04_object_crop"
    ] = object_crop

    # ============================================================
    # 5. OBJECT CROP + SQUARE PAD
    # ============================================================

    h2, w2 = object_crop.shape[:2]

    square_size = max(
        h2,
        w2
    )

    object_square = np.zeros(
        (
            square_size,
            square_size,
            3
        ),
        dtype=np.uint8
    )

    yy = (
        square_size - h2
    ) // 2

    xx = (
        square_size - w2
    ) // 2

    object_square[
        yy:
        yy + h2,

        xx:
        xx + w2
    ] = object_crop

    variants[
        "05_object_crop_square"
    ] = object_square

    # ============================================================
    # SAVE
    # ============================================================

    paths = {}

    for name, variant in variants.items():

        path = (
            OUTPUT_DIR
            / f"{name}.jpg"
        )

        cv2.imwrite(
            str(path),
            variant
        )

        paths[name] = path

        print(
            f"\n{name}:"
        )

        print(
            f"  size = "
            f"{variant.shape[1]} x "
            f"{variant.shape[0]}"
        )

        print(
            f"  saved = {path}"
        )

    return paths


# ================================================================
# RUN PATCHCORE
# ================================================================

def run_patchcore(
    image_path,
    checkpoint
):

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
            / "preprocessing_experiment"
        )
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

    prediction = predictions[0]

    return extract_score(
        prediction
    )


# ================================================================
# RUN PADIM
# ================================================================

def run_padim(
    image_path,
    checkpoint
):

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
            / "preprocessing_experiment"
        )
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

    prediction = predictions[0]

    return extract_score(
        prediction
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("\n")
    print("=" * 70)
    print("EXPERIMENT 2")
    print("EXTERNAL NORMAL BOTTLE PREPROCESSING")
    print("=" * 70)

    # ============================================================
    # CALIBRATION
    # ============================================================

    calibration = load_calibration()

    # ============================================================
    # CHECKPOINTS
    # ============================================================

    patchcore_checkpoint = find_checkpoint(
        PATCHCORE_DIR
    )

    padim_checkpoint = find_checkpoint(
        PADIM_DIR
    )

    print(
        "\nPatchCore checkpoint:"
    )

    print(
        patchcore_checkpoint
    )

    print(
        "\nPaDiM checkpoint:"
    )

    print(
        padim_checkpoint
    )

    # ============================================================
    # CREATE IMAGES
    # ============================================================

    variants = create_variants()

    # ============================================================
    # RESULTS
    # ============================================================

    results = []

    # ============================================================
    # TEST EACH VARIANT
    # ============================================================

    for name, image_path in variants.items():

        print("\n")
        print("=" * 70)

        print(
            f"TESTING: {name}"
        )

        print("=" * 70)

        # --------------------------------------------------------
        # PATCHCORE
        # --------------------------------------------------------

        patchcore_raw = run_patchcore(
            image_path,
            patchcore_checkpoint
        )

        patchcore_norm = normalize_score(

            patchcore_raw,

            calibration[
                "patchcore_p1"
            ],

            calibration[
                "patchcore_p99"
            ]
        )

        patchcore_label = (

            "ANOMALY"
            if patchcore_norm >= 0.5
            else "NORMAL"
        )

        # --------------------------------------------------------
        # PADIM
        # --------------------------------------------------------

        padim_raw = run_padim(
            image_path,
            padim_checkpoint
        )

        padim_norm = normalize_score(

            padim_raw,

            calibration[
                "padim_p1"
            ],

            calibration[
                "padim_p99"
            ]
        )

        padim_label = (

            "ANOMALY"
            if padim_norm >= 0.5
            else "NORMAL"
        )

        # --------------------------------------------------------
        # AGREEMENT
        # --------------------------------------------------------

        score_agreement = (

            1.0
            -
            abs(
                patchcore_norm
                -
                padim_norm
            )
        )

        prediction_agreement = (

            1.0
            if patchcore_label
            ==
            padim_label

            else 0.0
        )

        # --------------------------------------------------------
        # PRINT
        # --------------------------------------------------------

        print(
            f"\nPatchCore raw: "
            f"{patchcore_raw:.6f}"
        )

        print(
            f"PatchCore normalized: "
            f"{patchcore_norm:.6f}"
        )

        print(
            f"PatchCore: "
            f"{patchcore_label}"
        )

        print(
            f"\nPaDiM raw: "
            f"{padim_raw:.6f}"
        )

        print(
            f"PaDiM normalized: "
            f"{padim_norm:.6f}"
        )

        print(
            f"PaDiM: "
            f"{padim_label}"
        )

        print(
            f"\nScore agreement: "
            f"{score_agreement:.4f}"
        )

        print(
            f"Prediction agreement: "
            f"{prediction_agreement:.4f}"
        )

        # --------------------------------------------------------
        # SAVE
        # --------------------------------------------------------

        results.append({

            "variant":
                name,

            "image":
                str(image_path),

            "PatchCore_raw":
                patchcore_raw,

            "PatchCore_normalized":
                patchcore_norm,

            "PatchCore_label":
                patchcore_label,

            "PaDiM_raw":
                padim_raw,

            "PaDiM_normalized":
                padim_norm,

            "PaDiM_label":
                padim_label,

            "score_agreement":
                score_agreement,

            "prediction_agreement":
                prediction_agreement
        })

    # ============================================================
    # SAVE JSON
    # ============================================================

    json_file = (
        OUTPUT_DIR
        / "experiment_2_results.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # ============================================================
    # SAVE CSV
    # ============================================================

    import csv

    csv_file = (
        OUTPUT_DIR
        / "experiment_2_results.csv"
    )

    fieldnames = list(
        results[0].keys()
    )

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ============================================================
    # FINAL TABLE
    # ============================================================

    print("\n")
    print("=" * 90)
    print("EXPERIMENT 2 — FINAL COMPARISON")
    print("=" * 90)

    print(
        f"{'Variant':<28}"
        f"{'PatchCore':<15}"
        f"{'PaDiM':<15}"
        f"{'PC Norm':<12}"
        f"{'PD Norm':<12}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['variant']:<28}"
            f"{result['PatchCore_label']:<15}"
            f"{result['PaDiM_label']:<15}"
            f"{result['PatchCore_normalized']:<12.3f}"
            f"{result['PaDiM_normalized']:<12.3f}"
        )

    print("=" * 90)

    print(
        "\nJSON saved:"
    )

    print(
        json_file
    )

    print(
        "\nCSV saved:"
    )

    print(
        csv_file
    )

    print("\nDONE.")


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()