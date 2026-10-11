# ================================================================
# EXTERNAL + MVTec INFERENCE
# AUGMENTED BOTTLE MODELS
#
# Uses:
#   - Augmented PatchCore
#   - Augmented PaDiM
#   - Augmented train/good calibration
#
# Tests:
#   1. MVTec normal bottle
#   2. External normal bottle
#   3. External broken bottle 1
#   4. External broken bottle 2
#
# IMPORTANT:
# Calibration uses ONLY:
#   data/bottle/train/good
#
# It does NOT use:
#   - MVTec test images
#   - external images
# ================================================================


from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image

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
# AUGMENTED MODEL DIRECTORIES
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
# NEW AUGMENTED CALIBRATION
# ================================================================

CALIBRATION_FILE = (
    RESULTS_DIR
    / "calibration"
    / "bottle_augmented_calibration_stats.json"
)


# ================================================================
# OUTPUT DIRECTORY
# ================================================================

OUTPUT_ROOT = (
    RESULTS_DIR
    / "external_inference_augmented"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# CHOOSE WHICH IMAGE TO TEST
#
# Change ONLY this line.
#
# Options:
#
# "mvtec_normal"
# "external_normal"
# "external_broken_1"
# "external_broken_2"
# ================================================================

TEST_NAME = "external_broken_2"


# ================================================================
# IMAGE PATHS
# ================================================================

TEST_IMAGES = {

    "mvtec_normal":
        PROJECT_ROOT
        / "data"
        / "bottle"
        / "test"
        / "good"
        / "000.png",

    "external_normal":
        PROJECT_ROOT
        / "external images"
        / "good bottle external image.jpg",

    "external_broken_1":
        PROJECT_ROOT
        / "external images"
        / "broken bottle external.webp",

    "external_broken_2":
        PROJECT_ROOT
        / "external images"
        / "broken bottle 2 external.jpg"
}


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
            f"\nModel directory not found:\n"
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

    print("\nCheckpoint:")
    print(checkpoint)

    return checkpoint


# ================================================================
# GET SCORE
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
            "\nCould not find anomaly score."
        )

    # ------------------------------------------------------------
    # Tensor -> numpy
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
# GET HEATMAP
# ================================================================

def extract_heatmap(
    prediction
):

    possible_names = [

        "anomaly_map",

        "anomaly_maps",

        "pred_mask",

        "pred_masks"
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
    # Object
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

        return None

    # ------------------------------------------------------------
    # Tensor
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
    )

    # Remove batch/channel dimensions
    # where appropriate.

    while value.ndim > 2:

        value = value[0]

    return value.astype(
        np.float32
    )


# ================================================================
# LOAD CALIBRATION
# ================================================================

def load_calibration():

    if not CALIBRATION_FILE.exists():

        raise FileNotFoundError(
            f"\nCalibration file not found:\n"
            f"{CALIBRATION_FILE}"
        )

    with open(
        CALIBRATION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        calibration = json.load(f)

    patchcore_stats = calibration[
        "PatchCore"
    ]

    padim_stats = calibration[
        "PaDiM"
    ]

    # Support both naming styles

    patchcore_p1 = patchcore_stats.get(
        "percentile_1",
        patchcore_stats.get("p1")
    )

    patchcore_p99 = patchcore_stats.get(
        "percentile_99",
        patchcore_stats.get("p99")
    )

    padim_p1 = padim_stats.get(
        "percentile_1",
        padim_stats.get("p1")
    )

    padim_p99 = padim_stats.get(
        "percentile_99",
        padim_stats.get("p99")
    )

    if (
        patchcore_p1 is None
        or patchcore_p99 is None
        or padim_p1 is None
        or padim_p99 is None
    ):

        raise RuntimeError(
            "\nCould not find percentile values "
            "inside calibration file."
        )

    print("\n")
    print("=" * 70)
    print("AUGMENTED CALIBRATION")
    print("=" * 70)

    print(
        f"\nPatchCore percentile 1: "
        f"{patchcore_p1:.6f}"
    )

    print(
        f"PatchCore percentile 99: "
        f"{patchcore_p99:.6f}"
    )

    print(
        f"\nPaDiM percentile 1: "
        f"{padim_p1:.6f}"
    )

    print(
        f"PaDiM percentile 99: "
        f"{padim_p99:.6f}"
    )

    print("=" * 70)

    return {

        "patchcore_p1":
            float(patchcore_p1),

        "patchcore_p99":
            float(patchcore_p99),

        "padim_p1":
            float(padim_p1),

        "padim_p99":
            float(padim_p99)
    }


# ================================================================
# NORMALIZE SCORE
# ================================================================

def normalize_score(
    score,
    p1,
    p99
):

    if p99 <= p1:

        return 0.0

    normalized = (
        score - p1
    ) / (
        p99 - p1
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return float(
        normalized
    )


# ================================================================
# HEATMAP IoU
# ================================================================

def calculate_heatmap_iou(
    heatmap_a,
    heatmap_b
):

    if (
        heatmap_a is None
        or heatmap_b is None
    ):

        return 0.0

    # ------------------------------------------------------------
    # Resize B to A
    # ------------------------------------------------------------

    if heatmap_a.shape != heatmap_b.shape:

        image_b = Image.fromarray(
            heatmap_b
        )

        image_b = image_b.resize(
            (
                heatmap_a.shape[1],
                heatmap_a.shape[0]
            )
        )

        heatmap_b = np.asarray(
            image_b
        )

    # ------------------------------------------------------------
    # Normalize each heatmap
    # ------------------------------------------------------------

    def normalize_heatmap(
        heatmap
    ):

        minimum = np.min(
            heatmap
        )

        maximum = np.max(
            heatmap
        )

        if maximum <= minimum:

            return np.zeros_like(
                heatmap,
                dtype=np.float32
            )

        return (
            heatmap - minimum
        ) / (
            maximum - minimum
        )

    a = normalize_heatmap(
        heatmap_a
    )

    b = normalize_heatmap(
        heatmap_b
    )

    # ------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------

    mask_a = (
        a >= 0.5
    )

    mask_b = (
        b >= 0.5
    )

    intersection = np.logical_and(
        mask_a,
        mask_b
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b
    ).sum()

    if union == 0:

        return 1.0

    return float(
        intersection / union
    )


# ================================================================
# RUN PATCHCORE
# ================================================================

def run_patchcore(
    image_path,
    checkpoint
):

    print("\n")
    print("=" * 70)
    print("PATCHCORE")
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
            / "external_inference_augmented"
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

    if (
        predictions is None
        or len(predictions) == 0
    ):

        raise RuntimeError(
            "PatchCore returned no prediction."
        )

    prediction = predictions[0]

    score = extract_score(
        prediction
    )

    heatmap = extract_heatmap(
        prediction
    )

    return score, heatmap


# ================================================================
# RUN PADIM
# ================================================================

def run_padim(
    image_path,
    checkpoint
):

    print("\n")
    print("=" * 70)
    print("PADIM")
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
            / "external_inference_augmented"
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

    if (
        predictions is None
        or len(predictions) == 0
    ):

        raise RuntimeError(
            "PaDiM returned no prediction."
        )

    prediction = predictions[0]

    score = extract_score(
        prediction
    )

    heatmap = extract_heatmap(
        prediction
    )

    return score, heatmap


# ================================================================
# MAIN
# ================================================================

def main():

    print("\n")
    print("=" * 70)
    print("AUGMENTED EXTERNAL / MVTec INFERENCE")
    print("=" * 70)

    # ============================================================
    # CHECK TEST NAME
    # ============================================================

    if TEST_NAME not in TEST_IMAGES:

        raise ValueError(
            "\nInvalid TEST_NAME.\n\n"
            "Choose one of:\n"
            "mvtec_normal\n"
            "external_normal\n"
            "external_broken_1\n"
            "external_broken_2"
        )

    image_path = TEST_IMAGES[
        TEST_NAME
    ]

    # ============================================================
    # CHECK IMAGE
    # ============================================================

    if not image_path.exists():

        raise FileNotFoundError(
            f"\nImage not found:\n"
            f"{image_path}"
        )

    print(
        f"\nTEST: {TEST_NAME}"
    )

    print(
        f"\nImage:\n"
        f"{image_path}"
    )

    # ============================================================
    # LOAD CALIBRATION
    # ============================================================

    calibration = (
        load_calibration()
    )

    # ============================================================
    # FIND CHECKPOINTS
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
    # PATCHCORE
    # ============================================================

    patchcore_raw, patchcore_heatmap = (
        run_patchcore(
            image_path,
            patchcore_checkpoint
        )
    )

    # ============================================================
    # PADIM
    # ============================================================

    padim_raw, padim_heatmap = (
        run_padim(
            image_path,
            padim_checkpoint
        )
    )

    # ============================================================
    # NORMALIZE
    # ============================================================

    patchcore_normalized = (
        normalize_score(
            patchcore_raw,
            calibration["patchcore_p1"],
            calibration["patchcore_p99"]
        )
    )

    padim_normalized = (
        normalize_score(
            padim_raw,
            calibration["padim_p1"],
            calibration["padim_p99"]
        )
    )

    # ============================================================
    # MODEL LABELS
    #
    # IMPORTANT:
    # Anomalib's original prediction label is retained here.
    # ============================================================

    # We use the normalized score as an additional
    # evidence signal.
    #
    # For this project, > 0.5 means the score is above
    # the middle of the calibrated range.

    patchcore_label = (
        "ANOMALY"
        if patchcore_normalized >= 0.5
        else "NORMAL"
    )

    padim_label = (
        "ANOMALY"
        if padim_normalized >= 0.5
        else "NORMAL"
    )

    # ============================================================
    # SCORE AGREEMENT
    # ============================================================

    score_agreement = (
        1.0
        - abs(
            patchcore_normalized
            - padim_normalized
        )
    )

    score_agreement = float(
        np.clip(
            score_agreement,
            0.0,
            1.0
        )
    )

    # ============================================================
    # PREDICTION AGREEMENT
    # ============================================================

    prediction_agreement = (

        1.0
        if patchcore_label
        == padim_label

        else 0.0
    )

    # ============================================================
    # HEATMAP AGREEMENT
    # ============================================================

    heatmap_iou = (
        calculate_heatmap_iou(
            patchcore_heatmap,
            padim_heatmap
        )
    )

    # ============================================================
    # CONSENSUS
    # ============================================================

    consensus_score = (

        0.40
        * score_agreement

        +

        0.35
        * prediction_agreement

        +

        0.25
        * heatmap_iou
    )

    consensus_score = float(
        np.clip(
            consensus_score,
            0.0,
            1.0
        )
    )

    # ============================================================
    # CONSENSUS LEVEL
    # ============================================================

    if consensus_score >= 0.80:

        consensus_level = "HIGH"

    elif consensus_score >= 0.60:

        consensus_level = "MEDIUM"

    else:

        consensus_level = "LOW"

    # ============================================================
    # UNCERTAINTY
    # ============================================================

    uncertainty_score = (
        1.0
        - consensus_score
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print("\n")
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"\nPatchCore raw score:"
        f" {patchcore_raw:.6f}"
    )

    print(
        f"PatchCore normalized:"
        f" {patchcore_normalized:.6f}"
    )

    print(
        f"PatchCore:"
        f" {patchcore_label}"
    )

    print(
        f"\nPaDiM raw score:"
        f" {padim_raw:.6f}"
    )

    print(
        f"PaDiM normalized:"
        f" {padim_normalized:.6f}"
    )

    print(
        f"PaDiM:"
        f" {padim_label}"
    )

    print(
        f"\nScore agreement:"
        f" {score_agreement:.4f}"
    )

    print(
        f"Prediction agreement:"
        f" {prediction_agreement:.4f}"
    )

    print(
        f"Heatmap IoU:"
        f" {heatmap_iou:.4f}"
    )

    print(
        f"\nConsensus score:"
        f" {consensus_score:.4f}"
    )

    print(
        f"Consensus level:"
        f" {consensus_level}"
    )

    print(
        f"Uncertainty:"
        f" {uncertainty_score:.4f}"
    )

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    output_dir = (
        OUTPUT_ROOT
        / TEST_NAME
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result = {

        "test_name":
            TEST_NAME,

        "image_path":
            str(image_path),

        "models":
            "augmented",

        "calibration_file":
            str(CALIBRATION_FILE),

        "calibration_source":
            "MVTec bottle train/good only",

        "PatchCore": {

            "raw_score":
                patchcore_raw,

            "normalized_score":
                patchcore_normalized,

            "label":
                patchcore_label
        },

        "PaDiM": {

            "raw_score":
                padim_raw,

            "normalized_score":
                padim_normalized,

            "label":
                padim_label
        },

        "score_agreement":
            score_agreement,

        "prediction_agreement":
            prediction_agreement,

        "heatmap_agreement_iou":
            heatmap_iou,

        "consensus_score":
            consensus_score,

        "consensus_level":
            consensus_level,

        "uncertainty_score":
            uncertainty_score,

        "interpretation_note":
            "Consensus and uncertainty represent "
            "agreement between model evidence. "
            "They are not calibrated probabilities."
    }

    output_file = (
        output_dir
        / "result.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    print("\n")
    print(
        "Result saved to:"
    )

    print(
        output_file
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()