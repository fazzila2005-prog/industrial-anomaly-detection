import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

EVIDENCE_FILE = Path(
    "results/consensus/per_image_evidence_fusion.csv"
)

CALIBRATION_FILE = Path(
    "results/calibration/calibration_stats.json"
)

OUTPUT_DIRECTORY = Path(
    "results/consensus"
)

OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "final_consensus_results.csv"
)


# ============================================================
# CONSENSUS WEIGHTS
# ============================================================

SCORE_AGREEMENT_WEIGHT = 0.40
PREDICTION_AGREEMENT_WEIGHT = 0.35
HEATMAP_AGREEMENT_WEIGHT = 0.25


# ============================================================
# CONSENSUS THRESHOLDS
# ============================================================

HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.60


# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_calibration():

    if not CALIBRATION_FILE.exists():

        raise FileNotFoundError(
            f"Calibration file not found:\n"
            f"{CALIBRATION_FILE}"
        )

    with open(
        CALIBRATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        calibration = json.load(file)

    return calibration


# ============================================================
# NORMALIZE SCORE
# ============================================================

def normalize_score(
    score,
    lower,
    upper
):

    if upper <= lower:

        return 0.0

    normalized = (
        (score - lower)
        / (upper - lower)
    )

    return float(
        np.clip(
            normalized,
            0.0,
            1.0
        )
    )


# ============================================================
# SCORE AGREEMENT
# ============================================================

def calculate_score_agreement(
    patchcore_score,
    padim_score,
    patchcore_calibration,
    padim_calibration
):

    patchcore_normalized = normalize_score(

        patchcore_score,

        patchcore_calibration[
            "percentile_1"
        ],

        patchcore_calibration[
            "percentile_99"
        ]
    )

    padim_normalized = normalize_score(

        padim_score,

        padim_calibration[
            "percentile_1"
        ],

        padim_calibration[
            "percentile_99"
        ]
    )

    score_agreement = (
        1.0
        - abs(
            patchcore_normalized
            - padim_normalized
        )
    )

    return (
        patchcore_normalized,
        padim_normalized,
        float(
            np.clip(
                score_agreement,
                0.0,
                1.0
            )
        )
    )


# ============================================================
# CONSENSUS LEVEL
# ============================================================

def get_consensus_level(
    consensus_score
):

    if consensus_score >= HIGH_THRESHOLD:

        return "HIGH"

    elif consensus_score >= MEDIUM_THRESHOLD:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("LEAKAGE-SAFE FINAL CONSENSUS CALCULATION")
    print("=" * 80)

    print(
        "\nScore normalization source:"
    )

    print(
        "MVTec train/good calibration statistics"
    )

    print(
        "\nMVTec test images are NOT used "
        "to create score ranges."
    )

    print(
        "\nExternal images are NOT used "
        "to create score ranges."
    )

    # ========================================================
    # CHECK INPUT FILES
    # ========================================================

    if not EVIDENCE_FILE.exists():

        raise FileNotFoundError(
            f"\nEvidence file not found:\n"
            f"{EVIDENCE_FILE}"
        )

    calibration = load_calibration()

    # ========================================================
    # LOAD EXISTING EVIDENCE
    # ========================================================

    print(
        "\nLoading existing evidence data..."
    )

    df = pd.read_csv(
        EVIDENCE_FILE
    )

    print(
        f"Evidence records found: "
        f"{len(df)}"
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = [

        "Category",
        "Image_Name",
        "Defect_Type",
        "Image_Path",

        "PatchCore_Raw_Score",
        "PatchCore_Label",

        "PaDiM_Raw_Score",
        "PaDiM_Label",

        "Heatmap_Agreement_IoU"
    ]

    missing_columns = [

        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # ========================================================
    # VERIFY CALIBRATION COVERAGE
    # ========================================================

    print(
        "\nChecking calibration coverage..."
    )

    missing_categories = []

    for category in sorted(
        df["Category"].unique()
    ):

        if category not in calibration:

            missing_categories.append(
                category
            )

    if missing_categories:

        raise ValueError(
            "\nThese categories do not have "
            "calibration statistics:\n"
            + "\n".join(
                missing_categories
            )
        )

    print(
        "✓ Calibration available for "
        "all categories."
    )

    # ========================================================
    # CALCULATE FINAL CONSENSUS
    # ========================================================

    final_records = []

    print(
        "\nCalculating leakage-safe "
        "consensus..."
    )

    for _, row in df.iterrows():

        category = row["Category"]

        # ----------------------------------------------------
        # RAW SCORES
        # ----------------------------------------------------

        patchcore_score = float(
            row["PatchCore_Raw_Score"]
        )

        padim_score = float(
            row["PaDiM_Raw_Score"]
        )

        # ----------------------------------------------------
        # CALIBRATION STATISTICS
        # ----------------------------------------------------

        patchcore_calibration = (
            calibration[
                category
            ]["PatchCore"]
        )

        padim_calibration = (
            calibration[
                category
            ]["PaDiM"]
        )

        # ----------------------------------------------------
        # NORMALIZE SCORES
        # ----------------------------------------------------

        (
            patchcore_normalized,
            padim_normalized,
            score_agreement
        ) = calculate_score_agreement(

            patchcore_score,

            padim_score,

            patchcore_calibration,

            padim_calibration
        )

        # ----------------------------------------------------
        # PREDICTION AGREEMENT
        # ----------------------------------------------------

        patchcore_label = str(
            row["PatchCore_Label"]
        ).upper()

        padim_label = str(
            row["PaDiM_Label"]
        ).upper()

        if patchcore_label == padim_label:

            prediction_agreement = 1.0

        else:

            prediction_agreement = 0.0

        # ----------------------------------------------------
        # HEATMAP AGREEMENT
        # ----------------------------------------------------

        heatmap_agreement = float(
            row["Heatmap_Agreement_IoU"]
        )

        heatmap_agreement = float(
            np.clip(
                heatmap_agreement,
                0.0,
                1.0
            )
        )

        # ----------------------------------------------------
        # FINAL CONSENSUS
        # ----------------------------------------------------

        consensus_score = (

            SCORE_AGREEMENT_WEIGHT
            * score_agreement

            +

            PREDICTION_AGREEMENT_WEIGHT
            * prediction_agreement

            +

            HEATMAP_AGREEMENT_WEIGHT
            * heatmap_agreement
        )

        consensus_score = float(
            np.clip(
                consensus_score,
                0.0,
                1.0
            )
        )

        # ----------------------------------------------------
        # CONSENSUS LEVEL
        # ----------------------------------------------------

        consensus_level = (
            get_consensus_level(
                consensus_score
            )
        )

        # ----------------------------------------------------
        # UNCERTAINTY
        # ----------------------------------------------------

        uncertainty_score = (
            1.0
            - consensus_score
        )

        # ----------------------------------------------------
        # SAVE RECORD
        # ----------------------------------------------------

        final_records.append({

            "Category":
                category,

            "Image_Name":
                row["Image_Name"],

            "Defect_Type":
                row["Defect_Type"],

            "Image_Path":
                row["Image_Path"],

            "PatchCore_Raw_Score":
                patchcore_score,

            "PaDiM_Raw_Score":
                padim_score,

            "PatchCore_Normalized_Score":
                patchcore_normalized,

            "PaDiM_Normalized_Score":
                padim_normalized,

            "PatchCore_Prediction":
                patchcore_label,

            "PaDiM_Prediction":
                padim_label,

            "Score_Agreement":
                score_agreement,

            "Prediction_Agreement":
                prediction_agreement,

            "Heatmap_Agreement_IoU":
                heatmap_agreement,

            "Consensus_Score":
                consensus_score,

            "Consensus_Level":
                consensus_level,

            "Uncertainty_Score":
                uncertainty_score
        })

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    final_df = pd.DataFrame(
        final_records
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL CONSENSUS CALCULATION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTotal records: "
        f"{len(final_df)}"
    )

    print(
        "\nCategories:"
    )

    print(
        final_df["Category"]
        .nunique()
    )

    print(
        "\nConsensus levels:"
    )

    print(
        final_df[
            "Consensus_Level"
        ].value_counts()
    )

    print(
        "\nConsensus score range:"
    )

    print(
        f"Minimum: "
        f"{final_df['Consensus_Score'].min():.4f}"
    )

    print(
        f"Maximum: "
        f"{final_df['Consensus_Score'].max():.4f}"
    )

    print(
        "\nUncertainty score range:"
    )

    print(
        f"Minimum: "
        f"{final_df['Uncertainty_Score'].min():.4f}"
    )

    print(
        f"Maximum: "
        f"{final_df['Uncertainty_Score'].max():.4f}"
    )

    print(
        "\nOutput file:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The consensus score is an "
        "evidence-consistency score."
    )

    print(
        "It is NOT a probability."
    )

    print(
        "\nThe uncertainty score is also "
        "an evidence-consistency indicator,"
    )

    print(
        "not a calibrated probability."
    )

    print(
        "\nScore normalization uses only "
        "MVTec train/good calibration data."
    )

    print(
        "\nNo MVTec test scores or external "
        "image scores were used to create "
        "the calibration ranges."
    )

    print(
        "\n" + "=" * 80
    )