import os
import numpy as np
import pandas as pd
import torch
import anomalib
from torch.serialization import add_safe_globals

# Ensure Anomalib structural properties deserialize correctly
add_safe_globals([anomalib.PrecisionType])

from anomalib.data import MVTecAD
from anomalib.models import Patchcore, Padim
from anomalib.engine import Engine


# ============================================================
# SETTINGS
# ============================================================

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile",
    "toothbrush", "transistor", "wood", "zipper"
]

OUTPUT_FILE = "results/consensus/per_image_evidence_fusion.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_checkpoints(category):

    p_path = (
        f"results/patchcore_{category}/Patchcore/MVTecAD/"
        f"{category}/latest/weights/lightning/model.ckpt"
    )

    d_path = (
        f"results/padim_{category}/Padim/MVTecAD/"
        f"{category}/latest/weights/lightning/model.ckpt"
    )

    return p_path, d_path


def extract_predictions(model, datamodule, ckpt_path):

    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir="results/consensus_temp"
    )

    return engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        return_predictions=True
    )


def flatten_predictions(predictions):

    flattened = []

    if predictions is None:
        return flattened

    for batch in predictions:

        if isinstance(batch, (list, tuple)):

            flattened.extend(batch)

        elif hasattr(batch, "keys") or isinstance(batch, dict):

            try:

                batch_size = len(next(iter(batch.values())))

                for i in range(batch_size):

                    item = {
                        k: v[i]
                        if hasattr(v, "__getitem__") and not isinstance(v, str)
                        else v
                        for k, v in batch.items()
                    }

                    flattened.append(item)

            except Exception:

                flattened.append(batch)

        else:

            try:

                batch_size = len(batch)

                for i in range(batch_size):

                    try:
                        flattened.append(batch[i])

                    except Exception:
                        flattened.append(batch)

            except Exception:

                flattened.append(batch)

    return flattened


def safe_convert(val, is_scalar=True):

    """
    Robust tensor-to-scalar conversion that handles
    multi-element batched prediction tensors.
    """

    if val is None:
        return 0 if is_scalar else 0.0

    if torch.is_tensor(val):

        val = val.detach().cpu().squeeze()

        if val.ndim > 0:
            val = val[0]

        return int(val.item()) if is_scalar else float(val.item())

    if isinstance(val, (np.ndarray, list)):

        val = np.squeeze(val)

        if val.ndim > 0:
            val = val[0]

        return int(val) if is_scalar else float(val)

    return int(val) if is_scalar else float(val)


def calculate_heatmap_overlap(map1, map2):

    if map1 is None or map2 is None:
        return 0.0

    m1 = torch.tensor(map1).detach().cpu().numpy()
    m2 = torch.tensor(map2).detach().cpu().numpy()

    if m1.ndim > 2:
        m1 = m1 if m1.ndim == 4 else m1

    if m2.ndim > 2:
        m2 = m2 if m2.ndim == 4 else m2

    if m1.shape != m2.shape:

        import cv2

        m2 = cv2.resize(
            m2,
            (m1.shape[1], m1.shape[0])
        )

    def norm(x):

        mn = np.min(x)
        mx = np.max(x)

        if (mx - mn) != 0:
            return (x - mn) / (mx - mn)

        return np.zeros_like(x)

    mask1 = norm(m1) >= 0.5
    mask2 = norm(m2) >= 0.5

    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return float(inter / union) if union > 0 else 1.0


# ============================================================
# MAIN ASSEMBLY
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    all_image_records = []

    print("\n" + "=" * 80)
    print("RUNNING PER-IMAGE LEVEL EVIDENCE FUSION LAYER")
    print("=" * 80)

    for category in CATEGORIES:

        p_ckpt, d_ckpt = get_checkpoints(category)

        if not os.path.exists(p_ckpt) or not os.path.exists(d_ckpt):

            print(
                f"\n⚠️ Checkpoint missing for {category.upper()} - skipped."
            )

            continue

        print(
            f"\n⚡ Extracting detailed prediction layers for: "
            f"{category.upper()}"
        )

        try:

            # ----------------------------------------------------
            # DATA
            # ----------------------------------------------------

            datamodule = MVTecAD(
                root="./data",
                category=category,
                train_batch_size=4,
                eval_batch_size=4,
                num_workers=0
            )

            datamodule.setup()


            # ----------------------------------------------------
            # MODEL CONFIGURATIONS
            # ----------------------------------------------------

            p_model = Patchcore(
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                coreset_sampling_ratio=0.01
            )

            d_model = Padim(
                backbone="resnet18"
            )


            # ----------------------------------------------------
            # GET PREDICTIONS
            # ----------------------------------------------------

            p_preds = flatten_predictions(
                extract_predictions(
                    p_model,
                    datamodule,
                    p_ckpt
                )
            )

            d_preds = flatten_predictions(
                extract_predictions(
                    d_model,
                    datamodule,
                    d_ckpt
                )
            )


            # ----------------------------------------------------
            # CHECK PREDICTION COUNTS
            # ----------------------------------------------------

            if len(p_preds) != len(d_preds):

                print(
                    f"   ⚠️ Prediction count mismatch for "
                    f"{category}: PatchCore={len(p_preds)}, "
                    f"PaDiM={len(d_preds)}"
                )

                continue


            # ----------------------------------------------------
            # RAW SCORES
            # ----------------------------------------------------

            p_raw_scores = [
                safe_convert(
                    obj.get("pred_score")
                    if isinstance(obj, dict)
                    else getattr(obj, "pred_score", 0),
                    is_scalar=False
                )
                for obj in p_preds
            ]

            d_raw_scores = [
                safe_convert(
                    obj.get("pred_score")
                    if isinstance(obj, dict)
                    else getattr(obj, "pred_score", 0),
                    is_scalar=False
                )
                for obj in d_preds
            ]


            # ----------------------------------------------------
            # CATEGORY SCORE NORMALIZATION LIMITS
            # ----------------------------------------------------

            p_min = min(p_raw_scores)
            p_max = max(p_raw_scores)

            d_min = min(d_raw_scores)
            d_max = max(d_raw_scores)


            # ====================================================
            # PROCESS EACH IMAGE
            # ====================================================

            for i in range(len(p_preds)):

                p_obj = p_preds[i]
                d_obj = d_preds[i]


                # ------------------------------------------------
                # IMAGE PATH / UNIQUE IMAGE IDENTIFICATION
                # ------------------------------------------------

                img_path = (
                    p_obj.get("image_path")
                    if isinstance(p_obj, dict)
                    else getattr(
                        p_obj,
                        "image_path",
                        "unknown"
                    )
                )

                # Handle lists / tuples / arrays
                if isinstance(
                    img_path,
                    (list, tuple, np.ndarray)
                ):

                    img_path = img_path[0]

                # Convert Windows path to standard slash format
                img_path = str(img_path).replace("\\", "/")


                # ------------------------------------------------
                # EXTRACT RELATIVE TEST PATH
                # ------------------------------------------------

                if "/test/" in img_path:

                    relative_test_path = img_path.split(
                        "/test/",
                        1
                    )[1]

                else:

                    relative_test_path = os.path.basename(
                        img_path
                    )


                # ------------------------------------------------
                # EXTRACT DEFECT TYPE
                # ------------------------------------------------

                path_parts = relative_test_path.split("/")

                if len(path_parts) >= 2:

                    defect_type = path_parts[0]

                else:

                    defect_type = "unknown"


                # ------------------------------------------------
                # IMAGE NAME
                # ------------------------------------------------

                img_name = os.path.basename(img_path)


                # ------------------------------------------------
                # UNIQUE IMAGE IDENTIFIER
                # ------------------------------------------------

                image_identifier = (
                    f"{category}/{relative_test_path}"
                )


                # ------------------------------------------------
                # LABELS
                # ------------------------------------------------

                p_label = safe_convert(
                    p_obj.get("pred_label")
                    if isinstance(p_obj, dict)
                    else getattr(
                        p_obj,
                        "pred_label",
                        0
                    )
                )

                d_label = safe_convert(
                    d_obj.get("pred_label")
                    if isinstance(d_obj, dict)
                    else getattr(
                        d_obj,
                        "pred_label",
                        0
                    )
                )


                # ------------------------------------------------
                # RAW SCORES
                # ------------------------------------------------

                p_score = p_raw_scores[i]
                d_score = d_raw_scores[i]


                # ------------------------------------------------
                # NORMALIZE ANOMALY SCORES
                # ------------------------------------------------

                p_norm = (
                    (p_score - p_min) /
                    (p_max - p_min)
                    if (p_max - p_min) != 0
                    else 0.5
                )

                d_norm = (
                    (d_score - d_min) /
                    (d_max - d_min)
                    if (d_max - d_min) != 0
                    else 0.5
                )


                # =================================================
                # 1. SCORE AGREEMENT
                # =================================================

                score_agreement = (
                    1.0 -
                    abs(p_norm - d_norm)
                )


                # =================================================
                # 2. PREDICTION AGREEMENT
                # =================================================

                pred_agreement = (
                    1.0
                    if p_label == d_label
                    else 0.0
                )


                # =================================================
                # 3. HEATMAP AGREEMENT
                # =================================================

                p_map = (
                    p_obj.get("anomaly_map")
                    if isinstance(p_obj, dict)
                    else getattr(
                        p_obj,
                        "anomaly_map",
                        None
                    )
                )

                d_map = (
                    d_obj.get("anomaly_map")
                    if isinstance(d_obj, dict)
                    else getattr(
                        d_obj,
                        "anomaly_map",
                        None
                    )
                )

                heatmap_agreement = (
                    calculate_heatmap_overlap(
                        p_map,
                        d_map
                    )
                )


                # =================================================
                # WEIGHTED CONSENSUS SCORE
                # =================================================

                consensus_score = (
                    (0.40 * score_agreement)
                    +
                    (0.35 * pred_agreement)
                    +
                    (0.25 * heatmap_agreement)
                )

                consensus_score = float(
                    np.clip(
                        consensus_score,
                        0.0,
                        1.0
                    )
                )


                # =================================================
                # CONSENSUS LEVEL
                # =================================================

                if consensus_score >= 0.80:

                    consensus_level = "HIGH"

                elif consensus_score >= 0.60:

                    consensus_level = "MEDIUM"

                else:

                    consensus_level = "LOW"


                # =================================================
                # UNCERTAINTY
                # =================================================

                uncertainty_score = (
                    1.0 -
                    consensus_score
                )


                # =================================================
                # SAVE RECORD
                # =================================================

                all_image_records.append({

                    "Category": category,

                    # Original filename
                    "Image_Name": img_name,

                    # NEW: defect folder
                    "Defect_Type": defect_type,

                    # NEW: unique full relative image path
                    "Image_Path": image_identifier,

                    # PatchCore evidence
                    "PatchCore_Raw_Score": round(
                        p_score,
                        4
                    ),

                    "PatchCore_Label": p_label,

                    # PaDiM evidence
                    "PaDiM_Raw_Score": round(
                        d_score,
                        4
                    ),

                    "PaDiM_Label": d_label,

                    # Evidence agreement
                    "Score_Agreement": round(
                        score_agreement,
                        4
                    ),

                    "Prediction_Agreement": int(
                        pred_agreement
                    ),

                    "Heatmap_Agreement_IoU": round(
                        heatmap_agreement,
                        4
                    ),

                    # Final evidence fusion
                    "Consensus_Score": round(
                        consensus_score,
                        4
                    ),

                    "Consensus_Level": consensus_level,

                    "Uncertainty_Score": round(
                        uncertainty_score,
                        4
                    )
                })


            print(
                f"   ✓ Fused {len(p_preds)} image rows successfully."
            )


        except Exception as e:

            print(
                f"   ❌ Errored loop component skipped: "
                f"{str(e)}"
            )

            continue


    # ============================================================
    # SAVE FINAL DATASET
    # ============================================================

    if all_image_records:

        master_df = pd.DataFrame(
            all_image_records
        )

        master_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\n" + "=" * 80)

        print(
            "🎉 SUCCESS: Per-Image Evidence Fusion matrix "
            f"saved cleanly to:\n{OUTPUT_FILE}"
        )

        print("=" * 80)

        print(
            f"\nTotal image records: {len(master_df)}"
        )

        print(
            "\nExample image identifiers:"
        )

        print(
            master_df[
                [
                    "Category",
                    "Defect_Type",
                    "Image_Name",
                    "Image_Path"
                ]
            ].head(10).to_string(index=False)
        )

    else:

        print(
            "\n❌ No image records were generated."
        )