import os
import numpy as np
import pandas as pd
import torch
import anomalib
from torch.serialization import add_safe_globals

# Ensure Anomalib internal configuration classes deserialize cleanly
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

RESULTS_DIR = "results/consensus"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_patchcore_checkpoint(category):
    return (
        f"results/patchcore_{category}/"
        f"Patchcore/MVTecAD/{category}/latest/"
        f"weights/lightning/model.ckpt"
    )


def get_padim_checkpoint(category):
    return (
        f"results/padim_{category}/"
        f"Padim/MVTecAD/{category}/latest/"
        f"weights/lightning/model.ckpt"
    )


def extract_predictions_fixed(model, datamodule, ckpt_path):
    """
    Passes the ckpt_path directly into engine.predict.
    This guarantees PyTorch Lightning restores model weights and memory banks cleanly.
    """
    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir="results/consensus_temp"
    )
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        return_predictions=True
    )
    return predictions


def flatten_predictions(predictions):
    """Convert prediction batches into a simple list."""
    flattened = []
    if predictions is None:
        return flattened
    for batch in predictions:
        if isinstance(batch, (list, tuple)):
            flattened.extend(batch)
        elif hasattr(batch, 'keys') or isinstance(batch, dict):
            try:
                batch_size = len(next(iter(batch.values())))
                for i in range(batch_size):
                    item = {k: v[i] if hasattr(v, '__getitem__') and not isinstance(v, str) else v for k, v in batch.items()}
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


def get_value(obj, name):
    """Safely obtain a value from an Anomalib prediction object."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def to_numpy(value):
    """Convert torch tensors / arrays into numpy arrays."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def calculate_heatmap_overlap(map1, map2):
    """
    Safely unpacks hidden batch dimensions from tensors to calculate 
    the true Intersection-over-Union spatial anomaly overlap.
    """
    if map1 is None or map2 is None:
        return np.nan

    map1 = to_numpy(map1)
    map2 = to_numpy(map2)

    # Safely slice down multi-channel batch shapes [1, 1, H, W] to clean 2D [H, W] matrices
    if map1.ndim > 2:
        map1 = map1[0, 0] if map1.ndim == 4 else map1[0]
    if map2.ndim > 2:
        map2 = map2[0, 0] if map2.ndim == 4 else map2[0]

    if map1.ndim != 2 or map2.ndim != 2:
        return np.nan

    # Resize if resolution dimensions mismatch between backbones
    if map1.shape != map2.shape:
        import cv2
        map2 = cv2.resize(map2, (map1.shape[1], map1.shape[0]))

    def normalize(x):
        x = x.astype(np.float32)
        min_val = np.min(x)
        max_val = np.max(x)
        if max_val - min_val == 0:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    map1 = normalize(map1)
    map2 = normalize(map2)

    # Threshold at 50% max anomaly confidence
    mask1 = map1 >= 0.5
    mask2 = map2 >= 0.5

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


# ============================================================
# MAIN ASSEMBLY
# ============================================================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    print("\n" + "=" * 70)
    print("PATCHCORE + PADIM MULTI-MODEL CONSENSUS EVALUATION")
    print("=" * 70)
    print(f"Total Target Categories: {len(CATEGORIES)}")
    print("Hardware Device Profile: CPU")
    print("=" * 70)

    for category in CATEGORIES:
        print("\n" + "=" * 70)
        print(f"PROCESSING CONSENSUS FOR: {category.upper()}")
        print("=" * 70)

        patchcore_ckpt = get_patchcore_checkpoint(category)
        padim_ckpt = get_padim_checkpoint(category)

        if not os.path.exists(patchcore_ckpt) or not os.path.exists(padim_ckpt):
            print(f"⚠️ Skipping {category}: Checkpoint weight paths missing.")
            continue

        try:
            # Initialize datamodule
            datamodule = MVTecAD(
                root="./data",
                category=category,
                train_batch_size=4,
                eval_batch_size=4,
                num_workers=0
            )
            datamodule.setup()

            # Create standard model shells (Weights will load cleanly in the engine via predict)
            patchcore = Patchcore(
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                coreset_sampling_ratio=0.01
            )
            padim = Padim(backbone="resnet18")

            # Extract evaluation predictions using explicit weight paths
            print("Extracting PatchCore predictions...")
            patchcore_predictions = flatten_predictions(
                extract_predictions_fixed(patchcore, datamodule, patchcore_ckpt)
            )
            
            print("Extracting PaDiM predictions...")
            padim_predictions = flatten_predictions(
                extract_predictions_fixed(padim, datamodule, padim_ckpt)
            )

            if len(patchcore_predictions) != len(padim_predictions):
                print(f"❌ Length mismatch for category {category}. Skipping data alignment.")
                continue

            overlaps = []
            class_agreements = 0
            total_samples = len(patchcore_predictions)

            # Robust unpacking tool for multi-dimensional tensor metrics
            def safe_scalar_convert(val):
                if val is None:
                    return 0
                if torch.is_tensor(val):
                    val = val.detach().cpu().squeeze()
                    if val.ndim > 0:
                        return int(val[0].item()) if hasattr(val, '__getitem__') else int(val.item())
                    return int(val.item())
                if isinstance(val, (np.ndarray, list)):
                    val = np.squeeze(val)
                    if val.ndim > 0:
                        return int(val[0])
                    return int(val)
                return int(val)

            for i in range(total_samples):
                p_obj = patchcore_predictions[i]
                d_obj = padim_predictions[i]

                p_map = get_value(p_obj, "anomaly_map")
                d_map = get_value(d_obj, "anomaly_map")
                p_pred = get_value(p_obj, "pred_label")
                d_pred = get_value(d_obj, "pred_label")

                p_val = safe_scalar_convert(p_pred)
                d_val = safe_scalar_convert(d_pred)

                if p_val == d_val:
                    class_agreements += 1

                iou_overlap = calculate_heatmap_overlap(p_map, d_map)
                if not np.isnan(iou_overlap):
                    overlaps.append(iou_overlap)

            mean_iou = np.mean(overlaps) if overlaps else 0.0
            agreement_rate = (class_agreements / total_samples) if total_samples > 0 else 0.0

            print(f"📊 Results [{category.upper()}]: Model Agreement Rate: {agreement_rate:.4f}, Mean Anomaly Map IoU Overlap: {mean_iou:.4f}")
            
            all_results.append({
                "Category": category,
                "Sample_Count": total_samples,
                "Model_Agreement_Rate": agreement_rate,
                "Mean_Heatmap_IoU": mean_iou
            })

        except Exception as e:
            print(f"❌ Failed to run evaluation loop for {category}: {str(e)}")
            continue

    # Compile Summary Data Report
    if all_results:
        summary_df = pd.DataFrame(all_results)
        csv_report_path = os.path.join(RESULTS_DIR, "consensus_results_summary.csv")
        summary_df.to_csv(csv_report_path, index=False)
        print("\n" + "=" * 70)
        print(f"🎉 SUCCESS: Consensus analysis matrix saved cleanly at: {csv_report_path}")
        print("=" * 70)
