# ============================================================
# EXTERNAL COMPARISON WITH RAW SCORES
#
# Why this exists: anomalib's predict() returns pred_score after
# min-max normalization (calibrated on MVTec). Outside images score
# above the MVTec maximum, so they get clipped to 1.0 and every
# image ties. Here we call the underlying model directly to get
# the RAW anomaly score, which keeps the ranking.
#
# Run in the project .venv (the one with anomalib).
# ============================================================

import os
import csv
import glob

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision import transforms as T

try:
    import pillow_avif  # noqa: F401  (pip install pillow-avif-plugin)
except ImportError:
    pass

from anomalib.models import Patchcore, Padim


# ============================================================
# 1. SETTINGS
# ============================================================
PROJECT_ROOT = r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"

# Start with one category. Add more later.
CATEGORIES = ["bottle"]

VALID_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "external_comparison_raw")
os.makedirs(OUT_DIR, exist_ok=True)

torch.set_grad_enabled(False)



# ============================================================
# 2. HELPERS
# ============================================================
def find_checkpoint(prefix, category):
    key = f"{prefix}_{category}".lower()
    matches = []
    for dirpath, _, files in os.walk(os.path.join(PROJECT_ROOT, "results")):
        if "model.ckpt" in files and key in dirpath.lower():
            full = os.path.join(dirpath, "model.ckpt")
            matches.append((os.path.getmtime(full), full))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for {key}")
    return sorted(matches)[-1][1]


def list_images(folder):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXT)
    )


def load_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    result = model.load_state_dict(state, strict=False)
    print("   loaded weights. missing:", len(result.missing_keys),
          "| unexpected:", len(result.unexpected_keys))
    model.eval()
    return model


def make_preprocess(model):
    """Use the model's own preprocessing if available, else a standard fallback."""
    pre = getattr(model, "pre_processor", None)
    if pre is not None:
        def fn(x):
            try:
                out = pre(x)
                return out.image if hasattr(out, "image") else out
            except Exception:
                return manual_pre(x)
        return fn
    return manual_pre


_manual = T.Compose([
    T.Resize((256, 256), antialias=True),
    T.CenterCrop((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def manual_pre(x):
    return _manual(x)


to_tensor = T.ToTensor()


def raw_score(model, preprocess, image_path):
    img = Image.open(image_path).convert("RGB")
    x = to_tensor(img).unsqueeze(0)
    x = preprocess(x)
    out = model.model(x)          # underlying torch model, raw output in eval mode
    score = out.pred_score if hasattr(out, "pred_score") else out["pred_score"]
    return float(score.flatten()[0])


def load_anomalyclip_scores(category):
    candidates = [
        os.path.join(PROJECT_ROOT, "results", "anomalyclip_external", category,
                     f"{category}_external_results.csv"),
        os.path.join(PROJECT_ROOT, "results", f"anomalyclip_external_{category}",
                     f"anomalyclip_external_{category}_results.csv"),
    ]
    candidates += glob.glob(
        os.path.join(PROJECT_ROOT, "results", "**", f"*{category}*.csv"), recursive=True
    )
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            name_col = next((c for c in cols if "name" in c.lower() or c.lower() == "image"), None)
            score_col = next(
                (c for c in cols if "score" in c.lower() and "thresh" not in c.lower()
                 and "patchcore" not in c.lower() and "padim" not in c.lower()), None)
            if not name_col or not score_col:
                continue
            scores = {}
            for row in reader:
                try:
                    scores[row[name_col]] = float(row[score_col])
                except (ValueError, TypeError):
                    pass
            if scores:
                print(f"   AnomalyCLIP scores read from: {path}")
                return scores
    print("   (No AnomalyCLIP CSV found; that column will be empty)")
    return {}


# ============================================================
# 3. RUN
# ============================================================
summary = []

for category in CATEGORIES:
    print("=" * 70)
    print("CATEGORY:", category)
    print("=" * 70)

    ext = os.path.join(PROJECT_ROOT, "external images", category)
    normal_files = list_images(os.path.join(ext, "test_normal"))
    anomaly_files = list_images(os.path.join(ext, "test_anomaly"))
    print(f"test_normal: {len(normal_files)} | test_anomaly: {len(anomaly_files)}")

    pc_ckpt = find_checkpoint("patchcore", category)
    pd_ckpt = find_checkpoint("padim", category)
    print("PatchCore ckpt:", pc_ckpt)
    print("PaDiM ckpt    :", pd_ckpt)

    print("Loading PatchCore...")
    patchcore = load_model(
        Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                  pre_trained=True, coreset_sampling_ratio=0.01, num_neighbors=9),
        pc_ckpt)
    print("Loading PaDiM...")
    padim = load_model(
        Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"]), pd_ckpt)

    pc_pre = make_preprocess(patchcore)
    pd_pre = make_preprocess(padim)
    clip_scores = load_anomalyclip_scores(category)

    rows = []
    items = [(p, 0) for p in normal_files] + [(p, 1) for p in anomaly_files]
    for i, (path, label) in enumerate(items, start=1):
        name = os.path.basename(path)
        try:
            pc = raw_score(patchcore, pc_pre, path)
            pd_ = raw_score(padim, pd_pre, path)
        except Exception as e:
            print(f"[{i}/{len(items)}] {name}\n    ERROR: {e}")
            continue
        clip = clip_scores.get(name, float("nan"))
        print(f"[{i}/{len(items)}] {name}")
        print(f"    PatchCore {pc:.4f} | PaDiM {pd_:.4f} | AnomalyCLIP {clip:.4f}")
        rows.append((name, label, pc, pd_, clip))

    if not rows:
        print("No images scored.")
        continue

    y = np.array([r[1] for r in rows])
    if len(set(y)) < 2:
        print("Need both normal and anomaly images.")
        continue

    with open(os.path.join(OUT_DIR, f"{category}_per_image_raw_scores.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_label", "PatchCore_raw", "PaDiM_raw", "AnomalyCLIP"])
        w.writerows(rows)

    # Sanity check: are the raw scores actually varied?
    for col, nm in ((2, "PatchCore"), (3, "PaDiM")):
        vals = np.array([r[col] for r in rows])
        print(f"{nm} raw score range: {vals.min():.4f} to {vals.max():.4f} "
              f"({len(np.unique(np.round(vals, 6)))} distinct values)")

    def auc_ap(col):
        s = np.array([r[col] for r in rows], dtype=float)
        m = ~np.isnan(s)
        if m.sum() < 2 or len(set(y[m])) < 2:
            return float("nan"), float("nan")
        return roc_auc_score(y[m], s[m]), average_precision_score(y[m], s[m])

    pc_auc, pc_ap = auc_ap(2)
    pd_auc, pd_ap = auc_ap(3)
    cl_auc, cl_ap = auc_ap(4)

    print()
    print("-" * 70)
    print(f"{category.upper()} - EXTERNAL AUROC (raw scores)")
    print(f"  PatchCore   : {pc_auc:.3f}  (AP {pc_ap:.3f})")
    print(f"  PaDiM       : {pd_auc:.3f}  (AP {pd_ap:.3f})")
    print(f"  AnomalyCLIP : {cl_auc:.3f}  (AP {cl_ap:.3f})")
    print("-" * 70)
    print()
    summary.append((category, pc_auc, pd_auc, cl_auc))

if summary:
    path = os.path.join(OUT_DIR, "external_comparison_raw_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Category", "PatchCore_AUROC", "PaDiM_AUROC", "AnomalyCLIP_AUROC"])
        w.writerows(summary)
    print("=" * 70)
    print("EXTERNAL AUROC SUMMARY")
    print("=" * 70)
    print(f"{'Category':<12}{'PatchCore':>12}{'PaDiM':>12}{'AnomalyCLIP':>14}")
    for c, a, b, d in summary:
        print(f"{c:<12}{a:>12.3f}{b:>12.3f}{d:>14.3f}")
    print("\nSaved:", path)