import os
import sys
import csv
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"
ANOMALYCLIP_ROOT = r"C:\Users\Fazzila\OneDrive\Desktop\AnomalyCLIP"

EXTERNAL_ROOT = os.path.join(
    PROJECT_ROOT, "external images", "cable"
)

NORMAL_DIR = os.path.join(
    EXTERNAL_ROOT, "test_normal"
)

ANOMALY_DIR = os.path.join(
    EXTERNAL_ROOT, "test_anomaly"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "external_comparison",
    "cable"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ANOMALYCLIP IMPORTS
# ============================================================

sys.path.insert(0, ANOMALYCLIP_ROOT)

from AnomalyCLIP_lib.prompt_learner import AnomalyCLIP_PromptLearner
from AnomalyCLIP_lib.anomalyclip import load

# ============================================================
# ANOMALYCLIP SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = os.path.join(
    ANOMALYCLIP_ROOT,
    "checkpoints",
    "9_12_4_multiscale_visa",
    "epoch_15.pth"
)

N_CTX = 12
T_N_CTX = 4
DEPTH = 9

FEATURES_LIST = [6, 12, 18, 24]

AnomalyCLIP_parameters = {
    "Prompt_length": N_CTX,
    "learnabel_text_embedding_length": T_N_CTX,
    "learnabel_text_embedding_depth": DEPTH,
    "DPAM_layer": 20,
}

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".avif",
)

# ============================================================
# FIND ANOMALIB CHECKPOINT
# ============================================================

def find_checkpoint(root, model_name):

    matches = []

    for dirpath, _, filenames in os.walk(root):

        for filename in filenames:

            if filename == "model.ckpt":

                if model_name.lower() in dirpath.lower():

                    matches.append(
                        os.path.join(
                            dirpath,
                            filename
                        )
                    )

    if not matches:

        raise FileNotFoundError(
            f"Could not find {model_name} checkpoint under {root}"
        )

    return matches[0]


PATCHCORE_CKPT = find_checkpoint(
    os.path.join(
        PROJECT_ROOT,
        "results"
    ),
    "patchcore_cable"
)

PADIM_CKPT = find_checkpoint(
    os.path.join(
        PROJECT_ROOT,
        "results"
    ),
    "padim_cable"
)

# ============================================================
# START
# ============================================================

print("=" * 70)
print("CABLE EXTERNAL THREE-MODEL COMPARISON")
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nPatchCore checkpoint:")
print(PATCHCORE_CKPT)

print("\nPaDiM checkpoint:")
print(PADIM_CKPT)

# ============================================================
# LOAD ANOMALYCLIP
# ============================================================

print("\n" + "=" * 70)
print("LOADING ANOMALYCLIP")
print("=" * 70)

model, preprocess = load(
    "ViT-L/14@336px",
    device=DEVICE,
    download_root=os.path.join(
        ANOMALYCLIP_ROOT,
        "pretrained"
    )
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["state_dict"],
    strict=False
)

model.eval()

prompt_learner = AnomalyCLIP_PromptLearner(
    model.to("cpu"),
    AnomalyCLIP_parameters
)

model.to(DEVICE)
prompt_learner.to(DEVICE)

with torch.no_grad():

    prompts, tokenized_prompts, compound_prompts_text = (
        prompt_learner(cls_id=None)
    )

    text_features = model.encode_text_learn(
        prompts,
        tokenized_prompts,
        compound_prompts_text
    ).float()

    text_features = torch.stack(
        torch.chunk(
            text_features,
            dim=0,
            chunks=2
        ),
        dim=1
    )

    text_features = (
        text_features /
        text_features.norm(
            dim=-1,
            keepdim=True
        )
    )

print("AnomalyCLIP loaded successfully.")

# ============================================================
# LOAD ANOMALIB
# ============================================================

print("\n" + "=" * 70)
print("LOADING PATCHCORE AND PADIM")
print("=" * 70)

import anomalib

from torch.serialization import add_safe_globals

add_safe_globals([
    anomalib.PrecisionType
])

from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim

# ------------------------------------------------------------
# PatchCore
# ------------------------------------------------------------

patchcore_model = Patchcore(
    backbone="wide_resnet50_2",
    layers=[
        "layer2",
        "layer3"
    ],
    pre_trained=True,
    coreset_sampling_ratio=0.01,
    num_neighbors=9,
)

# ------------------------------------------------------------
# PaDiM
# ------------------------------------------------------------

padim_model = Padim(
    backbone="resnet18",
    layers=[
        "layer1",
        "layer2",
        "layer3"
    ]
)

patchcore_engine = Engine(
    accelerator="cpu",
    devices=1
)

padim_engine = Engine(
    accelerator="cpu",
    devices=1
)

print("PatchCore loaded.")
print("PaDiM loaded.")

# ============================================================
# GET EXTERNAL IMAGES
# ============================================================

def get_images(folder):

    files = []

    if not os.path.exists(folder):

        raise FileNotFoundError(
            f"Folder does not exist:\n{folder}"
        )

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

print("\n" + "=" * 70)
print("IMAGE COUNTS")
print("=" * 70)

print(
    f"Normal test images  : {len(normal_images)}"
)

print(
    f"Anomaly test images : {len(anomaly_images)}"
)

# ============================================================
# ANOMALYCLIP SCORE
# ============================================================

def get_anomalyclip_score(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = preprocess(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        image_features, patch_features = (
            model.encode_image(
                image_tensor,
                FEATURES_LIST,
                DPAM_layer=20
            )
        )

        image_features = (
            image_features /
            image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        text_probs = (
            image_features @
            text_features.permute(
                0,
                2,
                1
            )
        )

        text_probs = (
            text_probs / 0.07
        ).softmax(dim=-1)

        score = text_probs[
            :, 0, 1
        ].item()

    return float(score)

# ============================================================
# PATCHCORE / PADIM SCORE
# ============================================================

def get_anomalib_score(
    engine,
    model,
    image_path,
    checkpoint
):

    predictions = engine.predict(
        model=model,
        data_path=image_path,
        ckpt_path=checkpoint
    )

    prediction = predictions[0]

    score = prediction.pred_score

    if torch.is_tensor(score):

        score = score.item()

    return float(score)

# ============================================================
# CREATE IMAGE LIST
# ============================================================

all_images = []

for image_path in normal_images:

    all_images.append(
        (
            image_path,
            0
        )
    )

for image_path in anomaly_images:

    all_images.append(
        (
            image_path,
            1
        )
    )

# ============================================================
# RUN ALL THREE MODELS
# ============================================================

results = []

print("\n" + "=" * 70)
print("RUNNING ALL THREE MODELS")
print("=" * 70)

for index, (image_path, true_label) in enumerate(
    all_images,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    print("\n" + "-" * 70)

    print(
        f"[{index}/{len(all_images)}] {filename}"
    )

    # --------------------------------------------------------
    # PatchCore
    # --------------------------------------------------------

    print(
        "    Running PatchCore..."
    )

    patchcore_score = get_anomalib_score(
        patchcore_engine,
        patchcore_model,
        image_path,
        PATCHCORE_CKPT
    )

    print(
        f"    PatchCore score : "
        f"{patchcore_score:.4f}"
    )

    # --------------------------------------------------------
    # PaDiM
    # --------------------------------------------------------

    print(
        "    Running PaDiM..."
    )

    padim_score = get_anomalib_score(
        padim_engine,
        padim_model,
        image_path,
        PADIM_CKPT
    )

    print(
        f"    PaDiM score     : "
        f"{padim_score:.4f}"
    )

    # --------------------------------------------------------
    # AnomalyCLIP
    # --------------------------------------------------------

    print(
        "    Running AnomalyCLIP..."
    )

    anomalyclip_score = get_anomalyclip_score(
        image_path
    )

    print(
        f"    AnomalyCLIP     : "
        f"{anomalyclip_score:.4f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.append({

        "image": filename,

        "path": image_path,

        "true_label": true_label,

        "PatchCore_score":
            patchcore_score,

        "PaDiM_score":
            padim_score,

        "AnomalyCLIP_score":
            anomalyclip_score,
    })

# ============================================================
# CALCULATE AUROC + AP
# ============================================================

y_true = np.array([
    result["true_label"]
    for result in results
])

model_names = [
    "PatchCore",
    "PaDiM",
    "AnomalyCLIP"
]

summary = []

for model_name in model_names:

    scores = np.array([

        result[
            f"{model_name}_score"
        ]

        for result in results

    ])

    auroc = roc_auc_score(
        y_true,
        scores
    )

    average_precision = (
        average_precision_score(
            y_true,
            scores
        )
    )

    summary.append({

        "Model":
            model_name,

        "AUROC":
            auroc,

        "Average_Precision":
            average_precision
    })

# ============================================================
# SAVE IMAGE-LEVEL CSV
# ============================================================

results_csv = os.path.join(
    OUTPUT_DIR,
    "cable_three_model_comparison.csv"
)

with open(
    results_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "path",
            "true_label",
            "PatchCore_score",
            "PaDiM_score",
            "AnomalyCLIP_score"
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )

# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_csv = os.path.join(
    OUTPUT_DIR,
    "cable_three_model_summary.csv"
)

with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Model",
            "AUROC",
            "Average_Precision"
        ]
    )

    writer.writeheader()

    writer.writerows(
        summary
    )

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n\n")

print("=" * 70)
print("FINAL CABLE THREE-MODEL COMPARISON")
print("=" * 70)

print()

for row in summary:

    print(
        f"{row['Model']:15s} | "
        f"AUROC: "
        f"{row['AUROC']:.4f} | "
        f"AP: "
        f"{row['Average_Precision']:.4f}"
    )

print("\n" + "-" * 70)

best_model = max(
    summary,
    key=lambda row: row["AUROC"]
)

print(
    f"Best AUROC: "
    f"{best_model['Model']} "
    f"({best_model['AUROC']:.4f})"
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(results_csv)
print(summary_csv)

print("\n" + "=" * 70)
print("CABLE COMPARISON COMPLETE")
print("=" * 70)