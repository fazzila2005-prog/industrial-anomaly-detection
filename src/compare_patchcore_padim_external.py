import os
import csv
import tempfile

import numpy as np
import torch

from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

import anomalib
from torch.serialization import add_safe_globals

from anomalib.models.image.patchcore import Patchcore
from anomalib.models.image.padim import Padim


# ============================================================
# CHANGE ONLY THIS
# ============================================================

CATEGORY = "cable"


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"

EXTERNAL_ROOT = os.path.join(
    PROJECT_ROOT,
    "external images",
    CATEGORY
)

NORMAL_DIR = os.path.join(
    EXTERNAL_ROOT,
    "test_normal"
)

ANOMALY_DIR = os.path.join(
    EXTERNAL_ROOT,
    "test_anomaly"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "external_comparison",
    CATEGORY
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".avif",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# FIND CHECKPOINT
# ============================================================

def find_checkpoint(folder):

    checkpoints = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(".ckpt"):

                checkpoints.append(
                    os.path.join(root, file)
                )

    if not checkpoints:

        raise FileNotFoundError(
            f"No checkpoint found inside:\n{folder}"
        )

    # Newest checkpoint
    checkpoints.sort(
        key=os.path.getmtime,
        reverse=True
    )

    return checkpoints[0]


# ============================================================
# FIND MODEL CHECKPOINT
# ============================================================

def find_model_checkpoint(model_name):

    # Prefer augmented model if it exists
    augmented_folder = os.path.join(
        PROJECT_ROOT,
        "results",
        f"{model_name}_{CATEGORY}_augmented"
    )

    normal_folder = os.path.join(
        PROJECT_ROOT,
        "results",
        f"{model_name}_{CATEGORY}"
    )

    if os.path.exists(augmented_folder):

        try:

            checkpoint = find_checkpoint(
                augmented_folder
            )

            print(
                f"Using augmented {model_name} checkpoint."
            )

            return checkpoint

        except FileNotFoundError:

            pass

    if os.path.exists(normal_folder):

        checkpoint = find_checkpoint(
            normal_folder
        )

        print(
            f"Using normal {model_name} checkpoint."
        )

        return checkpoint

    raise FileNotFoundError(
        f"Could not find {model_name} checkpoint "
        f"for category '{CATEGORY}'."
    )


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    if not os.path.exists(folder):

        raise FileNotFoundError(
            f"Folder does not exist:\n{folder}"
        )

    images = []

    for filename in sorted(
        os.listdir(folder)
    ):

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in IMAGE_EXTENSIONS:

            images.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    return images


# ============================================================
# AVIF SUPPORT
# ============================================================

def prepare_image(path):

    extension = os.path.splitext(
        path
    )[1].lower()

    if extension != ".avif":

        return path, None

    try:

        import pillow_avif

    except ImportError:

        raise ImportError(
            "\nAVIF support is missing.\n"
            "Run:\n"
            "pip install pillow-avif-plugin\n"
        )

    image = Image.open(
        path
    ).convert("RGB")

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    )

    temp_path = temp_file.name

    temp_file.close()

    image.save(
        temp_path,
        format="JPEG"
    )

    return temp_path, temp_path


# ============================================================
# LOAD CHECKPOINT DIRECTLY
# ============================================================

def load_checkpoint_directly(
    model,
    checkpoint_path,
    model_name
):

    print()
    print(
        f"Loading {model_name} checkpoint..."
    )

    print(
        checkpoint_path
    )

    # PyTorch 2.6 compatibility
    add_safe_globals(
        [anomalib.PrecisionType]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    # Lightning checkpoints normally contain:
    # checkpoint["state_dict"]
    if isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    # Load the trained weights
    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    print(
        f"{model_name} checkpoint loaded."
    )

    # Show only if something unusual happened
    if result.missing_keys:

        print()
        print(
            f"WARNING: {len(result.missing_keys)} "
            f"missing keys."
        )

        print(
            result.missing_keys[:10]
        )

    if result.unexpected_keys:

        print()
        print(
            f"WARNING: {len(result.unexpected_keys)} "
            f"unexpected keys."
        )

        print(
            result.unexpected_keys[:10]
        )

    model.eval()

    return model


# ============================================================
# LOAD PATCHCORE
# ============================================================

def load_patchcore(
    checkpoint_path
):

    print()
    print("=" * 60)
    print("Loading PatchCore")
    print("=" * 60)

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=[
            "layer2",
            "layer3"
        ],
        # IMPORTANT:
        # We load the backbone weights from the checkpoint.
        # This avoids unnecessary Hugging Face downloading.
        pre_trained=False,
        coreset_sampling_ratio=0.01,
        num_neighbors=9
    )

    model = load_checkpoint_directly(
        model,
        checkpoint_path,
        "PatchCore"
    )

    return model


# ============================================================
# LOAD PADIM
# ============================================================

def load_padim(
    checkpoint_path
):

    print()
    print("=" * 60)
    print("Loading PaDiM")
    print("=" * 60)

    model = Padim(
        backbone="resnet18",
        layers=[
            "layer1",
            "layer2",
            "layer3"
        ],
        pre_trained=False
    )

    model = load_checkpoint_directly(
        model,
        checkpoint_path,
        "PaDiM"
    )

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image_path
):

    temporary_path = None

    try:

        usable_path, temporary_path = prepare_image(
            image_path
        )

        image = Image.open(
            usable_path
        ).convert("RGB")

        # Resize exactly as used by our previous
        # raw-score comparison
        image = image.resize(
            (256, 256)
        )

        image = np.asarray(
            image
        ).astype(
            np.float32
        ) / 255.0

        # HWC -> CHW
        image = torch.from_numpy(
            image
        ).permute(
            2,
            0,
            1
        )

        # Add batch dimension
        image = image.unsqueeze(0)

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32
        ).view(
            1,
            3,
            1,
            1
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32
        ).view(
            1,
            3,
            1,
            1
        )

        image = (
            image - mean
        ) / std

        return image

    finally:

        if temporary_path is not None:

            try:

                os.remove(
                    temporary_path
                )

            except OSError:

                pass


# ============================================================
# EXTRACT SCORE FROM OUTPUT
# ============================================================

def extract_score(
    output,
    model_name
):

    # Normal Anomalib output
    if hasattr(
        output,
        "pred_score"
    ):

        score = output.pred_score

    # Dictionary output
    elif isinstance(
        output,
        dict
    ) and "pred_score" in output:

        score = output[
            "pred_score"
        ]

    # Tuple/list output
    elif isinstance(
        output,
        (tuple, list)
    ):

        found = None

        for item in output:

            if torch.is_tensor(item):

                if item.numel() == 1:

                    found = item

                    break

        if found is None:

            raise RuntimeError(
                f"Could not find a scalar score "
                f"in {model_name} output."
            )

        score = found

    else:

        raise RuntimeError(
            f"Unknown {model_name} output type: "
            f"{type(output)}"
        )

    score = score.detach().cpu()

    return float(
        score.reshape(-1)[0]
    )


# ============================================================
# GET PATCHCORE SCORE
# ============================================================

def get_patchcore_score(
    model,
    image_path
):

    image = preprocess_image(
        image_path
    )

    device = next(
        model.parameters()
    ).device

    image = image.to(
        device
    )

    with torch.no_grad():

        output = model.model(
            image
        )

    return extract_score(
        output,
        "PatchCore"
    )


# ============================================================
# GET PADIM SCORE
# ============================================================

def get_padim_score(
    model,
    image_path
):

    image = preprocess_image(
        image_path
    )

    device = next(
        model.parameters()
    ).device

    image = image.to(
        device
    )

    with torch.no_grad():

        output = model.model(
            image
        )

    return extract_score(
        output,
        "PaDiM"
    )


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print(
    f"PATCHCORE vs PADIM — EXTERNAL {CATEGORY.upper()}"
)
print("=" * 70)


# ------------------------------------------------------------
# Get external test images
# ------------------------------------------------------------

normal_images = get_images(
    NORMAL_DIR
)

anomaly_images = get_images(
    ANOMALY_DIR
)

print()
print(
    f"Normal images : {len(normal_images)}"
)

print(
    f"Anomaly images: {len(anomaly_images)}"
)

print(
    f"Total images  : "
    f"{len(normal_images) + len(anomaly_images)}"
)


if len(normal_images) == 0:

    raise RuntimeError(
        "No normal images found."
    )


if len(anomaly_images) == 0:

    raise RuntimeError(
        "No anomaly images found."
    )


# ------------------------------------------------------------
# Find checkpoints
# ------------------------------------------------------------

patchcore_checkpoint = find_model_checkpoint(
    "patchcore"
)

padim_checkpoint = find_model_checkpoint(
    "padim"
)


# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------

patchcore = load_patchcore(
    patchcore_checkpoint
)

padim = load_padim(
    padim_checkpoint
)


# ------------------------------------------------------------
# Build test set
# ------------------------------------------------------------

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
# SCORE IMAGES
# ============================================================

print()
print("=" * 70)
print("SCORING EXTERNAL IMAGES")
print("=" * 70)

records = []

for index, (
    image_path,
    true_label
) in enumerate(
    all_images,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    print()
    print(
        f"[{index}/{len(all_images)}] {filename}"
    )

    # --------------------------------------------------------
    # PatchCore
    # --------------------------------------------------------

    patchcore_score = get_patchcore_score(
        patchcore,
        image_path
    )

    print(
        f"PatchCore raw score: "
        f"{patchcore_score:.6f}"
    )

    # --------------------------------------------------------
    # PaDiM
    # --------------------------------------------------------

    padim_score = get_padim_score(
        padim,
        image_path
    )

    print(
        f"PaDiM raw score:     "
        f"{padim_score:.6f}"
    )

    records.append(
        {
            "Image_Name": filename,
            "True_Label": true_label,
            "PatchCore_Raw_Score": patchcore_score,
            "PaDiM_Raw_Score": padim_score
        }
    )


# ============================================================
# METRICS
# ============================================================

y_true = np.array(
    [
        record["True_Label"]
        for record in records
    ]
)

patchcore_scores = np.array(
    [
        record["PatchCore_Raw_Score"]
        for record in records
    ]
)

padim_scores = np.array(
    [
        record["PaDiM_Raw_Score"]
        for record in records
    ]
)


patchcore_auroc = roc_auc_score(
    y_true,
    patchcore_scores
)

patchcore_ap = average_precision_score(
    y_true,
    patchcore_scores
)

padim_auroc = roc_auc_score(
    y_true,
    padim_scores
)

padim_ap = average_precision_score(
    y_true,
    padim_scores
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    RESULT_DIR,
    f"{CATEGORY}_patchcore_padim_scores.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Image_Name",
            "True_Label",
            "PatchCore_Raw_Score",
            "PaDiM_Raw_Score"
        ]
    )

    writer.writeheader()

    writer.writerows(
        records
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULT_DIR,
    f"{CATEGORY}_patchcore_padim_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"PATCHCORE vs PADIM — "
        f"EXTERNAL {CATEGORY.upper()}\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Normal images: {len(normal_images)}\n"
    )

    file.write(
        f"Anomaly images: {len(anomaly_images)}\n"
    )

    file.write(
        f"Total images: {len(records)}\n\n"
    )

    file.write(
        "PATCHCORE\n"
    )

    file.write(
        f"AUROC: {patchcore_auroc:.4f}\n"
    )

    file.write(
        f"Average Precision: {patchcore_ap:.4f}\n\n"
    )

    file.write(
        "PADIM\n"
    )

    file.write(
        f"AUROC: {padim_auroc:.4f}\n"
    )

    file.write(
        f"Average Precision: {padim_ap:.4f}\n\n"
    )

    file.write(
        "EVALUATION PROTOCOL\n"
    )

    file.write(
        "PatchCore and PaDiM were evaluated on "
        "the exact same external test images.\n"
    )

    file.write(
        "External anomaly images were NOT used "
        "for training.\n"
    )

    file.write(
        "Adaptation images were NOT included "
        "in the external test.\n"
    )

    file.write(
        "Metrics were calculated from raw anomaly "
        "scores using AUROC and Average Precision.\n"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print()
print("=" * 70)
print("EXTERNAL COMPARISON COMPLETE")
print("=" * 70)

print()
print(
    f"Category: {CATEGORY}"
)

print()
print("PATCHCORE")
print(
    f"  AUROC: {patchcore_auroc:.4f}"
)
print(
    f"  AP:    {patchcore_ap:.4f}"
)

print()
print("PADIM")
print(
    f"  AUROC: {padim_auroc:.4f}"
)
print(
    f"  AP:    {padim_ap:.4f}"
)

print()
print(
    "CSV saved to:"
)

print(
    csv_path
)

print()
print(
    "Summary saved to:"
)

print(
    summary_path
)

print()
print("=" * 70)