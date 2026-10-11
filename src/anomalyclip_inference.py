import os
import sys
import random
import torch
import numpy as np
import cv2

from PIL import Image
from scipy.ndimage import gaussian_filter


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"

ANOMALYCLIP_DIR = r"C:\Users\Fazzila\OneDrive\Desktop\AnomalyCLIP"

CHECKPOINT_PATH = os.path.join(
    ANOMALYCLIP_DIR,
    "checkpoints",
    "9_12_4_multiscale_visa",
    "epoch_15.pth"
)

# ============================================================
# HEATMAP FOLDER
# ============================================================

HEATMAP_DIR = r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project\results\heatmaps"


# ============================================================
# ANOMALYCLIP PATH
# ============================================================

if ANOMALYCLIP_DIR not in sys.path:
    sys.path.insert(0, ANOMALYCLIP_DIR)


# ============================================================
# IMPORTS
# ============================================================

import AnomalyCLIP_lib

from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform, normalize


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 518

FEATURES_LIST = [6, 12, 18, 24]

FEATURE_MAP_LAYER = [0, 1, 2, 3]

SIGMA = 4

DEPTH = 9

N_CTX = 12

T_N_CTX = 4

DPAM_LAYER = 20


# ============================================================
# LOCKED THRESHOLDS
# ============================================================

THRESHOLDS = {
    "bottle": 0.8651,
    "cable": 0.9346,
    "capsule": 0.7881,
    "carpet": 0.9469,
    "grid": 0.5531,
    "hazelnut": 0.7630,
    "leather": 0.5711,
    "metal_nut": 0.9368,
    "pill": 0.9574,
    "screw": 0.7531,
    "tile": 0.5484,
    "toothbrush": 0.6963,
    "transistor": 0.7800,
    "wood": 0.4597,
    "zipper": 0.9909,
}


# ============================================================
# GLOBAL MODEL
# ============================================================

_model = None
_prompt_learner = None
_text_features = None
_preprocess = None


# ============================================================
# RANDOM SEED
# ============================================================

def setup_seed(seed=111):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    random.seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD ANOMALYCLIP
# ============================================================

def load_anomalyclip():

    global _model
    global _prompt_learner
    global _text_features
    global _preprocess

    if _model is not None:

        return (
            _model,
            _prompt_learner,
            _text_features,
            _preprocess
        )

    print("\nLoading AnomalyCLIP...")

    setup_seed()

    anomalyclip_parameters = {
        "Prompt_length": N_CTX,
        "learnabel_text_embedding_depth": DEPTH,
        "learnabel_text_embedding_length": T_N_CTX
    }

    # --------------------------------------------------------
    # Base model
    # --------------------------------------------------------

    _model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px",
        device=DEVICE,
        design_details=anomalyclip_parameters
    )

    _model.eval()

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    class Args:
        image_size = IMAGE_SIZE

    transform_args = Args()

    _preprocess, _ = get_transform(transform_args)

    # --------------------------------------------------------
    # Prompt learner
    # --------------------------------------------------------

    _prompt_learner = AnomalyCLIP_PromptLearner(
        _model.to("cpu"),
        anomalyclip_parameters
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    _prompt_learner.load_state_dict(
        checkpoint["prompt_learner"]
    )

    _prompt_learner.to(DEVICE)

    _model.to(DEVICE)

    # --------------------------------------------------------
    # DAPM
    # --------------------------------------------------------

    _model.visual.DAPM_replace(
        DPAM_layer=DPAM_LAYER
    )

    # --------------------------------------------------------
    # Text features
    # --------------------------------------------------------

    prompts, tokenized_prompts, compound_prompts_text = (
        _prompt_learner(cls_id=None)
    )

    _text_features = _model.encode_text_learn(
        prompts,
        tokenized_prompts,
        compound_prompts_text
    ).float()

    _text_features = torch.stack(
        torch.chunk(
            _text_features,
            dim=0,
            chunks=2
        ),
        dim=1
    )

    _text_features = (
        _text_features /
        _text_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    print("AnomalyCLIP loaded successfully.")

    return (
        _model,
        _prompt_learner,
        _text_features,
        _preprocess
    )


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap_overlay(image_path, anomaly_map):

    # --------------------------------------------------------
    # Read image using PIL.
    #
    # This supports .avif input images.
    # --------------------------------------------------------

    try:

        pil_image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as e:

        raise ValueError(
            f"Could not read image:\n"
            f"{image_path}\n"
            f"Reason: {e}"
        )

    # PIL RGB -> NumPy RGB
    original = np.array(
        pil_image
    )

    # RGB -> BGR for OpenCV
    original = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    original = cv2.resize(
        original,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    anomaly_map = np.asarray(
        anomaly_map
    )

    # Remove unnecessary dimensions
    anomaly_map = np.squeeze(
        anomaly_map
    )

    # Make absolutely sure it is 2D
    if anomaly_map.ndim != 2:

        raise ValueError(
            f"Heatmap has invalid shape: "
            f"{anomaly_map.shape}"
        )

    heatmap = normalize(
        anomaly_map
    )

    heatmap = (
        heatmap * 255
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.resize(
        heatmap,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    overlay = cv2.addWeighted(
        original,
        0.5,
        heatmap,
        0.5,
        0
    )

    return overlay


# ============================================================
# SAVE HEATMAP
# ============================================================

def save_heatmap(
    image_path,
    anomaly_map,
    output_path
):

    overlay = create_heatmap_overlay(
        image_path,
        anomaly_map
    )

    # Make absolutely sure the output is JPG.
    # OpenCV cannot write AVIF.
    output_path = os.path.splitext(
        output_path
    )[0] + ".jpg"

    success = cv2.imwrite(
        output_path,
        overlay
    )

    if not success:

        raise RuntimeError(
            f"OpenCV could not save heatmap:\n"
            f"{output_path}"
        )

    return output_path


# ============================================================
# PREDICT
# ============================================================

def predict(
    image_path,
    category,
    heatmap_output=None
):

    category = category.lower().strip()

    # --------------------------------------------------------
    # Validate category
    # --------------------------------------------------------

    if category not in THRESHOLDS:

        raise ValueError(
            f"Unknown category: {category}\n"
            f"Available categories:\n"
            f"{list(THRESHOLDS.keys())}"
        )

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not os.path.isfile(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        prompt_learner,
        text_features,
        preprocess
    ) = load_anomalyclip()

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    img = Image.open(
        image_path
    ).convert("RGB")

    img_tensor = preprocess(
        img
    )

    image = img_tensor.reshape(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        image_features, patch_features = (
            model.encode_image(
                image,
                FEATURES_LIST,
                DPAM_layer=DPAM_LAYER
            )
        )

        image_features = (
            image_features /
            image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        # ----------------------------------------------------
        # IMAGE SCORE
        # ----------------------------------------------------

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
        ).softmax(-1)

        anomaly_score = (
            text_probs[:, 0, 1]
        ).item()

        # ----------------------------------------------------
        # ANOMALY MAP
        # ----------------------------------------------------

        anomaly_map_list = []

        for idx, patch_feature in enumerate(
            patch_features
        ):

            if idx >= FEATURE_MAP_LAYER[0]:

                patch_feature = (
                    patch_feature /
                    patch_feature.norm(
                        dim=-1,
                        keepdim=True
                    )
                )

                similarity, _ = (
                    AnomalyCLIP_lib.compute_similarity(
                        patch_feature,
                        text_features[0]
                    )
                )

                similarity_map = (
                    AnomalyCLIP_lib.get_similarity_map(
                        similarity[:, 1:, :],
                        IMAGE_SIZE
                    )
                )

                anomaly_map = (
                    similarity_map[..., 1]
                    +
                    1
                    -
                    similarity_map[..., 0]
                ) / 2.0

                anomaly_map_list.append(
                    anomaly_map
                )

        anomaly_map = torch.stack(
            anomaly_map_list
        )

        anomaly_map = anomaly_map.sum(
            dim=0
        )

        # ----------------------------------------------------
        # SMOOTHING
        # ----------------------------------------------------

        anomaly_map = torch.stack([
            torch.from_numpy(
                gaussian_filter(
                    i,
                    sigma=SIGMA
                )
            )
            for i in anomaly_map.detach().cpu()
        ])

    # ========================================================
    # PREDICTION
    # ========================================================

    threshold = THRESHOLDS[
        category
    ]

    if anomaly_score >= threshold:

        prediction = "ANOMALY"

    else:

        prediction = "NORMAL"

    # ========================================================
    # SAVE HEATMAP
    # ========================================================

    saved_heatmap_path = heatmap_output

    if heatmap_output is not None:

        saved_heatmap_path = save_heatmap(
            image_path,
            anomaly_map.numpy(),
            heatmap_output
        )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "category": category,
        "anomaly_score": round(
            float(anomaly_score),
            4
        ),
        "threshold": threshold,
        "prediction": prediction,
        "heatmap_path": saved_heatmap_path
    }


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print("\n======================================")
    print("       ANOMALYCLIP TEST")
    print("======================================\n")

    # --------------------------------------------------------
    # Check heatmap folder BEFORE doing anything
    # --------------------------------------------------------

    print("Checking heatmap folder...")

    print(
        f"Expected location:\n{HEATMAP_DIR}\n"
    )

    if not os.path.exists(
        HEATMAP_DIR
    ):

        print(
            "ERROR: The heatmaps folder does not exist."
        )

        print(
            "\nCreate this folder manually:"
        )

        print(
            r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project\results\heatmaps"
        )

        sys.exit(1)

    if not os.path.isdir(
        HEATMAP_DIR
    ):

        print(
            "ERROR: 'heatmaps' exists but is not a folder."
        )

        sys.exit(1)

    print(
        "Heatmap folder OK.\n"
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    image_path = input(
        "Enter image path: "
    ).strip().strip('"')

    category = input(
        "Enter category: "
    ).strip().lower()

    # --------------------------------------------------------
    # Heatmap filename
    # --------------------------------------------------------

    filename = os.path.basename(
        image_path
    )

    # Remove original extension.
    # Example:
    # image.avif -> image
    filename_without_extension = os.path.splitext(
        filename
    )[0]

    # ALWAYS save heatmap as JPG.
    heatmap_name = (
        "anomaly_map_"
        + category
        + "_"
        + filename_without_extension
        + ".jpg"
    )

    heatmap_path = os.path.join(
        HEATMAP_DIR,
        heatmap_name
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    try:

        result = predict(
            image_path=image_path,
            category=category,
            heatmap_output=heatmap_path
        )

        print("\n======================================")
        print("             RESULT")
        print("======================================")

        print(
            f"Category      : "
            f"{result['category']}"
        )

        print(
            f"Anomaly Score : "
            f"{result['anomaly_score']}"
        )

        print(
            f"Threshold     : "
            f"{result['threshold']}"
        )

        print(
            f"Prediction    : "
            f"{result['prediction']}"
        )

        print(
            f"Heatmap       : "
            f"{result['heatmap_path']}"
        )

        print(
            "======================================\n"
        )

    except Exception as e:

        print("\n======================================")
        print("ERROR")
        print("======================================")
        print(e)