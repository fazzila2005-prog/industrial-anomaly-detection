import os
import sys
import uuid
import shutil
import io
import base64

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from groq import Groq
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"
)

SRC_DIR = PROJECT_DIR / "src"

RESULTS_DIR = PROJECT_DIR / "results"
HEATMAP_DIR = RESULTS_DIR / "heatmaps"
UPLOAD_DIR = RESULTS_DIR / "uploads"

HEATMAP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ALLOWED IMAGE FORMATS
# ============================================================

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".avif",
}


# ============================================================
# ANOMALYCLIP
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anomalyclip_inference import predict


# ============================================================
# GROQ
# ============================================================

ENV_FILE = PROJECT_DIR / ".env"

load_dotenv(ENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        f"GROQ_API_KEY was not found in:\n{ENV_FILE}"
    )

groq_client = Groq(
    api_key=GROQ_API_KEY,
    timeout=30.0
)

GROQ_MODEL = "qwen/qwen3.8-27b"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Industrial Anomaly Inspection API",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEATMAP STATIC FILES
# ============================================================

app.mount(
    "/heatmaps",
    StaticFiles(
        directory=str(HEATMAP_DIR)
    ),
    name="heatmaps"
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Industrial Anomaly Inspection API"
    }


# ============================================================
# GROQ EXPLANATION
# ============================================================

def generate_explanation(
    image_path,
    category,
    prediction,
    anomaly_score,
    threshold
):

    try:

        # --------------------------------------------------------
        # Open image
        # --------------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # --------------------------------------------------------
        # Convert image to JPEG
        # --------------------------------------------------------

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=90
        )

        image_bytes = buffer.getvalue()

        # --------------------------------------------------------
        # Convert image to Base64
        # --------------------------------------------------------

        base64_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        # --------------------------------------------------------
        # Prompt
        # --------------------------------------------------------

        prompt = f"""
You are an AI assistant helping with industrial visual inspection.

Analyze the uploaded industrial product image.

Category: {category}

Detection result: {prediction}

Anomaly score: {anomaly_score:.4f}

Decision threshold: {threshold:.4f}

IMPORTANT RULES:

- Do NOT change the given detection result.
- Do NOT invent a defect.
- Only describe what is visually supported by the image.
- If the result is ANOMALY, describe the suspicious visible
  region or irregularity.
- If the result is NORMAL, explain that no obvious abnormality
  is visible.
- Do not mention datasets.
- Do not mention training.
- Do not mention AnomalyCLIP.
- Do not mention PatchCore.
- Do not mention PaDiM.
- Keep the explanation simple for a human inspector.

Return exactly:

Result:
Observation:
Reason:
Recommended Action:
"""

        # --------------------------------------------------------
        # GROQ VISION REQUEST
        # --------------------------------------------------------

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64,"
                                    + base64_image
                                )
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_completion_tokens=512,
            reasoning_effort="none"
        )

        # --------------------------------------------------------
        # Read response
        # --------------------------------------------------------

        explanation = (
            completion
            .choices[0]
            .message
            .content
        )

        if explanation:
            return explanation.strip()

        return (
            "AI explanation could not be generated."
        )

    except Exception as e:

        print("=" * 60)
        print("GROQ ERROR")
        print(repr(e))
        print("=" * 60)

        return f"Groq error: {e}"


# ============================================================
# INSPECT IMAGE
# ============================================================

@app.post("/inspect")
async def inspect_image(
    file: UploadFile = File(...),
    category: str = Form(...)
):

    allowed_categories = {
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper"
    }

    # --------------------------------------------------------
    # Validate category
    # --------------------------------------------------------

    if category not in allowed_categories:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {category}"
        )

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No image uploaded."
        )

    # --------------------------------------------------------
    # Get extension
    # --------------------------------------------------------

    extension = Path(
        file.filename
    ).suffix.lower()

    # --------------------------------------------------------
    # Validate format
    # --------------------------------------------------------

    if extension not in ALLOWED_IMAGE_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Please upload JPG, JPEG, PNG, WEBP, or AVIF."
            )
        )

    # ========================================================
    # UNIQUE ID
    # ========================================================

    unique_id = str(
        uuid.uuid4()
    )

    # ========================================================
    # UPLOAD PATH
    # ========================================================

    upload_filename = (
        f"upload_{category}_{unique_id}{extension}"
    )

    upload_path = (
        UPLOAD_DIR / upload_filename
    )

    # ========================================================
    # HEATMAP PATH
    # ========================================================

    heatmap_filename = (
        f"anomaly_map_{category}_{unique_id}.jpg"
    )

    heatmap_output_path = (
        HEATMAP_DIR / heatmap_filename
    )

    # ========================================================
    # SAVE UPLOADED IMAGE
    # ========================================================

    with open(
        upload_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # ========================================================
    # VERIFY ACTUAL IMAGE
    # ========================================================

    try:

        with Image.open(
            upload_path
        ) as test_image:

            test_image.verify()

    except Exception:

        try:
            upload_path.unlink()
        except Exception:
            pass

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is not a valid "
                "readable image."
            )
        )

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    try:

        # ====================================================
        # 1. ANOMALYCLIP
        # ====================================================

        print()
        print("=" * 60)
        print("RUNNING ANOMALYCLIP")
        print("=" * 60)

        result = predict(
            image_path=str(upload_path),
            category=category,
            heatmap_output=str(
                heatmap_output_path
            )
        )

        # ----------------------------------------------------
        # Detection result
        # ----------------------------------------------------

        prediction = result["prediction"]

        anomaly_score = float(
            result["anomaly_score"]
        )

        threshold = float(
            result["threshold"]
        )

        print()
        print("AnomalyCLIP result:")
        print(result)

        # ====================================================
        # 2. VERIFY HEATMAP
        # ====================================================

        print()
        print("=" * 60)
        print("CHECKING HEATMAP")
        print("=" * 60)

        # We already gave AnomalyCLIP the exact output path.
        # No searching and no guessing.

        if not heatmap_output_path.exists():

            raise RuntimeError(
                "AnomalyCLIP completed, but the expected "
                f"heatmap was not created:\n"
                f"{heatmap_output_path}"
            )

        if not heatmap_output_path.is_file():

            raise RuntimeError(
                "The expected heatmap path is not a file:\n"
                f"{heatmap_output_path}"
            )

        print(
            "Heatmap created successfully:"
        )

        print(
            heatmap_output_path
        )

        # ====================================================
        # 3. HEATMAP URL
        # ====================================================

        heatmap_url = (
            f"/heatmaps/{heatmap_filename}"
        )

        # ====================================================
        # 4. GROQ EXPLANATION
        # ====================================================

        print()
        print("=" * 60)
        print("RUNNING GROQ EXPLANATION")
        print("=" * 60)

        explanation = generate_explanation(
            image_path=str(upload_path),
            category=category,
            prediction=prediction,
            anomaly_score=anomaly_score,
            threshold=threshold
        )

        # ====================================================
        # 5. DELETE TEMP UPLOAD
        # ====================================================

        try:
            upload_path.unlink()
        except Exception:
            pass

        # ====================================================
        # 6. FINAL RESPONSE
        # ====================================================

        return {
            "category": category,
            "anomaly_score": round(
                anomaly_score,
                4
            ),
            "threshold": round(
                threshold,
                4
            ),
            "prediction": prediction,
            "heatmap_url": heatmap_url,
            "explanation": explanation
        }

    except Exception as e:

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        try:
            upload_path.unlink()
        except Exception:
            pass

        # ----------------------------------------------------
        # Print error
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("INSPECTION ERROR")
        print(repr(e))
        print("=" * 60)
        print()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )