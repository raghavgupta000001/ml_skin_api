# app.py
import os
import io
import json
import tempfile
import zipfile
import shutil
import requests
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Teachable Machine Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- in prod replace "*" with your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.getenv("MODEL_DIR", "model")
MODEL_URL = os.getenv("MODEL_URL", None)   # optional: zip or .h5 URL
CLASS_FILE = os.getenv("CLASS_FILE", "class_names.json")


def download_and_extract(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    logger.info("Downloading model from %s", url)
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download model (status {r.status_code})")
    tmp = tempfile.NamedTemporaryFile(delete=False)
    for chunk in r.iter_content(8192):
        tmp.write(chunk)
    tmp.flush()
    tmp.close()
    # if zip extract
    if tmp.name.endswith(".zip") or url.lower().endswith(".zip"):
        with zipfile.ZipFile(tmp.name, "r") as z:
            z.extractall(target_dir)
    else:
        # assume it's a single file like model.h5
        shutil.move(tmp.name, os.path.join(target_dir, "model.h5"))


def load_class_names(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "labels" in data:
                return data["labels"]
    return []


def load_model():
    if MODEL_URL and not os.path.exists(MODEL_DIR):
        download_and_extract(MODEL_URL, MODEL_DIR)

    # load .h5 or SavedModel dir
    if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb")):
        m = tf.keras.models.load_model(MODEL_DIR)
    elif os.path.exists(os.path.join(MODEL_DIR, "model.h5")):
        m = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model.h5"))
    elif os.path.isdir(MODEL_DIR) and any(name.endswith(".h5") for name in os.listdir(MODEL_DIR)):
        # pick first .h5
        for f in os.listdir(MODEL_DIR):
            if f.endswith(".h5"):
                m = tf.keras.models.load_model(os.path.join(MODEL_DIR, f))
                break
    else:
        raise FileNotFoundError(f"No model found in {MODEL_DIR}. Place model.h5 or SavedModel or set MODEL_URL.")
    return m


# Try to load model but don't crash the whole process — log and keep running
try:
    model = load_model()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Could not load model at startup. The server will start but /predict will return 503 until a model is available.")
    model = None

class_names = load_class_names(CLASS_FILE)
if model and not class_names:
    try:
        class_names = [f"class_{i}" for i in range(model.output_shape[-1])]
    except Exception:
        class_names = []
elif not class_names:
    class_names = []

# detect input size (width, height) used by PIL.resize
if model:
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        # shape typical: (None, H, W, C)
        H = shape[1] if len(shape) > 1 else None
        W = shape[2] if len(shape) > 2 else None
        H = int(H) if H else 224
        W = int(W) if W else 224
        INPUT_SIZE = (W, H)
    except Exception:
        logger.exception("Error determining model input size; falling back to 224x224.")
        INPUT_SIZE = (224, 224)
else:
    INPUT_SIZE = (224, 224)


def preprocess_image_bytes(b: bytes):
    img = Image.open(io.BytesIO(b)).convert("RGB")
    img = img.resize(INPUT_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)


@app.get("/")
def root():
    return {"status": "ok", "model_loaded": bool(model), "classes": len(class_names)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    contents = await file.read()
    x = preprocess_image_bytes(contents)
    preds = model.predict(x)
    preds = np.array(preds).flatten()
    idx = preds.argsort()[-top_k:][::-1]
    out = [
        {
            "label": class_names[int(i)] if i < len(class_names) else f"class_{i}",
            "probability": float(preds[int(i)]),
        }
        for i in idx
    ]
    return {"predictions": out}


if __name__ == "__main__":
    # Ensure we bind to the Render-provided PORT (or default 10000 for local tests)
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    logger.info("Starting uvicorn on 0.0.0.0:%s", port)
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
