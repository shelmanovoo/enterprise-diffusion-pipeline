# server.py

import os
import base64
import threading
from io import BytesIO
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from diffusers import StableDiffusionXLPipeline
from modelscope.hub.snapshot_download import snapshot_download


# =========================
# CONFIG
# =========================

MODELSCOPE_MODEL_ID = "AI-ModelScope/stable-diffusion-xl-base-1.0"

# куда скачивать модель
MODEL_CACHE_DIR = os.environ.get(
    "MODEL_CACHE_DIR",
    "/mnt/models/sdxl-base"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# глобальный singleton pipeline
pipe: Optional[StableDiffusionXLPipeline] = None

# lock для thread-safe init
pipe_lock = threading.Lock()


# =========================
# MODEL DOWNLOAD
# =========================

def download_model():
    """
    Download model from ModelScope CDN if not exists.
    """
    if os.path.exists(MODEL_CACHE_DIR) and len(os.listdir(MODEL_CACHE_DIR)) > 0:
        print(f"[ModelScope] Using cached model: {MODEL_CACHE_DIR}")
        return MODEL_CACHE_DIR

    print(f"[ModelScope] Downloading model: {MODELSCOPE_MODEL_ID}")

    path = snapshot_download(
        MODELSCOPE_MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        revision="master"
    )

    print(f"[ModelScope] Download complete: {path}")

    return path


# =========================
# PIPELINE INIT
# =========================

def init_pipeline():
    """
    Initialize Stable Diffusion pipeline.
    Thread-safe singleton.
    """
    global pipe

    if pipe is not None:
        return pipe

    with pipe_lock:

        if pipe is not None:
            return pipe

        model_path = download_model()

        print(f"[SDXL] Loading pipeline from: {model_path}")
        print(f"[SDXL] Device: {DEVICE}")

        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            use_safetensors=True,
            local_files_only=True
        )

        pipe = pipe.to(DEVICE)

        # memory optimizations
        pipe.enable_attention_slicing()

        if DEVICE == "cuda":
            pipe.enable_vae_slicing()

        print("[SDXL] Pipeline loaded")

        return pipe


# =========================
# FASTAPI
# =========================

app = FastAPI(
    title="SDXL ModelScope API",
    version="2.0"
)


# =========================
# REQUEST MODEL
# =========================

class GenerateRequest(BaseModel):

    prompt: str

    steps: int = 30

    width: int = 1024

    height: int = 1024


# =========================
# STARTUP EVENT
# =========================

@app.on_event("startup")
def startup_event():
    """
    Download and load model at startup.
    """
    init_pipeline()


# =========================
# GENERATE ENDPOINT
# =========================

@app.post("/generate")
def generate(req: GenerateRequest):

    global pipe

    if pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:

        with torch.inference_mode():

            image = pipe(
                prompt=req.prompt,
                num_inference_steps=req.steps,
                width=req.width,
                height=req.height
            ).images[0]

        buffered = BytesIO()

        image.save(buffered, format="PNG")

        img_base64 = base64.b64encode(
            buffered.getvalue()
        ).decode()

        return {
            "image_base64": img_base64
        }

    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail="CUDA out of memory"
        )


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": pipe is not None,
        "model_path": MODEL_CACHE_DIR
    }