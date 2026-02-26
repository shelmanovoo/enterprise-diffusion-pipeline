# server.py

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline
import base64
from io import BytesIO

# ---- CONFIG ----

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- LOAD MODEL (singleton) ----

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    use_safetensors=True
)

pipe = pipe.to(DEVICE)

# optional performance optimization
pipe.enable_attention_slicing()

# ---- API ----

app = FastAPI(title="SDXL API", version="1.0")


class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 30
    width: int = 1024
    height: int = 1024


@app.post("/generate")
def generate(req: GenerateRequest):

    image = pipe(
        prompt=req.prompt,
        num_inference_steps=req.steps,
        width=req.width,
        height=req.height
    ).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")

    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return {
        "image_base64": img_base64
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}