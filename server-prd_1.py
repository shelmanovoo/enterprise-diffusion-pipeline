import os
import torch
from io import BytesIO
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from diffusers import StableDiffusionXLPipeline


# ========================
# CONFIG
# ========================

MODEL_PATH = "/mnt/usb/models/sdxl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_STEPS = 50
MAX_RESOLUTION = 1024


# ========================
# LOAD MODEL (singleton)
# ========================

print("Loading SDXL pipeline...")

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    local_files_only=True,
    use_safetensors=True
)

pipe = pipe.to(DEVICE)

# memory optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

if DEVICE == "cuda":
    pipe.enable_xformers_memory_efficient_attention()

# PyTorch 2 optimization
if hasattr(torch, "compile"):
    pipe.unet = torch.compile(
        pipe.unet,
        mode="reduce-overhead",
        fullgraph=True
    )

print(f"SDXL loaded on {DEVICE}")


# ========================
# THREAD SAFETY
# ========================

pipe_lock = Lock()


# ========================
# FASTAPI INIT
# ========================

app = FastAPI(
    title="SDXL Inference API",
    version="3.0",
)


# ========================
# REQUEST MODEL
# ========================

class GenerateRequest(BaseModel):

    prompt: str = Field(..., max_length=1000)

    steps: int = Field(
        default=30,
        ge=1,
        le=MAX_STEPS
    )

    width: int = Field(
        default=1024,
        ge=256,
        le=MAX_RESOLUTION
    )

    height: int = Field(
        default=1024,
        ge=256,
        le=MAX_RESOLUTION
    )

    seed: int | None = None


# ========================
# PNG GENERATION ENDPOINT
# ========================

@app.post("/generate", response_class=StreamingResponse)
async def generate(req: GenerateRequest):

    generator = None

    if req.seed is not None:
        generator = torch.Generator(
            device=DEVICE
        ).manual_seed(req.seed)

    try:

        with pipe_lock:

            with torch.inference_mode():

                image = pipe(
                    prompt=req.prompt,
                    num_inference_steps=req.steps,
                    width=req.width,
                    height=req.height,
                    generator=generator
                ).images[0]

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=generated.png",
                "Cache-Control": "no-store"
            }
        )

    except torch.cuda.OutOfMemoryError:

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail="GPU out of memory"
        )


# ========================
# HEALTH ENDPOINT
# ========================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "vram_allocated_mb":
            torch.cuda.memory_allocated() // 1024 // 1024
            if DEVICE == "cuda" else 0
    }


# ========================
# WARMUP
# ========================

@app.on_event("startup")
def warmup():

    print("Warmup started...")

    with pipe_lock:

        with torch.inference_mode():

            pipe(
                prompt="warmup",
                num_inference_steps=1,
                width=512,
                height=512
            )

    print("Warmup complete")