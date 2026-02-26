import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline
import base64
from io import BytesIO
import threading

# ---- CONFIG ----

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ---- GLOBAL STATE ----

pipe = None
pipe_lock = threading.Lock()


# ---- LOAD MODEL ----

def load_pipeline():
    global pipe

    if pipe is not None:
        return pipe

    with pipe_lock:
        if pipe is not None:
            return pipe

        print("Loading SDXL pipeline...")

        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            use_safetensors=True,
            variant="fp16" if DEVICE == "cuda" else None,
            local_files_only=False
        )

        pipe = pipe.to(DEVICE)

        # ---- PERFORMANCE OPTIMIZATIONS ----

        if DEVICE == "cuda":

            # faster inference
            pipe.enable_attention_slicing()

            # memory optimization
            pipe.enable_vae_slicing()

            # PyTorch 2.x optimization
            try:
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            except Exception:
                pass

        else:
            pipe.enable_attention_slicing()

        pipe.set_progress_bar_config(disable=True)

        print("SDXL pipeline loaded")

        return pipe


# ---- FASTAPI ----

app = FastAPI(
    title="SDXL API",
    version="1.0"
)


# ---- REQUEST MODEL ----

class GenerateRequest(BaseModel):

    prompt: str

    steps: int = 30

    width: int = 1024

    height: int = 1024


# ---- GENERATE ENDPOINT ----

@app.post("/generate")
def generate(req: GenerateRequest):

    try:

        pipeline = load_pipeline()

        with torch.inference_mode():

            image = pipeline(
                prompt=req.prompt,
                num_inference_steps=req.steps,
                width=req.width,
                height=req.height
            ).images[0]

        buffered = BytesIO()

        image.save(buffered, format="PNG")

        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "image_base64": img_base64,
            "device": DEVICE
        }

    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail="CUDA out of memory"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---- HEALTH ----

@app.get("/health")
def health():

    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": pipe is not None
    }