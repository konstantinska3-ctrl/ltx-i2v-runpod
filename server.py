"""FastAPI server for LTX-Video image-to-video on RunPod Pod."""

from __future__ import annotations

import base64
import gc
import io
import os
import threading
import time
from typing import Any

import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

MODEL_ID = os.getenv("LTX_MODEL_ID", "Lightricks/LTX-Video")
WIDTH = int(os.getenv("LTX_WIDTH", "768"))
HEIGHT = int(os.getenv("LTX_HEIGHT", "512"))
NUM_FRAMES = int(os.getenv("LTX_NUM_FRAMES", "49"))
FPS = int(os.getenv("LTX_FPS", "24"))
NUM_INFERENCE_STEPS = int(os.getenv("LTX_STEPS", "30"))
GUIDANCE_SCALE = float(os.getenv("LTX_GUIDANCE", "3.0"))
NEGATIVE_PROMPT = os.getenv(
    "LTX_NEGATIVE_PROMPT",
    "worst quality, inconsistent motion, blurry, jittery, distorted",
)

app = FastAPI(title="LTX I2V RunPod Server", version="1.0.0")
PIPELINE: LTXImageToVideoPipeline | None = None
LOADING = False
LOAD_ERROR: str | None = None


class GenerateRequest(BaseModel):
    image_base64: str = Field(..., description="PNG/JPEG as base64 (with or without data: prefix)")
    prompt: str
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    fps: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None


class GenerateResponse(BaseModel):
    video_base64: str
    duration_seconds: float
    generation_seconds: float


def _decode_image(data: str) -> Image.Image:
    payload = data.split(",", 1)[1] if "," in data else data
    raw = base64.b64decode(payload)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _load_pipeline() -> LTXImageToVideoPipeline:
    global PIPELINE, LOAD_ERROR
    if PIPELINE is not None:
        return PIPELINE
    if LOAD_ERROR:
        raise RuntimeError(LOAD_ERROR)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available inside container")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {MODEL_ID} ({dtype})...")
    t0 = time.time()
    try:
        pipe = LTXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
    except Exception as exc:  # noqa: BLE001
        LOAD_ERROR = str(exc)
        raise
    PIPELINE = pipe
    print(f"Model ready in {time.time() - t0:.1f}s")
    return PIPELINE


def _load_pipeline_background() -> None:
    global LOADING, LOAD_ERROR
    LOADING = True
    try:
        _load_pipeline()
    except Exception as exc:  # noqa: BLE001
        LOAD_ERROR = str(exc)
        print(f"Model load failed: {exc}")
    finally:
        LOADING = False


@app.on_event("startup")
def warmup() -> None:
    threading.Thread(target=_load_pipeline_background, daemon=True).start()


@app.get("/health")
def health() -> dict[str, Any]:
    if LOAD_ERROR:
        raise HTTPException(status_code=500, detail=LOAD_ERROR)
    ready = PIPELINE is not None and not LOADING
    return {
        "status": "ok" if ready else "loading",
        "model": MODEL_ID,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        image = _decode_image(req.image_base64)
        w = req.width or WIDTH
        h = req.height or HEIGHT
        image = image.resize((w, h), Image.Resampling.LANCZOS)
        if LOADING or PIPELINE is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        pipe = _load_pipeline()
        t0 = time.time()
        frames = pipe(
            image=image,
            prompt=req.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=w,
            height=h,
            num_frames=req.num_frames or NUM_FRAMES,
            frame_rate=req.fps or FPS,
            num_inference_steps=req.num_inference_steps or NUM_INFERENCE_STEPS,
            guidance_scale=req.guidance_scale or GUIDANCE_SCALE,
        ).frames[0]
        tmp_path = "/tmp/out.mp4"
        export_to_video(frames, tmp_path, fps=req.fps or FPS)
        video_b64 = base64.b64encode(open(tmp_path, "rb").read()).decode("ascii")
        gen_sec = time.time() - t0
        duration = (req.num_frames or NUM_FRAMES) / float(req.fps or FPS)
        return GenerateResponse(
            video_base64=video_b64,
            duration_seconds=round(duration, 3),
            generation_seconds=round(gen_sec, 1),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
