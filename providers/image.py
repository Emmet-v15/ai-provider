"""
SDXL image generation provider — wraps StableDiffusionXLPipeline from
a single-file checkpoint cached via HuggingFace Hub.

Model: John6666/ramthrusts-nsfw-pink-alchemy-mix-169-sdxl
       (RAMTHRUST_S-NSFW-PINK-ALCHEMY-MIX.safetensors, 6.5 GB)
Architecture: SDXL (UNet + CLIP-L/14 text encoders + VAE)
VRAM estimate: ~9 GB in bf16

Supports txt2img generation with standard SDXL parameters.
"""

from __future__ import annotations

import gc
import io
import os
import time
import asyncio
import base64
import logging
from typing import Any

import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

import admission

logger = logging.getLogger("provider.image")

# ── config ────────────────────────────────────────────────────────────
MODEL_REPO = "John6666/ramthrusts-nsfw-pink-alchemy-mix-169-sdxl"
MODEL_FILE = "RAMTHRUST_S-NSFW-PINK-ALCHEMY-MIX.safetensors"
DTYPE = torch.bfloat16
VRAM_GB = 12.0  # peak ~12 GB during inference (9 GB idle + ~3 GB compute buffers)
IMAGE_MAX_QUEUE = int(os.getenv("IMAGE_MAX_QUEUE", "8"))

# One pipeline, one generation at a time.  Two concurrent to_thread() calls
# into the same StableDiffusionXLPipeline share its scheduler state and
# corrupt each other, besides each needing their own ~3 GB of activations.
gate = admission.gate("sdxl", 1, max_queue=IMAGE_MAX_QUEUE, initial_service_s=30.0)


# ── state ─────────────────────────────────────────────────────────────
class ImageState:
    pipe: StableDiffusionXLPipeline | None = None
    loaded_at: float | None = None


state = ImageState()

# Serialises load/unload.  A double load would build a second pipeline and
# overwrite state.pipe, stranding ~9 GB of VRAM with no reference to free it.
_lock = asyncio.Lock()


# ── cache resolution ──────────────────────────────────────────────────
def _resolve_model_path() -> str:
    """Return the local cached path to the .safetensors checkpoint."""
    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        library_name="diffusers",
        local_files_only=True,
    )


# ── load / unload (called by ModelManager) ────────────────────────────
def _load_sync() -> Any:
    """Synchronous load — runs in thread pool to avoid blocking the event loop."""
    path = _resolve_model_path()
    logger.info("Loading SDXL from %s (%.1f GB)", path,
                os.path.getsize(path) / (1024 ** 3))

    pipe = StableDiffusionXLPipeline.from_single_file(
        path,
        torch_dtype=DTYPE,
        use_safetensors=True,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Minor perf tweaks (no compile/quant for now — kept simple)
    pipe.unet.to(memory_format=torch.channels_last)

    state.pipe = pipe
    state.loaded_at = time.time()
    logger.info("SDXL pipeline loaded")
    return pipe


async def load_model() -> Any:
    """Async load — runs directly on the event-loop thread.

    PyTorch CUDA operations are NOT thread-safe; running them in a
    thread-pool worker segfaults the process. We synchronously load
    on the main thread via the lifespan context.
    """
    async with _lock:
        if state.pipe is not None:
            logger.info("SDXL already loaded")
            return state.pipe
        return _load_sync()


async def unload_model() -> None:
    """Move pipeline to CPU, delete, and clear CUDA cache."""
    async with _lock:
        if state.pipe is not None:
            logger.info("Unloading SDXL ...")
            state.pipe.to("cpu")
            del state.pipe
            state.pipe = None
            state.loaded_at = None
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info("SDXL unloaded")


async def register_models() -> None:
    """Register the SDXL model with the global ModelManager."""
    from model_manager import get_manager
    mgr = get_manager()
    mgr.register(
        name="sdxl",
        size_gb=VRAM_GB,
        load_fn=load_model,
        unload_fn=unload_model,
        # Keeps the manager's view in sync with state.pipe, which is what
        # /health and the image endpoint actually read.
        health_check=lambda: state.pipe is not None,
    )
    logger.info("SDXL model registered (%.1f GB)", VRAM_GB)


# ── inference ─────────────────────────────────────────────────────────
def _generate_sync(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.0,
    seed: int | None = None,
) -> bytes:
    """Run SDXL inference and return PNG bytes. Blocking — call via threadpool."""
    pipe = state.pipe
    if pipe is None:
        raise RuntimeError("SDXL not loaded — POST /models/load with name 'sdxl' first")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


async def generate_txt2img(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.0,
    seed: int | None = None,
    *,
    request=None,
) -> bytes:
    """Async wrapper around _generate_sync, one generation at a time."""
    return await gate.run(
        lambda: asyncio.to_thread(
            _generate_sync,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        ),
        request=request,
        interruptible=False,
    )