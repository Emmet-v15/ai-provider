"""
Speech-to-Text provider — uses faster-whisper (or openai-whisper fallback)
for local transcription on the RTX 5090.

Automatic speech recognition with model size configurable via env var
STT_MODEL_SIZE (default: large-v3-turbo).
"""

from __future__ import annotations

import os
import asyncio
import logging
import numpy as np

logger = logging.getLogger("provider.stt")

STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "large-v3-turbo")
# ~3 GB VRAM for large-v3-turbo (turbo is smaller than original large-v3)
STT_VRAM_GB = float(os.getenv("STT_VRAM_GB", "3"))

_WHISPER_MODEL = None
_FASTER_WHISPER_AVAILABLE = False


def _ensure_cublas_on_path():
    """Add nvidia-cublas-cu12 DLLs to PATH if installed via pip."""
    import importlib.util as _util
    spec = _util.find_spec("nvidia.cublas")
    if spec and spec.submodule_search_locations:
        pkg_dir = list(spec.submodule_search_locations)[0]
        dll_dir = os.path.join(pkg_dir, "bin")
        if os.path.isdir(dll_dir):
            os.environ.setdefault("PATH", "")
            if dll_dir not in os.environ["PATH"]:
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]


def _import_whisper():
    global _FASTER_WHISPER_AVAILABLE
    _ensure_cublas_on_path()
    try:
        import faster_whisper
        _FASTER_WHISPER_AVAILABLE = True
        return faster_whisper
    except ImportError:
        try:
            import whisper
            return whisper
        except ImportError:
            raise ImportError(
                "No Whisper backend found. Install one: "
                "`pip install faster-whisper` or `pip install openai-whisper`"
            )


async def load_models():
    """Register the STT model with ModelManager and load it."""
    from model_manager import get_manager
    mgr = get_manager()

    async def _load():
        global _WHISPER_MODEL
        logger.info("Loading Whisper model: %s ...", STT_MODEL_SIZE)
        mod = _import_whisper()
        if _FASTER_WHISPER_AVAILABLE:
            # faster-whisper: more efficient on GPU
            _WHISPER_MODEL = mod.WhisperModel(
                STT_MODEL_SIZE,
                device="cuda",
                compute_type="float16",
            )
        else:
            _WHISPER_MODEL = mod.load_model(STT_MODEL_SIZE, device="cuda")
        logger.info("Whisper model loaded")
        return _WHISPER_MODEL

    async def _unload():
        global _WHISPER_MODEL
        if _WHISPER_MODEL is not None:
            if hasattr(_WHISPER_MODEL, "cpu"):
                _WHISPER_MODEL.cpu()
            del _WHISPER_MODEL
        _WHISPER_MODEL = None
        import gc
        gc.collect()
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    mgr.register(
        "stt", STT_VRAM_GB, _load, _unload,
        health_check=lambda: _WHISPER_MODEL is not None,
    )


async def unload_models():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        if hasattr(_WHISPER_MODEL, "cpu"):
            _WHISPER_MODEL.cpu()
        del _WHISPER_MODEL
    _WHISPER_MODEL = None
    import gc
    gc.collect()
    import torch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _transcribe_sync(
    audio_bytes: bytes,
    language: str | None,
) -> tuple[str, float]:
    """Blocking transcription — call via a thread, never on the event loop."""
    import tempfile

    # Write to temp file for whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        if _FASTER_WHISPER_AVAILABLE:
            segments, info = _WHISPER_MODEL.transcribe(
                tmp_path,
                language=language,
                beam_size=5,
            )
            segments = list(segments)
            text = " ".join(seg.text for seg in segments)
            duration = info.duration if hasattr(info, "duration") else 0.0
        else:
            result = _WHISPER_MODEL.transcribe(tmp_path, language=language)
            text = result.get("text", "")
            duration = result.get("duration", 0.0)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return text, duration


async def transcribe(
    audio_bytes: bytes,
    language: str | None = None,
    response_format: str = "json",
) -> dict:
    """Transcribe audio bytes and return OpenAI-compatible response."""
    if _WHISPER_MODEL is None:
        raise RuntimeError("STT model not loaded")

    # Whisper inference is fully synchronous and takes seconds; running it
    # inline would block the event loop and stall every other request.
    text, duration = await asyncio.to_thread(_transcribe_sync, audio_bytes, language)

    if response_format == "verbose_json":
        return {"text": text, "duration": duration, "language": language or "en"}
    elif response_format == "text":
        return text
    else:
        return {"text": text.strip()}