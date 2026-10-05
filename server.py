"""
Unified AI Provider API — clean RESTful + OpenAI-compatible + legacy.

All providers register with the VRAM-aware ModelManager and load on demand.

Sections (top to bottom):
  1. Imports & config
  2. Lifespan (startup / shutdown)
  3. Models  — GET /models, POST /models/{name}/load|unload
  4. Health  — GET /health
  5. Audio   — POST /v1/audio/speech, /v1/audio/transcriptions
  6. Voices  — GET|POST|PATCH|DELETE /audio/voices
  7. Chat    — POST /v1/chat/completions
  8. Embed   — POST /v1/embeddings
  9. Image   — POST /v1/images/generations
  10. Legacy — popcorn4 compatibility wrappers (deprecated)
  11. Docs   — GET /, /documentation, /documentation/{name}, /SKILL.md
"""

from __future__ import annotations

import os
import sys
import time
import base64
import asyncio
import logging
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

import admission
import documentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("ai-provider")

# httpx logs every internal backend call (llama-server, TTS worker, ...);
# that's plumbing noise — real inbound traffic is logged by the middleware below.
logging.getLogger("httpx").setLevel(logging.WARNING)

HOST = os.getenv("AI_PROVIDER_HOST", "0.0.0.0")
PORT = int(os.getenv("AI_PROVIDER_PORT", "8765"))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "1024"))

# Admission control lives in admission.py: every backend has a gate sized to
# what it can run at once (llama-server's slot count, one TTS/STT/SDXL job),
# with a bounded FIFO behind it.  Queue limits: CHAT_MAX_QUEUE, EMBED_MAX_QUEUE,
# TTS_MAX_QUEUE, STT_MAX_QUEUE, IMAGE_MAX_QUEUE.


# ── orphan prevention ──────────────────────────────────────────────────
import ctypes
import subprocess as sp
import io as _io
import csv as _csv


def _setup_job_object():
    """Windows Job Object: auto-kill child processes when this process exits.

    Uses KILL_ON_JOB_CLOSE so that when the server dies (taskkill /F,
    crash, SIGKILL), Windows terminates all spawned subprocesses.
    Falls back silently if the process is already in a job (VS Code
    terminal, etc.).
    """
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return

        class _BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class _IO(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class _EXT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC),
                ("IoInfo", _IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = _EXT()
        info.BasicLimitInformation.LimitFlags = 0x00002000

        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            return

        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(job)
            return

        # Keep a reference so the handle stays alive until process exit
        _setup_job_object._job = job
        logger.info("Job Object active — child processes will be killed on exit")
    except Exception as e:
        logger.debug("Job Object setup failed (non-critical): %s", e)


def _cleanup_orphans():
    """Kill orphaned subprocesses from earlier server runs.

    Handles the case where the previous server was killed without
    going through the graceful shutdown path (lifespan yield → shutdown).
    """
    sp.run(["taskkill", "/F", "/IM", "llama-server.exe"],
            capture_output=True, timeout=5)

    try:
        result = sp.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*tts_worker*' } | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip():
            for pid in result.stdout.strip().split():
                pid = pid.strip()
                if pid:
                    sp.run(["taskkill", "/F", "/PID", pid],
                            capture_output=True, timeout=5)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
#  1. LIFESPAN
# ══════════════════════════════════════════════════════════════════════════

# ── notifications (Windows toast) ──────────────────────────────────────
TEMP_ALERT_C = float(os.getenv("AI_TEMP_ALERT_C", "80"))
TEMP_ALERT_COOLDOWN_S = int(os.getenv("AI_TEMP_ALERT_COOLDOWN", "300"))


def _windows_toast(title: str, message: str) -> bool:
    """Fire a Windows toast notification.  Returns False on failure."""
    esc = lambda s: s.replace("'", "''")[:180]
    ps = (
        "[void][Windows.UI.Notifications.ToastNotificationManager,"
        "Windows.UI.Notifications,ContentType=WindowsRuntime];"
        "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
        "$t.GetElementsByTagName('text').Item(0).AppendChild("
        "$t.CreateTextNode('" + esc(title) + "'))|Out-Null;"
        "$t.GetElementsByTagName('text').Item(1).AppendChild("
        "$t.CreateTextNode('" + esc(message) + "'))|Out-Null;"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        "'AI Provider').Show([Windows.UI.Notifications.ToastNotification]::new($t))"
    )
    try:
        sp.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
            stdout=sp.DEVNULL, stderr=sp.DEVNULL,
            creationflags=getattr(sp, "CREATE_NO_WINDOW", 0),
        )
        return True
    except Exception as e:
        logger.warning("toast failed: %s", e)
        return False


def _gpu_temp() -> float | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return None


async def _temp_watcher():
    """Background loop: toast when GPU temp crosses TEMP_ALERT_C (with cooldown)."""
    last_alert = 0.0
    hot = False
    while True:
        await asyncio.sleep(15)
        temp = _gpu_temp()
        if temp is None:
            continue
        now = time.time()
        if temp >= TEMP_ALERT_C and not hot and now - last_alert > TEMP_ALERT_COOLDOWN_S:
            hot = True
            last_alert = now
            _windows_toast("GPU running hot", f"{temp:.0f}°C (alert at {TEMP_ALERT_C:.0f}°C)")
            logger.warning("GPU temp alert: %.0f°C", temp)
        elif temp < TEMP_ALERT_C - 5:
            hot = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    from model_manager import get_manager, unload_all
    mgr = get_manager()

    # Kill orphaned subprocesses from previous runs, then set up a
    # Windows Job Object so new child processes die with the parent.
    _cleanup_orphans()
    _setup_job_object()

    # Clear any stale loaded state from a previous process
    await unload_all()

    from providers import tts as tts_provider
    await tts_provider.load_models()
    logger.info("TTS provider ready")

    from providers import llm as llm_provider
    await llm_provider.load_models()
    logger.info("LLM provider registered")

    from providers import embeddings as embed_provider
    await embed_provider.load_models()
    logger.info("Embeddings provider registered")

    from providers import stt as stt_provider
    try:
        await stt_provider.load_models()
        logger.info("STT provider registered")
    except ImportError as e:
        logger.warning("STT not available: %s", e)

    from providers import image as image_provider
    await image_provider.register_models()
    logger.info("Image provider registered")

    watcher = asyncio.create_task(_temp_watcher())
    yield
    watcher.cancel()

    logger.info("Shutting down AI provider ...")
    for mod_name in ("tts", "llm", "embeddings", "stt", "image"):
        try:
            mod = __import__(f"providers.{mod_name}", fromlist=[""])
            unloader = getattr(mod, "unload_models" if mod_name != "image" else "unload_model", None)
            if unloader:
                await unloader()
        except Exception:
            pass
    from model_manager import unload_all
    await unload_all()


app = FastAPI(
    title="AI Provider",
    version="0.8.0",
    description=(
        "VRAM-aware REST API for local AI on an RTX 5090 (32 GB). All modalities:\n\n"
        "- **Chat** (text in/out) and **vision** (image in) — Qwen3.8-27B via llama-server\n"
        "- **TTS** (audio out) — Qwen3-TTS voice clone\n"
        "- **STT** (audio in) — faster-whisper\n"
        "- **Embeddings** — nomic-embed-text\n"
        "- **Image generation** — SDXL\n\n"
        "Models are loaded/unloaded on demand via `/models/{name}/load`;"
        " TTS and image generation auto-load. See the Models section for VRAM status.\n\n"
        "**Full documentation** is served by this API: [/documentation](/documentation)"
        " (index), [overview](/documentation/readme),"
        " [API reference](/documentation/api),"
        " [engineering notes](/documentation/agents) — raw markdown for API clients,"
        " rendered HTML for browsers.\n\n"
        # Lifted from API.md so the two can't drift apart.
        + documentation.openapi_section("api", "Queueing & rate limits")
    ),
    lifespan=lifespan,
    docs_url="/docs",
    openapi_tags=[
        {"name": "Health", "description": "Server & GPU health"},
        {"name": "Models", "description": "VRAM-aware model lifecycle"},
        {"name": "Audio", "description": "TTS & STT (OpenAI-compatible)"},
        {"name": "Voices", "description": "Clone voice management"},
        {"name": "Chat", "description": "LLM chat completions (OpenAI-compatible)"},
        {"name": "Embeddings", "description": "Text embeddings (OpenAI-compatible)"},
        {"name": "Image", "description": "Image generation (OpenAI-compatible)"},
        {"name": "Legacy", "description": "Deprecated popcorn4-compat endpoints"},
        {"name": "Documentation", "description": "This project's guides, as markdown or HTML"},
    ],
)

# OpenAPI entries for the admission outcomes every gated endpoint can return
# (see admission.py).  499 is left out: it is only ever logged, since the
# client it would be sent to has already gone.
ADMISSION_RESPONSES: dict[int | str, dict] = {
    429: {
        "description": "The backend's queue is full. Retry after `Retry-After` seconds, "
                       "derived from the measured service time.",
        "headers": {"Retry-After": {"schema": {"type": "integer"},
                                    "description": "Seconds until a queue place should open."}},
    },
    409: {"description": "Cancelled server-side (e.g. `POST /v1/chat/completions/cancel`)."},
}


@app.exception_handler(admission.QueueFull)
@app.exception_handler(admission.ClientGone)
@app.exception_handler(admission.Cancelled)
async def admission_error_handler(request: Request, exc: Exception):
    """429 (with Retry-After) / 499 / 409 for refused, abandoned, cancelled."""
    err = admission.http_error(exc)
    return JSONResponse({"detail": err.detail}, status_code=err.status_code, headers=err.headers)


@app.middleware("http")
async def access_log_middleware(request, call_next):
    """One INFO line per inbound request: client, method, path, status, duration."""
    path = request.url.path
    if path in ("/docs", "/redoc", "/openapi.json"):
        return await call_next(request)
    t0 = time.perf_counter()
    response = await call_next(request)
    dur_ms = (time.perf_counter() - t0) * 1000
    client = request.client.host if request.client else "-"
    logger.info("%s %s %s -> %d (%.0f ms)",
                client, request.method, path, response.status_code, dur_ms)
    return response


@app.post("/notify", tags=["Health"])
async def notify(req: dict):
    """Show a Windows toast: {"title": "...", "message": "..."} (message optional)."""
    title = str(req.get("title") or "AI Provider")
    message = str(req.get("message") or "")
    if not _windows_toast(title, message):
        raise HTTPException(500, "Failed to launch notification")
    return {"sent": True, "title": title}


# ══════════════════════════════════════════════════════════════════════════
#  2. MODELS  (resource lifecycle)
# ══════════════════════════════════════════════════════════════════════════

@app.get("/models", tags=["Models"])
async def list_models():
    from model_manager import get_manager
    mgr = get_manager()
    return {"models": mgr.list_all(), "loaded": mgr.list_loaded()}


@app.get("/models/llm-chat/variants", tags=["Models"])
async def chat_variants():
    """Which chat models `llm-chat` can serve, and which one is selected.

    Both variants share one llama-server on the chat port, so exactly one
    can be loaded at a time — selecting is a choice, not an addition.
    """
    from providers.llm import list_variants
    return list_variants()


@app.post("/models/llm-chat/variant/{key}", tags=["Models"])
async def chat_variant_select(key: str):
    """Point `llm-chat` at a different chat model. Unload it first."""
    from providers.llm import select_variant
    try:
        return select_variant(key)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/models/{name}/load", tags=["Models"])
async def models_load(name: str, force: bool = Query(False)):
    from model_manager import get_manager
    mgr = get_manager()
    try:
        await mgr.load(name, force=force)
        # Auto-load companion models (tiny ones that share the runtime)
        companions = {"llm-chat": ["llm-embed"], "llm-embed": ["llm-chat"]}
        for comp in companions.get(name, []):
            if not mgr.is_loaded(comp):
                try:
                    await mgr.load(comp, force=force)
                except RuntimeError:
                    pass  # companion may not fit; not fatal
        return {"status": "loaded", "name": name}
    except KeyError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except ImportError as e:
        raise HTTPException(501, str(e))


@app.post("/models/{name}/unload", tags=["Models"])
async def models_unload(name: str):
    from model_manager import get_manager
    mgr = get_manager()
    await mgr.unload(name)
    # Also unload companions that depend on this model
    dependants = {"llm-chat": ["llm-embed"]}
    for dep in dependants.get(name, []):
        await mgr.unload(dep)
    return {"status": "unloaded", "name": name}


# ══════════════════════════════════════════════════════════════════════════
#  3. HEALTH
# ══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Health"])
async def health():
    import torch
    from model_manager import get_manager
    from providers.image import state as img_state

    mgr = get_manager()

    cuda_avail = torch.cuda.is_available()
    total_gb = mgr.max_vram_gb
    if cuda_avail:
        free_gb, total_gpu = torch.cuda.mem_get_info(0)
        free_gb = free_gb / (1024 ** 3)
        total_gpu = total_gpu / (1024 ** 3)
    else:
        free_gb = total_gb - mgr.loaded_gb
        total_gpu = total_gb

    loaded_models = {
        s["name"]: {
            "vram_gb": s["size_gb"],
            "loaded_at": s["loaded_at"],
        } for s in mgr.list_loaded()
    }

    # GPU telemetry via pynvml
    gpu = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu = {
            "name": pynvml.nvmlDeviceGetName(handle),
            "temp_c": temp,
            "gpu_util_pct": util.gpu,
            "mem_util_pct": util.memory,
            "mem_used_gb": round(mem.used / (1024 ** 3), 1),
            "mem_total_gb": round(mem.total / (1024 ** 3), 1),
            "idle": util.gpu < 5 and util.memory < 10,
        }
    except Exception as e:
        logger.warning("health gpu telemetry failed: %s", e)
        gpu = {"name": "unknown", "idle": True}

    return {
        "status": "ok",
        "vram": {
            "max_vram_gb": mgr.max_vram_gb,
            "total_gb": round(total_gpu, 1),
            "free_gb": round(free_gb, 1),
            "loaded_gb": round(mgr.loaded_gb, 1),
            "per_model": loaded_models,
        },
        "gpu": gpu,
        "sdxl_loaded": img_state.pipe is not None,
        # Per backend: slots, running, queued, mean service time, est. wait.
        "queues": admission.all_stats(),
    }


# ══════════════════════════════════════════════════════════════════════════
#  4. AUDIO  (OpenAI-compatible TTS + STT)
# ══════════════════════════════════════════════════════════════════════════

# ── TTS ────────────────────────────────────────────────────────────────

class SpeechRequest(BaseModel):
    """OpenAI-compatible TTS request."""
    input: str = Field(..., description="Text to speak.")
    voice: str = Field(default="axel", description="Tag of a cloned voice (see /audio/voices).")
    language: str = Field(default="English")
    response_format: str = Field(default="opus", description='"opus" (ogg) or "wav".')
    model: str = Field(default="", description="Ignored; TTS model is fixed.")
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"input": "Hello there.", "voice": "roxy", "response_format": "opus"}
            ]
        }
    }


@app.post(
    "/v1/audio/speech",
    tags=["Audio"],
    responses={200: {"content": {"audio/ogg": {}, "audio/wav": {}},
                     "description": "Audio bytes; X-Waveform and X-Duration-Seconds headers."},
               **ADMISSION_RESPONSES},
)
async def v1_audio_speech(req: SpeechRequest, request: Request):
    text = req.input
    if not text:
        raise HTTPException(400, "input is required")
    voice = req.voice
    language = req.language
    fmt = "opus" if req.response_format != "wav" else "wav"

    from providers.tts import synthesize_audio, is_running
    from model_manager import get_manager

    # Auto-load TTS if not running
    if not is_running():
        try:
            await get_manager().load("tts")
        except RuntimeError as e:
            raise HTTPException(503, f"TTS failed to load: {e}")

    data, media_type, dur, wf = await synthesize_audio(
        text, language, voice, fmt=fmt, request=request,
    )
    return Response(
        content=data,
        media_type=media_type,
        headers={"X-Waveform": wf, "X-Duration-Seconds": str(dur)},
    )


# ── STT ────────────────────────────────────────────────────────────────

@app.post("/v1/audio/transcriptions", tags=["Audio"], responses=ADMISSION_RESPONSES)
async def v1_audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str = Form(None),
    response_format: str = Form("json"),
):
    from providers.stt import transcribe
    audio_bytes = await file.read()
    try:
        result = await transcribe(
            audio_bytes, language=language, response_format=response_format, request=request,
        )
        return Response(content=result, media_type="text/plain") if isinstance(result, str) else result
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ImportError as e:
        raise HTTPException(501, str(e))


# ══════════════════════════════════════════════════════════════════════════
#  5. VOICES  (clone management — CRUD sub-resource of audio)
# ══════════════════════════════════════════════════════════════════════════

@app.get("/audio/voices", tags=["Voices"])
async def audio_voices_list():
    """List all voices (built-in presets + cloned)."""
    from providers.tts import get_all_voices, cloned_voices as cvs
    return {
        "voices": sorted(get_all_voices()),
        "clones": {
            tag: {
                "has_transcript": bool(v.get("ref_text")),
                "clone_of": v.get("clone_of", ""),
                "cloned_from_msg": v.get("cloned_from_msg", ""),
            }
            for tag, v in cvs.items()
        },
    }


@app.post("/audio/voices", tags=["Voices"], responses=ADMISSION_RESPONSES)
async def audio_voices_save(
    tag: str = Form(...),
    new_tag: str = Form(default=""),
    ref_text: str = Form(default=""),
    clone_of: str = Form(default=""),
    cloned_from_msg: str = Form(default=""),
    ref_audio: UploadFile = File(None),
):
    """Create, overwrite, rename, or update metadata on a cloned voice."""
    from providers.tts import REFS_DIR, cloned_voices, _save_clones, FFMPEG
    import subprocess as sp
    import torch

    tag = tag.strip().lower()
    if not tag:
        raise HTTPException(400, "Tag cannot be empty")

    # ── rename-only ─────
    if new_tag and tag in cloned_voices:
        new_tag = new_tag.strip().lower()
        if not new_tag:
            raise HTTPException(400, "new_tag cannot be empty")
        if new_tag in cloned_voices and new_tag != tag:
            raise HTTPException(409, f"Clone '{new_tag}' already exists")
        for ext in (".wav", ".pt"):
            old = os.path.join(REFS_DIR, f"{tag}{ext}")
            new_ = os.path.join(REFS_DIR, f"{new_tag}{ext}")
            if os.path.exists(old):
                os.rename(old, new_)
        cloned_voices[new_tag] = cloned_voices.pop(tag)
        cloned_voices[new_tag]["prompt_path"] = cloned_voices[new_tag]["prompt_path"].replace(tag, new_tag)
        _save_clones()
        return {"status": "renamed", "old_tag": tag, "new_tag": new_tag}

    # ── metadata-only update ─────
    if tag in cloned_voices and not ref_audio:
        cv = cloned_voices[tag]
        if ref_text:
            cv["ref_text"] = ref_text
        if clone_of:
            cv["clone_of"] = clone_of
        if cloned_from_msg:
            cv["cloned_from_msg"] = cloned_from_msg
        _save_clones()
        return {"status": "updated", "tag": tag}

    # ── create or overwrite ─────
    if not ref_audio or not ref_audio.filename:
        if tag in cloned_voices:
            raise HTTPException(400, "No audio file provided for overwrite")
        raise HTTPException(404, f"Clone '{tag}' not found — provide ref_audio to create it")

    os.makedirs(REFS_DIR, exist_ok=True)
    wav_path = os.path.join(REFS_DIR, f"{tag}.wav")

    try:
        audio_bytes = await ref_audio.read()
        proc = sp.run(
            [FFMPEG, "-y", "-i", "pipe:0", "-ar", "24000", "-ac", "1",
             "-af", "loudnorm=I=-16:LRA=1:TP=-1.5", "-c:a", "pcm_s16le", wav_path],
            input=audio_bytes, capture_output=True, timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {proc.stderr.decode(errors='replace')[:200]}")
    except Exception as e:
        raise HTTPException(400, f"Failed to process audio: {e}")

    from providers import tts as tts_provider
    from model_manager import get_manager
    if not tts_provider.is_running():
        try:
            await get_manager().load("tts")
        except RuntimeError as e:
            raise HTTPException(503, f"TTS failed to load: {e}")

    try:
        result = await tts_provider.precompute_clone(tag, ref_text)
        pt_path = result["prompt_path"]
        mode = result["mode"]
    except Exception as e:
        raise HTTPException(500, f"Failed to precompute voice prompt: {e}")

    cloned_voices[tag] = {
        "prompt_path": pt_path,
        "ref_text": ref_text or "",
        "clone_of": clone_of or "",
        "cloned_from_msg": cloned_from_msg or "",
    }
    _save_clones()
    return {"status": "saved", "tag": tag, "mode": mode}


@app.patch("/audio/voices/{tag}", tags=["Voices"])
async def audio_voices_update(
    tag: str,
    new_tag: str = Form(default=""),
    ref_text: str = Form(default=""),
    clone_of: str = Form(default=""),
    cloned_from_msg: str = Form(default=""),
):
    """Update metadata on a cloned voice (rename, transcript, attribution)."""
    from providers.tts import REFS_DIR, cloned_voices, _save_clones
    tag = tag.strip().lower()
    if tag not in cloned_voices:
        raise HTTPException(404, f"Clone '{tag}' not found")

    cv = cloned_voices[tag]

    # rename
    if new_tag:
        new_tag = new_tag.strip().lower()
        if not new_tag:
            raise HTTPException(400, "new_tag cannot be empty")
        if new_tag in cloned_voices and new_tag != tag:
            raise HTTPException(409, f"Clone '{new_tag}' already exists")
        for ext in (".wav", ".pt"):
            old = os.path.join(REFS_DIR, f"{tag}{ext}")
            new_ = os.path.join(REFS_DIR, f"{new_tag}{ext}")
            if os.path.exists(old):
                os.rename(old, new_)
        cloned_voices[new_tag] = cv
        del cloned_voices[tag]
        cloned_voices[new_tag]["prompt_path"] = cloned_voices[new_tag]["prompt_path"].replace(tag, new_tag)
        tag = new_tag

    if ref_text:
        cv["ref_text"] = ref_text
    if clone_of:
        cv["clone_of"] = clone_of
    if cloned_from_msg:
        cv["cloned_from_msg"] = cloned_from_msg

    _save_clones()
    return {"status": "updated", "tag": tag}


@app.delete("/audio/voices/{tag}", tags=["Voices"])
async def audio_voices_delete(tag: str):
    """Delete a cloned voice — moves its files to references/_deleted/ and removes from the list."""
    from providers.tts import cloned_voices, REFS_DIR, _save_clones
    import shutil
    tag = tag.strip().lower()
    if tag not in cloned_voices:
        raise HTTPException(404, f"Clone '{tag}' not found")

    # Move files to a _deleted subfolder instead of erasing
    deleted_dir = os.path.join(REFS_DIR, "_deleted")
    os.makedirs(deleted_dir, exist_ok=True)
    for ext in (".pt", ".wav"):
        src = os.path.join(REFS_DIR, f"{tag}{ext}")
        if os.path.exists(src):
            dst = os.path.join(deleted_dir, f"{tag}{ext}")
            # Avoid overwrite collisions
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)

    del cloned_voices[tag]
    _save_clones()
    return {"status": "deleted", "tag": tag}


# ══════════════════════════════════════════════════════════════════════════
#  6. CHAT  (LLM — OpenAI-compatible)
# ══════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    model: str = Field(default="qwen3.8-27b")
    messages: list = Field(
        ...,
        description=(
            "OpenAI-style messages. `content` may be a plain string, or an array of "
            "parts for multimodal input: {\"type\": \"text\", \"text\": ...} and "
            "{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,...\"}} "
            "(vision requires llm-chat loaded). Responses include `reasoning_content` "
            "alongside `content`."
        ),
    )
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048, description="Server clamps this to MAX_COMPLETION_TOKENS.")
    stream: bool = Field(default=False)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "qwen3.8-27b",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What is in this image?"},
                                {"type": "image_url",
                                 "image_url": {"url": "data:image/png;base64,<base64 data>"}},
                            ],
                        }
                    ],
                    "max_tokens": 1024,
                }
            ]
        }
    }


@app.post("/v1/chat/completions", tags=["Chat"], responses=ADMISSION_RESPONSES)
async def v1_chat_completions(req: ChatRequest, request: Request):
    from providers.llm import get_provider
    prov = get_provider()
    if not prov.is_running:
        raise HTTPException(503, "LLM server not running. POST /models/llm-chat/load to start it.")
    _log_chat_body(req)
    # Clamp runaway outputs: popcorn4 sends max_tokens=4096 for short
    # classification answers and retries before they finish, pinning all
    # 4 slots indefinitely.
    req.max_tokens = min(req.max_tokens, MAX_COMPLETION_TOKENS)
    # Waits in a FIFO for a free llama-server slot and is dispatched the
    # moment one opens; dropped if the client disconnects while waiting.
    try:
        return await prov.chat_completions(req.model_dump(), request=request)
    except admission.ADMISSION_ERRORS:
        raise
    except Exception as e:
        raise HTTPException(502, f"LLM proxy error: {e}")


# Diagnostic log: what clients actually ask for.  Disable with LOG_CHAT_CONTENT=0.
CHAT_BODY_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_requests.log")


def _log_chat_body(req: ChatRequest):
    if os.getenv("LOG_CHAT_CONTENT", "1") == "0":
        return
    try:
        import json as _json
        msgs = req.messages[-3:]  # last few messages are the live conversation
        summary = [
            {
                "role": m.get("role") if isinstance(m, dict) else "?",
                "text": (m.get("content") if isinstance(m, dict) else str(m)),
                "has_image": isinstance(m, dict) and isinstance(m.get("content"), list)
                and any(p.get("type") == "image_url" for p in m["content"] if isinstance(p, dict)),
            }
            for m in msgs
        ]
        # Truncate each text to keep the file sane
        for m in summary:
            if isinstance(m["text"], str):
                m["text"] = m["text"][:500]
            elif isinstance(m["text"], list):
                m["text"] = [p.get("text", "<image>")[:200] if isinstance(p, dict) else "?"
                             for p in m["text"]][:6]
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_messages": len(req.messages),
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "last_messages": summary,
        }
        with open(CHAT_BODY_LOG, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("chat body log failed: %s", e)


@app.post("/v1/chat/completions/cancel", tags=["Chat"])
async def v1_chat_cancel():
    """Cancel every chat generation running or queued.

    Running generations are aborted in llama-server (their slots free up
    immediately); the cancelled requests answer 409.
    """
    from providers.llm import get_provider
    prov = get_provider()
    n = prov.cancel_current()
    return {"cancelled": n > 0, "count": n}


# ══════════════════════════════════════════════════════════════════════════
#  7. EMBEDDINGS  (OpenAI-compatible)
# ══════════════════════════════════════════════════════════════════════════

class EmbedRequest(BaseModel):
    model: str = Field(default="nomic-embed-text-v1.5")
    input: str | list[str] = Field(...)


@app.post("/v1/embeddings", tags=["Embeddings"], responses=ADMISSION_RESPONSES)
async def v1_embeddings(req: EmbedRequest, request: Request):
    from providers.embeddings import get_provider
    prov = get_provider()
    if not prov.is_running:
        raise HTTPException(503, "Embeddings server not running. POST /models/llm-embed/load to start it.")
    try:
        return await prov.embeddings(req.model_dump(), request=request)
    except admission.ADMISSION_ERRORS:
        raise
    except Exception as e:
        raise HTTPException(502, f"Embeddings proxy error: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  8. IMAGE  (txt2img — OpenAI-compatible)
# ══════════════════════════════════════════════════════════════════════════

class ImageGenRequest(BaseModel):
    model: str = Field(default="sdxl")
    prompt: str = Field(...)
    n: int = Field(default=1, ge=1, le=4)
    size: str = Field(default="1024x1024", pattern=r"^\d+x\d+$")
    negative_prompt: str = Field(default="")
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    guidance_scale: float = Field(default=7.0, ge=0.0, le=30.0)
    seed: int | None = Field(default=None)


@app.post("/v1/images/generations", tags=["Image"], responses=ADMISSION_RESPONSES)
async def v1_images_generations(req: ImageGenRequest, request: Request):
    from providers.image import generate_txt2img, state as img_state
    from model_manager import get_manager

    if img_state.pipe is None:
        mgr = get_manager()
        try:
            await mgr.load("sdxl")
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        except KeyError:
            raise HTTPException(404, "SDXL model not registered")

    w_str, h_str = req.size.split("x")
    width, height = int(w_str), int(h_str)

    try:
        png_bytes = await generate_txt2img(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=width, height=height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            request=request,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {
        "created": int(time.time()),
        "data": [{"b64_json": b64, "revised_prompt": None} for _ in range(req.n)],
    }


# ══════════════════════════════════════════════════════════════════════════
#  9. LEGACY  (popcorn4 bot compatibility — deprecated)
# ══════════════════════════════════════════════════════════════════════════

# ── TTS ────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = Field(default="English")
    speaker: str = Field(default="axel")


@app.post("/tts", tags=["Legacy"], deprecated=True, responses=ADMISSION_RESPONSES)
async def legacy_tts(
    req: TTSRequest, request: Request,
    format: str = Query("opus", pattern="^(wav|opus)$"),
):
    from providers.tts import synthesize_audio, is_running
    from model_manager import get_manager

    # Auto-load TTS if not running
    if not is_running():
        try:
            await get_manager().load("tts")
        except RuntimeError as e:
            raise HTTPException(503, f"TTS failed to load: {e}")

    data, media_type, dur, wf = await synthesize_audio(
        req.text, req.language, req.speaker, fmt=format, request=request,
    )
    return Response(
        content=data, media_type=media_type,
        headers={"X-Waveform": wf, "X-Duration-Seconds": str(dur)},
    )


# ── one-shot clone ─────────────────────────────────────────────────────

class CloneRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = Field(default="English")
    ref_audio: str
    ref_text: str = ""
    x_vector_only_mode: bool = False


@app.post("/clone", tags=["Legacy"], deprecated=True, responses=ADMISSION_RESPONSES)
async def legacy_clone(
    req: CloneRequest, request: Request,
    format: str = Query("opus", pattern="^(wav|opus)$"),
):
    from providers.tts import synthesize_clone
    data, media_type, dur, wf = await synthesize_clone(
        req.text, req.language, req.ref_audio, req.ref_text, fmt=format, request=request,
    )
    return Response(
        content=data, media_type=media_type,
        headers={"X-Waveform": wf, "X-Duration-Seconds": str(dur)},
    )


# ── clone/save → POST /audio/voices ────────────────────────────────────

@app.post("/clone/save", tags=["Legacy"], deprecated=True, responses=ADMISSION_RESPONSES)
async def legacy_clone_save(
    tag: str = Form(...),
    new_tag: str = Form(default=""),
    ref_text: str = Form(default=""),
    clone_of: str = Form(default=""),
    cloned_from_msg: str = Form(default=""),
    ref_audio: UploadFile = File(None),
):
    """Deprecated — use POST /audio/voices instead."""
    return await audio_voices_save(tag, new_tag, ref_text, clone_of, cloned_from_msg, ref_audio)


# ── clone/list → GET /audio/voices ─────────────────────────────────────

@app.post("/clone/list", tags=["Legacy"], deprecated=True)
async def legacy_clone_list():
    from providers.tts import cloned_voices
    return {"clones": {
        tag: {
            "has_transcript": bool(v.get("ref_text")),
            "clone_of": v.get("clone_of", ""),
            "cloned_from_msg": v.get("cloned_from_msg", ""),
        } for tag, v in cloned_voices.items()
    }}


# ── clone/delete → DELETE /audio/voices/{tag} ──────────────────────────

@app.post("/clone/delete", tags=["Legacy"], deprecated=True)
async def legacy_clone_delete(tag: str = Form(...)):
    return await audio_voices_delete(tag)


# ── /voices → GET /audio/voices ────────────────────────────────────────

@app.post("/voices", tags=["Legacy"], deprecated=True)
async def legacy_list_voices():
    from providers.tts import get_all_voices
    return {"voices": sorted(get_all_voices())}


# ══════════════════════════════════════════════════════════════════════════
#  10. DOCUMENTATION  (the repo's markdown, served from disk)
# ══════════════════════════════════════════════════════════════════════════

_FORMAT_QUERY = Query(
    None, pattern="^(md|html)$",
    description="`md` or `html`. Default: HTML if the client accepts `text/html` "
                "(a browser), otherwise markdown.",
)


@app.get("/", tags=["Documentation"], include_in_schema=False)
@app.get("/documentation", tags=["Documentation"])
async def documentation_index(request: Request, format: str | None = _FORMAT_QUERY):
    """Every document this API serves, plus links to the OpenAPI views."""
    if documentation.wants_html(request.headers.get("accept"), format):
        return HTMLResponse(documentation.render_index())
    return {
        "documents": documentation.index(),
        "openapi": {"swagger_ui": "/docs", "redoc": "/redoc", "json": "/openapi.json"},
    }


@app.get(
    "/SKILL.md",
    tags=["Documentation"],
    response_class=Response,
    responses={200: {"content": {"text/markdown": {}},
                     "description": "Agent Skills-format SKILL.md for this API."}},
)
async def skill_md():
    """How an AI agent should use this API, as a SKILL.md.

    Always raw markdown regardless of `Accept`, so it can be fetched or saved
    straight into a skills directory. Rendered: `/documentation/skill`.
    """
    doc = documentation.DOCS["skill"]
    try:
        return Response(doc.read(), media_type="text/markdown; charset=utf-8")
    except OSError as e:
        raise HTTPException(500, f"Could not read {doc.filename}: {e}")


@app.get(
    "/documentation/{name}",
    tags=["Documentation"],
    responses={200: {"content": {"text/markdown": {}, "text/html": {}},
                     "description": "The document, as markdown or rendered HTML."}},
)
async def documentation_get(name: str, request: Request, format: str | None = _FORMAT_QUERY):
    """One document by name (`readme`, `api`, `agents`, `skill`) or filename (`API.md`).

    Read from disk per request, so it is always the current text.
    """
    doc = documentation.find(name)
    if doc is None:
        raise HTTPException(
            404, f"No document {name!r}. Available: {', '.join(documentation.DOCS)}",
        )
    try:
        if documentation.wants_html(request.headers.get("accept"), format):
            return HTMLResponse(documentation.render_doc(doc))
        return Response(doc.read(), media_type="text/markdown; charset=utf-8")
    except OSError as e:
        raise HTTPException(500, f"Could not read {doc.filename}: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  11. ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", access_log=False)