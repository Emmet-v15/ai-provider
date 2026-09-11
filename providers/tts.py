"""
Qwen3-TTS provider — spawns a worker subprocess for GPU isolation.

The worker (tts_worker.py) handles all model inference.  Killing the worker
instantly releases VRAM.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import socket
import subprocess as sp
import sys
import logging

import httpx
from fastapi import HTTPException

logger = logging.getLogger("provider.tts")

# ── config ────────────────────────────────────────────────────────────
TTS_SIZE_GB = 4.5  # Base model in bfloat16 (~3.9 GB load + ~0.5 GB gen overhead)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLONES_FILE = os.path.join(_BASE_DIR, "cloned_voices.json")
REFS_DIR = os.path.join(_BASE_DIR, "references")

# Preset voice aliases (removed — only clones are supported now)
VOICE_ALIASES: dict[str, str] = {}

# FFmpeg path (used by server.py for clone-audio conversion)
FFMPEG = os.getenv("QWEN_TTS_FFMPEG") or next(
    (p for p in [
        r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
        r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\yt-dlp.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-N-124279-g0f6ba39122-win64-gpl\bin\ffmpeg.exe",
        "ffmpeg",
    ] if os.path.exists(p) or p == "ffmpeg"),
    "ffmpeg",
)

# ── worker management ─────────────────────────────────────────────────
_proc: sp.Popen | None = None
_worker_url: str | None = None
_log_fh = None
WORKER_START_TIMEOUT = 90  # single model load

# Serialises worker start/stop.  Without it two overlapping loads each pick
# their own free port and spawn a worker; the second overwrites the _proc /
# _worker_url globals and the first is orphaned holding its VRAM with no
# reference left to kill it.
_worker_lock = asyncio.Lock()

cloned_voices: dict[str, dict[str, str]] = {}


def resolve_voice(name: str) -> str | None:
    return None  # preset voices removed; only clones supported


# ── cloned voices persistence ─────────────────────────────────────────
def _load_clones():
    global cloned_voices
    try:
        if os.path.exists(CLONES_FILE):
            with open(CLONES_FILE) as f:
                cloned_voices = json.load(f)
    except Exception as e:
        logger.error("Failed to load clones: %s", e)
        cloned_voices = {}

    # prompt_path is persisted as an absolute path, so a moved project dir
    # leaves every entry stale.  Remap missing paths to the same filename in
    # the current REFS_DIR and write the fix back once.
    changed = False
    for tag, cv in cloned_voices.items():
        p = cv.get("prompt_path", "")
        if p and not os.path.exists(p):
            rep = os.path.join(REFS_DIR, os.path.basename(p))
            if os.path.exists(rep):
                logger.info("Remapped clone '%s' prompt_path to %s", tag, rep)
                cv["prompt_path"] = rep
                changed = True
            else:
                logger.warning("Clone '%s': prompt file missing: %s", tag, p)
    if changed:
        _save_clones()


def _save_clones():
    try:
        with open(CLONES_FILE, "w") as f:
            json.dump(cloned_voices, f, indent=2)
    except Exception as e:
        logger.error("Failed to save clones: %s", e)


# ── helpers ───────────────────────────────────────────────────────────
def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_worker(url: str, timeout: int = WORKER_START_TIMEOUT) -> bool:
    async with httpx.AsyncClient() as client:
        for _ in range(timeout):
            try:
                r = await client.get(f"{url}/health", timeout=3)
                if r.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
                pass
            await asyncio.sleep(1)
    return False


async def start_worker() -> str:
    """Spawn the worker subprocess, wait for it to be ready, return its URL."""
    async with _worker_lock:
        return await _start_worker_locked()


async def _start_worker_locked() -> str:
    """Body of :func:`start_worker`.  Caller must hold ``_worker_lock``."""
    global _proc, _worker_url, _log_fh

    if _worker_url:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{_worker_url}/health", timeout=3)
                if r.status_code == 200:
                    return _worker_url
        except Exception:
            pass
        await _stop_worker_locked()

    _load_clones()
    port = _find_free_port()
    worker_script = os.path.join(_BASE_DIR, "providers", "tts_worker.py")
    log_path = os.path.join(_BASE_DIR, "tts_worker.log")

    _log_fh = open(log_path, "a")
    _proc = sp.Popen(
        [sys.executable, "-u", worker_script, "--port", str(port)],
        stdout=_log_fh,
        stderr=sp.STDOUT,
        cwd=_BASE_DIR,
    )

    url = f"http://127.0.0.1:{port}"
    if not await _wait_for_worker(url):
        await _stop_worker_locked()
        raise RuntimeError("TTS worker failed to start within %ds" % WORKER_START_TIMEOUT)

    _worker_url = url
    logger.info("TTS worker ready at %s (pid %d)", url, _proc.pid)
    return url


async def stop_worker():
    """Kill the worker subprocess, releasing all VRAM."""
    async with _worker_lock:
        await _stop_worker_locked()


async def _stop_worker_locked():
    """Body of :func:`stop_worker`.  Caller must hold ``_worker_lock``."""
    global _proc, _worker_url, _log_fh
    if _worker_url:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{_worker_url}/shutdown", timeout=3)
        except Exception:
            pass
        _worker_url = None
    if _proc:
        _proc.kill()
        try:
            _proc.wait(timeout=5)
        except sp.TimeoutExpired:
            _proc.kill()
            _proc.wait()
        _proc = None
    if _log_fh is not None:
        # Otherwise every start/stop cycle leaks a handle on tts_worker.log.
        try:
            _log_fh.close()
        except Exception:
            pass
        _log_fh = None
    gc.collect()


def get_worker_url() -> str:
    if not _worker_url:
        raise RuntimeError("TTS worker not running")
    return _worker_url


def is_running() -> bool:
    return _worker_url is not None and _proc is not None and _proc.poll() is None


# ── synthesis proxy ───────────────────────────────────────────────────
async def _proxy_synthesize(payload: dict) -> tuple[bytes, str, float, str]:
    url = f"{get_worker_url()}/synthesize"
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(url, json=payload)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", str(r.content[:300]))
        except Exception:
            detail = str(r.content[:300])
        raise HTTPException(r.status_code, detail)
    wf = r.headers.get("X-Waveform", "")
    dur_str = r.headers.get("X-Duration-Seconds", "0")
    dur = float(dur_str) if dur_str else 0.0
    mt = r.headers.get("content-type", "audio/ogg")
    return r.content, mt, dur, wf


async def synthesize_audio(
    text: str,
    language: str,
    speaker: str,
    *,
    fmt: str = "opus",
) -> tuple[bytes, str, float, str]:
    tag = speaker.strip().lower()

    _load_clones()  # refresh from disk — cloned_voices.json may be updated externally

    if tag in cloned_voices:
        cv = cloned_voices[tag]
        return await _proxy_synthesize({
            "text": text,
            "language": language,
            "clone_prompt_path": cv.get("prompt_path", ""),
            "fmt": fmt,
        })

    raise HTTPException(400, f"Unknown voice '{speaker}'. Available: {', '.join(get_all_voices())}")


async def synthesize_clone(
    text: str,
    language: str,
    ref_audio: str | None = None,
    ref_text: str = "",
    *,
    prompt_path: str | None = None,
    fmt: str = "opus",
) -> tuple[bytes, str, float, str]:
    if prompt_path:
        return await _proxy_synthesize({
            "text": text,
            "language": language,
            "clone_prompt_path": prompt_path,
            "fmt": fmt,
        })
    if ref_audio:
        return await _proxy_synthesize({
            "text": text,
            "language": language,
            "clone_ref_audio_b64": ref_audio,
            "clone_ref_text": ref_text,
            "fmt": fmt,
        })
    raise HTTPException(400, "One of prompt_path or ref_audio is required for clone synthesis")


async def precompute_clone(tag: str, ref_text: str = "") -> dict:
    """Call the worker to precompute a voice prompt from the saved wav file on disk.

    The caller MUST have already saved ``{REFS_DIR}/{tag}.wav``.
    Returns ``{"prompt_path": str, "mode": str}``.
    """
    url = f"{get_worker_url()}/clone-precompute"
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json={
            "tag": tag,
            "ref_text": ref_text,
        })
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", str(r.content[:300]))
        except Exception:
            detail = str(r.content[:300])
        raise HTTPException(r.status_code, detail)
    return r.json()


def get_all_voices() -> list[str]:
    return sorted(cloned_voices.keys())


# ── lifecycle ─────────────────────────────────────────────────────────
async def load_models():
    """Register TTS as a single model with ModelManager (worker-based)."""
    _load_clones()
    logger.info("Loaded %d saved cloned voices.", len(cloned_voices))

    from model_manager import get_manager
    mgr = get_manager()

    async def _load():
        await start_worker()
        return _worker_url

    async def _unload():
        await stop_worker()

    mgr.register("tts", TTS_SIZE_GB, _load, _unload, health_check=is_running)
    logger.info("TTS model registered (worker-based, ~%d GB)", TTS_SIZE_GB)


async def unload_models():
    await stop_worker()
    from model_manager import get_manager
    await get_manager().unload("tts")
    gc.collect()