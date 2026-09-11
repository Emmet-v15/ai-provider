"""
TTS worker subprocess — runs in a dedicated process with its own CUDA context.
Killing this process releases all VRAM instantly.

Protocol: HTTP JSON on a local port.  The main process spawns this worker,
waits for /health, then proxies synthesis requests here.
"""

from __future__ import annotations

import os
import sys
import json
import base64
import gc
import io
import logging
import subprocess as sp
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("tts-worker")

# ── config ────────────────────────────────────────────────────────────────
MODEL_BASE   = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = int(os.getenv("QWEN_TTS_MAX_TOKENS", "512"))

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLONES_FILE = os.path.join(_BASE_DIR, "cloned_voices.json")
REFS_DIR = os.path.join(_BASE_DIR, "references")

# Preset voice aliases (removed — all voices must be saved clones)
VOICE_ALIASES: dict[str, str] = {}

# FFmpeg for opus encoding
FFMPEG = os.getenv("QWEN_TTS_FFMPEG") or next(
    (p for p in [
        r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
        r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\yt-dlp.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-N-124279-g0f6ba39122-win64-gpl\bin\ffmpeg.exe",
        "ffmpeg",
    ] if os.path.exists(p) or p == "ffmpeg"),
    "ffmpeg",
)


# ── state ─────────────────────────────────────────────────────────────────
class WorkerState:
    base: Qwen3TTSModel | None = None


state = WorkerState()
app = FastAPI(title="tts-worker")


# ── helpers ───────────────────────────────────────────────────────────────
def compute_waveform(wav: np.ndarray, sr: int) -> str:
    samples = len(wav)
    if samples == 0:
        return base64.b64encode(bytes([0])).decode()
    duration = samples / sr
    bins = min(256, max(32, int(duration * 10)))
    per = max(1, samples // bins)
    pcm = np.clip(np.round(wav * 32768), -32768, 32767).astype(np.int16)
    out = bytearray(bins)
    for b in range(bins):
        start = b * per
        seg = pcm[start: start + per]
        peak = int(np.max(np.abs(seg.astype(np.int32)))) if len(seg) else 0
        out[b] = min(255, round((peak / 32768) * 255))
    return base64.b64encode(bytes(out)).decode()


def compute_duration(wav: np.ndarray, sr: int) -> float:
    return len(wav) / sr if len(wav) else 0.0


def encode_opus(wav: bytes, sample_rate: int) -> bytes:
    proc = sp.run(
        [FFMPEG, "-y",
         "-i", "pipe:0",
         "-ar", "48000",
         "-c:a", "libopus",
         "-b:a", "24k",
         "-application", "lowdelay",
         "-vbr", "on",
         "-compression_level", "0",
         "-f", "ogg",
         "pipe:1"],
        input=wav,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg opus encode failed: {proc.stderr.decode(errors='replace')[:200]}")
    return proc.stdout


# ── synthesis ─────────────────────────────────────────────────────────────
def resolve_prompt_path(prompt_path: str) -> str:
    """Accept stale absolute paths (project dir moved) by falling back to
    the same filename in this worker's REFS_DIR."""
    if os.path.exists(prompt_path):
        return prompt_path
    rep = os.path.join(REFS_DIR, os.path.basename(prompt_path))
    if os.path.exists(rep):
        return rep
    raise HTTPException(404, f"Voice prompt file not found: {prompt_path}")


def synthesize_clone_from_prompt(
    text: str, language: str,
    prompt_path: str, fmt: str = "opus",
):
    if state.base is None:
        raise RuntimeError("Base model not loaded")
    prompt_data = torch.load(resolve_prompt_path(prompt_path), map_location="cpu", weights_only=True)
    prompt_item = VoiceClonePromptItem(
        ref_code=prompt_data["ref_code"],
        ref_spk_embedding=prompt_data["ref_spk_embedding"],
        x_vector_only_mode=prompt_data.get("x_vector_only_mode", not bool(prompt_data.get("ref_text"))),
        icl_mode=prompt_data.get("icl_mode", False),
        ref_text=prompt_data.get("ref_text", ""),
    )
    wavs, sr = state.base.generate_voice_clone(
        text=text, language=language,
        voice_clone_prompt=[prompt_item],
        non_streaming_mode=True,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    wavs[0] = np.concatenate([np.zeros(int(sr * 0.1)), wavs[0]])
    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    raw_wav = buf.getvalue()
    wf = compute_waveform(wavs[0], sr)
    dur = compute_duration(wavs[0], sr)
    if fmt == "opus":
        data = encode_opus(raw_wav, sr)
        return data, "audio/ogg", dur, wf
    return raw_wav, "audio/wav", dur, wf


def synthesize_clone_one_shot(
    text: str, language: str,
    ref_audio_b64: str, ref_text: str = "",
    fmt: str = "opus",
):
    if state.base is None:
        raise RuntimeError("Base model not loaded")
    import tempfile
    ref_audio = base64.b64decode(ref_audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(ref_audio)
        tmp_path = tmp.name
    try:
        xvec = not bool(ref_text)
        prompt_items = state.base.create_voice_clone_prompt(
            ref_audio=tmp_path,
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=xvec,
        )
        wavs, sr = state.base.generate_voice_clone(
            text=text, language=language,
            voice_clone_prompt=prompt_items,
            non_streaming_mode=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    finally:
        os.unlink(tmp_path)
    wavs[0] = np.concatenate([np.zeros(int(sr * 0.1)), wavs[0]])
    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    raw_wav = buf.getvalue()
    wf = compute_waveform(wavs[0], sr)
    dur = compute_duration(wavs[0], sr)
    if fmt == "opus":
        data = encode_opus(raw_wav, sr)
        return data, "audio/ogg", dur, wf
    return raw_wav, "audio/wav", dur, wf


# ── models ────────────────────────────────────────────────────────────────
def load_models():
    logger.info("Loading Base model: %s ...", MODEL_BASE)
    m = Qwen3TTSModel.from_pretrained(
        MODEL_BASE,
        device_map="cuda:0",
        dtype=DTYPE,
        attn_implementation="sdpa",
    )
    torch.set_float32_matmul_precision("high")
    if hasattr(torch, "compile"):
        try:
            m = torch.compile(m, mode="default")
            logger.info("Base model compiled with torch.compile")
        except Exception as e:
            logger.warning("torch.compile skipped: %s", e)
    state.base = m
    logger.info("TTS Base model loaded in worker")


# ── HTTP endpoints ────────────────────────────────────────────────────────

class SynthRequest(BaseModel):
    text: str
    language: str = "English"
    voice: str = ""                     # preset name (e.g. "alex")
    clone_prompt_path: str = ""         # saved clone .pt path
    clone_ref_audio_b64: str = ""       # one-shot clone audio
    clone_ref_text: str = ""
    fmt: str = "opus"


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    try:
        if req.clone_prompt_path:
            data, mt, dur, wf = synthesize_clone_from_prompt(
                req.text, req.language, req.clone_prompt_path, fmt=req.fmt,
            )
        elif req.clone_ref_audio_b64:
            data, mt, dur, wf = synthesize_clone_one_shot(
                req.text, req.language, req.clone_ref_audio_b64, req.clone_ref_text, fmt=req.fmt,
            )
        elif req.voice:
            # CustomVoice removed — all voices must be saved clones now
            raise HTTPException(400, f"Preset voices are no longer available. Use a cloned voice (saved via the clone endpoint).")
        else:
            raise HTTPException(400, "one of voice, clone_prompt_path, or clone_ref_audio_b64 is required")

        from fastapi.responses import Response
        return Response(
            content=data,
            media_type=mt,
            headers={"X-Waveform": wf, "X-Duration-Seconds": str(dur)},
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))


class CloneSaveRequest(BaseModel):
    tag: str
    ref_audio_b64: str = ""
    ref_text: str = ""


@app.post("/clone-precompute")
async def clone_precompute(req: CloneSaveRequest):
    """Precompute voice prompt from audio and save to disk. Returns the .pt path.

    If *ref_audio_b64* is empty, reads ``{REFS_DIR}/{tag}.wav`` from disk
    (saved there by the parent process).  Otherwise decodes the base64 audio
    and writes it to disk first.
    """
    if state.base is None:
        raise HTTPException(503, "Base model not loaded")
    os.makedirs(REFS_DIR, exist_ok=True)
    wav_path = os.path.join(REFS_DIR, f"{req.tag}.wav")

    if req.ref_audio_b64:
        audio_bytes = base64.b64decode(req.ref_audio_b64)
        proc = sp.run(
            [FFMPEG, "-y", "-i", "pipe:0", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
            input=audio_bytes, capture_output=True, timeout=30,
        )
        if proc.returncode != 0:
            raise HTTPException(400, f"FFmpeg conversion failed: {proc.stderr.decode(errors='replace')[:200]}")
    elif not os.path.exists(wav_path):
        raise HTTPException(400, f"No audio provided and no existing wav found for '{req.tag}'")
    xvec = not bool(req.ref_text)
    prompt_items = state.base.create_voice_clone_prompt(
        ref_audio=wav_path,
        ref_text=req.ref_text if req.ref_text else None,
        x_vector_only_mode=xvec,
    )
    pt_path = os.path.join(REFS_DIR, f"{req.tag}.pt")
    prompt_dict = {
        "ref_code": prompt_items[0].ref_code,
        "ref_spk_embedding": prompt_items[0].ref_spk_embedding,
        "x_vector_only_mode": prompt_items[0].x_vector_only_mode,
        "icl_mode": prompt_items[0].icl_mode,
        "ref_text": prompt_items[0].ref_text,
    }
    torch.save(prompt_dict, pt_path)
    mode = "ICL" if req.ref_text else "x-vector"
    return {"prompt_path": pt_path, "mode": mode}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "base_loaded": state.base is not None,
    }


@app.post("/shutdown")
async def shutdown():
    logger.info("Worker shutting down on request")
    if state.base is not None:
        state.base.cpu()
        del state.base
        state.base = None
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    import os
    os._exit(0)


# ── entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    port = args.port
    load_models()
    logger.info("Worker ready on port %d", port)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info", access_log=False)