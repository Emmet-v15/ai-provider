# ai-provider

VRAM-aware, OpenAI-compatible local inference gateway. Multiplexes five models —
LLM, embeddings, TTS (with voice cloning), STT and image generation — on a single
32 GB GPU, with server-enforced VRAM budgeting, on-demand model load/unload and
health-checked subprocess management.

## Why

One GPU, many models, not enough VRAM for all of them. ai-provider registers each
model with a VRAM budget, refuses loads that would exceed it, auto-loads on demand,
and unloads on request — exposing everything through an OpenAI-compatible API so any
standard client works without knowing what is running underneath.

## Architecture

```
server.py          — FastAPI app, routes, lifespan (entrypoint)
model_manager.py   — VRAM-aware model registry (ModelManager singleton)
providers/
  tts.py           — Qwen3-TTS worker management + voice synthesis
  tts_worker.py    — TTS inference subprocess
  llm.py           — llama-server chat completions proxy
  embeddings.py    — nomic-embed-text proxy
  stt.py           — faster-whisper large-v3-turbo
  image.py         — SDXL txt2img pipeline
```

## Design principles

- **No auto-load on startup** — lifespan only *registers* every model; a fresh server
  reports `loaded: false` for all five.
- **VRAM budgeting** — each model registers a size in GB; loading is refused if the
  total would exceed 32 GB unless `force=true` is passed.
- **Companion loading** — `llm-chat` and `llm-embed` load/unload as a pair.
- **Auto-load on demand** — TTS synthesis and image generation auto-load their model.
- **Serialised load/unload per model** — one `asyncio.Lock` per model name with the
  `loaded` check re-run inside the lock: concurrent load calls run the load exactly
  once and every caller receives the same handle.

See [AGENTS.md](AGENTS.md) for the full engineering notes and [API.md](API.md) for
endpoint documentation.

## Endpoints (OpenAI-compatible)

`/v1/chat/completions` (+ `/cancel`) · `/v1/embeddings` · `/v1/images/generations` ·
`/v1/audio/speech` · `/v1/audio/transcriptions` · `/models` (load/unload) ·
`/health` (GPU telemetry) · voice-clone CRUD under `/audio/voices`

## Models

| Model | Provider | Type |
|---|---|---|
| Dolphin-Mistral-24B (GGUF via llama-server) | `llm.py` | chat completions |
| nomic-embed-text | `embeddings.py` | embeddings |
| Qwen3-TTS-1.7B | `tts.py` | speech + voice cloning |
| whisper large-v3-turbo | `stt.py` | transcriptions |
| SDXL | `image.py` | image generation |

## Running

```pwsh
uv run server.py        # binds 0.0.0.0:8765 — docs at /docs
uv run pytest tests/    # integration tests
```

Hardware reference: RTX 5090 (32 GB). Any CUDA GPU with headroom for the registered
VRAM budgets works; budgets are configured per model in `model_manager.py`.

## Notes

- Windows-specific detail: the server uses Job Objects to prevent orphaned
  subprocesses on shutdown — the listening PID after `uv run` re-execs twice is *not*
  the venv python path, which matters for firewall rules.
- AI-assisted development: architecture, VRAM model and security decisions are human;
  implementation drafted with LLM assistance and reviewed.
