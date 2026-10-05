# AI Provider API

A VRAM-aware REST API for running AI models on an RTX 5090 (32 GB). Models load on demand and are tracked by the `ModelManager`.

**Base URL:** `http://<host>:8765`
**Docs:** `http://<host>:8765/docs` (Swagger UI) · `/redoc` · `/openapi.json`
**Guides:** `http://<host>:8765/documentation` — this file and the rest of the project docs, served by the API (see [Documentation](#documentation))

## Queueing & rate limits

Every inference endpoint is admitted by **capacity, not by time**. Each backend
runs as many requests as it physically can at once — `llm-chat`/`llm-embed` one
per llama-server slot (read from its `/props` at load, 4 by default), `tts`,
`stt` and `sdxl` one at a time — and further requests wait in a FIFO queue. A
request is dispatched the instant a slot frees, so there is no need to retry on
a timer: send it once and wait for the response.

| Status | Meaning |
|---|---|
| `429` | The queue for that backend is full. `Retry-After` (seconds) is derived from the measured average service time ÷ slots — when the next queue place should open. |
| `409` | The request was cancelled server-side (`POST /v1/chat/completions/cancel`). |
| `499` | The client disconnected first (only visible in the server's access log). |

Disconnecting abandons the request: a queued one gives up its place at once and
never reaches the backend; a running chat/embedding request is aborted in
llama-server and its slot freed. (TTS, STT and SDXL work can't be interrupted
mid-run, so it finishes and keeps its slot until it does.)

Queue limits: `CHAT_MAX_QUEUE` (32), `EMBED_MAX_QUEUE` (64), `TTS_MAX_QUEUE`
(16), `STT_MAX_QUEUE` (16), `IMAGE_MAX_QUEUE` (8). Live figures are under
`queues` in `GET /health`.

---

## Models (VRAM Lifecycle)

Models register a VRAM budget. The API refuses to load if the total would exceed 32 GB unless `force=true` is passed.

| Model | VRAM | Type |
|-------|------|------|
| `tts` | 4.5 GB | Qwen3-TTS worker process |
| `llm-chat` | 18.0 / 21.0 GB | llama-server; whichever chat variant is selected (see below) |
| `llm-embed` | 0.5 GB | llama-server (nomic-embed-text-v1.5) |
| `stt` | 3.0 GB | faster-whisper large-v3-turbo |
| `sdxl` | 12.0 GB | StableDiffusionXLPipeline (ramthrusts) |

**Companion loading:** loading `llm-chat` auto-loads `llm-embed` (and vice versa). Unloading `llm-chat` also unloads `llm-embed`.

**Chat variants:** `llm-chat` can serve either of two models. They share one
llama-server on `LLM_CHAT_PORT`, so exactly one is loadable at a time —
selecting is a choice, not an addition, and the slot's VRAM budget moves with
the selection.

| Key | Model | VRAM | Context | Vision |
|---|---|---|---|---|
| `qwen3` (default) | Qwen3.8-27B UD-Q4_K_XL | 18.0 GB | 131072 | yes |
| `mistral` | Dolphin-Mistral-24B-Venice-Edition Q6_K | 21.0 GB | 32768 | no |

Q6_K (18.0 GiB of weights) is the largest Mistral quant that leaves room for a
usable context on a 32 GB card: KV at `q8_0` costs ~80 KiB/token, so 32k adds
~2.5 GiB. Q8_0 would fit the weights but strand the context.

### `GET /models/llm-chat/variants`
List the chat variants, which one is active, and whether the server is running.

```json
{"active": "qwen3", "loaded": false,
 "variants": [{"key": "qwen3", "label": "...", "vram_gb": 18.0, "ctx": 131072,
               "vision": true, "present": true}]}
```

### `POST /models/llm-chat/variant/{key}`
Select a chat variant. `404` if the key is unknown or its weights are missing,
`409` if `llm-chat` is currently loaded — unload it first, since the running
process *is* the old weights.

### `GET /models`
List all registered models and their load status.

### `POST /models/{name}/load?force=false`
Load a model by name. Returns `400` if insufficient VRAM.

### `POST /models/{name}/unload`
Unload a model, releasing its VRAM.

---

## Health

### `GET /health`
Returns GPU telemetry + VRAM status per model.

```json
{
  "status": "ok",
  "vram": {
    "max_vram_gb": 32,
    "total_gb": 31.8,
    "free_gb": 23.2,
    "loaded_gb": 16.5,
    "per_model": { "tts": { "vram_gb": 4.5, "loaded_at": ... } }
  },
  "gpu": {
    "name": "NVIDIA GeForce RTX 5090",
    "temp_c": 40,
    "gpu_util_pct": 22,
    "mem_util_pct": 3,
    "mem_used_gb": 25.6,
    "mem_total_gb": 31.8,
    "idle": false
  },
  "sdxl_loaded": true,
  "queues": {
    "llm-chat": {
      "capacity": 4, "running": 4, "queued": 2, "max_queue": 32,
      "avg_service_s": 6.1, "est_wait_s": 6.1,
      "completed": 1520, "rejected": 0, "abandoned": 3
    }
  }
}
```

`queues` has one entry per backend that has served a request: slots
(`capacity`), what is `running` and `queued` now, the moving-average service
time, the wait a new request should expect, and lifetime counters.

---

## Audio (TTS)

### `POST /v1/audio/speech`
OpenAI-compatible TTS. Body:

```json
{
  "model": "qwen3-tts",
  "input": "Hello world",
  "voice": "axel",
  "language": "English",
  "response_format": "opus"
}
```

Returns `audio/opus` (or `audio/wav`). Headers: `X-Waveform`, `X-Duration-Seconds`.

**Voices:** any cloned voice tag created via `POST /audio/voices`.

---

## Audio (STT)

### `POST /v1/audio/transcriptions`
OpenAI-compatible speech-to-text. Multipart form with `file`, `model`, `language`.

---

## Voices (Clone Management)

### `GET /audio/voices`
List all voices with clone metadata.

### `POST /audio/voices`
Create or update a cloned voice. Multipart form fields:

| Field | Required | Description |
|-------|----------|-------------|
| `tag` | yes | Voice name (lowercase, no spaces) |
| `ref_audio` | for create | Audio file for cloning |
| `ref_text` | no | Optional transcript (HCL mode) |
| `new_tag` | no | Rename destination |
| `clone_of` | no | Discord user ID |
| `cloned_from_msg` | no | Discord message ID |

### `PATCH /audio/voices/{tag}`
Update metadata (rename, transcript, attribution). No audio re-upload needed.

### `DELETE /audio/voices/{tag}`
Delete a cloned voice and its prompt files.

---

## Chat (LLM)

### `POST /v1/chat/completions`
OpenAI-compatible chat. Body:

```json
{
  "model": "dolphin-mistral-24b",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "max_tokens": 2048
}
```

Requires `llm-chat` model to be loaded first.

### `POST /v1/chat/completions/cancel`
Cancel every chat request currently running or queued, freeing all slots for new
requests. Returns `{"cancelled": true, "count": 3}`. Running generations are
stopped by closing their connection to llama-server; each cancelled request
answers `409`. Use this to interrupt long responses and send a new request with
different context.

Not needed just to give up on one request: disconnecting does the same for that
request alone.

---

## Embeddings

### `POST /v1/embeddings`
OpenAI-compatible embeddings.

```json
{
  "model": "nomic-embed-text-v1.5",
  "input": "text to embed"
}
```

Requires `llm-embed` model to be loaded first.

---

## Image Generation

### `POST /v1/images/generations`
OpenAI-compatible txt2img.

```json
{
  "prompt": "a cat",
  "size": "1024x1024",
  "negative_prompt": "",
  "num_inference_steps": 25,
  "guidance_scale": 7.0
}
```

Returns base64-encoded PNG in `data[0].b64_json`. Auto-loads `sdxl` if not already loaded.

---

## Documentation

All of the project's documentation is served by the API itself, read from the
repo's markdown files on every request — so it is always the running code's docs.

### `GET /documentation`
Index of documents (`GET /` is the same).

```json
{
  "documents": [
    {"name": "readme", "title": "Overview", "file": "README.md", "url": "/documentation/readme", "summary": "..."},
    {"name": "api", "title": "API reference", "file": "API.md", "url": "/documentation/api", "summary": "..."},
    {"name": "agents", "title": "Engineering notes", "file": "AGENTS.md", "url": "/documentation/agents", "summary": "..."},
    {"name": "skill", "title": "Agent skill", "file": "SKILL.md", "url": "/documentation/skill", "summary": "..."}
  ],
  "openapi": {"swagger_ui": "/docs", "redoc": "/redoc", "json": "/openapi.json"}
}
```

### `GET /documentation/{name}?format=md|html`
One document, by name (`readme`, `api`, `agents`, `skill`) or filename (`API.md`, any case).
Returns `text/markdown` by default and rendered HTML to browsers (anything sending
`Accept: text/html`); `?format=` overrides. Unknown names → `404` listing the valid ones.

```bash
curl http://<host>:8765/documentation/api          # this reference, as markdown
```

### `GET /SKILL.md`
An [Agent Skills](https://agentskills.io)-format `SKILL.md` that teaches an AI agent how
to use this API: what to check first, load rules, queueing etiquette, a call for every
modality and an error table. Always raw markdown, so it can be read directly or
saved into a skills directory:

```bash
mkdir -p ~/.claude/skills/ai-provider
curl -s http://emperor.empirenet:8765/SKILL.md -o ~/.claude/skills/ai-provider/SKILL.md
```

The OpenAPI schema (`/docs`, `/redoc`) carries the queueing section above in its
description and documents the `429`/`409` responses on every queued endpoint.

---

## Legacy (Deprecated)

These endpoints exist for popcorn4 bot compatibility. Use the modern equivalents instead.

| Legacy | Modern |
|--------|--------|
| `POST /tts` | `POST /v1/audio/speech` |
| `POST /clone` | `POST /v1/audio/speech` with clone voice |
| `POST /clone/save` | `POST /audio/voices` |
| `POST /clone/list` | `GET /audio/voices` |
| `POST /clone/delete` | `DELETE /audio/voices/{tag}` |
| `POST /voices` | `GET /audio/voices` |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_PROVIDER_HOST` | `0.0.0.0` | Bind address |
| `AI_PROVIDER_PORT` | `8765` | Port |
| `MAX_VRAM_GB` | `32` | VRAM budget |
| `LLM_CHAT_VARIANT` | `qwen3` | Chat variant selected at startup |
| `LLM_CHAT_MODEL` | `models/Qwen3.8-27B-UD-Q4_K_XL.gguf` | `qwen3` weights (.gguf path or HF ref) |
| `LLM_CHAT_CTX` | `131072` | `qwen3` context length |
| `LLM_CHAT_VRAM_GB` | `18` | `qwen3` VRAM budget |
| `LLM_CHAT_MMPROJ` | `models/mmproj-F16.gguf` | `qwen3` vision projector |
| `LLM_CHAT_MISTRAL_MODEL` | `models/...Dolphin-Mistral-24B-Venice-Edition-Q6_K.gguf` | `mistral` weights |
| `LLM_CHAT_MISTRAL_CTX` | `32768` | `mistral` context length |
| `LLM_CHAT_MISTRAL_VRAM_GB` | `21` | `mistral` VRAM budget |
| `LLM_CHAT_PORT` | `8000` | llama-server chat port (shared by both variants) |
| `EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0` | HF model ref |
| `EMBED_PORT` | `8001` | llama-server embed port |
| `STT_MODEL_SIZE` | `large-v3-turbo` | Whisper model size |
| `QWEN_TTS_FFMPEG` | auto-detected | FFmpeg path |