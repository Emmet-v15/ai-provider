---
name: ai-provider
description: Use the local GPU inference server on emperor.empirenet (RTX 5090) for LLM chat and vision, embeddings, text-to-speech with cloned voices, speech-to-text and SDXL image generation through an OpenAI-compatible HTTP API. Use whenever a task needs local AI inference, a cloned voice, a transcription, an image, or to check, load or unload models on that GPU.
---

# AI Provider (emperor.empirenet)

An OpenAI-compatible gateway in front of five models that share one 32 GB GPU.

**Base URL:** `http://emperor.empirenet:8765`. Use the MagicDNS name over Tailscale, not
the raw IP. No auth.

This skill is served live by the API at `/SKILL.md`. The full reference is at
`http://emperor.empirenet:8765/documentation/api`, and the OpenAPI schema is at `/openapi.json`.

## 1. Check state before you start

```bash
curl -s http://emperor.empirenet:8765/models    # loaded / busy per model
curl -s http://emperor.empirenet:8765/health    # free VRAM, GPU temp, queue depths
```

| Model | VRAM | Loads | Serves |
|---|---|---|---|
| `llm-chat` | 18 GB (`qwen3`) / 21 GB (`mistral`) | **manually** | `/v1/chat/completions` |
| `llm-embed` | 0.5 GB | with `llm-chat` | `/v1/embeddings` |
| `tts` | 4.5 GB | automatically on first use | `/v1/audio/speech` |
| `stt` | 3 GB | **manually** | `/v1/audio/transcriptions` |
| `sdxl` | 12 GB | automatically on first use | `/v1/images/generations` |

- **Loading:** `POST /models/{name}/load` loads a model and is safe to call twice. `POST /models/{name}/unload` frees it.
  - `llm-chat` and `llm-embed` always load and unload together.
- **VRAM limit:** a load that would push the total past 32 GB is refused with `400` and the free/needed figures. Unload something first.
  - `llm-chat` and `sdxl` don't fit together. Don't pass `force=true` unless the user asks for it.
  - These are other people's models too. Unload only what you loaded, and only when you need the room.
- **Not loaded:** a request to a manually loaded model that isn't up returns `503`. Load it and send the request again. A `busy: true` model is mid-load, so wait for that load instead of starting another.

## 2. Queueing: send once, then wait

Each backend runs as many requests as it can at once: 4 chat slots, and 1 job each for TTS, STT and SDXL. Requests beyond that queue first-come-first-served and start the moment a slot frees.

- **Don't add a short client timeout or retry loop.** A request that's waiting is in line, not stuck.
  - A completion can take tens of seconds and an image about 30 s, so use a timeout of at least 300 s.
- **`429` means the queue is full.** Wait the number of seconds in the `Retry-After` header, then send it once more.
- **Disconnecting cancels.** Closing the connection drops a queued request, and stops a running chat or embedding generation.
- **`409` means someone called `POST /v1/chat/completions/cancel`**, which cancels every running and queued chat request.
- **`GET /health` → `queues`** shows `running`, `queued` and `est_wait_s` for each backend.

## 3. Calls

### Chat (and vision)

```bash
curl -s http://emperor.empirenet:8765/models/llm-chat/load -X POST
curl -s http://emperor.empirenet:8765/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

- **Streaming:** keep `"stream": false`, because streaming is not supported. The reply is one JSON body, and `choices[0].message.content` is the answer.
  - `qwen3` also returns `reasoning_content`.
- **Token cap:** the server caps `max_tokens` at 1024, so ask for what you need.
- **Vision (`qwen3` only):** use content parts.
  - `{"type":"text","text":"..."}`
  - `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`
- **Variants:**
  - `GET /models/llm-chat/variants` shows the active variant: `qwen3` is 131k context with vision, `mistral` is 32k and text only.
  - Switching takes three calls: `POST /models/llm-chat/unload`, `POST /models/llm-chat/variant/{key}`, then load again.
  - Switching is refused with `409` while the model is loaded.

### Embeddings

```bash
curl -s http://emperor.empirenet:8765/v1/embeddings \
  -H 'content-type: application/json' -d '{"input":["first text","second text"]}'
```

The vectors are in `data[i].embedding` (nomic-embed-text-v1.5). It needs `llm-embed` loaded. Loading either `llm-embed` or `llm-chat` brings up both, about 18.5 GB in total.

### Text-to-speech

```bash
curl -s http://emperor.empirenet:8765/audio/voices          # list voice tags
curl -s http://emperor.empirenet:8765/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"Hello there.","voice":"<tag>","response_format":"opus"}' -o out.ogg
```

- **Formats:** `opus` returns ogg and `wav` returns wav. An unknown voice returns `400` with the list of valid ones.
- **Response headers:** `X-Duration-Seconds` and `X-Waveform`.
- **New voice:** `POST /audio/voices`, a multipart form with `tag`, a `ref_audio` file and an optional `ref_text` transcript.
- **Changing voices:** `PATCH /audio/voices/{tag}` renames a voice or updates its metadata. `DELETE /audio/voices/{tag}` deletes it.

### Speech-to-text

```bash
curl -s http://emperor.empirenet:8765/models/stt/load -X POST
curl -s http://emperor.empirenet:8765/v1/audio/transcriptions -F file=@clip.wav -F language=en
```

The response is `{"text": "..."}`. `response_format` takes `json` (the default), `text` or `verbose_json`.

### Images (SDXL)

```bash
curl -s http://emperor.empirenet:8765/v1/images/generations \
  -H 'content-type: application/json' \
  -d '{"prompt":"a lighthouse at dusk","size":"1024x1024","num_inference_steps":25}'
```

- **Output:** base64 PNG in `data[0].b64_json`.
- **Options:** `negative_prompt`, `guidance_scale` (default 7) and `seed`.
- **`n`:** use `n=1`. Higher `n` repeats the same image.
- **VRAM:** the first call loads 12 GB. That fails with `400` if `llm-chat` is loaded.

## Errors at a glance

| Code | Meaning | Do |
|---|---|---|
| 400 | Bad input, or not enough VRAM to load | Read `detail` and fix the input or unload something |
| 404 | Unknown model, voice tag or variant | Use the names from `/models` or `/audio/voices` |
| 409 | Cancelled, or variant switch while loaded | Don't retry blindly |
| 429 | Queue full | Wait `Retry-After` seconds, then send once |
| 502 | Backend error (e.g. context too long) | Read `detail`; a retry won't change the result |
| 503 | Model not loaded / failed to load | Load it, or check `/health` |
