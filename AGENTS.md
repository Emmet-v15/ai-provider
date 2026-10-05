# AI Provider — agent guidelines

A VRAM-aware REST API server for running AI models on an RTX 5090 (32 GB). Written in Python with FastAPI.

## Architecture

```
server.py          — FastAPI app, routes, lifespan (entrypoint)
model_manager.py   — VRAM-aware model registry (ModelManager singleton)
ctl/               — `ai-provider` CLI + tray icon (separate uv tool, not imported by the server)
providers/
  tts.py           — Qwen3-TTS worker management + voice synthesis
  tts_worker.py    — TTS inference subprocess
  llm.py           — llama-server chat completions proxy
  embeddings.py    — nomic-embed-text proxy
  stt.py           — faster-whisper large-v3-turbo
  image.py         — SDXL txt2img pipeline
```

## Key Principles

- **No auto-load on startup** — nothing is loaded at startup. Lifespan only *registers*
  every model (including TTS: `tts.load_models()` registers, it does not load). A fresh
  server reports `loaded: false` for all five.
- **VRAM budgeting** — each model registers a size in GB. Loading is refused if the total would exceed 32 GB unless `force=true` is passed.
- **Companion loading** — `llm-chat` and `llm-embed` load/unload as a pair.
- **Auto-load on demand** — TTS synthesis and image generation endpoints auto-load their model if not already running.
- **Load/unload is serialised per model** — `ModelManager` holds one `asyncio.Lock` per
  model name and re-checks `loaded` *inside* it. Concurrent `load` calls therefore run the
  load function exactly once; the other callers wait and get the same handle. This matters
  because the check and the state update are separated by an `await`: without the lock, N
  concurrent requests each saw `loaded == False` and each ran a full load. The providers
  hold their own locks too, so calling `start_worker()` / `provider.start()` directly is
  equally safe.

## Running the server

- Normally runs in the background via the `ai-provider` CLI (`ctl/`, installed with
  `uv tool install --editable .\ctl`): `start`, `stop`, `restart`, `status [--json]`,
  `logs [-f]`, `tray`, `autostart on|off`. `start` launches `.venv\Scripts\python.exe
  server.py` with no console, logging to `logs/server.log`. `stop` kills the server's
  whole process tree, so it works on a hand-started `uv run server.py` too (it finds the
  server by whoever owns the port). Autostart is a per-user `HKCU\...\Run` entry for
  `ai-provider-tray --login`, not a Windows service: the models live under the user's
  profile (HF cache, uv's Python), and it needs no admin rights. The tray's watchdog
  restarts a crashed server only while `want == "running"` in
  `%LOCALAPPDATA%\ai-provider\state.json`; `stop` sets it to `stopped`.
- Never run a second server alongside a live one, even on another port: startup's
  orphan cleanup kills *every* `llama-server.exe` and `tts_worker` on the machine.
- Foreground: `uv run server.py` (binds `0.0.0.0:8765`)
- `uv run` re-execs twice: `uv.exe` → `.venv\Scripts\python.exe` (a `py.exe` launcher
  shim) → the uv-managed CPython that actually owns the socket. So the listening PID is
  *not* the venv path — matters when matching a firewall rule to the binary, and it's why
  a single TTS worker shows up as two `python.exe` processes sharing one `--port`.
- The server runs on the Windows dev machine and is accessed by popcorn4 via Tailscale.
- The service is managed via systemd on the host — do not start/stop manually on the remote side.

### Iterating (Development restarts)

The server spawns subprocesses that hold GPU VRAM (`tts_worker.py`, `llama-server.exe` ×2).
When restarting, always use `taskkill /F /PID <pid>` or Ctrl+C — never close the terminal
window without stopping the server first, as that can orphan the subprocesses.

**On startup**, the server automatically:
1. Kills any orphaned `llama-server.exe` and `tts_worker.py` processes from previous runs.
2. Sets up a Windows Job Object with `KILL_ON_JOB_CLOSE` — if the server process dies
   for any reason (taskkill, crash, Ctrl+C), Windows terminates all child processes.

The model manager also verifies subprocess health via `health_check` callbacks:
if a backend dies externally, it's automatically treated as unloaded on the next
query — stale in-memory state won't persist. All five models register one
(`sdxl` checks `state.pipe`, `stt` checks the Whisper handle), so the manager's view
can't drift from what the endpoints actually read.

Note `health_check` only ever *demotes* a slot from loaded to unloaded — it never
promotes. A backend left running from an earlier process shows as `loaded: false`
until something loads it; the provider-level start functions are idempotent, so that
load adopts the running backend instead of spawning a second one.

**Unload → load within one server lifetime works.** It didn't before 2026-08-07: the
llm and embeddings providers closed their shared `httpx.AsyncClient` in `stop()` and
never rebuilt it, so every readiness probe after an unload raised
`Cannot send a request, as the client has been closed`, was swallowed by the
`except Exception: pass` in the retry loop, and the load failed after the full timeout
while llama-server was in fact running. The client is now recreated on demand.

**If VRAM is still maxed out after a restart**, run this to check for leftover processes:
```powershell
tasklist /FI "IMAGENAME eq llama-server.exe"
tasklist /V /FO CSV | findstr "tts_worker"
```
And kill any with `taskkill /F /PID <pid>`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | GPU telemetry + VRAM per model |
| GET | `/models` | List all models, load status, and `busy` (load/unload in flight) |
| POST | `/models/{name}/load?force=false` | Load a model (safe to call concurrently) |
| POST | `/models/{name}/unload` | Unload a model (idempotent) |
| POST | `/v1/audio/speech` | TTS (OpenAI-compatible) |
| POST | `/v1/audio/transcriptions` | STT (OpenAI-compatible) |
| GET | `/audio/voices` | List all voices/clones |
| POST | `/audio/voices` | Create/update a cloned voice |
| PATCH | `/audio/voices/{tag}` | Update voice metadata |
| DELETE | `/audio/voices/{tag}` | Delete a cloned voice |
| POST | `/v1/chat/completions` | LLM chat |
| POST | `/v1/embeddings` | Text embeddings |
| POST | `/v1/images/generations` | txt2img (auto-loads SDXL) |

Legacy endpoints (`/tts`, `/clone`, `/clone/save`, `/clone/list`, `/clone/delete`, `/voices`) exist for popcorn4 backward compatibility.

## Models & VRAM

| Model | VRAM | Backend |
|-------|------|---------|
| `tts` | 4.5 GB | Qwen3-TTS worker subprocess |
| `llm-chat` | 18.0 / 21.0 GB + KV | llama-server; one of two selectable chat variants |
| `llm-embed` | 0.5 GB | llama-server (nomic-embed-text v1.5 Q8_0) |
| `stt` | 3.0 GB | faster-whisper large-v3-turbo |
| `sdxl` | 12.0 GB | StableDiffusionXLPipeline (ramthrusts) |

`llm-chat` serves one of two **chat variants**, declared in
`providers/llm.py:CHAT_VARIANTS`:

| Key | Weights | VRAM | Context | Vision |
|---|---|---|---|---|
| `qwen3` (default) | `models/Qwen3.8-27B-UD-Q4_K_XL.gguf` | 18.0 GB | 131072 | yes, via `--mmproj` |
| `mistral` | `models/...Dolphin-Mistral-24B-Venice-Edition-Q6_K.gguf` | 21.0 GB | 32768 | no |

Both drive the *same* llama-server on `LLM_CHAT_PORT`, so they are mutually
exclusive by construction — there is one slot, not two, and `GET/POST
/models/llm-chat/variant*` chooses which weights it loads. Switching is refused
while the model is loaded: the running process is the old weights, so the
registered VRAM figure would otherwise describe something that is not resident.
`ModelManager.set_size()` moves the budget with the selection; without it an
18 GB budget would be checked against a 21 GB load.

A variant carries its own context and projector. `qwen3` accepts OpenAI-style
`image_url` content parts; `mistral` is text-only and must not be handed Qwen's
`--mmproj`. KV cache is q8_0 for both (~80 KiB/token at 24-27B), which is why
`mistral` runs at 32k rather than 131k — Q6_K weights plus a 131k cache would
not leave room for the `llm-embed` companion.

Override per variant with `LLM_CHAT_MODEL` / `_CTX` / `_VRAM_GB` / `_MMPROJ` and
`LLM_CHAT_MISTRAL_MODEL` / `_CTX` / `_VRAM_GB`; pick the startup default with
`LLM_CHAT_VARIANT`. llama-server logs go to `llama_chat.log` (not DEVNULL).

## Verification

- `pytest tests/` — run the test suite (requires the server to be running)
- Check `/health` for GPU state and loaded models
- Models are cached via Hugging Face cache (shared with other projects)

## Cloned voices

Cloned voice data lives in `references/` (audios + `.pt` prompts). Metadata is persisted in `cloned_voices.json`. Deleted voices are moved to `references/_deleted/` rather than erased.
