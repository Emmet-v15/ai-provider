"""
LLM provider — wraps llama-server.exe as a subprocess for chat completions.

Registers with ModelManager, launches the server on port 8000, and
proxies OpenAI-compatible /v1/chat/completions requests.
"""

from __future__ import annotations

import os
import asyncio
import subprocess as sp
import time
import logging
import httpx
from dataclasses import dataclass
from urllib.parse import urljoin

logger = logging.getLogger("provider.llm")

# ── config ────────────────────────────────────────────────────────────
_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
CHAT_PORT = int(os.getenv("LLM_CHAT_PORT", "8000"))
CHAT_EXTRA = os.getenv(
    "LLM_CHAT_EXTRA",
    "-fa on --cache-type-k q8_0 --cache-type-v q8_0",
)
LLAMA_SERVER_EXE = os.getenv("LLAMA_SERVER_EXE") or next(
    (p for p in [
        r"C:\llama-cpp\llama-server.exe",
        "llama-server",  # fall back to PATH lookup
    ] if os.path.exists(p) or p == "llama-server"),
    "llama-server",
)


@dataclass(frozen=True)
class ChatVariant:
    """One selectable set of chat weights.

    ``model`` is either a local ``.gguf`` path (passed to llama-server as
    ``-m``) or an HF ``repo[:quant]`` reference (passed as ``-hf``), which
    is the same distinction ``start()`` already made for the single model.

    ``vram_gb`` is the figure ModelManager budgets against and covers the
    weights *plus* the KV cache at ``ctx`` — a 24B at q8_0 KV costs about
    80 KiB per token (2 x 40 layers x 8 KV heads x 128 dim), so context is
    not a rounding error at these sizes.
    """

    key: str
    label: str
    model: str
    vram_gb: float
    ctx: int
    mmproj: str = ""  # vision projector; "" or a missing file = text only

    @property
    def is_local(self) -> bool:
        return self.model.lower().endswith(".gguf")

    @property
    def present(self) -> bool:
        """Whether the weights are usable now.

        An HF reference always counts as present: llama-server downloads it
        on first load. A local path has to actually exist.
        """
        return os.path.isfile(self.model) if self.is_local else bool(self.model)

    @property
    def vision(self) -> bool:
        return bool(self.mmproj) and os.path.isfile(self.mmproj)


# Both variants drive the same llama-server on CHAT_PORT, so they are
# mutually exclusive by construction — selecting one unloads the other.
CHAT_VARIANTS: dict[str, ChatVariant] = {
    "qwen3": ChatVariant(
        key="qwen3",
        label="Qwen3.8-27B UD-Q4_K_XL (vision)",
        model=os.getenv(
            "LLM_CHAT_MODEL", os.path.join(_MODELS_DIR, "Qwen3.8-27B-UD-Q4_K_XL.gguf")
        ),
        vram_gb=float(os.getenv("LLM_CHAT_VRAM_GB", "18")),
        ctx=int(os.getenv("LLM_CHAT_CTX", "131072")),
        mmproj=os.getenv("LLM_CHAT_MMPROJ", os.path.join(_MODELS_DIR, "mmproj-F16.gguf")),
    ),
    "mistral": ChatVariant(
        key="mistral",
        label="Dolphin-Mistral-24B-Venice-Edition Q6_K",
        model=os.getenv(
            "LLM_CHAT_MISTRAL_MODEL",
            os.path.join(
                _MODELS_DIR,
                "cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q6_K.gguf",
            ),
        ),
        # 18.0 GiB of weights + ~2.5 GiB of KV at 32k, leaving room for the
        # llm-embed companion that loads alongside it.
        vram_gb=float(os.getenv("LLM_CHAT_MISTRAL_VRAM_GB", "21")),
        ctx=int(os.getenv("LLM_CHAT_MISTRAL_CTX", "32768")),
        mmproj="",  # Mistral-Small-24B is text-only
    ),
}

DEFAULT_VARIANT = os.getenv("LLM_CHAT_VARIANT", "qwen3")
if DEFAULT_VARIANT not in CHAT_VARIANTS:
    logger.warning(
        "LLM_CHAT_VARIANT=%r is not one of %s — falling back to 'qwen3'",
        DEFAULT_VARIANT, list(CHAT_VARIANTS),
    )
    DEFAULT_VARIANT = "qwen3"

MODEL_NAME = "llm-chat"


class LLMProvider:
    """Manages a llama-server.exe subprocess on the configured port."""

    def __init__(self) -> None:
        self._proc: sp.Popen | None = None
        self._port = CHAT_PORT
        self._variant: ChatVariant = CHAT_VARIANTS[DEFAULT_VARIANT]
        self._client: httpx.AsyncClient | None = None
        # Persistent bounded pool for chat completions — a per-request client
        # leaks server-side sockets when requests are aborted/cancelled.
        self._chat_client: httpx.AsyncClient | None = None
        self._inflight_client: httpx.AsyncClient | None = None
        # Serialises start/stop so two overlapping loads cannot both spawn a
        # llama-server on the same port.
        self._lock = asyncio.Lock()

    @property
    def client(self) -> httpx.AsyncClient:
        """The shared client, (re)created on demand.

        ``stop()`` closes it to release its connections, and a closed
        httpx client cannot be reused — so a later ``start()`` has to get a
        fresh one, otherwise every readiness probe raises and the server is
        reported as failing to start when it is in fact running.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def variant(self) -> ChatVariant:
        """The weights the next ``start()`` will load, or the running ones."""
        return self._variant

    def select(self, key: str) -> ChatVariant:
        """Point the provider at a different variant.

        Refuses while the server is up: the running process *is* the old
        weights, so switching under it would leave ``variant`` describing
        something that is not loaded. Callers unload first.
        """
        try:
            variant = CHAT_VARIANTS[key]
        except KeyError:
            raise KeyError(
                f"Unknown chat variant {key!r} — expected one of {list(CHAT_VARIANTS)}"
            ) from None
        if self.is_running:
            raise RuntimeError(
                f"Cannot switch to {key!r} while {self._variant.key!r} is loaded — "
                f"unload llm-chat first"
            )
        if not variant.present:
            raise FileNotFoundError(
                f"Variant {key!r} has no weights at {variant.model!r}"
            )
        self._variant = variant
        logger.info("chat variant set to %r (%s)", key, variant.label)
        return variant

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    async def start(self) -> None:
        if self.is_running:
            return
        async with self._lock:
            if self.is_running:
                logger.info("LLM already running on port %d", self._port)
                return

            exe = LLAMA_SERVER_EXE
            if os.path.dirname(exe) and not os.path.exists(exe):
                raise FileNotFoundError(f"llama-server.exe not found at {exe}")

            variant = self._variant
            if not variant.present:
                raise FileNotFoundError(
                    f"Chat variant {variant.key!r} has no weights at {variant.model!r}"
                )

            # A variant's model is either a local GGUF file (-m) or an HF
            # repo:quant that llama-server fetches itself (-hf).
            model_args = (
                ["-m", variant.model]
                if os.path.isfile(variant.model)
                else ["-hf", variant.model]
            )
            args = [
                exe,
                *model_args,
                "-ngl", "99",
                "-c", str(variant.ctx),
                "--host", "0.0.0.0",
                "--port", str(self._port),
            ]
            if CHAT_EXTRA:
                args.extend(CHAT_EXTRA.split())
            if variant.vision:
                args.extend(["--mmproj", variant.mmproj])

            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llama_chat.log")
            log_fh = open(log_path, "a")
            logger.info(
                "Starting LLM server [%s]: %s ...",
                variant.key, " ".join(str(a) for a in args[:6]),
            )
            proc = sp.Popen(
                args,
                stdout=log_fh,
                stderr=sp.STDOUT,
            )
            self._proc = proc

            # Wait for the server to be ready
            t0 = time.time()
            while time.time() - t0 < 300:  # 5 min timeout for model download+load
                if proc.poll() is not None:
                    self._proc = None
                    raise RuntimeError(
                        f"LLM server exited during startup (code {proc.returncode}) — "
                        f"port {self._port} may already be in use"
                    )
                try:
                    r = await self.client.get(
                        f"{self.base_url}/v1/models",
                        timeout=5.0,
                    )
                    if r.status_code == 200:
                        logger.info("LLM server ready on port %d (%.1fs)", self._port, time.time() - t0)
                        return
                except Exception:
                    pass
                await _sleep(5)

            # Don't leave an orphan holding VRAM behind on timeout.
            await self._kill_proc()
            raise RuntimeError("LLM server failed to start within 300s")

    async def _kill_proc(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except sp.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None

    async def stop(self) -> None:
        async with self._lock:
            self.cancel_current()
            await self._kill_proc()
            if self._client is not None:
                await self._client.aclose()
                self._client = None
            if self._chat_client is not None:
                await self._chat_client.aclose()
                self._chat_client = None

    def cancel_current(self) -> bool:
        """Cancel the in-flight chat completion request, if any."""
        client = self._inflight_client
        if client is not None:
            self._inflight_client = None
            import asyncio
            asyncio.ensure_future(client.aclose())
            return True
        return False

    async def chat_completions(self, body: dict) -> dict:
        if not self.is_running:
            raise RuntimeError("LLM server not running")
        if self._chat_client is None or self._chat_client.is_closed:
            self._chat_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0),
                limits=httpx.Limits(max_connections=8),
            )
        # Shared so cancel_current() can abort the in-flight request
        client = self._chat_client
        self._inflight_client = client
        try:
            r = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
            )
            r.raise_for_status()
            return r.json()
        finally:
            self._inflight_client = None

    async def models(self) -> dict:
        if not self.is_running:
            raise RuntimeError("LLM server not running")
        r = await self.client.get(f"{self.base_url}/v1/models")
        r.raise_for_status()
        return r.json()


_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = LLMProvider()
    return _provider


async def load_models():
    """Register the LLM provider with ModelManager and launch the server."""
    from model_manager import get_manager
    mgr = get_manager()

    prov = get_provider()

    async def _load():
        await prov.start()
        return prov

    async def _unload():
        await prov.stop()

    mgr.register(
        MODEL_NAME,
        prov.variant.vram_gb,
        _load,
        _unload,
        health_check=lambda: prov.is_running,
    )


def list_variants() -> dict:
    """Every selectable chat variant plus which one is active."""
    prov = get_provider()
    return {
        "active": prov.variant.key,
        "loaded": prov.is_running,
        "variants": [
            {
                "key": v.key,
                "label": v.label,
                "model": v.model,
                "vram_gb": v.vram_gb,
                "ctx": v.ctx,
                "vision": v.vision,
                "present": v.present,
            }
            for v in CHAT_VARIANTS.values()
        ],
    }


def select_variant(key: str) -> dict:
    """Select ``key`` and re-budget the ``llm-chat`` slot to match it.

    The slot's size has to move with the variant or the VRAM budget would
    keep checking loads against whichever model happened to be registered
    at startup — 18 GB for Qwen against 21 GB for Mistral is exactly the
    kind of gap that turns a refused load into an OOM.
    """
    from model_manager import get_manager

    prov = get_provider()
    variant = prov.select(key)
    get_manager().set_size(MODEL_NAME, variant.vram_gb)
    return list_variants()


async def unload_models():
    await get_provider().stop()


async def _sleep(secs: float):
    await __import__("asyncio").sleep(secs)