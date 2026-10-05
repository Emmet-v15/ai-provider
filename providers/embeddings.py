"""
Embeddings provider — wraps a second llama-server.exe instance for embedding.

Registers with ModelManager, launches on port 8001, and proxies
OpenAI-compatible /v1/embeddings requests.
"""

from __future__ import annotations

import os
import asyncio
import subprocess as sp
import time
import logging
import httpx

import admission
from providers.llm import fetch_total_slots, wait_ready

logger = logging.getLogger("provider.embeddings")

# ── config ────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0",
)
EMBED_PORT = int(os.getenv("EMBED_PORT", "8001"))
EMBED_CTX = int(os.getenv("EMBED_CTX", "2048"))
LLAMA_SERVER_EXE = os.getenv("LLAMA_SERVER_EXE") or next(
    (p for p in [
        r"C:\llama-cpp\llama-server.exe",
        "llama-server",  # fall back to PATH lookup
    ] if os.path.exists(p) or p == "llama-server"),
    "llama-server",
)

EMBED_VRAM_GB = float(os.getenv("EMBED_VRAM_GB", "0.5"))
EMBED_MAX_QUEUE = int(os.getenv("EMBED_MAX_QUEUE", "64"))


class EmbeddingsProvider:
    """Manages a llama-server.exe subprocess with --embedding flag."""

    def __init__(self) -> None:
        self._proc: sp.Popen | None = None
        self._port = EMBED_PORT
        self._client: httpx.AsyncClient | None = None
        # Serialises start/stop so two overlapping loads cannot both spawn a
        # llama-server on the same port.
        self._lock = asyncio.Lock()
        # Sized from /props once the server is up.
        self.gate = admission.gate(
            "llm-embed", 4, max_queue=EMBED_MAX_QUEUE, initial_service_s=0.2,
        )

    @property
    def client(self) -> httpx.AsyncClient:
        """The shared client, (re)created on demand.

        ``stop()`` closes it, and a closed httpx client cannot be reused —
        so a later ``start()`` has to get a fresh one, otherwise every
        readiness probe raises and the server is reported as failing to
        start when it is in fact running.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._client

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    async def start(self) -> None:
        if self.is_running:
            return
        async with self._lock:
            if self.is_running:
                logger.info("Embeddings server already running on port %d", self._port)
                return

            exe = LLAMA_SERVER_EXE
            if not os.path.exists(exe):
                raise FileNotFoundError(f"llama-server.exe not found at {exe}")

            args = [
                exe,
                "-hf", EMBED_MODEL,
                "--embedding",
                "-ngl", "99",
                "-c", str(EMBED_CTX),
                "--host", "0.0.0.0",
                "--port", str(self._port),
            ]

            logger.info("Starting embeddings server on port %d ...", self._port)
            proc = sp.Popen(
                args,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
            self._proc = proc

            t0 = time.time()
            if await wait_ready(proc, self.client, self.base_url, deadline_s=120):
                logger.info("Embeddings server ready on port %d (%.1fs)", self._port, time.time() - t0)
                slots = await fetch_total_slots(self.client, self.base_url)
                if slots:
                    self.gate.resize(slots)
                return
            if proc.poll() is not None:
                self._proc = None
                raise RuntimeError(
                    f"Embeddings server exited during startup (code {proc.returncode}) — "
                    f"port {self._port} may already be in use"
                )

            # Don't leave an orphan holding VRAM behind on timeout.
            await self._kill_proc()
            raise RuntimeError("Embeddings server failed to start within 120s")

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
            self.gate.cancel_all()
            await self._kill_proc()
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    async def embeddings(self, body: dict, *, request=None) -> dict:
        if not self.is_running:
            raise RuntimeError("Embeddings server not running")

        async def _call() -> dict:
            r = await self.client.post(
                f"{self.base_url}/v1/embeddings",
                json=body,
            )
            r.raise_for_status()
            return r.json()

        return await self.gate.run(_call, request=request)


_provider: EmbeddingsProvider | None = None


def get_provider() -> EmbeddingsProvider:
    global _provider
    if _provider is None:
        _provider = EmbeddingsProvider()
    return _provider


async def load_models():
    from model_manager import get_manager
    mgr = get_manager()
    prov = get_provider()

    async def _load():
        await prov.start()
        return prov

    async def _unload():
        await prov.stop()

    mgr.register("llm-embed", EMBED_VRAM_GB, _load, _unload, health_check=lambda: prov.is_running)


async def unload_models():
    await get_provider().stop()