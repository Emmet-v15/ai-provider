"""
VRAM-aware model registry for the RTX 5090 (32 GB VRAM).

Tracks which models are loaded, which GPU slot they occupy, and whether
loading another model would exceed the budget.

Each model registers a name, size-estimate (GB), load function, and
unload function.  The manager refuses to load if the total would exceed
MAX_VRAM_GB — caller can override per request via the `force` flag.

Models can optionally supply a ``health_check`` callback (a sync
``Callable[[], bool]``).  When provided, the manager re-verifies the
model's ``loaded`` status every time it is queried — if the subprocess
or backend died externally, it is automatically treated as unloaded.

Concurrency: ``load``/``unload`` are serialised per model by an
``asyncio.Lock``.  Without it, two overlapping requests both observe
``loaded == False`` (the check and the state update are separated by an
``await``) and both run the load function — loading the model twice and
leaking the first copy's VRAM.  Callers that arrive while a load is in
flight wait for it and receive the same handle.
"""

from __future__ import annotations

import os
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger("model_manager")

# ── config ────────────────────────────────────────────────────────────
MAX_VRAM_GB = int(os.getenv("MAX_VRAM_GB", "32"))


@dataclass
class ModelSlot:
    name: str
    size_gb: float
    loaded: bool = False
    loaded_at: float | None = None
    load_fn: Callable[[], Awaitable[Any]] | None = None
    unload_fn: Callable[[], Awaitable[None]] | None = None
    health_check: Callable[[], bool] | None = None
    handle: Any = None  # the loaded model instance / subprocess ref


class ModelManager:
    """Singleton-ish registry — one instance per process tracks all models."""

    def __init__(self, max_vram_gb: int = MAX_VRAM_GB):
        self.max_vram_gb = max_vram_gb
        self._slots: dict[str, ModelSlot] = {}
        # Keyed by model name rather than held on the slot so that
        # re-registering a model cannot drop a lock that is currently held.
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, name: str) -> asyncio.Lock:
        lock = self._locks.get(name)
        if lock is None:
            lock = self._locks[name] = asyncio.Lock()
        return lock

    def is_busy(self, name: str) -> bool:
        """True while a load or unload for ``name`` is in flight."""
        lock = self._locks.get(name)
        return lock is not None and lock.locked()

    # ── registration ──────────────────────────────────────────────────
    def register(
        self,
        name: str,
        size_gb: float,
        load_fn: Callable[[], Awaitable[Any]],
        unload_fn: Callable[[], Awaitable[None]] | None = None,
        *,
        health_check: Callable[[], bool] | None = None,
    ) -> None:
        if name in self._slots:
            logger.warning("overwriting registered model %r", name)
        self._slots[name] = ModelSlot(
            name=name,
            size_gb=size_gb,
            load_fn=load_fn,
            unload_fn=unload_fn,
            health_check=health_check,
        )
        logger.info("registered %r (%.1f GB)", name, size_gb)

    def get(self, name: str) -> ModelSlot | None:
        return self._slots.get(name)

    def set_size(self, name: str, size_gb: float) -> None:
        """Re-budget a registered slot.

        For models whose footprint depends on configuration rather than
        identity — ``llm-chat`` serves whichever chat variant is selected,
        and they differ by several GB. Refused while the model is loaded,
        because the figure would then describe weights other than the ones
        actually occupying VRAM.
        """
        slot = self._slots.get(name)
        if slot is None:
            raise KeyError(f"Unknown model: {name!r}")
        if slot.loaded:
            raise RuntimeError(f"Cannot resize {name!r} while it is loaded")
        if size_gb != slot.size_gb:
            logger.info("%r re-budgeted %.1f GB -> %.1f GB", name, slot.size_gb, size_gb)
            slot.size_gb = size_gb

    # ── health syncing ──────────────────────────────────────────────────
    def _sync_health(self, slot: ModelSlot) -> None:
        """If the slot has a health_check and it returns False, mark as unloaded."""
        if slot.loaded and slot.health_check is not None:
            if not slot.health_check():
                logger.warning(
                    "%r health check failed — marking as unloaded", slot.name
                )
                slot.handle = None
                slot.loaded = False
                slot.loaded_at = None

    @property
    def loaded_gb(self) -> float:
        total = 0.0
        for s in self._slots.values():
            self._sync_health(s)
            if s.loaded:
                total += s.size_gb
        return total

    @property
    def free_gb(self) -> float:
        # rough: torch reports free VRAM; fall back to max - loaded
        try:
            import torch
            free = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
            return free
        except Exception:
            return max(0.0, self.max_vram_gb - self.loaded_gb)

    # ── load / unload ─────────────────────────────────────────────────
    async def load(self, name: str, *, force: bool = False) -> Any:
        slot = self._slots.get(name)
        if not slot:
            raise KeyError(f"Unknown model: {name!r}")

        # Fast path — already loaded and healthy, no need to take the lock.
        self._sync_health(slot)
        if slot.loaded:
            logger.info("%r already loaded", name)
            return slot.handle

        async with self._lock_for(name):
            # Re-check under the lock.  A concurrent request may have
            # completed the load while we were waiting for it; the VRAM
            # budget check below must also see that model's usage.
            self._sync_health(slot)
            if slot.loaded:
                logger.info("%r already loaded (completed by a concurrent request)", name)
                return slot.handle

            needed = slot.size_gb
            current_free = self.free_gb

            if needed > current_free and not force:
                raise RuntimeError(
                    f"Cannot load {name!r} ({needed:.1f} GB): "
                    f"only {current_free:.1f} GB free (loaded: {self.loaded_gb:.1f} GB / "
                    f"{self.max_vram_gb} GB max)"
                )

            logger.info("loading %r (%.1f GB, free: %.1f GB) …", name, needed, current_free)
            t0 = time.perf_counter()
            try:
                handle = await slot.load_fn()
            except Exception:
                # Leave the slot cleanly unloaded so the next caller retries
                # rather than inheriting a half-loaded model.
                slot.handle = None
                slot.loaded = False
                slot.loaded_at = None
                logger.exception("failed to load %r", name)
                raise
            slot.handle = handle
            slot.loaded = True
            slot.loaded_at = time.time()  # wall clock — this is exposed via /health
            logger.info("loaded %r in %.1fs", name, time.perf_counter() - t0)
            return handle

    async def unload(self, name: str) -> None:
        slot = self._slots.get(name)
        if not slot:
            return
        async with self._lock_for(name):
            if not slot.loaded:
                return
            try:
                if slot.unload_fn:
                    await slot.unload_fn()
            except Exception:
                logger.exception("error unloading %r", name)
            slot.handle = None
            slot.loaded = False
            slot.loaded_at = None
            logger.info("unloaded %r", name)

    def is_loaded(self, name: str) -> bool:
        slot = self._slots.get(name)
        if slot:
            self._sync_health(slot)
            return slot.loaded
        return False

    def list_loaded(self) -> list[dict]:
        result = []
        for s in self._slots.values():
            self._sync_health(s)
            if s.loaded:
                result.append({
                    "name": s.name,
                    "size_gb": s.size_gb,
                    "loaded": s.loaded,
                    "loaded_at": s.loaded_at,
                })
        return result

    def list_all(self) -> list[dict]:
        result = []
        for s in self._slots.values():
            self._sync_health(s)
            result.append({
                "name": s.name,
                "size_gb": s.size_gb,
                "loaded": s.loaded,
                # Lets a client distinguish "not loaded" from "load in flight"
                # instead of firing a second load request at it.
                "busy": self.is_busy(s.name),
            })
        return result


# ── global instance ───────────────────────────────────────────────────
_manager: ModelManager | None = None


def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


async def unload_all() -> None:
    mgr = get_manager()
    for name in list(mgr._slots):
        await mgr.unload(name)