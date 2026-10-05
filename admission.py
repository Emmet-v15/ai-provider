"""
Capacity-based admission control for the inference backends.

Each backend has a fixed number of things it can do at once — llama-server
has ``--parallel`` slots, the TTS worker and the SDXL pipeline run one job at
a time.  A :class:`Gate` admits exactly that many requests to the backend and
holds the rest in a FIFO queue.  When a running request finishes, its slot is
handed *directly* to the oldest waiter, so a slot never sits idle while
someone is queued and nobody has to poll or retry on a timer to notice it
freed up.

What this replaces, and why:

* A fixed in-flight counter that answered 429 past 8 requests.  Clients
  retried on a ~250 ms timer, so whether a request got a slot depended on
  when its retry happened to land rather than on when it arrived.
* Requests beyond the 4 llama-server slots were forwarded anyway and queued
  *inside* llama-server, invisible to us and with nothing tracking whether
  their client was still there.  Abandoned requests ran to completion, kept
  slots pinned, and the ones behind them hit the 300 s read timeout
  (~860 of the 502s in ``server.err.log``).

Now only requests that hold a slot reach the backend, so the upstream read
timeout measures generation and not queueing.  A waiter whose client
disconnects is dropped from the queue, and a running request whose client
disconnects is cancelled, which closes the upstream connection and makes
llama-server stop generating.  Queue length is bounded by *count*
(``max_queue``); past that the caller gets 429 with a ``Retry-After``
derived from measured service times, not a constant.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Any, Awaitable, Callable

from fastapi import HTTPException, Request


class QueueFull(Exception):
    """The gate is at capacity and its queue is full."""

    def __init__(self, gate: "Gate", retry_after: int) -> None:
        self.gate = gate
        self.retry_after = retry_after
        super().__init__(
            f"{gate.name}: {gate.running} running, {gate.queued} queued "
            f"(capacity {gate.capacity}, queue limit {gate.max_queue})"
        )


class Gate:
    """FIFO admission gate with direct slot hand-off.

    ``capacity`` is how many requests may run on the backend at once and can
    be changed at runtime (``resize``) once the backend reports its real slot
    count.  ``max_queue`` bounds how many may wait; ``None`` means unbounded.
    """

    # Weight of the newest sample in the service-time average.
    _EWMA_ALPHA = 0.2

    def __init__(
        self,
        name: str,
        capacity: int,
        *,
        max_queue: int | None = None,
        initial_service_s: float = 10.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.name = name
        self.capacity = capacity
        self.max_queue = max_queue
        self.running = 0
        self._waiters: deque[asyncio.Future[None]] = deque()
        self._service_s = initial_service_s
        self._tasks: set[asyncio.Task[Any]] = set()
        # Counters for /health.
        self.completed = 0
        self.rejected = 0
        self.abandoned = 0

    # ── introspection ─────────────────────────────────────────────────
    @property
    def queued(self) -> int:
        return len(self._waiters)

    def estimated_wait_s(self, position: int | None = None) -> float:
        """How long a request joining at ``position`` should expect to wait.

        Each slot clears one request per mean service time, so the n-th
        request in line (0-based) gets a slot after about
        ``(n // capacity + 1)`` service times if all slots are busy.
        """
        if position is None:
            position = self.queued
        if self.running < self.capacity and position == 0:
            return 0.0
        return (position // self.capacity + 1) * self._service_s

    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "running": self.running,
            "queued": self.queued,
            "max_queue": self.max_queue,
            "avg_service_s": round(self._service_s, 2),
            "est_wait_s": round(self.estimated_wait_s(), 1),
            "completed": self.completed,
            "rejected": self.rejected,
            "abandoned": self.abandoned,
        }

    # ── capacity ──────────────────────────────────────────────────────
    def resize(self, capacity: int) -> None:
        """Change the slot count, admitting waiters if it grew.

        Shrinking never preempts: running requests finish and the surplus
        drains as they release.
        """
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._dispatch()

    def _dispatch(self) -> None:
        """Admit waiters while there is a free slot."""
        while self.running < self.capacity and self._waiters:
            fut = self._waiters.popleft()
            if fut.done():  # cancelled while queued
                continue
            self.running += 1
            fut.set_result(None)

    # ── acquire / release ─────────────────────────────────────────────
    async def acquire(self) -> None:
        if self.running < self.capacity and not self._waiters:
            self.running += 1
            return
        if self.max_queue is not None and len(self._waiters) >= self.max_queue:
            self.rejected += 1
            raise QueueFull(self, self._retry_after())

        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # The slot was handed to us in the same tick we were
                # cancelled — pass it on rather than leaking it.
                self.release()
            else:
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise

    def release(self, service_s: float | None = None) -> None:
        if service_s is not None:
            self._service_s += self._EWMA_ALPHA * (service_s - self._service_s)
            self.completed += 1
        self.running -= 1
        self._dispatch()

    def _retry_after(self) -> int:
        # When a queue position opens: the head of the queue is admitted as
        # soon as any running request finishes.
        return max(1, math.ceil(self._service_s / self.capacity))

    # ── running work ──────────────────────────────────────────────────
    async def run(
        self,
        fn: Callable[[], Awaitable[Any]],
        *,
        request: Request | None = None,
        interruptible: bool = True,
    ) -> Any:
        """Queue for a slot, then run ``fn()`` while holding it.

        With ``request``, the request is abandoned the moment its client
        disconnects: a queued one gives up its place at once.  A running one
        is cancelled if ``interruptible``; otherwise it is left to finish and
        *keeps its slot until it does*.  That is the right call for work that
        cannot actually be stopped — a ``to_thread`` inference call, or a
        backend that keeps computing after its connection closes — because
        releasing the slot early would start the next job on top of it.
        """

        async def _body() -> Any:
            await self.acquire()
            t0 = time.perf_counter()
            work = asyncio.ensure_future(fn())

            def _done(w: asyncio.Future[Any]) -> None:
                # Only successful runs feed the service-time estimate — a
                # cancelled or failed one says nothing about how long a real
                # request takes.  (Reading exception() also marks it
                # retrieved when nobody is left to await the work.)
                ok = not w.cancelled() and w.exception() is None
                self.release(time.perf_counter() - t0 if ok else None)

            work.add_done_callback(_done)
            if interruptible:
                # Cancelling _body cancels `work` too; the slot is freed once
                # it has actually unwound (e.g. the upstream socket closed).
                return await work
            return await asyncio.shield(work)

        task = asyncio.ensure_future(_body())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        watcher = (
            asyncio.ensure_future(_wait_disconnect(request)) if request is not None else None
        )
        try:
            await asyncio.wait(
                {task} if watcher is None else {task, watcher},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not task.done():
                self.abandoned += 1
                raise ClientGone()
            if task.cancelled():
                raise Cancelled(f"{self.name} request was cancelled")
            return task.result()
        finally:
            if watcher is not None:
                watcher.cancel()
            # Covers both a vanished client and this handler itself being
            # cancelled.
            if not task.done():
                task.cancel()

    def cancel_all(self) -> int:
        """Cancel every request running or queued on this gate."""
        n = 0
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
                n += 1
        return n


class ClientGone(Exception):
    """The client disconnected before its request finished."""


class Cancelled(Exception):
    """The request was cancelled server-side (``Gate.cancel_all``)."""


async def _wait_disconnect(request: Request) -> None:
    """Return when the client goes away.

    The body has already been read by the time a handler runs, so the only
    message left for ``receive()`` to deliver is ``http.disconnect`` — this
    waits on that event instead of polling ``is_disconnected()``.
    """
    while True:
        msg = await request.receive()
        if msg["type"] == "http.disconnect":
            return


def http_error(exc: Exception) -> HTTPException:
    """Map an admission outcome to the HTTP error a handler should raise."""
    if isinstance(exc, QueueFull):
        return HTTPException(
            429,
            f"Queue full — {exc}. Retry in ~{exc.retry_after}s.",
            headers={"Retry-After": str(exc.retry_after)},
        )
    if isinstance(exc, ClientGone):
        # Nobody is listening; the code only shows up in the access log.
        return HTTPException(499, "Client closed request")
    if isinstance(exc, Cancelled):
        return HTTPException(409, str(exc))
    raise TypeError(f"not an admission error: {exc!r}")


ADMISSION_ERRORS = (QueueFull, ClientGone, Cancelled)


# ── registry ──────────────────────────────────────────────────────────
_gates: dict[str, Gate] = {}


def gate(name: str, capacity: int = 1, **kw: Any) -> Gate:
    """The process-wide gate for ``name``, created on first use."""
    g = _gates.get(name)
    if g is None:
        g = _gates[name] = Gate(name, capacity, **kw)
    return g


def all_stats() -> dict:
    return {name: g.stats() for name, g in _gates.items()}
