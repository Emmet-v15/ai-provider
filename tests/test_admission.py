"""
Unit tests for admission.Gate — no server, no GPU.

The properties that matter: never more than ``capacity`` running, FIFO
order, a freed slot goes straight to the next waiter, a full queue refuses
with a usable Retry-After, and requests whose client leaves stop costing
anything (except work that physically cannot be stopped, which must keep
its slot until it ends).
"""

import asyncio

import pytest

from admission import Cancelled, ClientGone, Gate, QueueFull


def run(coro):
    return asyncio.run(coro)


class FakeRequest:
    """Stands in for starlette's Request: receive() yields a disconnect on cue."""

    def __init__(self) -> None:
        self.gone = asyncio.Event()

    async def receive(self) -> dict:
        await self.gone.wait()
        return {"type": "http.disconnect"}


async def _tick(n: int = 3) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


def test_never_exceeds_capacity_and_serves_in_arrival_order():
    async def main():
        g = Gate("t", 2)
        running = peak = 0
        started: list[int] = []
        release = asyncio.Event()

        async def job(i):
            nonlocal running, peak
            started.append(i)
            running += 1
            peak = max(peak, running)
            await release.wait()
            running -= 1
            return i

        tasks = []
        for i in range(6):
            tasks.append(asyncio.ensure_future(g.run(lambda i=i: job(i))))
            await _tick()
        assert started == [0, 1]
        assert g.queued == 4
        release.set()
        assert await asyncio.gather(*tasks) == list(range(6))
        assert peak == 2
        assert started == list(range(6))
        assert g.running == 0 and g.queued == 0

    run(main())


def test_a_freed_slot_goes_to_the_next_waiter_without_any_delay():
    async def main():
        g = Gate("t", 1)
        first_done = asyncio.Event()
        second_started = asyncio.Event()

        async def first():
            await first_done.wait()

        async def second():
            second_started.set()

        t1 = asyncio.ensure_future(g.run(first))
        t2 = asyncio.ensure_future(g.run(second))
        await _tick()
        assert not second_started.is_set()
        first_done.set()
        # No sleep, no poll interval: a handful of loop iterations at most.
        await _tick(6)
        assert second_started.is_set()
        await asyncio.gather(t1, t2)

    run(main())


def test_full_queue_is_refused_with_a_measured_retry_after():
    async def main():
        g = Gate("t", 2, max_queue=1, initial_service_s=9.0)
        hold = asyncio.Event()
        tasks = [asyncio.ensure_future(g.run(hold.wait)) for _ in range(3)]
        await _tick()
        assert (g.running, g.queued) == (2, 1)
        with pytest.raises(QueueFull) as e:
            await g.run(hold.wait)
        # 9 s per request across 2 slots -> one frees about every 4.5 s.
        assert e.value.retry_after == 5
        assert g.rejected == 1
        hold.set()
        await asyncio.gather(*tasks)

    run(main())


def test_a_waiter_whose_client_disconnects_leaves_the_queue():
    async def main():
        g = Gate("t", 1)
        hold = asyncio.Event()
        ran = []
        t1 = asyncio.ensure_future(g.run(hold.wait))
        req = FakeRequest()

        async def never():
            ran.append(True)

        t2 = asyncio.ensure_future(g.run(never, request=req))
        await _tick()
        assert g.queued == 1
        req.gone.set()
        with pytest.raises(ClientGone):
            await t2
        assert g.queued == 0
        assert g.abandoned == 1
        hold.set()
        await t1
        assert ran == [], "an abandoned request must never reach the backend"
        assert g.running == 0

    run(main())


def test_a_running_request_whose_client_disconnects_is_cancelled_and_frees_its_slot():
    async def main():
        g = Gate("t", 1)
        cancelled = asyncio.Event()

        async def generate():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        req = FakeRequest()
        t = asyncio.ensure_future(g.run(generate, request=req))
        await _tick()
        assert g.running == 1
        req.gone.set()
        with pytest.raises(ClientGone):
            await t
        await _tick()
        assert cancelled.is_set()
        assert g.running == 0

    run(main())


def test_uninterruptible_work_keeps_its_slot_until_it_really_ends():
    async def main():
        g = Gate("t", 1)
        finish = asyncio.Event()
        second_started = asyncio.Event()

        async def thread_like():
            # Ignores cancellation, like a to_thread() call would.
            await asyncio.shield(finish.wait())

        async def second():
            second_started.set()

        req = FakeRequest()
        t1 = asyncio.ensure_future(g.run(thread_like, request=req, interruptible=False))
        await _tick()
        t2 = asyncio.ensure_future(g.run(second))
        await _tick()
        req.gone.set()
        with pytest.raises(ClientGone):
            await t1
        await _tick()
        assert g.running == 1, "slot must stay held while the work is still on the GPU"
        assert not second_started.is_set()
        finish.set()
        await t2
        assert second_started.is_set()

    run(main())


def test_cancel_all_aborts_running_and_queued():
    async def main():
        g = Gate("t", 1)
        forever = asyncio.Event()
        t1 = asyncio.ensure_future(g.run(forever.wait))
        t2 = asyncio.ensure_future(g.run(forever.wait))
        await _tick()
        assert g.cancel_all() == 2
        for t in (t1, t2):
            with pytest.raises(Cancelled):
                await t
        await _tick()
        assert (g.running, g.queued) == (0, 0)

    run(main())


def test_growing_capacity_admits_waiters_immediately():
    async def main():
        g = Gate("t", 1)
        hold = asyncio.Event()
        tasks = [asyncio.ensure_future(g.run(hold.wait)) for _ in range(4)]
        await _tick()
        assert (g.running, g.queued) == (1, 3)
        g.resize(4)
        assert (g.running, g.queued) == (4, 0)
        hold.set()
        await asyncio.gather(*tasks)

    run(main())


def test_failures_release_the_slot_and_do_not_skew_the_estimate():
    async def main():
        g = Gate("t", 1, initial_service_s=5.0)

        async def boom():
            raise ValueError("backend said no")

        with pytest.raises(ValueError):
            await g.run(boom)
        await _tick()
        assert g.running == 0
        assert g.stats()["avg_service_s"] == 5.0
        assert g.completed == 0

    run(main())
