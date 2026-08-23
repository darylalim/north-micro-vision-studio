"""Regression test: abandoning a stream must not wedge the MLX worker.

`nmv.runtime` funnels every MLX call through one process-wide worker thread.
`iterate` hands its producer a bounded `emit`, so a consumer that walks away
cannot block it — but the sentinel put in the producer's `finally` was
originally a plain `channel.put()`. Whenever a consumer abandoned a *full*
channel, that call blocked forever, and with a single worker it took down model
loading and generation for every browser session until the server restarted.

Everything here is deadlock-prone by nature, so each check runs on its own
thread with a join timeout: a regression fails the run instead of hanging it.

Run:  uv run python tests/test_runtime_abandon.py
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nmv import runtime  # noqa: E402

CHANNEL_LIMIT = 128  # nmv.runtime.iterate's queue maxsize
PROBE_TIMEOUT = 15.0

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}{f' — {detail}' if detail else ''}"
    )
    if not condition:
        failures.append(name)


def worker_responds(timeout: float = PROBE_TIMEOUT) -> bool:
    """True if the shared MLX worker can still accept and finish a task."""
    answered = threading.Event()

    def probe() -> None:
        runtime.call(lambda: None)
        answered.set()

    thread = threading.Thread(target=probe, daemon=True)
    thread.start()
    thread.join(timeout)
    return answered.is_set()


print("abandoning a full channel")


def flood(emit) -> None:
    for i in range(CHANNEL_LIMIT * 20):
        if not emit(i):
            return


gen = runtime.iterate(flood)
next(gen)
time.sleep(1.0)  # let the producer saturate the channel
gen.close()  # what Streamlit does when it interrupts a script
check(
    "worker survives an abandoned full stream",
    worker_responds(),
    f"probe answered within {PROBE_TIMEOUT:.0f}s",
)

print("normal completion")
drained: list = []


def drain() -> None:
    def counter(emit) -> None:
        for i in range(500):
            emit(i)

    drained.extend(runtime.iterate(counter))


t = threading.Thread(target=drain, daemon=True)
t.start()
t.join(PROBE_TIMEOUT)
check(
    "a fully consumed stream yields every item",
    drained == list(range(500)),
    f"{len(drained)} items",
)

print("producer failure")
raised: list = []


def expect_raise() -> None:
    def boom(emit):
        emit("first")
        raise RuntimeError("producer exploded")

    try:
        list(runtime.iterate(boom))
    except RuntimeError as error:
        raised.append(str(error))


t = threading.Thread(target=expect_raise, daemon=True)
t.start()
t.join(PROBE_TIMEOUT)
check(
    "a producer exception reaches the consumer",
    raised == ["producer exploded"],
    str(raised),
)
check("worker still usable after a failure", worker_responds())

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all runtime checks passed")
