"""Every MLX call runs on one dedicated, long-lived thread.

MLX binds each thread's default stream to that thread, and an array that has
not been evaluated yet is work queued on one of those streams: only that
thread can evaluate it. Anywhere else raises

    RuntimeError: There is no Stream(gpu, 1) in current thread.

mlx-vlm's ``load()`` leaves arrays like that behind. It evaluates
``model.parameters()``, which skips underscore-named attributes, so the text
model's shared ``rotary_embeddings["sliding_attention"]._inv_freq`` stays
pending until the first forward pass reads it. Streamlit starts a fresh
ScriptRunner thread for each rerun the browser requests, and
``st.cache_resource`` loads on whichever rerun asks first, so page code that
drives mlx-vlm directly fails on its first reply. Running the import, the load
and every later call on a single worker means nothing pending ever changes
threads.

The arrangement pays for itself twice over: MLX generation is not reentrant,
so funnelling every session through one worker also stops two browser tabs
from trampling each other's decode.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")

logger = logging.getLogger("nmv.runtime")

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-worker")
_BOOT_LOCK = threading.Lock()
_booted = False

# How long a stalled producer waits between checks that the consumer is gone.
_PUT_TIMEOUT = 0.1


def _import_mlx() -> None:
    """First touch of mlx-vlm, on the worker, so import-time MLX work is too."""
    import mlx_vlm  # noqa: F401


def _ensure_booted() -> None:
    global _booted
    if _booted:
        return
    with _BOOT_LOCK:
        if not _booted:
            _EXECUTOR.submit(_import_mlx).result()
            _booted = True


def submit(fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
    """Queue work on the MLX worker. Never call this from the worker itself."""
    _ensure_booted()
    return _EXECUTOR.submit(fn, *args, **kwargs)


def call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` on the MLX worker and wait for its result."""
    return submit(fn, *args, **kwargs).result()


def iterate(
    produce: Callable[[Callable[[Any], bool]], object],
) -> Generator[Any, None, None]:
    """Stream items out of a worker-side producer.

    ``produce`` is handed an ``emit(item) -> bool`` callback and runs on the
    worker; this generator yields each emitted item on the calling thread.
    ``emit`` returns False once the consumer has walked away, which lets a
    generation loop bail out instead of wedging the single worker on a full
    queue — the case that matters when a user navigates away mid-response.

    The return type is ``Generator`` rather than ``Iterator`` because closing
    it is part of the contract: that is how Streamlit signals abandonment when
    it interrupts a script mid-stream.
    """
    channel: queue.Queue = queue.Queue(maxsize=128)
    finished = object()
    abandoned = threading.Event()

    def emit(item: Any) -> bool:
        while not abandoned.is_set():
            try:
                channel.put(item, timeout=_PUT_TIMEOUT)
                return True
            except queue.Full:
                continue
        return False

    def run() -> None:
        try:
            produce(emit)
        except BaseException as error:
            # Hand the failure to the consumer. If nobody is listening any more
            # the traceback would otherwise vanish silently, so log it instead.
            if not emit(error):
                logger.exception("MLX worker failed after the consumer left")
        finally:
            # Bounded, exactly like every other put. A plain `channel.put()`
            # here blocks forever when the consumer abandons a full channel —
            # and because the executor has a single worker, that wedges MLX for
            # every session in the process until the server restarts.
            emit(finished)

    future = submit(run)
    try:
        while True:
            item = channel.get()
            if item is finished:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        abandoned.set()

    future.result()
