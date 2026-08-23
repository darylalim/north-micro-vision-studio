"""Loading and streaming for the North Micro Vision bf16 checkpoint.

Only bf16 is used here — the full-precision MLX conversion, ~5 GB resident.
On a 32 GB Apple Silicon machine that leaves ample headroom for the KV cache,
and it keeps results comparable to the published numbers, which is what a
research studio wants. Quantised conversions exist upstream if the footprint
ever needs to come down.

Every mlx-vlm import in this module is deliberately function-local and runs on
the worker thread from ``nmv.runtime``. Hoisting one to module scope would bind
mlx-vlm's thread-local GPU stream to a Streamlit ScriptRunner thread and break
generation on the next rerun.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import streamlit as st
from PIL import Image

from nmv import runtime

MODEL_REPO = "mlx-community/North-Micro-Vision-Instruct-bf16"

# The checkpoint's own generation_config.json. Cohere also notes the model was
# not trained to follow system prompts, so the studio never sends one — task
# instructions are prepended to the user turn instead.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20


@dataclass
class Studio:
    """The loaded model plus everything needed to prompt it."""

    model: Any
    processor: Any
    config: dict


@dataclass(frozen=True)
class Sampling:
    max_tokens: int = 512
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    repetition_penalty: float = 1.0
    seed: int | None = None

    def as_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self.repetition_penalty
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs


@dataclass
class RunStats:
    """Timings reported by the last generation."""

    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    finish_reason: str | None = None

    def absorb(self, chunk: Any) -> None:
        for name in (
            "prompt_tokens",
            "generation_tokens",
            "prompt_tps",
            "generation_tps",
            "peak_memory",
            "finish_reason",
        ):
            value = getattr(chunk, name, None)
            if value:
                setattr(self, name, value)


def _load(repo_id: str) -> Studio:
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model, processor = load(repo_id)
    return Studio(model=model, processor=processor, config=load_config(repo_id))


@st.cache_resource(show_spinner=False)
def load_studio(repo_id: str = MODEL_REPO) -> Studio:
    """Load weights once per server process.

    ``st.cache_resource`` rather than ``cache_data``: the model is a live,
    unserialisable object graph, and every session should share the one copy
    instead of paying 5 GB again.
    """
    return runtime.call(_load, repo_id)


def is_cached(repo_id: str = MODEL_REPO) -> bool:
    """True when weights are already on disk, so the UI can warn before a 5 GB pull."""
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)


def user_turn(text: str, image_count: int = 0) -> dict:
    """Build a user message that declares its own images.

    mlx-vlm reads these ``{"type": "image"}`` markers to work out which turn
    each image belongs to, so a conversation can carry pictures in several
    turns and still line up. Without markers every image would be attached to
    the most recent user message.
    """
    content: list[dict] = [{"type": "image"} for _ in range(image_count)]
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def assistant_turn(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def stream_reply(
    studio: Studio,
    messages: list[dict],
    images: list[Image.Image],
    sampling: Sampling,
    stats: RunStats | None = None,
) -> Iterator[str]:
    """Yield response deltas, suitable for ``st.write_stream``."""

    def produce(emit):
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(
            studio.processor, studio.config, messages, num_images=len(images)
        )
        for chunk in stream_generate(
            studio.model,
            studio.processor,
            # apply_chat_template is annotated `list | str | Any`; it returns a
            # str whenever return_messages is False, which is the default.
            cast(str, prompt),
            # mlx-vlm annotates `image` as str | list[str], but prepare_inputs
            # routes every entry through process_image, which accepts PIL images
            # directly. Passing them avoids round-tripping uploads via temp files.
            image=cast(Any, list(images) or None),
            **sampling.as_kwargs(),
        ):
            if not emit(chunk):  # consumer walked away — stop decoding
                return

    for chunk in runtime.iterate(produce):
        if stats is not None:
            stats.absorb(chunk)
        yield chunk.text
