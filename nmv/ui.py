"""Shared chrome: the model gate and the sidebar controls both pages use."""

from __future__ import annotations

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from nmv.imaging import MAX_PIXELS, MEGAPIXEL, PIXELS_PER_TOKEN
from nmv.model import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MODEL_REPO,
    RunStats,
    Sampling,
    Studio,
    is_cached,
    load_studio,
)

MAX_MEGAPIXELS = MAX_PIXELS / MEGAPIXEL


def ensure_studio() -> Studio:
    """Load the checkpoint, warning first if it still has to be downloaded."""
    if not is_cached():
        st.warning(
            f"`{MODEL_REPO}` is not in the Hugging Face cache yet — the first run "
            "downloads about 5 GB.",
            icon=":material/cloud_download:",
        )
    with st.spinner("Loading North Micro Vision (bf16)…"):
        return load_studio()


def _budget_control() -> int:
    """Pixel budget per image, the single biggest lever on speed and memory."""
    st.subheader("Image budget", divider="gray")
    megapixels = st.slider(
        "Pixels per image",
        min_value=0.10,
        max_value=MAX_MEGAPIXELS,
        value=MAX_MEGAPIXELS,
        step=0.05,
        format="%.2f MP",
        key="budget_mp",
        help=(
            "The encoder spends one token per 32×32 px block, so tokens scale "
            "with area. The ceiling is an A4 page at 200 dpi — the largest "
            "resolution this model was trained on."
        ),
    )
    budget = int(megapixels * MEGAPIXEL)
    st.caption(
        f"Up to **{budget // PIXELS_PER_TOKEN:,} tokens** per image"
        + ("  ·  native ceiling" if megapixels >= MAX_MEGAPIXELS else "")
    )
    return budget


def _sampling_control() -> Sampling:
    st.subheader("Sampling", divider="gray")
    greedy = st.toggle(
        "Deterministic",
        value=False,
        key="greedy",
        help="Temperature 0 — repeatable runs, better for grounding and OCR.",
    )
    max_tokens = st.slider("Max new tokens", 64, 2048, 512, step=64, key="max_tokens")

    if greedy:
        st.caption("Temperature pinned to 0; nucleus and top-k inactive.")
        return Sampling(max_tokens=max_tokens, temperature=0.0, top_p=1.0, top_k=0)

    temperature = st.slider(
        "Temperature", 0.0, 1.5, DEFAULT_TEMPERATURE, 0.05, key="temp"
    )
    top_p = st.slider("Top-p", 0.05, 1.0, DEFAULT_TOP_P, 0.05, key="top_p")
    top_k = st.slider("Top-k", 0, 100, DEFAULT_TOP_K, 1, key="top_k")
    penalty = st.slider(
        "Repetition penalty",
        1.0,
        1.5,
        1.0,
        0.01,
        key="rep_pen",
        help="1.0 disables it. Cohere suggests a mild penalty for long outputs.",
    )
    return Sampling(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=penalty,
    )


def render_stats(slot: DeltaGenerator, stats: RunStats | None) -> None:
    """Fill the reserved sidebar slot with timings from a run.

    The slot is claimed while the sidebar draws, before generation starts, so
    a fresh run can write into it afterwards. Without that, the panel would
    always trail one interaction behind — it would render the *previous* run's
    numbers under a "Last run" heading.
    """
    if stats is None:
        return
    with slot.container():
        st.subheader("Last run", divider="gray")
        with st.container(horizontal=True):
            st.metric("Prefill tokens", f"{stats.prompt_tokens:,}")
            st.metric("Decode tok/s", f"{stats.generation_tps:.0f}")
        with st.container(horizontal=True):
            st.metric("Prefill tok/s", f"{stats.prompt_tps:.0f}")
            st.metric("Peak memory GB", f"{stats.peak_memory:.2f}")


def sidebar() -> tuple[int, Sampling, DeltaGenerator]:
    """Render every shared control.

    Returns the image budget, the sampling config, and a slot the caller fills
    with run statistics once generation finishes.
    """
    with st.sidebar:
        st.caption("Model")
        st.code(MODEL_REPO, language=None, wrap_lines=True)
        st.caption("2.4B · bf16 · Apache 2.0 · native resolution")
        budget = _budget_control()
        sampling = _sampling_control()
        stats_slot = st.empty()

    render_stats(stats_slot, st.session_state.get("last_stats"))
    return budget, sampling, stats_slot
