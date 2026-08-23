"""Multi-turn visual chat.

Images are attached to the turn they arrived on. mlx-vlm reads the per-message
image markers we emit and hands each picture to the right turn, so a follow-up
question can refer back to a page uploaded three turns ago.
"""

import streamlit as st
from PIL import Image, UnidentifiedImageError

from nmv.imaging import AspectRatioError, prepare
from nmv.model import RunStats, assistant_turn, stream_reply, user_turn
from nmv.ui import ensure_studio, render_stats, sidebar

IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]

SUGGESTIONS = {
    ":blue[:material/description:] Read a document": (
        "Transcribe all text in this image, preserving the layout."
    ),
    ":green[:material/bar_chart:] Explain a chart": (
        "Describe this chart and state the trend it shows."
    ),
    ":orange[:material/table:] Extract a table": (
        "Extract every table in this image as markdown."
    ),
}

st.title("Chat")
st.caption(
    "Document Q&A, OCR, charts and captioning. Attach images with the paperclip; "
    "they stay attached to that turn for the rest of the conversation."
)

budget, sampling, stats_slot = sidebar()

if "chat" not in st.session_state:
    st.session_state.chat = []

studio = ensure_studio()

if st.session_state.chat:
    st.button(
        "Clear conversation",
        icon=":material/delete_sweep:",
        on_click=st.session_state.chat.clear,
    )

for record in st.session_state.chat:
    with st.chat_message(record["role"]):
        if record["images"]:
            st.image(record["images"], width=180)
        if record.get("note"):
            st.caption(record["note"])
        st.markdown(record["text"])
        if record.get("stats_line"):
            st.caption(record["stats_line"])

queued = st.session_state.pop("queued_prompt", None)

if not st.session_state.chat and not queued:
    picked = st.pills(
        "Try asking",
        list(SUGGESTIONS),
        label_visibility="collapsed",
        key="chat_suggestion",
    )
    if picked:
        st.session_state.queued_prompt = SUGGESTIONS[picked]
        st.rerun()

submission = st.chat_input(
    "Ask about a page, chart, or photo",
    accept_file="multiple",
    file_type=IMAGE_TYPES,
    submit_mode="disable",
)

if submission is not None:
    text = (submission.text or "").strip()
    uploads = list(submission.files or [])
elif queued:
    text, uploads = queued, []
else:
    text, uploads = "", []

if text or uploads:
    images, notes, failures = [], [], []
    for upload in uploads:
        try:
            image, resolved = prepare(Image.open(upload), budget)
        except (AspectRatioError, UnidentifiedImageError, OSError, ValueError) as error:
            failures.append(f"**{upload.name}** — {error}")
            continue
        images.append(image)
        note = (
            f"{upload.name} · {resolved.target[0]}×{resolved.target[1]} · "
            f"{resolved.tokens:,} tokens"
        )
        if resolved.resized:
            note += f" (from {resolved.source[0]}×{resolved.source[1]})"
        notes.append(note)

    for failure in failures:
        st.error(failure, icon=":material/broken_image:")

    if not text:
        text = "Describe this image." if images else ""

    if text:
        st.session_state.chat.append(
            {
                "role": "user",
                "text": text,
                "images": images,
                "note": " · ".join(f"`{n}`" for n in notes) if notes else "",
            }
        )
        with st.chat_message("user"):
            if images:
                st.image(images, width=180)
            if notes:
                st.caption(" · ".join(f"`{n}`" for n in notes))
            st.markdown(text)

        # Rebuild the whole conversation each turn. Images are collected in the
        # same order the markers appear, which is how they get matched up.
        messages, ordered_images = [], []
        for record in st.session_state.chat:
            if record["role"] == "user":
                messages.append(user_turn(record["text"], len(record["images"])))
                ordered_images.extend(record["images"])
            else:
                messages.append(assistant_turn(record["text"]))

        stats = RunStats()
        with st.chat_message("assistant"):
            reply = st.write_stream(
                stream_reply(studio, messages, ordered_images, sampling, stats)
            )
            stats_line = (
                f"{stats.prompt_tokens:,} prompt tokens · "
                f"{stats.generation_tokens:,} generated at "
                f"{stats.generation_tps:.0f} tok/s · peak {stats.peak_memory:.2f} GB"
            )
            st.caption(stats_line)

        st.session_state.last_stats = stats
        render_stats(stats_slot, stats)
        st.session_state.chat.append(
            {"role": "assistant", "text": reply, "images": [], "stats_line": stats_line}
        )
