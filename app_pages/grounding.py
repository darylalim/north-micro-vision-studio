"""Visual grounding with a bounding-box overlay.

The model returns boxes on a 0-1000 grid rather than in pixels, so the same
numbers apply to any rendering of the image. That means the overlay can be
drawn on the full-resolution upload even though the encoder saw a downscaled
copy — no coordinate rescaling between the two.
"""

from io import BytesIO

import streamlit as st
from PIL import Image

from nmv.grounding import draw_boxes, parse_boxes
from nmv.imaging import IMAGE_ERRORS, IMAGE_TYPES, encode, normalize
from nmv.model import RunStats, stream_reply, user_turn
from nmv.ui import ensure_studio, render_stats, sidebar

PRESETS = {
    "Objects": (
        "Detect every distinct object in this image. Return a JSON list of "
        '{"label": ..., "bbox_2d": [x1, y1, x2, y2]}.'
    ),
    "Text blocks": (
        "Locate every block of text. Return a JSON list of "
        '{"label": ..., "bbox_2d": [x1, y1, x2, y2]} where the label is the text.'
    ),
    "Stamps and signatures": (
        "Find any stamps, seals, logos or handwritten signatures. Return a JSON "
        'list of {"label": ..., "bbox_2d": [x1, y1, x2, y2]}.'
    ),
    "Custom": "Locate the ",
}

st.title("Grounding")
st.caption(
    "Boxes come back on a 0–1000 grid and are drawn over your original upload. "
    "Deterministic sampling gives steadier coordinates."
)

budget, sampling, stats_slot = sidebar()
studio = ensure_studio()

upload = st.file_uploader("Image", type=IMAGE_TYPES, key="grounding_upload")

preset = st.segmented_control(
    "Task", list(PRESETS), default="Objects", key="grounding_preset"
)
preset = preset or "Objects"

# Keying the text area by preset means picking a new preset genuinely resets
# the instruction instead of leaving the previous one stranded in the widget.
instruction = st.text_area(
    "Instruction",
    value=PRESETS[preset],
    key=f"grounding_prompt_{preset}",
    height=110,
)

# Any input that changes what the boxes *mean* invalidates the cached result.
# Keying on the upload alone let a new instruction, preset or image budget
# render a fresh "Encoding at ..." caption directly above the previous run's
# overlay, table and download — stale output indistinguishable from current.
# "Show labels" is deliberately absent: it only re-renders what is already
# there, and re-running the model for it would be wasteful.
signature = (getattr(upload, "file_id", None), instruction, budget)
if st.session_state.get("grounding_signature") != signature:
    st.session_state.grounding_signature = signature
    st.session_state.pop("grounding_result", None)

run = st.button(
    "Find boxes",
    icon=":material/frame_inspect:",
    type="primary",
    disabled=upload is None or not instruction.strip(),
)

if upload is None:
    st.info("Upload an image to begin.", icon=":material/image:")
    st.stop()

# Normalise first so the overlay is drawn on exactly the orientation the
# encoder saw; otherwise EXIF-rotated photos get boxes on the wrong axis.
try:
    original = normalize(Image.open(upload))
    encoded, resolved = encode(original, budget)  # already normalised
except IMAGE_ERRORS as error:
    st.error(str(error), icon=":material/broken_image:")
    st.stop()

st.caption(
    f"Encoding at **{resolved.target[0]}×{resolved.target[1]}** "
    f"({resolved.tokens:,} visual tokens)"
    + (
        f" · {resolved.direction} from {resolved.source[0]}×{resolved.source[1]}"
        if resolved.resized
        else ""
    )
)

if run:
    stats = RunStats()
    with st.spinner("Locating…"):
        raw = "".join(
            stream_reply(studio, [user_turn(instruction, 1)], [encoded], sampling, stats)
        )
    st.session_state.last_stats = stats
    render_stats(stats_slot, stats)
    st.session_state.grounding_result = {
        "raw": raw,
        "boxes": parse_boxes(raw),
        "truncated": stats.truncated,
    }

result = st.session_state.get("grounding_result")
if result is None:
    st.image(original, caption="Ready", width="stretch")
    st.stop()

boxes = result["boxes"]
show_labels = st.toggle("Show labels", value=True, key="grounding_labels")

overlay, details = st.columns([3, 2], gap="medium")

with overlay:
    if boxes:
        # Render and PNG-encode once per (result, label setting) rather than on
        # every rerun. Without this, nudging any sidebar widget re-ran an RGBA
        # convert, an alpha composite and a full PNG encode of the original —
        # hundreds of milliseconds for a UI action that changed nothing here.
        overlay_key = (st.session_state.get("grounding_signature"), show_labels)
        if st.session_state.get("grounding_overlay_key") != overlay_key:
            annotated = draw_boxes(original, boxes, show_labels=show_labels)
            buffer = BytesIO()
            annotated.save(buffer, format="PNG")
            st.session_state.grounding_overlay = (annotated, buffer.getvalue())
            st.session_state.grounding_overlay_key = overlay_key

        annotated, png_bytes = st.session_state.grounding_overlay
        st.image(annotated, width="stretch")
        st.download_button(
            "Download overlay",
            png_bytes,
            file_name=f"{upload.name.rsplit('.', 1)[0]}-boxes.png",
            mime="image/png",
            icon=":material/download:",
        )
    else:
        st.image(original, width="stretch")

with details:
    if boxes:
        st.dataframe(
            [
                {
                    "#": index + 1,
                    "label": box.label or "—",
                    "grid": f"{box.x1:.0f}, {box.y1:.0f}, {box.x2:.0f}, {box.y2:.0f}",
                    "pixels": "{}, {}, {}, {}".format(
                        *box.to_pixels(original.width, original.height)
                    ),
                }
                for index, box in enumerate(boxes)
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        # Diagnose the actual cause. A response cut off at max_tokens leaves a
        # half-written JSON array, which looks identical to the model simply
        # answering in prose — and the remedies are opposite.
        if result.get("truncated"):
            st.warning(
                "The response stopped at the max-tokens limit, so the box list was "
                "cut off mid-array. Raise **Max new tokens** in the sidebar.",
                icon=":material/content_cut:",
            )
        else:
            st.warning(
                "No boxes parsed from the response. The model sometimes answers in "
                "prose — try the deterministic toggle, or ask explicitly for JSON.",
                icon=":material/search_off:",
            )

    with st.expander("Raw response"):
        st.code(result["raw"] or "(empty)", language=None, wrap_lines=True)
