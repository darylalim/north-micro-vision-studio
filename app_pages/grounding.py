"""Visual grounding with a bounding-box overlay.

The model returns boxes on a 0-1000 grid rather than in pixels, so the same
numbers apply to any rendering of the image. That means the overlay can be
drawn on the full-resolution upload even though the encoder saw a downscaled
copy — no coordinate rescaling between the two.
"""

from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from nmv.grounding import draw_boxes, parse_boxes
from nmv.imaging import AspectRatioError, normalize, prepare
from nmv.model import RunStats, stream_reply, user_turn
from nmv.ui import ensure_studio, render_stats, sidebar

IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]

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

# A new upload must not inherit the previous image's boxes.
if st.session_state.get("grounding_file_id") != getattr(upload, "file_id", None):
    st.session_state.grounding_file_id = getattr(upload, "file_id", None)
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
    encoded, resolved = prepare(original, budget)
except (AspectRatioError, UnidentifiedImageError, OSError, ValueError) as error:
    st.error(str(error), icon=":material/broken_image:")
    st.stop()

st.caption(
    f"Encoding at **{resolved.target[0]}×{resolved.target[1]}** "
    f"({resolved.tokens:,} visual tokens)"
    + (
        f" · downscaled from {resolved.source[0]}×{resolved.source[1]}"
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
        "stats": stats,
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
        annotated = draw_boxes(original, boxes, show_labels=show_labels)
        st.image(annotated, width="stretch")
        buffer = BytesIO()
        annotated.save(buffer, format="PNG")
        st.download_button(
            "Download overlay",
            buffer.getvalue(),
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
        st.warning(
            "No boxes parsed from the response. The model sometimes answers in "
            "prose — try the deterministic toggle, or ask explicitly for JSON.",
            icon=":material/search_off:",
        )

    with st.expander("Raw response"):
        st.code(result["raw"] or "(empty)", language=None, wrap_lines=True)
