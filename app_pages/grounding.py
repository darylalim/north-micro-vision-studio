"""Visual grounding with a bounding-box overlay.

The model returns boxes on a 0-1000 grid rather than in pixels, so the same
numbers apply to any rendering of the image. That means the overlay can be
drawn on the full-resolution upload even though the encoder saw a downscaled
copy — no coordinate rescaling between the two.
"""

from io import BytesIO

import streamlit as st
from PIL import Image
from streamlit.typing import UploadedFile

from nmv.grounding import Box, draw_boxes, parse_boxes
from nmv.imaging import IMAGE_ERRORS, IMAGE_TYPES, ImagePlan, encode, normalize
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


@st.cache_data(show_spinner=False, max_entries=4)
def _encode_upload(
    _upload: UploadedFile, file_id: str, budget_pixels: int
) -> tuple[Image.Image, Image.Image, ImagePlan]:
    """Normalise and resize an upload, once per (file, budget) pair.

    Uncached, this ran on *every* rerun: nudging a sampling slider or flipping
    "Show labels" paid a full EXIF transpose and resize of the original. Measured
    on an M2 Max, encoding costs 27 ms for the sample invoice and 100 ms for a
    12 MP phone photo, against ~2 ms and ~9 ms to unpickle the cached result.

    ``_upload`` is underscore-prefixed so Streamlit skips hashing it. The bytes
    are the expensive part to hash -- 7 ms for a 5 MB upload, which would eat
    most of the win -- and ``file_id`` already identifies them uniquely.

    ``max_entries`` is not optional here. An entry holds the full-resolution
    original *and* the encoded copy: 13 MB for the sample invoice, 48 MB for a
    12 MP photo. Dragging the budget slider would otherwise mint one per step.
    """
    # Normalise first so the overlay is drawn on exactly the orientation the
    # encoder saw; otherwise EXIF-rotated photos get boxes on the wrong axis.
    original = normalize(Image.open(_upload))
    encoded, resolved = encode(original, budget_pixels)  # already normalised
    return original, encoded, resolved


@st.cache_data(show_spinner=False, max_entries=4)
def _render_overlay(
    _image: Image.Image, file_id: str, boxes: tuple[Box, ...], show_labels: bool
) -> tuple[Image.Image, bytes]:
    """Composite the overlay and PNG-encode it, once per (image, boxes, labels).

    An RGBA convert, an alpha composite and a full PNG encode of the original:
    25 ms for the sample invoice, and proportionally more for a phone photo,
    against ~1 ms to unpickle the result. ``_image`` skips hashing for the same
    reason the upload does in :func:`_encode_upload` -- ``file_id`` identifies
    it, and hashing megabytes of pixels would cost more than it saves.

    Only two entries are useful at a time (labels on and off for the current
    result), since changing the instruction clears ``grounding_result`` outright.
    Four leaves room for a re-run without unbounded growth at ~7 MB an entry.
    """
    annotated = draw_boxes(_image, list(boxes), show_labels=show_labels)
    buffer = BytesIO()
    annotated.save(buffer, format="PNG")
    return annotated, buffer.getvalue()


@st.fragment
def _overlay_panel(image: Image.Image, file_id: str, boxes: list[Box], stem: str) -> None:
    """The overlay column, isolated so "Show labels" reruns only this much.

    The toggle lives inside the fragment on purpose: a fragment only scopes
    reruns triggered by widgets it owns, so leaving the toggle outside would
    rerun the whole page -- re-encoding the upload and re-parsing the response
    to redraw the one thing that actually changed.

    A full-app rerun (any sidebar widget) still re-executes this body, which is
    why :func:`_render_overlay` is cached rather than merely fragment-scoped.
    """
    show_labels = st.toggle("Show labels", value=True, key="grounding_labels")
    if not boxes:
        st.image(image, width="stretch")
        return
    annotated, png_bytes = _render_overlay(image, file_id, tuple(boxes), show_labels)
    st.image(annotated, width="stretch")
    st.download_button(
        "Download overlay",
        png_bytes,
        file_name=f"{stem}-boxes.png",
        mime="image/png",
        icon=":material/download:",
    )


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

try:
    original, encoded, resolved = _encode_upload(upload, upload.file_id, budget)
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

overlay, details = st.columns([3, 2], gap="medium")

# Claimed during the full run so the fragment has a stable slot to rerun into.
with overlay:
    _overlay_panel(original, upload.file_id, boxes, upload.name.rsplit(".", 1)[0])

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
            key="grounding_boxes",
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

    raw_panel = st.expander("Raw response", on_change="rerun")
    if raw_panel.open:
        with raw_panel:
            st.code(result["raw"] or "(empty)", language=None, wrap_lines=True)
