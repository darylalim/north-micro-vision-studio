"""Parse and render North Micro Vision's grounding output.

The model emits bounding boxes as ``[x1, y1, x2, y2]`` normalised to a 0-1000
grid regardless of the image's real size, so converting back is
``x_px = x / 1000 * width``. Because the grid is resolution independent, boxes
can be drawn on the *original* upload rather than the resized copy the encoder
saw — no rescaling between the two is needed.

Phrasing varies run to run (bare JSON, a fenced block, prose with inline
lists), so parsing is deliberately forgiving: structured JSON first, loose
regex as a fallback.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

GRID = 1000.0

_BOX_KEYS = ("bbox_2d", "box_2d", "bbox", "box", "rect", "coordinates")
_LABEL_KEYS = ("label", "name", "text", "object", "category", "class", "caption")

# Four numbers in brackets, allowing decimals and negative signs.
_QUAD = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

PALETTE = [
    (255, 92, 92),
    (56, 189, 248),
    (74, 222, 128),
    (250, 204, 21),
    (192, 132, 252),
    (251, 146, 60),
    (45, 212, 191),
    (244, 114, 182),
]


@dataclass(frozen=True)
class Box:
    """A detection in the model's 0-1000 coordinate space."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.x1 / GRID * width),
            round(self.y1 / GRID * height),
            round(self.x2 / GRID * width),
            round(self.y2 / GRID * height),
        )

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def _make_box(values, label: str) -> Box | None:
    """Validate four raw numbers into a Box, or reject them."""
    try:
        x1, y1, x2, y2 = (float(v) for v in values)
    except (TypeError, ValueError):
        return None

    # Models occasionally emit corners in the wrong order.
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    if not all(-1.0 <= v <= GRID + 1.0 for v in (x1, y1, x2, y2)):
        return None  # Not on the 0-1000 grid — probably pixels or noise.
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None  # Degenerate.

    clamp = lambda v: max(0.0, min(GRID, v))  # noqa: E731
    return Box(clamp(x1), clamp(y1), clamp(x2), clamp(y2), str(label or "").strip())


def _label_from(node: dict) -> str:
    for key in _LABEL_KEYS:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _walk(node, inherited: str, out: list[Box]) -> None:
    """Recursively pull boxes out of arbitrarily-shaped decoded JSON."""
    if isinstance(node, dict):
        label = _label_from(node) or inherited
        for key in _BOX_KEYS:
            value = node.get(key)
            if isinstance(value, (list, tuple)):
                if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
                    if (box := _make_box(value, label)) is not None:
                        out.append(box)
                        break
                else:  # e.g. {"boxes": [[...], [...]]}
                    for item in value:
                        _walk(item, label, out)
                    break
        else:
            for value in node.values():
                _walk(value, label, out)
        return

    if isinstance(node, (list, tuple)):
        if len(node) == 4 and all(isinstance(v, (int, float)) for v in node):
            if (box := _make_box(node, inherited)) is not None:
                out.append(box)
            return
        for item in node:
            _walk(item, inherited, out)


def _json_candidates(text: str):
    """Yield decoded JSON found in fenced blocks or balanced brackets."""
    for fenced in _FENCE.findall(text):
        with contextlib.suppress(json.JSONDecodeError):
            yield json.loads(fenced)

    openers = {"[": "]", "{": "}"}
    for start, char in enumerate(text):
        if char not in openers:
            continue
        depth, in_string, escaped = 0, False, False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current in openers:
                depth += 1
            elif current in openers.values():
                depth -= 1
                if depth == 0:
                    with contextlib.suppress(json.JSONDecodeError):
                        yield json.loads(text[start : end + 1])
                    break


def parse_boxes(text: str) -> list[Box]:
    """Extract every 0-1000 box from a model response, de-duplicated."""
    if not text:
        return []

    found: list[Box] = []
    for decoded in _json_candidates(text):
        _walk(decoded, "", found)

    # Always run the loose pass too. It cannot introduce boxes the JSON scanner
    # missed at the same coordinates (the merge below keys on geometry), but it
    # can recover a label from prose like "signature: [812, 640, 960, 700]" and
    # can rescue quads out of JSON that failed to decode.
    for match in _QUAD.finditer(text):
        prefix = text[max(0, match.start() - 80) : match.start()]
        label = ""
        if hits := re.findall(
            r"([A-Za-z][\w \-/']{1,40}?)\s*[:\-\u2013\u2014]\s*$", prefix
        ):
            label = hits[-1].strip()
        if (box := _make_box(match.groups(), label)) is not None:
            found.append(box)

    # The bracket scanner sees both a labelled object and the bare coordinate
    # array nested inside it, so the same box arrives twice — once with its
    # label, once without. Merge on geometry and keep whichever carries a label.
    merged: dict[tuple[int, int, int, int], Box] = {}
    for box in found:
        key = (round(box.x1), round(box.y1), round(box.x2), round(box.y2))
        existing = merged.get(key)
        if existing is None or (not existing.label and box.label):
            merged[key] = box
    return list(merged.values())


def _font(size: int) -> ImageFont.BaseImageFont:
    """Pick a legible font, degrading to whatever Pillow can supply.

    The return type is ``BaseImageFont`` rather than ``ImageFont`` because
    ``truetype()`` and ``load_default()`` both hand back ``FreeTypeFont``, which
    is a *sibling* of ``ImageFont``, not a subclass. ``BaseImageFont`` is their
    shared base and is what ``ImageDraw.text(font=...)`` accepts.
    """
    for path in (
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_boxes(
    image: Image.Image, boxes: list[Box], *, show_labels: bool = True
) -> Image.Image:
    """Composite translucent boxes and label chips over a copy of ``image``."""
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    longest = max(canvas.size)
    stroke = max(2, round(longest / 400))
    text_size = max(12, round(longest / 55))
    font = _font(text_size)
    pad = max(3, stroke)

    for index, box in enumerate(boxes):
        colour = PALETTE[index % len(PALETTE)]
        x1, y1, x2, y2 = box.to_pixels(*canvas.size)
        # A dark halo just outside the coloured stroke. Without it a box whose
        # palette colour happens to match what it surrounds (a red box over a
        # red stamp) becomes invisible.
        draw.rectangle(
            (x1 - stroke, y1 - stroke, x2 + stroke, y2 + stroke),
            outline=(15, 15, 20, 170),
            width=stroke,
        )
        draw.rectangle(
            (x1, y1, x2, y2), fill=(*colour, 38), outline=(*colour, 255), width=stroke
        )

        if not show_labels:
            continue
        caption = box.label or f"#{index + 1}"
        left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
        chip_w, chip_h = right - left + 2 * pad, bottom - top + 2 * pad

        # Prefer a chip above the box; drop it inside when there is no room.
        chip_x = min(max(0, x1), max(0, canvas.width - chip_w))
        chip_y = y1 - chip_h if y1 - chip_h >= 0 else min(y1, canvas.height - chip_h)
        draw.rectangle(
            (chip_x, chip_y, chip_x + chip_w, chip_y + chip_h), fill=(*colour, 235)
        )
        draw.text(
            (chip_x + pad - left, chip_y + pad - top),
            caption,
            font=font,
            fill=(17, 17, 17, 255),
        )

    return Image.alpha_composite(canvas, overlay).convert("RGB")
