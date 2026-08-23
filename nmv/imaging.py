"""Image budgeting for North Micro Vision's native-resolution pipeline.

North Micro Vision does not squash images into a fixed square. It cuts them
into 16 px patches, merges every 2x2 block, and spends one language-model
token per merged block. Token count therefore scales with *pixel area*, and
capping area is the only lever on prefill latency and KV-cache memory.

One token covers a 32x32 px block, so ``tokens == pixels / 1024``.

This module is deliberately free of any mlx-vlm import. mlx-vlm binds a
thread-local GPU stream the moment it is imported, so it may only ever be
imported on the worker thread (see ``nmv.runtime``); keeping the geometry here
as plain arithmetic removes the hazard of a page pulling it in first.
``smart_resize`` therefore mirrors mlx-vlm's implementation rather than
importing it — ``tests/test_resize_parity.py`` asserts the two stay identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageOps

PATCH_SIZE = 16
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32 — resized edges snap to this
PIXELS_PER_TOKEN = FACTOR * FACTOR  # 1024

# mlx-vlm's own bounds for this checkpoint. The ceiling is 1654 x 2339, an A4
# page at 200 dpi, which is also the largest resolution Cohere trained on.
# (The HF `preprocessor_config.json` advertises a far larger
# `size.longest_edge` of 16,777,216 px, but mlx-vlm only reads keys literally
# named `min_pixels`/`max_pixels`, which that file lacks — so this is the
# budget that actually applies.)
MIN_PIXELS = 16_384
MAX_PIXELS = 3_868_706
MAX_ASPECT_RATIO = 200

# The multimodal context Cohere validated, even though the LM backbone claims
# 128K. Used to warn when a conversation's accumulated images pass it.
VALIDATED_CONTEXT_TOKENS = 8192

MEGAPIXEL = 1_000_000

# Upload formats both pages accept. Kept here beside the other encoder
# constraints so the two pages cannot drift apart.
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]

# Everything PIL can raise for a corrupt, hostile or unsupported upload.
# DecompressionBombError descends straight from Exception -- not OSError, not
# ValueError -- so it must be named explicitly or it escapes the handler and
# aborts the entire Streamlit script run. (OSError already covers
# UnidentifiedImageError, and ValueError covers our own AspectRatioError.)
IMAGE_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    ValueError,
    Image.DecompressionBombError,
)


class AspectRatioError(ValueError):
    """Raised for images too elongated for the vision encoder."""


def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Snap a size to multiples of ``factor`` inside the pixel budget.

    A mirror of ``mlx_vlm.models.cohere_compass.processing_cohere_compass``.
    Returns ``(height, width)``, matching upstream's argument order.
    """
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise AspectRatioError(
            f"absolute aspect ratio must be smaller than {MAX_ASPECT_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@dataclass(frozen=True)
class ImagePlan:
    """What a given image will cost once the encoder sees it."""

    source: tuple[int, int]  # (width, height) as uploaded
    target: tuple[int, int]  # (width, height) actually encoded
    tokens: int
    budget_pixels: int

    @property
    def resized(self) -> bool:
        return self.source != self.target

    @property
    def source_pixels(self) -> int:
        return self.source[0] * self.source[1]

    @property
    def target_pixels(self) -> int:
        return self.target[0] * self.target[1]

    @property
    def direction(self) -> str:
        """``"downscaled"``, ``"upscaled"`` or ``"unchanged"``.

        Not always a downscale: ``smart_resize`` snaps each edge to a multiple
        of 32 and rounds *up* whenever that still fits the budget, so most
        uploads whose dimensions are not already multiples of 32 grow slightly.
        """
        if self.target_pixels < self.source_pixels:
            return "downscaled"
        if self.target_pixels > self.source_pixels:
            return "upscaled"
        return "unchanged"


def tokens_for(width: int, height: int) -> int:
    """Visual tokens for an already-snapped (multiple-of-32) size."""
    return (height // PATCH_SIZE) * (width // PATCH_SIZE) // (MERGE_SIZE**2)


def plan(width: int, height: int, budget_pixels: int) -> ImagePlan:
    """Resolve the encoded size and token cost without touching pixel data."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Image has a zero dimension: {width}x{height}")

    ratio = max(width, height) / min(width, height)
    if ratio > MAX_ASPECT_RATIO:
        raise AspectRatioError(
            f"Aspect ratio {ratio:.0f}:1 exceeds the encoder's "
            f"{MAX_ASPECT_RATIO}:1 limit. Crop or split the image first."
        )

    budget = max(MIN_PIXELS, min(int(budget_pixels), MAX_PIXELS))
    new_h, new_w = smart_resize(height, width, max_pixels=budget)

    # We resize before handing the image over, but mlx-vlm still runs its own
    # pass over whatever it receives — using the *checkpoint's* bounds, not our
    # budget. A tight budget can land below MIN_PIXELS (a 640x480 at the floor
    # resolves to 128x96 = 12,288 px), and mlx-vlm then scales it back up.
    # Model that second pass here, or the size and token count reported to the
    # user are not the ones the encoder actually sees.
    final_h, final_w = smart_resize(
        new_h, new_w, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    return ImagePlan(
        source=(width, height),
        target=(final_w, final_h),
        tokens=tokens_for(final_w, final_h),
        budget_pixels=budget,
    )


def normalize(image: Image.Image) -> Image.Image:
    """Apply the EXIF orientation tag and force RGB.

    This matters more than it looks: phone photos carry an orientation tag, and
    without it the model reads a sideways page while an overlay draws boxes on
    the upright copy — every coordinate lands wrong. Anything that renders
    boxes must use this same normalised image, not the raw upload.
    """
    image = ImageOps.exif_transpose(image) or image
    return image if image.mode == "RGB" else image.convert("RGB")


def encode(image: Image.Image, budget_pixels: int) -> tuple[Image.Image, ImagePlan]:
    """Resize an **already normalised** image to the token budget.

    Split out from :func:`prepare` so a caller that needs both the normalised
    original (to draw overlays on) and the encoded copy does not pay for a
    second full-resolution normalisation — Pillow's ``exif_transpose`` returns
    ``image.copy()`` even when there is no orientation tag left to apply, so
    the second call is a pure memcpy of every pixel.

    Resizing here rather than leaving it to mlx-vlm is safe because
    :func:`plan` already accounts for mlx-vlm's own pass: its target is a fixed
    point of ``smart_resize`` under the checkpoint's bounds, so mlx-vlm's
    internal call is a no-op and the resolution reported to the user is exactly
    the one the encoder sees. ``tests/test_resize_parity.py`` enforces that.
    """
    resolved = plan(image.width, image.height, budget_pixels)
    if resolved.resized:
        image = image.resize(resolved.target, Image.Resampling.BICUBIC)
    return image, resolved


def prepare(image: Image.Image, budget_pixels: int) -> tuple[Image.Image, ImagePlan]:
    """Normalise orientation and colour, then resize to the token budget."""
    return encode(normalize(image), budget_pixels)
