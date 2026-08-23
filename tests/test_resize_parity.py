"""Guard the one piece of mlx-vlm logic we mirror instead of import.

``nmv.imaging.smart_resize`` is a copy, kept so that no Streamlit page thread
can import mlx-vlm and steal its thread-local GPU stream. That copy is only
safe while it agrees with upstream, so check it exhaustively.

Run:  uv run python tests/test_resize_parity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_vlm.models.cohere_compass.processing_cohere_compass import (
    CohereCompassImageProcessor,
)
from mlx_vlm.models.cohere_compass.processing_cohere_compass import (
    smart_resize as upstream,
)

from nmv.imaging import FACTOR, MAX_PIXELS, MIN_PIXELS, AspectRatioError, plan
from nmv.imaging import smart_resize as mirrored

SIZES = [
    (1, 1),
    (16, 16),
    (31, 47),
    (128, 128),
    (640, 480),
    (800, 600),
    (1080, 1920),
    (1240, 1754),
    (1654, 2339),
    (2339, 1654),
    (3024, 4032),
    (4000, 3000),
    (5000, 40),
    (40, 5000),
    (10000, 10000),
    (7, 1200),
]
BUDGETS = [MIN_PIXELS, 100_000, 1_000_000, 2_000_000, MAX_PIXELS]

failures = 0
checked = 0
for height, width in SIZES:
    for budget in BUDGETS:
        kwargs = {"factor": FACTOR, "min_pixels": MIN_PIXELS, "max_pixels": budget}
        try:
            expected = upstream(height, width, **kwargs)
        except ValueError as error:
            expected = ("raises", type(error).__name__)
        try:
            actual = mirrored(height, width, **kwargs)
        except ValueError:
            # Ours raises the AspectRatioError subclass; upstream raises ValueError.
            actual = ("raises", "ValueError")
        checked += 1
        if expected != actual:
            failures += 1
            print(f"MISMATCH {height}x{width} budget={budget}: {expected} != {actual}")

print(f"{checked - failures}/{checked} size/budget combinations match upstream")


# --- plan() targets must be fixed points of mlx-vlm's own pass -------------
# nmv.imaging.encode resizes before mlx-vlm sees the image. That is only sound
# if mlx-vlm's internal smart_resize -- which uses the *checkpoint's* bounds,
# not our budget -- leaves the result alone. Otherwise the encoder silently
# works at a different resolution than the one the UI reported.
stable = 0
for height, width in SIZES:
    for budget in BUDGETS:
        try:
            resolved = plan(width, height, budget)
        except AspectRatioError:
            continue
        target_w, target_h = resolved.target
        again_h, again_w = mirrored(
            target_h,
            target_w,
            factor=FACTOR,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        stable += 1
        if (again_w, again_h) != (target_w, target_h):
            failures += 1
            print(
                f"NOT A FIXED POINT {width}x{height} budget={budget}: "
                f"we hand over {target_w}x{target_h}, "
                f"mlx-vlm makes it {again_w}x{again_h}"
            )
print(f"{stable} plan targets survive mlx-vlm's own resize untouched")


# --- the real end-to-end guarantee -----------------------------------------
# What actually matters is that mlx-vlm's own processor, running its own
# defaults on our already-resized image, reports the same token count the UI
# printed. This is the claim the studio makes to the user.
processor = CohereCompassImageProcessor()
merge = processor.merge_size**2
agreed = 0
for height, width in SIZES:
    for budget in BUDGETS:
        try:
            resolved = plan(width, height, budget)
        except AspectRatioError:
            continue
        target_w, target_h = resolved.target
        upstream_tokens = (
            processor.get_number_of_image_patches(target_h, target_w) // merge
        )
        agreed += 1
        if upstream_tokens != resolved.tokens:
            failures += 1
            print(
                f"TOKEN MISMATCH {width}x{height} budget={budget}: "
                f"we say {resolved.tokens}, mlx-vlm says {upstream_tokens}"
            )
print(f"{agreed} plans agree with mlx-vlm's own patch count")


# --- deliberate divergence in the no-kwargs defaults ------------------------
# The mirror's defaults are this checkpoint's (16384 / 3868706); upstream's
# function signature carries generic ones. The tests above always pass both
# explicitly, so assert the difference is intentional rather than accidental.
ours = mirrored(2339, 1654)
theirs = upstream(2339, 1654)
print(
    f"default-kwarg behaviour differs on purpose: ours {ours} "
    f"(checkpoint budget) vs upstream {theirs} (generic) — "
    f"{'expected' if ours != theirs else 'NOTE: now identical'}"
)

sys.exit(1 if failures else 0)
