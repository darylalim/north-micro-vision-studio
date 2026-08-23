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
    smart_resize as upstream,
)

from nmv.imaging import FACTOR, MAX_PIXELS, MIN_PIXELS
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
sys.exit(1 if failures else 0)
