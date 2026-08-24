# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit app for running Cohere's North Micro Vision (2.4B native-resolution
vision-language model) locally on Apple Silicon via MLX. Two pages: multi-turn visual
chat, and grounding with a bounding-box overlay.

**bf16 only** — `mlx-community/North-Micro-Vision-Instruct-bf16`, ~5 GB, ~6.2 GB peak.
This is a deliberate choice so results stay comparable to Cohere's published numbers.
Do not add quantised variants without being asked.

## Commands

```bash
uv sync                                        # create venv, install pinned deps
uv run streamlit run streamlit_app.py          # run the app (localhost:8501)

uv run python tests/test_resize_parity.py      # resize/token invariants; plain script, no pytest
uv run python tests/test_runtime_abandon.py    # MLX worker must survive abandoned streams
uv run ruff check .                            # lint  (E, F, I, UP, B, SIM, C4)
uv run ruff format .                           # format (line-length 90)
uv run ty check                                # type check (targets 3.11)

uv run hf download mlx-community/North-Micro-Vision-Instruct-bf16   # prefetch weights
```

ruff and ty are locked dev dependencies, so use `uv run ruff` / `uv run ty`, not the
`uvx` forms — those resolve independently of `uv.lock` and will drift the day a new
release ships. Reach for `uvx` only for tools the project does not declare.

ty runs clean. Two call sites in `nmv/model.py` use `cast` because mlx-vlm's
annotations are narrower than its runtime behaviour (`apply_chat_template` returns a
`str` under the default `return_messages=False`; `image=` accepts PIL objects because
`prepare_inputs` routes them through `process_image`). Keep the casts and their
comments rather than replacing them with blanket ignores.

**Restart the server after editing anything under `nmv/`.** Streamlit hot-reloads
`streamlit_app.py` and `app_pages/*`, but `nmv/*` are ordinary imported modules cached
in `sys.modules`; edits there surface as a stale `ImportError` until you restart.

When working with Python, invoke the relevant `/astral:<skill>` for `uv`, `ty`, and
`ruff` to ensure best practices are followed.

## CI and releases

`.github/workflows/ci.yml` runs every command above on an Apple Silicon runner, plus the
module-scope MLX import grep. There is no Linux job and cannot be one: `mlx` ships arm64
macOS wheels only, so `uv sync --locked` will not resolve on Linux, and `ty` needs
mlx-vlm importable to check `nmv/model.py`.

**Editing `version` in `pyproject.toml` is a publish action.** On `main`, once CI passes,
the workflow tags that commit and publishes a GitHub release with notes built from commit
subjects — there is no draft to approve and no undo that unpublishes a tag others may
have fetched. Do not bump the version to test something or as an incidental tidy-up; bump
it when you mean to ship. The workflow refuses a version that is not valid PEP 440 or not
ahead of the highest existing tag, and marks pre-releases so they do not take "Latest",
but nothing checks whether you *meant* it.

**A bump is two files.** `uv.lock` records the project's own version alongside its
dependencies, so run `uv lock` and commit the result with the `pyproject.toml` change:

```bash
uv lock && git add pyproject.toml uv.lock
```

Skipping it fails CI before anything publishes: `uv sync --locked` reports drift
(`The lockfile at uv.lock needs to be updated`), that fails `check`, and `release`
declares `needs: check`. So the outcome of a half-finished bump is a red run and *no
release at all* rather than a broken one — safe, but worth recognising, since uv's hint
names `uv lock` without connecting it to the version you just changed.

## The constraint that shapes the architecture

mlx-vlm builds its generation stream at import time:

```python
# mlx_vlm/generate/common.py:20
generation_stream = mx.new_thread_local_stream(mx.default_device())
```

That stream belongs to whichever thread **first imported** the package. Streamlit runs
every rerun on a fresh ScriptRunner thread, so a naive app works exactly once and then
fails with `RuntimeError: There is no Stream(gpu, 1) in current thread.`

`nmv/runtime.py` pins the import *and* every later MLX call to one long-lived worker
thread, streaming tokens back over a bounded queue. It also serialises GPU work across
browser tabs, which matters because MLX generation is not reentrant.

**Invariant: mlx-vlm is imported only inside worker functions, never at module scope.**
Check it before committing:

```bash
grep -rn --include='*.py' "^from mlx_vlm\|^import mlx_vlm\|^from mlx\." nmv/ app_pages/ streamlit_app.py
```

That must return nothing.

If you ever broaden the ruff `select` set, note that `PLC0415`
(`import-outside-top-level`) fires on all six function-local mlx-vlm imports. Those
are deliberate — ignore the rule, never hoist the imports to satisfy it.

Consequences:

- `nmv/model.py` keeps every `from mlx_vlm import ...` inside a function body.
- `nmv/imaging.py` **mirrors** mlx-vlm's `smart_resize` as plain arithmetic rather than
  importing it. `tests/test_resize_parity.py` holds the copy to upstream across 80
  size/budget combinations — run it after touching either.
- Anything new that calls MLX goes through `runtime.call()` (blocking) or
  `runtime.iterate()` (streaming). Never call those *from* the worker: single worker,
  instant deadlock.

## Module roles

| File | Responsibility |
|---|---|
| `nmv/runtime.py` | MLX worker thread. Read first. |
| `nmv/model.py` | Loading (`st.cache_resource`), `Sampling`, `RunStats`, streaming |
| `nmv/imaging.py` | Token budgeting, `encode`/`prepare`, EXIF normalisation. **No MLX import.** |
| `nmv/grounding.py` | 0–1000 box parsing and PIL overlay rendering |
| `nmv/ui.py` | Shared sidebar; returns `(budget, sampling, stats_slot)` |
| `app_pages/*.py` | Page scripts — UI only, direct scripts with no `main()` |

## Domain rules that are easy to get wrong

**Token count scales with pixel area.** One token per 32×32 px block, so
`tokens = pixels / 1024`. The image budget slider is the only lever on prefill latency
and KV-cache memory. Ceiling is 3,868,706 px (A4 @ 200 dpi). The checkpoint's
`preprocessor_config.json` advertises a far larger `size.longest_edge`, but mlx-vlm
only honours keys literally named `min_pixels`/`max_pixels`, which that file lacks — so
that larger number never applies and should not be reintroduced as a limit.

**Pre-resizing before handing images to mlx-vlm is safe only because `plan()` models
mlx-vlm's own second pass.** mlx-vlm re-runs `smart_resize` on whatever it receives
using the *checkpoint's* bounds, not our budget — so a tight budget that lands below
`MIN_PIXELS` gets scaled back up, and the size and token count shown to the user would
be wrong. `plan()` therefore applies our budget, then mlx-vlm's bounds, and reports that
fixed point. `tests/test_resize_parity.py` asserts every target survives mlx-vlm's pass
untouched and that the token count matches its own patch arithmetic.

**Grounding boxes are on a 0–1000 grid**, not in pixels, so they apply to any rendering
of the image. Overlays are therefore drawn on the full-resolution upload. Both the
encoded copy and the overlay copy must come from `imaging.normalize()` — skipping EXIF
transposition puts every box on the wrong axis for rotated phone photos.

**The model emits bare coordinate arrays** like `[659, 671, 921, 799]` even when asked
for labelled JSON. Labels are optional throughout `nmv/grounding.py`; don't add code
that assumes they exist.

**Never send a system prompt.** Cohere states the model was not trained to follow them.
Task instructions belong in the user turn.

**Per-turn image markers matter.** `model.user_turn(text, image_count)` emits
`{"type": "image"}` markers so mlx-vlm binds each image to the turn it arrived on.
Without them every image collapses onto the most recent user message and multi-turn
conversations with images break.

## Streamlit conventions in use

Targeting Streamlit 1.62. `use_container_width` is deprecated — use `width="stretch"`
or `width="content"`. Sidebar stats use a reserved `st.empty()` slot filled *after*
generation, otherwise the panel trails one interaction behind. Prefer native elements
over custom HTML/CSS. `st.chat_input(accept_file=...)` renders its attach control as a
`+`, not a paperclip — check `assets/screenshot-*.png` before describing any UI
affordance in prose or captions.

**A `key` does not survive a page switch — every sidebar widget needs
`persist_state="session"`.** Streamlit folds the active page's script hash into every
element id, "even if key_as_main_identity is specified"
(`streamlit/elements/lib/utils.py`), so `budget_mp` on Chat and `budget_mp` on Grounding
are two different widgets. `nmv/ui.py` is rendered from both pages, so without session
scope a page switch silently reset the image budget to the ceiling — a ~7x change in
prefill cost with nothing on screen to say so. The same rule covers the sliders
`_sampling_control` hides in deterministic mode, which were dropped on every toggle.
Anything new in that sidebar carries `persist_state="session"` too.

**The app is deliberately dark-only.** `.streamlit/config.toml` holds a single `[theme]`
block (Nord), and a `[theme]` with no `[theme.light]`/`[theme.dark]` siblings removes the
appearance switcher from the app menu entirely and ignores the viewer's system
preference. Nord is chosen for background luminance: uploads are mostly white document
scans and the grounding overlay is drawn on the full-resolution original, so `#2e3440`
keeps that near-white rectangle from reading as glare. Do not swap the accent back to a
teal — on a dark ground teal lands on `#2dd4bf`, which *is* `PALETTE[6]` in
`nmv/grounding.py`, so the chrome accent and a detection box would be the same colour.
Restart the server after editing the file; theme changes do not hot-reload.

For any non-trivial Streamlit work, invoke the `developing-with-streamlit` skill — it
routes to version-matched reference docs bundled inside the installed package.

## Measured baseline (M2 Max, 32 GB)

Load 3.4 s · prefill ~590 tok/s · decode 72–88 tok/s · peak 6.17 GB. Use
`assets/sample-invoice.png` to reproduce; its PAID stamp sits at `(820, 1180, 1140, 1400)`,
which grounding should hit within a few pixels.
