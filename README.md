# north-micro-vision-studio

[![CI](https://github.com/darylalim/north-micro-vision-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/north-micro-vision-studio/actions/workflows/ci.yml)

Streamlit application for research and development using
[Cohere North Micro Vision](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct)
on Apple Silicon with MLX.

North Micro Vision is a 2.4B-parameter, Apache-2.0 vision-language model that reads
images at their **native resolution** rather than squashing them to a fixed square.
Its strengths are documents, charts, OCR and visual grounding — which is exactly the
workload where running locally beats shipping scans to an API.

This studio runs the **bf16** conversion only
(`mlx-community/North-Micro-Vision-Instruct-bf16`, ~5 GB), so results stay comparable
to Cohere's published numbers. Quantised conversions exist upstream if you ever need
the footprint down.

## Requirements

- Apple Silicon Mac. 16 GB unified memory is enough; peak usage is ~6.2 GB.
- Python 3.11–3.13 (the pinned venv uses 3.12).
- [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

The first launch pulls ~5 GB of weights into the Hugging Face cache. To fetch them
ahead of time:

```bash
uv run hf download mlx-community/North-Micro-Vision-Instruct-bf16
```

## The two pages

**Chat** — multi-turn document Q&A, OCR, charts and captioning. Attach images with the
paperclip; each one stays bound to the turn it arrived on, so a later question can
refer back to a page uploaded several turns earlier. Responses stream token by token.

**Grounding** — single-shot detection with a bounding-box overlay, a table of parsed
coordinates, and a PNG download. Turn on **Deterministic** for steadier boxes.

`assets/sample-invoice.png` is a synthetic page for exercising both.

## Things worth knowing

### Token count scales with pixel area

The encoder cuts an image into 16 px patches, merges each 2×2 block, and spends one
language-model token per merged block. One token therefore covers 32×32 px:

```
tokens = pixels / 1024
```

The **Image budget** slider is the single biggest lever on prefill latency and
KV-cache memory. Its ceiling is 3,868,706 px — 1654 × 2339, an A4 page at 200 dpi,
the largest resolution Cohere trained on, and mlx-vlm's own default cap.

Note that the checkpoint's `preprocessor_config.json` advertises a far larger
`size.longest_edge` of 16,777,216 px (16,384 tokens — double the 8K context the model
was actually validated at for multimodal input). mlx-vlm never reads it: its processor
only honours keys literally named `min_pixels`/`max_pixels`, which that file lacks. The
effective budget is the sane one.

| Input | Encoded | Tokens |
|---|---|---:|
| A4 scan @ 200 dpi | 1632 × 2336 | 3,723 |
| 1080p photo | 1920 × 1088 | 2,040 |
| 12 MP phone photo | 2240 × 1696 | 3,710 |

### Grounding coordinates are resolution independent

Boxes come back as `[x1, y1, x2, y2]` on a **0–1000** grid, not in pixels, so
`x_px = x / 1000 * width` against whichever rendering you like. That is why the overlay
can be drawn on your full-resolution upload even though the encoder saw a downscaled
copy — nothing has to be rescaled between the two.

Two traps this codebase handles and you should keep handling:

- **EXIF orientation.** Phone photos carry a rotation tag. Normalise it *before*
  encoding, and draw boxes on that same normalised image — otherwise the model reads a
  sideways page while the overlay is upright and every coordinate lands wrong.
- **Bare arrays.** Despite being asked for labelled JSON, the model often replies with
  a plain `[659, 671, 921, 799]`. `nmv/grounding.py` parses JSON, fenced blocks and
  loose prose, and treats labels as optional.

### MLX streams are thread-local — this shapes the whole architecture

mlx-vlm creates its generation stream at import time:

```python
# mlx_vlm/generate/common.py
generation_stream = mx.new_thread_local_stream(mx.default_device())
```

That stream belongs to whichever thread first imported the package. Streamlit runs
**every rerun on a fresh ScriptRunner thread**, so a naive app works exactly once and
then dies with:

```
RuntimeError: There is no Stream(gpu, 1) in current thread.
```

`nmv/runtime.py` fixes this by pinning the import *and* every subsequent MLX call to
one long-lived worker thread, streaming tokens back over a bounded queue. It also
serialises GPU work across browser tabs, which matters because MLX generation is not
reentrant.

**The invariant:** mlx-vlm is imported only inside worker functions, never at module
scope. `nmv/imaging.py` therefore mirrors mlx-vlm's `smart_resize` as plain arithmetic
instead of importing it; `tests/test_resize_parity.py` asserts the copy stays identical
to upstream.

### The model's own limits

- No system prompts — Cohere notes it was not trained to follow them, so the studio
  never sends one. Task instructions go in the user turn.
- Weak text-only reasoning (MMLU 0.504). No tool calling, no agentic workflows, limited
  maths and code.
- 8K context was validated for multimodal input, even though the LM backbone claims 128K.

## Layout

```
streamlit_app.py          entry point and router
app_pages/chat.py         multi-turn visual chat
app_pages/grounding.py    detection with box overlay
nmv/runtime.py            the MLX worker thread (read this first)
nmv/model.py              loading, sampling, streaming
nmv/imaging.py            token budgeting and resizing (no MLX import)
nmv/grounding.py          0–1000 box parsing and rendering
nmv/ui.py                 shared sidebar
tests/                    resize/token invariants, worker-abandonment regression
```

## Development

```bash
uv run python tests/test_resize_parity.py
uv run python tests/test_runtime_abandon.py
uv run ruff check .
uv run ruff format .
uv run ty check
```

## CI and releases

Every push and pull request runs the checks above, plus a grep for module-scope MLX
imports, on an Apple Silicon runner — `mlx` ships arm64 macOS wheels only, so there is
no Linux job. `uv sync --locked` fails the build if `uv.lock` has drifted from
`pyproject.toml`.

**Bumping `version` in `pyproject.toml` on `main` publishes a GitHub release.** Once CI
is green the workflow tags that commit, builds notes from the commit subjects since the
previous tag, and publishes — there is no draft to approve. It refuses a version that is
not valid PEP 440 or that is not ahead of the highest existing tag, and marks a
pre-release (`0.2.0rc1`, `0.3.0.dev1`) as such so it does not take the "Latest" badge.

## Measured on an M2 Max (32 GB)

| | |
|---|---|
| Weight load (cached) | ~3.4 s |
| Prefill | ~590 tok/s |
| Decode | 72–88 tok/s |
| Peak memory | 6.17 GB |

## License

The studio's own code is Apache-2.0 — see [`LICENSE`](LICENSE).

The weights are a separate matter, and none are vendored here: `uv sync` installs no
checkpoint, and the ~5 GB of weights arrive from the Hugging Face cache on first run.
Both [CohereLabs/North-Micro-Vision-Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct)
and the [mlx-community bf16 conversion](https://huggingface.co/mlx-community/North-Micro-Vision-Instruct-bf16)
this studio loads are Apache-2.0 as well, so the two halves line up — but they are
governed by their own model cards, not by the file above.

Apache-2.0 rather than MIT for the patent grant (§3) and, more to the point here, for
the explicit statement that no trademark rights travel with it (§6). *Cohere* and
*North Micro Vision* are Cohere's marks; this is an unaffiliated project that runs their
published checkpoint.
