# Third-Party Notices

Tab Agent Pro is MIT-licensed (see `LICENSE`). It builds on the following
projects. Counts reflect `requirements.lock` (the exact shipped set).

All licenses below were verified from the installed distribution metadata
(`importlib.metadata`) on the reference environment, not from memory.

## Direct runtime dependencies

| Package | Version | License | Copyright |
|---|---|---|---|
| [basic-pitch](https://github.com/spotify/basic-pitch) | 0.4.0 | Apache-2.0 | Spotify AB |
| [demucs](https://github.com/facebookresearch/demucs) | 4.1.0 | MIT | Meta Platforms, Inc. |
| [torch](https://github.com/pytorch/pytorch) | 2.5.1 (CPU/GPU per install) | BSD-3-Clause | Meta Platforms, Inc. and contributors |
| [torchaudio](https://github.com/pytorch/audio) | 2.5.1 | BSD-2-Clause | Meta Platforms, Inc. and contributors |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | 1.31.1 | MIT | Microsoft Corporation |
| [librosa](https://github.com/librosa/librosa) | 0.10.2 | ISC | Brian McFee and contributors |
| [soundfile](https://github.com/bastibe/python-soundfile) | 0.12.1 | BSD-3-Clause | Bastian Bechtold |
| [scipy](https://github.com/scipy/scipy) | 1.17.1 | BSD-3-Clause | SciPy Developers |
| [numpy](https://github.com/numpy/numpy) | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | NumPy Developers |
| [pretty-midi](https://github.com/craffel/pretty-midi) | 0.2.10 | MIT | Colin Raffel |
| [note-seq](https://github.com/magenta/note-seq) | 0.0.5 | Apache-2.0 | Google LLC |
| [mir_eval](https://github.com/craffel/mir_eval) | 0.8.2 | MIT | Colin Raffel and contributors |
| [mido](https://github.com/mido/mido) | 1.3.3 | MIT | Ole Martin Bjørndalen |
| [pydub](https://github.com/jiaaro/pydub) | 0.25.1 | MIT | James Robert |
| [resampy](https://github.com/bmcfee/resampy) | 0.4.2 | ISC | Brian McFee |
| [numba](https://github.com/numba/numba) | 0.68.1 | BSD-2-Clause | Anaconda, Inc. |
| [gradio](https://github.com/gradio-app/gradio) | 6.31.0 | Apache-2.0 | Gradio Inc. |
| [gradio-client](https://github.com/gradio-app/gradio) | 2.7.2 | Apache-2.0 | Gradio Inc. |
| [fastapi](https://github.com/fastapi/fastapi) | 0.162.0 | MIT | Sebastián Ramírez |
| [uvicorn](https://github.com/encode/uvicorn) | 0.54.0 | BSD-3-Clause | Tom Christie |

## Copyleft / reciprocal licenses in the shipped set

Two shipped packages carry reciprocal licenses. Neither restricts distributing
Tab Agent Pro under MIT, provided they are used unmodified and, for the LGPL
library, dynamically linked — both conditions hold here.

| Package | License | How it is used | Obligation |
|---|---|---|---|
| [soxr](https://github.com/dofuuz/python-soxr) | **LGPL-2.1-or-later** | Dynamically loaded by `librosa` for resampling at audio-load time. Shipped as an unmodified binary wheel, never statically linked into our code and never modified. | Keep it replaceable (it is: a separate wheel, loaded at runtime) and retain this notice. |
| [tqdm](https://github.com/tqdm/tqdm) | **MPL-2.0** AND MIT | Progress bars in transcription/printing paths. Unmodified. | File-level copyleft only; modifications to tqdm files would need to be published — we make none. |

**No GPL, AGPL, SSPL or other network-copyleft license appears anywhere in the
shipped dependency tree** (`requirements.lock`). Verified by scanning the
license metadata of all installed distributions.

## Model weights and data

| Asset | License | Notes |
|---|---|---|
| Basic Pitch ONNX model | Apache-2.0 | Downloaded at install time with the `basic-pitch` wheel. |
| Demucs `htdemucs` weights | MIT | Downloaded by `demucs` on first use. |
| `examples/guitar_solo.wav`, `examples/bass_groove.wav` | MIT (this repository) | Synthesised for the test suite; no third-party audio. |

## NVIDIA CUDA redistributables (GPU installs only)

Installing `torch` from the default index pulls NVIDIA CUDA runtime wheels
(`nvidia-*`). These are distributed under the NVIDIA Software License Agreement
for CUDA Toolkit redistributables, which permits redistribution as part of an
application. They are not present in the CPU-only image built by the provided
`Dockerfile`.

## Regenerating this file

```bash
python - <<'PY'
import importlib.metadata as md
for dist in sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower()):
    m = dist.metadata
    print(m["Name"], dist.version, m.get("License-Expression") or m.get("License"))
PY
```

Anything new that appears with a GPL/AGPL/SSPL license is a release blocker —
`tools/check_vulns.py` covers vulnerabilities, and license review is part of the
dependency-refresh checklist in `docs/RUNBOOK.md`.
