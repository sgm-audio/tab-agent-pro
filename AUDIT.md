# Tab Agent Pro — Documentation & Claims Audit

**Subject:** `sgm-audio/tab-agent-pro` @ `c503e0b` (`fix: clear all 57 mypy errors on the hardened gate`)
**Audit date:** 2026-10-04
**Method:** every claim in `README.md`, `CHANGELOG.md`, `index.xml`, `pyproject.toml`, `Dockerfile`, `.env.example`, `run.sh`, CI/pre-commit config, and the module docstrings was checked against the code, against a
from-scratch install of the declared dependencies, against the public GitHub repo + CI history, and against the live
Hugging Face Space.

**Verdict key** — ✅ verified · 🟡 partially true / true with material caveats · ❌ contradicted by reality · ⚪ couldn't be verified from this environment

| Severity | Count | Meaning |
|---|---|---|
| **P0 — product-breaking** | 4 | A primary advertised workflow does not work as documented |
| **P1 — high** | 9 | A documented feature is inert, wrong, or silently degrades |
| **P2 — medium/minor** | 5 groups | Drift, stale metadata, doc nits |
| **Verified-accurate** | 16 | Claims that hold up under test |

**Headline:** the test suite, CI, linting, packaging metadata and the CLI are all in good shape and the "194
tests" badge is honest — but *both* advertised entry points are broken as shipped: the web UI's Transcribe
button raises on every click (**P0-1**), and the Docker/Space image installs Basic Pitch without two of its
required modules, so it cannot transcribe at all (**P0-2**). On top of that the container binds to
`127.0.0.1` (**P0-3**) and the "Zero GPU acceleration" claim is false in the code and in the deployment
(**P0-4**). The live Hugging Face demo is up, but it runs a stale revision of the project (**P2-3**), which is
why its button still works. Everything else is drift between docs and implementation, itemised below with
evidence and suggested fixes.

---

## 0. Environment used for verification

| Item | Value |
|---|---|
| Repo state | `c503e0b` on `arena/01a105b4-tab-agent-pro`; remote is public `github.com/sgm-audio/tab-agent-pro` (default `main`, pushed 2026-09-25T03:53:15Z) |
| Python | 3.11.2 (declared floor is 3.10) |
| Deps installed | `gradio 6.29.1`, `spaces 0.51.3`, `transformers 5.18.0`, `huggingface_hub 1.33.0` (also probed 2.1.1), `onnxruntime 1.30.0`, `basic-pitch 0.4.0` (installed `--no-deps`, as the Dockerfile does), `demucs 4.1.0`, `torch` (CPU), `librosa 0.10.2`, `note-seq 0.0.5`, `pretty-midi 0.2.10`, `setuptools 80.10.2` |
| Tooling | `ruff 0.11.5`, `black 26.5.1`, `mypy 2.4.0`, `bandit 1.9.4`, `pytest` (latest) — the same commands CI runs |
| Not available here | Docker/podman (no image build), Demucs weight download (`dl.fbaipublicfiles.com` blocked by sandbox TLS), HF Hub egress for the YourMT3+ checkpoint, HTTP POST to the live Space |

---

## 1. P0 — Primary advertised workflows broken

### P0-1 ❌ The Gradio web UI cannot transcribe anything (`README` → "Web UI (recommended)")

`app.py` wires the button to `_process_audio_impl` with **5 inputs**, but that function takes **6 required
parameters** (`... , progress`) — the `progress=gr.Progress()` default exists only on the unused
`process_audio()` wrappers:

```python
# app.py:437-441
transcribe_btn.click(
    fn=_process_audio_impl,                      # 6 params, progress has NO default
    inputs=[audio_input, instrument_type, export_midi, export_tab, export_json],   # 5 inputs
    outputs=[status_output, download_output],
)
```

Evidence (fresh install, `HOST=0.0.0.0 python app.py`, then the real Gradio HTTP API):

```
UserWarning: Expected 6 arguments for function _process_audio_impl, received 5.
$ python -c "import app; app._process_audio_impl('examples/guitar_solo.wav','Guitar',True,True,True)"
TypeError: _process_audio_impl() missing 1 required positional argument: 'progress'

# through Gradio's own API (exactly what the browser does):
$ gradio_client ... predict(handle_file("examples/guitar_solo.wav"), "Guitar", True, True, True,
                            api_name="/_process_audio_impl")
("❌ **Transcription failed:** 'NoneType' object is not callable", None)
```

The endpoint receives `None` for `progress` (Gradio supplies `None` for the parameter it cannot map), so the
exception surfaces at the first `progress(0.1, desc=...)` call. The button fails instantly, on every upload.
(The older build deployed on the Space wires the *wrapper* `process_audio`, which does have the default — see
P2-3 — so this is a regression introduced in the current revision, not a design limitation.)

**Why the tests missed it:** `tests/test_app_integration.py::test_process_audio_non_gpu_wrapper` supplies a
`MagicMock()` as the 6th argument and the `_run_mocked_pipeline` helper also passes a mock `progress`, so the
suite never reproduces the 5-argument call the UI makes.

**Resolve (either):**
```python
# app.py — preferred: make the UI entry point self-sufficient
def _process_audio_impl(..., progress=gr.Progress()):   # noqa: B008
    ...
```
or wire the button to `process_audio` (which has the default) and delete the duplicated wrapper bodies — but then
keep P0-4 in mind. Add a regression test that calls `_process_audio_impl` with **exactly five** arguments.

---

### P0-2 ❌ The Docker/Space image contains no working transcription backend (`Dockerfile`, `requirements.txt`, `CHANGELOG` → "Dockerfile with multi-stage build for Space deployment")

The Dockerfile installs Basic Pitch with `--no-deps`:

```dockerfile
# ponytail: basic-pitch declares tensorflow dep but uses ONNX on Linux
RUN pip install --no-deps basic-pitch>=0.4.0 && \
    grep -v '^basic-pitch' requirements.txt > /tmp/req-nobp.txt && \
    pip install -r /tmp/req-nobp.txt
```

Two independent problems:

1. **Shell redirection bug.** Unquoted `basic-pitch>=0.4.0` is parsed by `/bin/sh` as `pip install --no-deps
   basic-pitch` with stdout redirected to a file named `=0.4.0`. The version pin is silently discarded and a
   stray file is created. Reproduced: `sh -c "echo hello basic-pitch>=0.4.0"` creates `=0.4.0`.
2. **`--no-deps` drops two runtime-required packages that `requirements.txt` does not list.**
   `basic_pitch/note_creation.py` imports `mir_eval` **and** `resampy`; neither is a dependency of anything else
   in the file (verified with `importlib.metadata`: only `basic-pitch` requires them; `librosa` lists `resampy`
   under its *tests* extra only).

Reproduced with the exact Docker-style install:

```
$ python -c "import agents; print(agents.BASIC_PITCH_AVAILABLE)"      → False
$ python -c "from basic_pitch.inference import predict"               → ModuleNotFoundError: No module named 'mir_eval'
   (after installing mir_eval, the same import fails on 'resampy')
$ EarAgent(device='cpu', prefer_yourmt3=False).transcribe_stem('examples/guitar_solo.wav')
RuntimeError: No transcription models available for guitar_solo.wav. ...
```

`agents.py` catches the `ImportError` and silently degrades (`BASIC_PITCH_AVAILABLE = False`), so the container
starts, serves the UI, and fails only at the first transcription. **The README's Docker quick-start and the
"deployed on Spaces" promise cannot produce a single tab.**

**Resolve:**
```dockerfile
RUN pip install --no-deps "basic-pitch==0.4.0" "mir_eval>=0.6" "resampy>=0.2.2,<0.4.3" && \
    grep -v '^basic-pitch' requirements.txt > /tmp/req-nobp.txt && \
    pip install -r /tmp/req-nobp.txt
```
(or add `mir_eval` and `resampy<0.4.3` to `requirements.txt` next to the basic-pitch comment). Also add a
container smoke test: `python -c "import agents; assert agents.BASIC_PITCH_AVAILABLE"`.

---

### P0-3 ❌ The container binds `127.0.0.1`, so the Docker Space is unreachable from the Hugging Face proxy / any external client

```python
# app.py:468
uvicorn.run(create_app(), host=os.getenv("HOST", "127.0.0.1"), port=7860)
```
The Dockerfile never sets `HOST`, and `.env.example` ships `HOST=127.0.0.1` with the comment
*"Host binding (0.0.0.0 for Docker, 127.0.0.1 for local)"* — i.e. the documented intent contradicts the shipped
default. Docker Spaces must listen on `0.0.0.0` on port 7860
([HF Spaces docs](https://huggingface.co/docs/hub/spaces-config-reference); see also
[1](https://techjacksolutions.com/ai-tools/hugging-face/hugging-face-spaces/), [2](https://valemicolgarcia.github.io/software%20&%20backend%20engineering/2026/01/31/deploy-hugging-face-spaces.html)).

Verified locally: with defaults the server logs `Uvicorn running on http://127.0.0.1:7860`; with `HOST=0.0.0.0`
the socket binds `0.0.0.0:7860` and the app is reachable externally.

**Resolve:** `ENV HOST=0.0.0.0` in the Dockerfile (and/or default to `0.0.0.0` in `app.py`), and honor the
documented `PORT`:
```python
uvicorn.run(create_app(), host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "7860")))
```

---

### P0-4 ❌ "Zero GPU acceleration" is not real in the code, in the deployment, or in the library semantics

Claims: `README` feature list *"Zero GPU acceleration — HuggingFace Spaces optimization for faster CPU
inference"*; `CHANGELOG` *"Gradio web UI: HuggingFace Spaces-ready with Zero GPU acceleration"*; `app.py`
docstring *"Optimized for Zero GPU deployment"*; the live Space's own banner *"⚡ Zero GPU: Faster processing
with Hugging Face Zero GPU"*.

Reality, three ways:

1. `GPU_AVAILABLE` is set by *importing `spaces`*, not by detecting a GPU. On a CPU box it prints `True`.
2. **The decorated function is dead code** — the UI calls `_process_audio_impl`, never `process_audio`, so no
   Zero GPU request is ever made (see P0-1; fixing the wiring by calling `process_audio` would change this).
3. The live Space is on **`cpu-basic` hardware**, and `spaces` itself turns the decorator into a pass-through
   unless the Space is ZeroGPU:
   ```python
   # spaces/zero/decorator.py
   def _GPU(task, duration, size):
       if not Config.zero_gpu:   # SPACES_ZERO_GPU unset on cpu-basic
           return task           # ← plain function, no GPU, no error
   ```

**Resolve (pick one, then fix the docs):**
- **Make it true:** request `zero-a10g`/ZeroGPU hardware for the Space, wire the button to the decorated
  `process_audio`, and keep a defaulted `progress` parameter.
- **Or remove the claim:** delete "Zero GPU" from `README`/`CHANGELOG`/`app.py` docstring, drop the `spaces`
  dependency or gate it behind a real check (`os.getenv("SPACES_ZERO_GPU")`), and describe the app as CPU-based.

---

## 2. P1 — Documented features that are inert, wrong, or silently degrade

### P1-1 ❌ `validate.py` cannot do what its docstring says ("prove the project is complete and functional")

Run in a clean checkout with all dependencies installed:

```
$ python validate.py
$ echo $?
1                     #  ← and it printed NOTHING
```

Instrumented run (`import validate; validate.main()`) → `PASS: 89, FAIL: 3, SKIP: 1`:

```
❌ File exists: test_pipeline.py      # file actually lives at tests/test_pipeline.py
❌ Directory: input/                  # gitignored; cannot exist in a fresh clone
❌ Directory: output/                 # gitignored; cannot exist in a fresh clone
⏭  Docker build — install podman or docker
```

Additional defects in the same file:

* Results are collected into `RESULTS` and then discarded (`for _r in RESULTS: pass`); the summary is three
  `pass` statements, so the exit code carries no diagnostic information.
* ~15 checks are vacuous: `with contextlib.suppress(Exception): check("numpy", True)` "verifies" imports that are
  never performed; `check("monitoring.py imports", True)` is a constant; the gradio/monitoring "import" checks
  are inside `try` blocks that cannot raise.
* The ASCII-tab check `"3s" in content or "slide" in content` passes trivially because the legend line always
  contains "slide".
* The Docker check shells out to `podman` only, so a host with only Docker is silently "skipped".
* It is not run by CI, and `README` never mentions it.

**Resolve:** print `RESULTS`; fix `test_pipeline.py` → `tests/test_pipeline.py`; drop the `input/`/`output/`
directory checks (or `os.makedirs(..., exist_ok=True)` them); replace the fake import checks with real
`importlib.import_module` calls; try `docker` then `podman`; return a non-zero code only for real failures. Add
`python validate.py` (or its assertions) to CI.

### P1-2 ❌ `--profile` does nothing, and profile thresholds are ignored

`main.py` parses `--profile` (help text: *"Preset profile name (see init_memory.py --list)"*) and never
references `args.profile` again (`grep -n profile main.py` → only the parser line). `init_memory.py` says profiles
"configure tuning, thresholds, and processing behaviour", but `load_user_memory()` only feeds
`guitar_tuning` / `bass_tuning` / `num_frets` into the pipeline; `onset_threshold`, `frame_threshold`,
`suno_aggressive_mode`, `technique_sensitivity`, `bass_num_strings`, `guitar_num_strings` are read into `config`
and then ignored (`app.py` likewise never touches them).

**Resolve:** make `--profile` call `init_memory.save_profile()` and reload, or pass the profile file path
directly; apply `config["onset_threshold"]`/`config["frame_threshold"]` when the CLI flags were not explicitly
given (use `argparse` sentinel `default=None`); pass `technique_sensitivity` into `TabAgent.generate_tab`; or
narrow the profile docstring to what is actually honored.

### P1-3 ❌ REAPER settings are write-only; `index.xml` advertises controls that do not exist

`Settings.lua` persists `profile`, `instrument`, `export_midi`, `export_tab`, `export_json`, `install_path`.
`TabAgent.lua` reads **only `install_path`** (`grep -c` of each key in `TabAgent.lua` → 0 for all the others), and
it invokes `python3 main.py "<audio>"` with no flags, so every non-path setting silently does nothing.

`index.xml` goes further and claims the Settings script provides *"onset/frame threshold controls"* — the
Settings dialog has exactly six fields (profile number, install path, instrument, MIDI, Tab, JSON); no
thresholds exist there at all.

**Resolve:** have `Settings.lua` write a `user_preferences.json` (or pass `-i`, `--onset`, `--frame`,
`--no-tab`… on the command line) and have `TabAgent.lua` read it; drop the onset/frame claim from `index.xml`;
extend the Settings dialog if those controls are wanted.

### P1-4 ❌ `index.xml` deviates from the ReaPack index format and likely will not install

Compared with real indexes produced/consumed by ReaPack — `reapack.com/index.xml`,
`Audiokinetic/Reaper-Tools`, and `Bird-Bird/ReaScript_Testing` (which shows the canonical multi-file script
layout: `<source main="main">https://…/Script.lua</source>` plus
`<source file="libraries/gui.lua">https://…/libraries/gui.lua</source>`):

| Item | Reference format | This repo | Risk |
|---|---|---|---|
| Root version | `version="1"` | `version="1.0"` | Unknown/unsupported index version to ReaPack |
| Root `commit` | present (`commit="b6e5…"`, `commit="eec56…"`) | absent | No provenance; tooling expects it |
| Package element | `<reapack name="…" type="script" desc="…">` in current indexes | `<reaper name="…" type="script">` | `<reaper>` is the legacy element — verify against the ReaPack version you target; regenerating removes the doubt |
| `<category>` children | none | `<description>…</description>` | Not part of the format |
| `<version>` attrs | `name`, `author`, `time` | `name`, `author`, `type` (no `time`) | `time` is part of the format |
| `<source>` | attribute (`main`/`file`) = **installation target path**, body = **URL/path of the file to fetch** (`<source main="main">https://…/BirdBird_Envelope%20Palette.lua</source>`) | `file="reaper/TabAgent.lua"`, body `TabAgent.lua` | Looks **inverted**: ReaPack would fetch `<index-url>/TabAgent.lua`, which does not exist |

The index itself is present on `main` (2020 bytes, sha `d74e3c6`), so the README's step 2 URL resolves; the
failure would occur when ReaPack tries to download the script. "Install **Tab Agent** → Run script" is therefore
unverified and likely broken.

**Resolve:** regenerate the index with the official [`reapack-index`](https://github.com/cfillion/reapack-index)
tool (it validates, computes `commit`, `time` and commit-pinned source URLs), or hand-fix to
`<index version="1" name="Tab Agent Pro"> … <source file="reaper/TabAgent.lua">reaper/TabAgent.lua</source>` and
`<source file="reaper/Settings.lua">reaper/Settings.lua</source>`, then install once from ReaPack to confirm.
Note the repo is **not** listed on `reapack.com/repos` (that list is curated; importing the raw URL manually is
still supported — the README's step 2 is fine).

### P1-5 🟡 YourMT3+ is documented as the primary model, but is unreachable in the shipped configuration

`agents.py` module docstring: *"Modernized with YourMT3+ … Improvements from Basic Pitch: … Better pitch bend
detection … Multi-track simultaneous transcription"*; `EarAgent` docstring: *"Primary model: YourMT3+ (custom
Lightning) / Fallback: Basic Pitch"*; `index.xml`: *"Basic Pitch / YourMT3+ transcription"*.

Reality:

* The runtime stage of the Dockerfile installs `libsndfile1` and `ffmpeg` — **no `git`** — but
  `_clone_yourmt3_codebase()` shells out to `git clone`. In the container the code path always fails → fallback.
* `snapshot_download(repo_id="mimbres/YourMT3")` has no `allow_patterns`, and the repo is **2.7 GB with five
  checkpoints**, all of which would be fetched; the code comment claims "resume support" but
  `resume_download=True` is deprecated and ignored (`huggingface_hub 1.33.0` → warning; the parameter is absent
  from the signature in 2.1.1).
* The `http_401` branch in `_download_checkpoint()` is a no-op (`if http_401: pass`), so failures are silent.
* No test exercises a successful load (only `test_build_yourmt3_args_returns_namespace`), and the acquisition
  couldn't be completed here (HF egress blocked) — so the "primary" claim is unproven for the shipped setup.
* The model repo and the Space repo it clones do exist and do contain the expected checkpoint
  (`logs/2024/notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72/checkpoints/model.ckpt` — the exact `exp_id`
  hardcoded in `_build_yourmt3_args`) and `amt/src/{model/ymt3.py,model/init_train.py,utils/task_manager.py}`,
  so the approach is plausible — it is simply not the configuration that runs.

**Resolve:** either (a) install `git` in the runtime image, pin the checkpoint with `allow_patterns`, drop the
deprecated argument, set `SPACES_ZERO_GPU`-independent expectations in docs, and add an integration test behind
an env flag; or (b) demote YourMT3+ in all docs/`index.xml` and delete the dead acquisition path. Also gate the
2.7 GB download behind an explicit opt-in.

### P1-6 ❌ `transcribe_stem`'s contract contradicts its docstring (and the docs' "pipeline continues" promise)

Docstring: *"YourMT3+ → Basic Pitch → empty list (with warning, pipeline continues)"*. Code: when both models are
unavailable it logs an error and **raises `RuntimeError`**, aborting the run (observed in P0-2). Nothing returns
an empty list.

**Resolve:** align docstring and behaviour — either return `[]` and let callers decide (the app catches the
exception and reports a failure, which is arguably the better UX), or restate the docstring.

### P1-7 🟡 "Demucs isolates guitar/bass" overstates what the code does

`README`: *"Stem Separation | Demucs | Isolates guitar/bass from the mix"* and *"Demucs separates guitar/bass
even from complete songs"*. HTDemucs has no guitar stem: the code maps the `other` stem to `"guitar"` and only
uses `other` + `bass` (`_separate_with_api`, `stem_map = {"other": "other", "bass": "bass"}`). "Guitar" therefore
means "everything that is not drums, bass or vocals". Lead/rhythm splitting is a heuristic mid-side
subtraction (`mid=(L+R)/2`, `side = ch − 0.8·mid`), which for centre-panned or mono material produces a *quieter
copy* rather than a separated part — the repo's own example (mono) hits exactly this case.

Also verified: when separation fails the pipeline copies the raw mix into `other.wav` and `bass.wav`
(`_raw_audio_fallback`), so a mixed track is transcribed as if it were guitar **and** bass. That is documented
behaviour ("graceful fallback") and it works — but it is worth stating in the README, because the "isolates
guitar/bass" claim implies separation always happens.

**Resolve:** reword to "Demucs separates the mix; the residual 'other' stem (guitars and other instruments) is
processed as guitar"; document the mid-side heuristic and its limits; document the raw-audio fallback.

### P1-8 ❌ Packaging is broken: the built wheel contains zero modules

```
$ pip wheel . --no-deps      →  tab_agent_pro-1.0.0-py3-none-any.whl
files in wheel: dist-info/{METADATA,WHEEL,top_level.txt,RECORD}
python modules included: []      # the project ships an EMPTY package
```

`pyproject.toml` declares `[tool.setuptools.packages.find] include = ["*"]`, but this project is a *flat-layout*
collection of modules (`agents.py`, `app.py`, …), and `py-modules` is never declared. Nothing is installed by
`pip install .`.

Also in `pyproject.toml`:
* `setuptools<81` (required in `requirements.txt` and demonstrably necessary — see Verified list) is missing.
* `transformers>=4.48.0` is declared and described in `requirements.txt` as "YourMT3+ transformer model support",
  but `transformers` is **never imported** by the code (grep), and `agents.py` itself states YourMT3+ is *not* a
  transformers model. ~1 GB of dead dependency on the Space's cold start.
* `IPython` is pulled in by `note-seq`, not by this project.

**Resolve:** add `py-modules = ["agents","app","main","monitoring","suno_postprocessor","init_memory"]` (or
restructure into a package), mirror `setuptools<81` in `[project.dependencies]`, and drop `transformers`
(and `IPython` unless it's an intentional pin). Verify with `pip install .` + `python -c "import agents"`.

### P1-9 ❌ A single unplayable note silently discards an entire track's tablature

`TabAgent.generate_tab()` returns `[]` if **any** note has no valid string/fret position:

```python
if not all(layers):
    [i for i, layer in enumerate(layers) if not layer]   # dead expression; a warning was clearly intended
    return []
```
Evidence: notes `[E2, E4, B0]` on a 6-string → `tab entries out: 0`, `tab file written: False`. `export_tab_to_txt`
and `export_tab_to_json` then no-op, while the app's status message still reports success ("Files Generated").
The README promises "Assigns notes to strings/frets **optimally**" — one bad note (a detuned low note below the
tuning, or an out-of-range pitch) degrades that to "nothing".

**Resolve:** drop the unplayable note and flag it (log + a `warnings` field in the JSON), or clamp/transpose it,
and keep the rest of the tab. At minimum, surface the dropped-track condition in the status message.

---

## 3. P2 — Drift, stale metadata, and doc nits

### P2-1 ❌ `.env.example` is inert; three of its five variables are never read

No code loads `.env` (no `python-dotenv` dependency, no `load_dotenv()` call anywhere). Only `HOST` is read
(`app.py`). `PORT`, `CACHE_DIR`, `DEVICE`, `HF_TOKEN` appear **nowhere** in the source (grep over all `*.py`).
The header says *"Copy to .env if you need to override defaults"* — that mechanism does not exist.
`CACHE_DIR` also contradicts `EarAgent.YOURMT3_CACHE` (hardcoded `~/.cache/tab_agent/yourmt3`).

**Resolve:** wire them up (`python-dotenv`; `PORT` → uvicorn; `CACHE_DIR` → model caches; `DEVICE` →
`EarAgent(device=…)` + `SplitterAgent`) or delete them from `.env.example`.

### P2-2 🟡 Unbounded `gradio>=4.0.0` already emits deprecation warnings on the version it resolves to

With Gradio 6.29.1 (today's resolution of `gradio>=4.0.0`):
```
UserWarning: The parameters have been moved from the Blocks constructor to the launch() method in
Gradio 6.0: theme, css. Please pass these parameters to launch() instead.
UserWarning: Expected 6 arguments for function _process_audio_impl, received 5.
```
The documented theme/CSS silently do not apply, and `gr.mount_gradio_app` behaviour is version-sensitive.
**Resolve:** pin `gradio>=4.44,<7` (or whatever is actually tested) and add a UI smoke test; move `theme`/`css`
to the launch path for Gradio 6.

### P2-3 ❌ The live demo is a stale, different build of this project

Hugging Face API (`ScottyMills/tab-agent-pro`): `sdk=docker`, `runtime.stage=RUNNING`, hardware `cpu-basic`,
`sha=46027444…`, `lastModified=2026-06-07T21:14:23Z`. Its file list is
`Dockerfile, README.md, agents.py, app.py, main.py, requirements.txt, suno_postprocessor.py, .gitignore` —
i.e. **no `monitoring.py`, no `init_memory.py`, no `tests/`, no `reaper/`, no `index.xml`, no `validate.py`**.

The served app is a different revision: its `/config` reports Gradio `6.16.0` and a `process_audio` endpoint
(the current repo's button is `_process_audio_impl`), its labels are "Instrument Type" / "Export Options" /
"Transcribe to Tablature" (the repo's are "Instrument" / "Export formats" / "Transcribe"), and **`/health`
returns `{"detail":"Not Found"}`** while `README`/`CHANGELOG`/`Dockerfile` advertise health endpoints.

So: the README's *"Try the live demo"* link works and the Space is up, but it does not run this repository's
current code, and nothing in the repo deploys it. **Resolve:** either sync the Space from this repo (add
`monitoring.py` etc. and a deployment step/CI job) or state in the README which revision the demo runs.

### P2-4 🟡 CI/pre-commit drift

| Item | CI | pre-commit | Note |
|---|---|---|---|
| Linter | `ruff==0.11.5` | `ruff-pre-commit v0.15.20` | different versions |
| Formatter | `black --check` (26.5.1) | `ruff-format` | two formatters; both currently clean, but they can disagree |
| mypy | `mypy .` with `strict_optional = true` (pyproject) | `--no-strict-optional` | the "hardened gate" is relaxed locally |
| Unpinned tools | `mypy`, `pytest`, `pytest-cov`, `bandit` installed without versions | n/a | today mypy 2.4.0 passes, but the gate is not reproducible |
| Coverage | `--cov=.` includes `validate.py` (0 %) | n/a | dilutes the number |
| `validate.py` | not run | n/a | see P1-1 |

`continue-on-error: false` on the mypy step is a no-op (already the default). `bandit -x tests/` is combined with
an explicit file list, so `-x` is redundant. `.pre-commit-config.yaml`'s `exclude:` references `AUDIT.md` and
`PLAN.md`, neither of which exists in the repo (this report takes the name the config already anticipates).

**Resolve:** pin the tool versions in one place (`requirements-dev.txt` / `pyproject` extras) and use the same
versions in both places; align mypy flags; add `validate.py`/UI smoke to CI.

### P2-5 🟡 Miscellaneous code/doc statements that do not hold

* `main.py`: three no-op blocks at the end (`if not args.no_midi: pass`, ×3) where a summary was intended;
  `parse_args().print_help()` re-parses arguments (use `parser.print_help()`); when `audio` is missing the CLI
  exits 1 with **no message**; a comment says "Use CLI-provided thresholds (or defaults)" while profile
  thresholds are ignored (P1-2).
* `agents.py`: `if self.model is None and not BASIC_PITCH_AVAILABLE: pass` (a warning was intended);
  `_convert_to_noteseq` / `_parse_mt3_tokens` are dead code for the YourMT3+ path (only tests call them);
  `_filter_by_instrument_range` / `humanize_and_clean` have empty `if removed_count > 0: pass` blocks;
  the `# nosec B603 — hardcoded cmd, no user input` comment is inaccurate — `output_dir` and `audio_path` are
  interpolated (the list form is safe, but the justification is wrong).
* `monitoring.py`: `health` is hard-set to `"up"` at import (the status never reflects reality) and components
  start as `"unknown"`; the module docstring's example uses `app.get(...)` which matches `app.py` — that part is
  fine.
* `run.sh` is fully functional (`./run.sh --web`, `./run.sh song.wav`) but is never mentioned in the README.
* README "Profile presets — 8 tuning profiles (standard, drop D, classical, 4/5-string bass, etc.)": there are
  indeed 8 `PROFILES`, but two are Suno-cleanup profiles, not tunings — minor wording.
* The `CHANGELOG`'s 1.0.0 "stable release" is dated 2026-07-01 while the Space was created 2026-06-07 and the
  GitHub repo's single commit is later still — the project's version story is not reflected in history.
* `Dockerfile`: `pip install torchcodec … || true` swallows any failure (and torchcodec isn't needed by the
  pinned torchaudio); the builder installs `git` (needed for the YourMT3+ clone) but the runtime stage does not
  (P1-5).
* The ASCII tablature has two real defects (found by re-rendering the CLI output from its JSON):
  * **Adjacent 2-character columns run together** — `col[s].center(w)` pads with spaces but adds no separator,
    so consecutive notes on the same string render as one number: the shipped example prints
    `E|1215-  1922- …` (frets 12 and 15 in adjacent columns look like "1215") and `E|157p22h-|`
    (frets 15 and 7p/22h merge). A reader cannot reliably parse these.
  * **Notes are silently overwritten within a 50 ms group** when two notes share a string:
    `t=0.65 str4 fret15 pick` was replaced by `str4 fret8 pull`, and `t=1.9 str3 fret19` by `str3 fret12 pull`.
    The ASCII tab drops notes that the JSON export still contains, so the two promised formats disagree
    ("Column-aligned tablature … easy to read" is not currently met).

---

## 4. ✅ Claims that hold up (verified)

| # | Claim | Evidence |
|---|---|---|
| 1 | *README*: `tests/ # 194 tests` | Static count = 194 `def test_` across 13 modules; `pytest` → **194 passed, 18 warnings in 48.55s**, 85 % coverage |
| 2 | CI gates (lint, format, type-check, test, security scan) | `ruff check .` → All checks passed; `black --check .` → 20 files unchanged; `mypy .` (mypy 2.4.0) → Success, 20 files; `bandit -r …` → No issues identified (2336 LOC, 9 `#nosec`); GitHub Actions run `36092180974` on this commit → **success, every step green** |
| 3 | Repo/branch claims | `origin` = public `github.com/sgm-audio/tab-agent-pro`, default branch `main`, `index.xml` present on `main` |
| 4 | CLI works end-to-end | `python main.py examples/guitar_solo.wav -o /tmp/tabclirun` → MIDI + `.tab` + `.json` for lead/rhythm L/R/bass, processed WAV and stems; 17/15/15/7 notes transcribed by Basic Pitch |
| 5 | MIDI/Tab/JSON formats | Files valid; tab shows the documented `E|…|` grid, `Legend: s=slide, h=hammer-on, p=pull-off`, column-aligned chords (50 ms grouping) |
| 6 | Technique detection | `slide`/`hammer`/`pull` produced and annotated (`3s`, `4h`, `5p` verified in exports) |
| 7 | Demucs fallback is graceful | API failure → CLI failure → raw-audio fallback, exactly as documented (`demucs_raw_fallback` log; pipeline still produced output) |
| 8 | Health endpoints | `/health` and `/health/metrics` → 200 with the documented JSON shape (localhost; also on `0.0.0.0`) |
| 9 | Structured JSON logging | Every log line is a JSON record with `ts/level/module/event/uptime_s`; stage metrics emitted |
| 10 | 8 preset profiles | `init_memory.PROFILES` has 8 entries; `--list`, `--profile`, interactive selection all function |
| 11 | `setuptools<81` pin is necessary | On setuptools 84: `pretty_midi` → `ModuleNotFoundError: pkg_resources`; on 80.10.2 it imports (warns) |
| 12 | "Basic Pitch ONNX, no TensorFlow" (for Linux) | With `onnxruntime` present and TF absent, `basic_pitch.ICASSP_2022_MODEL_PATH` resolves to `…/nmp.onnx` and inference runs |
| 13 | `run.sh` behaviour | Executable, installs deps if missing, `--web` → `app.py`, otherwise → `main.py "$@"` |
| 14 | ReaPack files exist and match `index.xml` | `reaper/TabAgent.lua` (169 lines), `reaper/Settings.lua` (145), both referenced; XML parses |
| 15 | License/version consistency | MIT in `LICENSE` + both badges; version 1.0.0 agrees across `pyproject.toml`, `--version`, `CHANGELOG`, `index.xml` |
| 16 | Python 3.10+ | `requires-python = ">=3.10"`; no >3.10 syntax; CI runs 3.10, verified locally on 3.11 |

---

## 5. Caveats on this audit (what was *not* verifiable here)

1. **Docker image build** — neither `docker` nor `podman` exists in this sandbox, so the multi-stage build was
   reviewed statically and its pip steps reproduced in a venv, but not built. P0-2/P0-3 were reproduced at the
   dependency/binding level, not inside a container.
2. **Demucs separation quality** — weights download (`dl.fbaipublicfiles.com`) is blocked here, so stem quality
   was not assessed; only the fallback chain was exercised.
3. **YourMT3+ end-to-end** — `huggingface.co` egress is blocked, so a real load was impossible; the model/Space
   repository contents were verified through the HF API, and the code path was reviewed.
4. **Live Space transcription** — the sandbox cannot POST to `*.hf.space`; only GET `/`, `/config` and `/health`
   were checked. The deployed build predates the current repo (P2-3), so its behaviour says nothing about HEAD.
5. **`tests/test_benchmark.py`** is a benchmark in name only: it monkey-patches `transcribe_stem`, so it measures
   nothing; its docstring advertises `--benchmark`, a pytest-benchmark flag that is neither installed nor
   declared. It also never prints its results. Left as-is in this audit but should be fixed or removed.

---

## 6. Suggested remediation order

| Order | Action | Files | Effort |
|---|---|---|---|
| 1 | Default `progress=gr.Progress()` on `_process_audio_impl` **+ regression test with 5 args** | `app.py`, `tests/test_app_integration.py` | 15 min |
| 2 | Fix Basic Pitch deps (`mir_eval`, `resampy<0.4.3`) and quote the version pin | `Dockerfile`, `requirements.txt` | 15 min |
| 3 | Bind `0.0.0.0` (and honor `PORT`) | `app.py`, `Dockerfile`, `.env.example` | 10 min |
| 4 | Decide Zero GPU: make it real, or delete the claim everywhere | `README`, `CHANGELOG`, `app.py`, Space README | 1 h |
| 5 | Fix `validate.py` (report, paths, real imports, docker/podman) and add to CI | `validate.py`, `.github/workflows/ci.yml` | 1–2 h |
| 6 | Make `--profile`/profile thresholds actually apply | `main.py`, `app.py`, `init_memory.py` | 2 h |
| 7 | Wire REAPER settings into the pipeline (or trim docs/`index.xml`) | `reaper/*.lua`, `index.xml` | 2 h |
| 8 | Regenerate/repair `index.xml` and field-test one ReaPack install | `index.xml` | 1 h |
| 9 | Fix packaging (`py-modules`, `setuptools<81`, drop `transformers`) + `pip install .` smoke test | `pyproject.toml`, `requirements.txt` | 1 h |
| 10 | Replace synthetic `examples/*.wav` with real DI clips; re-tune the Suno detector so the demo isn't self-flagged | `examples/`, `suno_postprocessor.py` | 2 h |
| 11 | Sync the HF Space with the repo, or state which revision it runs | Space repo / CI | 1 h |
| 12 | Honour the remaining docstrings/`.env` items (P1-6, P1-9, P2-1, P2-2, P2-4, P2-5) | various | 3–4 h |

---

## Appendix A — exact commands used

```bash
# suite + gates (matches .github/workflows/ci.yml)
pip install -r requirements.txt            # via a fresh venv, plus torch/torchaudio/demucs
python -m pytest tests/ -q --no-header     # 194 passed in 48.55s, 85% coverage
ruff check . && black --check . && mypy . && bandit -r agents.py main.py app.py monitoring.py \
    suno_postprocessor.py init_memory.py -x tests/

# validate.py
python validate.py; echo $?                                   # 1, no output
python -c "import validate; validate.main(); print(validate.PASS, validate.FAIL, validate.SKIP)"

# web UI
HOST=0.0.0.0 python app.py                 # then GET /health, /health/metrics, /config
python -c "import app; app._process_audio_impl('examples/guitar_solo.wav','Guitar',True,True,True)"
                                           # TypeError: missing 1 required positional argument: 'progress'
python -c "from gradio_client import Client, handle_file; ..."
   # Client('http://127.0.0.1:7860/').predict(handle_file(...), 'Guitar', True, True, True,
   #                                          api_name='/_process_audio_impl')
   # → "❌ Transcription failed: 'NoneType' object is not callable"

# CLI end-to-end
python main.py examples/guitar_solo.wav --output-dir /tmp/tabclirun
ls /tmp/tabclirun                          # *.mid, *.tab, *.json × 4 parts + processed wav + stems/

# Docker-equivalent dependency check (this is what the image ships)
pip install --no-deps basic-pitch==0.4.0 && python -c "import agents; print(agents.BASIC_PITCH_AVAILABLE)"  # False
python -c "from basic_pitch.inference import predict"        # ModuleNotFoundError: mir_eval (then resampy)

# Suno detector on the shipped examples
python -c "from suno_postprocessor import process_suno_audio; print(process_suno_audio('examples/guitar_solo.wav'))"
                                           # is_suno=True (flatness 1.2e-8); bass_groove.wav → False

# packaging
pip wheel . --no-deps -w /tmp/w && unzip -l /tmp/w/*.whl     # wheel contains no .py modules

# external facts
gh repo view sgm-audio/tab-agent-pro; gh run view 36092180974     # public repo; CI run success
curl https://huggingface.co/api/spaces/ScottyMills/tab-agent-pro  # RUNNING, cpu-basic, sha 46027444
curl https://scottymills-tab-agent-pro.hf.space/health            # {"detail":"Not Found"}
python -c "import inspect,huggingface_hub as h; print('resume_download' in inspect.signature(h.snapshot_download).parameters)"  # False
```

## Appendix B — claim-by-claim status for README's top-level promises

| README claim | Status |
|---|---|
| "Upload guitar or bass audio. Get tablature, MIDI, and JSON back." | ❌ via the web UI (P0-1); ✅ via the CLI |
| "Works on anything from a clean DI track to a full mix with drums and vocals." | 🟡 depends on Demucs; degrades to whole-mix transcription on failure (P1-7) |
| Web UI recommendation + `python app.py` | 🟡 server runs, but transcription from the UI fails |
| Docker quick start | ❌ image cannot transcribe (P0-2) |
| Pipeline table: quality analysis → stems → transcription → tab → MIDI/Tab/JSON | ✅ (with wording fixes for peak separation) |
| "Basic Pitch … ONNX model, no GPU required" | ✅ on Linux with `onnxruntime`; ❌ as shipped in Docker (P0-2) |
| "Full-mix processing — Demucs separates guitar/bass" | 🟡 `other`/`bass` stems only (P1-7) |
| "Multi-track output — Lead, rhythm L/R, bass" | ✅ (CLI and app both emit these) |
| "Column-aligned tablature" | ✅ |
| "AI audio support — detection and cleanup" | ✅ mechanism exists; ❌ false-positives on the repo's own demo asset |
| "Technique detection" | ✅ |
| "REAPER integration via ReaPack" | 🟡 scripts exist; settings inert (P1-3) and `index.xml` likely won't install (P1-4) |
| "Zero GPU acceleration" | ❌ (P0-4) |
| "Profile presets — 8 tuning profiles" | 🟡 8 profiles exist; `--profile` and most fields are inert (P1-2) |
| "194 tests" | ✅ |
| CI badge | ✅ (badge target exists; run is green) |
| HF Space demo link | 🟡 link is live, but runs a stale/different revision (P2-3) |
| MIT license, `.env.example`, `run.sh` | ✅ license and `run.sh`; ❌ `.env.example` (P2-1) |

---

*No application code was modified by this audit; all fixes above are proposals. `AUDIT.md` already appears in
`.pre-commit-config.yaml`'s `exclude:` list, so this file will not trip the local hooks.*
