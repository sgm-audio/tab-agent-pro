# Tab Agent Pro — Independent Verification & Validation Report

**Subject:** `sgm-audio/tab-agent-pro` @ `a4814b2` ("refactor: remove Hugging Face, add local GPU, stream REAPER console")
**Branch audited:** `arena/91504cfe-tab-agent-pro` (base commit `a4814b2`, `main` at time of audit)
**Report date:** 2026-11-18
**Scope:** (1) *Verification* — does the code do what `README.md` and `AUDIT.md` claim? (2) *Validation* — does it actually produce playable guitar/bass tablature on real audio?

> **Note on a moving target.** While this audit ran, `main` advanced by one commit,
> `6d1dbd9` ("Audit and strengthen transcription and REAPER safety tests (#11)"), which
> fixes several of the gate failures below and *independently documents* the same core
> transcription defects found here (`docs/TRANSCRIPTION_TEST_AUDIT.md`). Every finding in
> this report was reproduced **locally against `a4814b2`**; where `6d1dbd9` changes the
> verdict, that is stated explicitly. Nothing below is copied from that commit.

---

## Reconciliation after PRs #20 and #22

This report records the original `a4814b2` audit and a #16-era follow-up. Its
measurements and verdicts are historical, not a new audit of current `main`.

- The chord regression checks the hard string-occupancy rule, pitch preservation
  and reporting of omissions. It covers all 300 dyads in MIDI pitches 40–64,
  63 selected triads, and one seven-note chord on six strings. It is not an
  exhaustive scan of the fretboard or a test of human playability.
- The CI harness drives the shipped Lua through Lupa with a real POSIX shell
  child. It verifies exact filename arguments, creation of a fresh run directory,
  and imports restricted to MIDI announced by a successful child in that
  directory. It does not exercise native REAPER, Windows or real inference.
- #22 already provides advisory ID/alias matching with expiry enforcement and
  regression tests. This reconciliation retains that implementation and the
  existing ReaPack index guard unchanged.

## Addendum (2026-11-18) — P0 remediation through PR #16

`#11`/`#12` addressed stale-MIDI import and command quoting; `#14` replaced the
per-note DP with `_assign_positions`, a beam search over per-string occupancy
states; and `#16` added chord-related costs (`CHORD_TOLERANCE`,
`SAME_STRING_CHORD_PENALTY`). The measurements in this addendum apply to #16's
code, not the later changes from #20.

| P0 | Original defect | Remediation | Regression on current `main` plus this reconciliation |
|---|---|---|---|
| **Chord collapse** | Independent note placement put simultaneous pitches on one string (375/990 tested dyads) | `_assign_positions` enforces per-string occupancy | `tests/test_tab_playability.py`, `tests/test_chord_playability_scan.py` |
| **Stale MIDI imported on failure** | `poll_pipeline` imported before checking exit status | Require a successful child, a same-run announcement and an output file in the fresh directory | `tools/check_reaper_safety.py` exercises failure with partial output, no output, stale files, and unannounced files |
| **Shell interpretation of filenames** | `quote()` did not pass all paths literally | POSIX single-quoting; Windows rejects ambiguous shell characters | `tools/check_reaper_safety.py` verifies literal POSIX arguments containing spaces, an apostrophe and variable syntax |

**Independent chord scan.** On #16, no simultaneous notes shared a string in
300 dyads or 63 selected triads. The solver kept both notes in 290 dyads and
reported the omitted pitch in each of the other 10. Four scan cases failed
when `_assign_positions`' occupancy guard was removed and passed when it was
restored. Setting `SAME_STRING_CHORD_PENALTY` to zero did not change the result:
the occupancy guard is the hard constraint, not the cost penalty.

**Fixture corrections.** An earlier simultaneous-note fixture used pitches
41 and 42, both available only on the low E string. It could not require both
notes to be kept. Another used E2 → A2, a five-fret leap, as a hammer-on
fixture. The two exact-optimum oracles also minimized over physically
infeasible assignments. The corrected oracles use `feasible_minimum` in
`tests/audio_fixtures.py` to restrict the comparison to feasible paths.

**Dependency scan failure.** The vulnerability gate failed during the audit
with identical dependency files because PyPI/OSV changed reported torch
advisory IDs from `PYSEC-*` to `GHSA-*`. The audit found that 26/26 advisories
for the locked `torch==2.5.1` matched accepted entries by alias. Earlier
passing results used dev `torch==2.15.0`, not the locked version. For the
subsequent repair and its tests, see [VULN_GATE_DIAGNOSIS.md](VULN_GATE_DIAGNOSIS.md).

**ReaPack correction.** The original sandbox was a shallow clone and could
not resolve the `index.xml` pin. After fetching history, both index tests
passed. The earlier 336-pass, two-failure run was therefore a local checkout
limitation; the subsequent run recorded 338 passes. The pin did not need
changing.

**Remaining evidence limits.** The scan verifies consistency of assigned
positions and explicit omissions, not Basic Pitch's transcription accuracy.
The historical recording of a six-note chord still yielded seven notes at
one onset. This audit did not establish reliable full-mix transcription or
human playability.

---

## 0. Method and environment

| Item | Value |
|---|---|
| Python | 3.11.2 |
| Install path | The project's own documented, CI/Dockerfile-matching path: `pip install --no-deps "basic-pitch>=0.4.0,<0.5"` then `pip install -r` (requirements minus basic-pitch) |
| Resolved versions | `basic-pitch 0.4.0`, `onnxruntime 1.32.0`, `demucs 4.1.0`, `torch 2.15.0` (CPU wheel), `torchaudio 2.11.1`, `gradio 6.32.0`, `librosa 0.10.2`, `note-seq 0.0.5`, `pretty-midi 0.2.10`, `setuptools 80.10.2` |
| CI toolchain | Reproduced exactly from `.github/workflows/ci.yml`: `ruff 0.11.5`, `black 26.5.1`, `mypy 2.4.0`, `bandit 1.9.4`, `pip-audit 2.10.1`, `pytest`, `pytest-cov`, `lupa` |
| Backends confirmed importable | `agents.BASIC_PITCH_AVAILABLE = True`, `agents.DEMUCS_API_AVAILABLE = True` |
| Blocked (sandbox egress) | `huggingface.co` and `dl.fbaipublicfiles.com` → **Demucs weights could not download**. Every run therefore exercised the documented *raw-audio passthrough* fallback. Stem-separation **quality** is consequently out of scope (see §6). |
| Not available | Docker/Podman (no image build), a real REAPER host, GPU hardware, a labelled multi-instrument corpus |

Original execution results and source inspections are summarised inline; commands
for the historical environment are in the appendix.

---

## 1. Headline verdict

| Question | Answer |
|---|---|
| Does the CLI produce tablature/MIDI/JSON from real audio? | **Yes.** End-to-end run on `examples/guitar_solo.wav` → 9 files, valid MIDI, in-range frets. |
| Does the web UI work? | **Yes.** Real HTTP round-trip via `gradio_client` → 14-file ZIP in 24.3 s. (This corrects AUDIT's original P0-1; that fix holds.) |
| Is the tablature **playable**? | **No, not reliably.** The "DP Viterbi" assigns **one note per onset layer**, so chords collapse onto a single string (§3.1). ~38 % of two-note chords and 100 % of a 5-note chord landed on one string. |
| Are techniques correctly annotated? | **No.** Legato labels are emitted from timing alone, ignoring note *end* times; two clearly separate picked notes 150 ms apart across silence are labelled `slide` (§3.2). |
| Is "full-mix processing" real? | **No as advertised.** `other`→guitar is honest in the code, but the shipped examples are 3-second **mono** files, and mid-side "lead/rhythm separation" is a mono fold-down that produces **bit-identical** L/R stems (§3.3). |
| Does REAPER integration pass settings through? | **Yes** (verified by executing the Lua). **But** it imports **stale MIDI on failure** and its command quoting permits **shell injection** from a filename (§4). |
| Is CI green at `a4814b2`? | **No.** 7 test failures, plus ruff/black/mypy failures under the exact pinned versions CI uses (§2.1). |
| Are the counts/metadata in the docs accurate? | **No.** Test count, lockfile size, advisory count all wrong (§2.3). |
| Does `AUDIT.md`'s "all fixed" appendix hold? | **Partially.** Most fixes are real; several specific claims in Appendix C are contradicted (§5). |

---

## 2. CI gates, tests, and repositories guards

### 2.1 Static gates (pinned CI versions)

| Gate | Command | Result |
|---|---|---|
| Ruff | `ruff check .` | **FAIL** — 8 errors: `agents.py:25` F401 `torch`, `agents.py:310/363/432` UP007, `agents.py:442` F401 `note_seq`, `app.py:17` I001, `device.py:30` F401 `logging`, `tests/test_hardening.py:17` F401 `os` |
| Black | `black --check .` | **FAIL** — would reformat `agents.py`, `app.py`, `suno_postprocessor.py` |
| mypy | `mypy .` | **FAIL** — 9 errors (`app.py` returns 3-tuples where `tuple[str, str \| None]` is declared, ×6; `tools/simulate_failures.py` references removed `EarAgent.model`, `.processor`, `_parse_mt3_tokens`) |
| Bandit | `bandit -r . -x ./tests,./.git,./.venv -ll` | **PASS** — 0 issues |
| pip-audit gate | `tools/check_vulns.py` | **PASS** — "167 packages scanned, 4 accepted, none unaccepted/expired" |

`README` claims the CI gates are "all blocking in CI" — they *are* configured as blocking, which means **CI cannot have been green on this commit**.

> **Superseded by `6d1dbd9`**, which fixes the ruff/black/mypy findings.

### 2.2 Test suite

```
$ python -m pytest tests/ -q --no-header
275 tests collected
7 failed, 268 passed — 93 % coverage (project-wide, --cov=.)
```

| Failure | Root cause | Real defect or stale test? |
|---|---|---|
| `test_tools.py::TestDockerInstallGuard::*` (×2) | `tools/check_docker_install.py` demands `RUN pip install torch` be index-pinned; the Dockerfile legitimately does `pip install torch==2.5.1 … --index-url "$TORCH_INDEX_URL"` | **Guard is a false negative** — CI would fail on a correct Dockerfile |
| `test_tools.py::TestEnvExampleGuard::*` (×2) | `.env.example` documents `CACHE_DIR`; nothing in `*.py`/`*.lua` reads it (only the Dockerfile sets it as a dead `ENV`) | **Real drift** |
| `test_tools.py::TestValidationSuite::test_validate_quick_passes` | `validate.py` asserts `reaper/TabAgent.lua` contains `os.execute(`, but the script was correctly migrated to `io.popen` + `reaper.defer` | **Guard is stale** — kills the CI "Run validation suite (quick)" step |
| `test_tools.py::TestReaPackIndex::*` (×2) | The shallow checkout could not resolve the `index.xml` pin `4061696b…` | **Incomplete checkout** (§4.4) |

### 2.3 Repository guard scripts (run individually)

| Script | README claim | Actual result |
|---|---|---|
| `tools/check_index.py` | "ReaPack index format" | **Initial FAIL** — pin unavailable in the shallow checkout; resolved after fetching history |
| `tools/check_env_example.py` | "every documented env var is read" | **FAIL** — `✗ CACHE_DIR (no reference in *.py, reaper/*.lua)` |
| `tools/check_docker_install.py` | "basic-pitch pin + required deps" | **FAIL** — false negative on the torch index line (see above) |
| `tools/check_lockfile.py --static` (CI) | — | **PASS** — "covers all 17 declared dependencies" |
| `tools/check_lockfile.py --verify` (CI) | "installed env == requirements.lock" | **FAIL** — `missing 4 shipped dependencies: … huggingface-hub==2.4.0`, `cuda-bindings`, `cuda-pathfinder`, `cuda-toolkit`; plus ~17 version drifts |
| `tools/check_vulns.py` | "advisories: none unaccepted/expired" | **PASS** |
| `tools/simulate_failures.py` | "failure modes still behave (15 checks)" | **FAIL** — `11/14`. Three simulations crash on removed API: `EarAgent` has no `model`/`processor`/`_parse_mt3_tokens`, `SplitterAgent` has no `device` |
| `validate.py` | "end-to-end checklist (PASS/FAIL summary)" | **PASS: 100 FAIL: 1 SKIP: 1**, exit code **1** (the stale `os.execute` check) |
| `tools/check_reaper_settings.py` | "REAPER settings reach main.py" | **PASS** — 15/15, executes the real Lua |

### 2.4 Stale counts in the docs

| Document | Claim | Measured |
|---|---|---|
| `README.md:88`, `:236` | "291 tests" | **275** collected (`268` `test_` functions) |
| `README.md:36`, `:241` | "118 distributions" / "118 exact pins" | **116** pins |
| `requirements.lock:21` | "131 packages" | **116** pins |
| `docs/ACCEPTED_RISKS.md:12` | "164 distributions" | **116** pins (167 scanned with tooling) |
| `docs/ACCEPTED_RISKS.md:13` | "70 advisories across 4 packages" | **66** entries in `.pip-audit-ignore` |
| `docs/ACCEPTED_RISKS.md:37` | "the 290-test suite" | **275** |
| `AUDIT.md` App. C (P2-2) | "290 tests" | **275** |

---

## 3. Critical flow — audio → stems → notes → tab → technique → export

### 3.1 `TabAgent.generate_tab` is **not** polyphonic — chords collapse onto one string 【P0】

**Claim:** *"DP Viterbi — Assigns notes to strings/frets **optimally**"* (README pipeline table); *"Viterbi-style DP for optimal fingering paths"* (`TabAgent` docstring).

**Evidence (`agents.py:682-714`).** The DP builds one "layer" per *note*, and the forward
pass computes `best_prev_idx` over `prev_layer` only. Nothing prevents the chosen path
from returning to the string the previous note already occupies — there is no
string-occupancy constraint within a simultaneous onset group. Rendering then joins the
colliding cells with `/` (`main.py:render_tab_text`), which *displays* the conflict
rather than resolving it.

**Reproduced:**

```
C-major voicing C3 E3 G3 C4 E4 (pitches 48,52,55,60,64) at one onset:
   {'string': 0, 'fret': 8}  {'string': 0, 'fret': 12}  {'string': 0, 'fret': 15}
   {'string': 0, 'fret': 20} {'string': 0, 'fret': 24}
   → all five notes on string 0; four labelled 'hammer'

Exhaustive two-note chord scan, 990 pairs playable on 6-string standard tuning:
   collapsed onto a single string: 375 / 990 (37.9 %)
```

**Real output from the shipped example** (`/tmp/out1/guitar_solo_lead_guitar.tab`):

```
B|-  -  15/19h 22 24 -      20 -  -
D|-  -  -      -  21 24/17p 21 -  -
```

`15/19h` is **two frets on one string in one 50 ms column** — physically unplayable, and the
README's own "Example tablature output" block advertises it. The bass run is worse:

```
D|9 -  14 7p - 7/2p/9h -  -  - -  - 16 9p/4p
```
`7/2p/9h` = three notes on one string inside one column.

**Verdict: Failed.** The optimisation is valid only for **monophonic** passages. For
chords the output is not playable tablature. *(Confirmed as an open P0 by `6d1dbd9`'s
`test_simultaneous_playable_chord_uses_distinct_strings`, marked strict-xfail.)*

### 3.2 Technique detection produces false positives 【P1】

**Claim:** *"Technique detection — Slides (`s`), hammer-ons (`h`), pull-offs (`p`) annotated in output."*

**Evidence (`agents.py:735-748`).** The annotation uses only `start_time`, `fret` and
`string`. Note *end* times (i.e. whether the previous note is actually still ringing) are
discarded at `agents.py:659` when notes are flattened to `{"pitch", "start"}`.

**Reproduced — two distinct attacks separated by silence:**

```
note A: fret 12, start 0.00, end 0.05   \
note B: fret 14, start 0.20, end 0.25   /  150 ms of silence between them
→ B labelled 'slide'

Same at 30 ms separation, 20 ms notes, no sustain overlap → 'slide'
```

A hammer-on or slide requires the first note to still be sounding. The implementation
cannot distinguish legato from separated picked notes. On the shipped example, the
`24/17p` pair is labelled `pull` between two notes with an **identical onset**
(`dt = 0.0000`).

**Verdict: Partial.** The mechanism exists and emits the documented `s`/`h`/`p` suffixes,
but the labels are not acoustically grounded.

### 3.3 Stem separation and "lead/rhythm" splitting 【P1】

**Claim:** *"Spatial Processing | Mid-Side | Splits lead (centre) from rhythm (sides) by subtraction"*; feature list *"Multi-track output — Lead guitar, rhythm L/R, bass in separate files"*.

**Evidence.**

1. **The shipped examples are mono.** `examples/guitar_solo.wav` and `bass_groove.wav` are
   both 3.00 s, 44.1 kHz, **1 channel**. `process_guitars` duplicates mono to stereo
   (`y = np.vstack((y, y))`), so `left == right`, hence `rhythm_l == rhythm_r`
   **bit-identically**:

   ```
   L == R (bit-identical)? True      corr(rhythmL, rhythmR) = 1.0
   rhythm peak / lead peak  = 0.20   (corr(lead, rhythmL) = 0.99999998)
   md5(guitar_solo_rhythm_L.mid) == md5(guitar_solo_rhythm_R.mid)
   ```

   The README's example is titled "three of the nine generated files omitted" —
   the omitted files are largely duplicates.

2. **Mid is not a centre extraction.** On a constructed stereo file (220 Hz centre,
   330 Hz hard left, 440 Hz hard right):

   | stem | 220 Hz (centre) | 330 Hz (left) | 440 Hz (right) |
   |---|---|---|---|
   | `lead` (mid) | 11025 | **5512** | **5512** |
   | `rhythm_L` | **2205** | 6615 | 4410 |
   | `rhythm_R` | **2205** | 4410 | 6615 |

   `mid = (L+R)/2` is a mono fold-down: the "lead" stem retains hard-panned content at
   −6 dB and the "rhythm" stems retain 20 % of the centre. This is a *documented formula*
   but not the "splits lead from rhythm" behaviour the pipeline table implies.

3. **Partial Demucs results are accepted.** When the API returns only `other` (no `bass`),
   `_separate_with_api` (`agents.py:163-171`) writes only what exists, then returns
   `{"guitar": …, "bass": <path that was never written>}` and
   `separate_stems` sets `health.set_component("demucs", "loaded")`:

   ```
   returned stems: {'guitar': '…/other.wav', 'bass': '…/bass.wav'}
     guitar: exists=True
     bass:   exists=False
   health demucs: loaded
   downstream:  bass: CRASH -> FileNotFoundError: …/bass.wav
   ```

**Verdict: Partial.** The `other`→guitar mapping and the raw-audio fallback are real and
graceful (both verified). The mid-side claim overstates what the code does, and the
multi-track output is duplicate content on mono input.

### 3.4 Skipped notes are recorded but never surfaced 【P1】

**Claim (`AUDIT.md` App. C, P1-5):** *"Notes outside the range are skipped, **counted in `last_skipped_notes` and reported**; regression test."*

**Evidence.** `TabAgent.last_skipped_notes` is set at `agents.py:668` and logged, but a
repo-wide grep shows the only non-test readers are the log line and `validate.py`.
Neither `main.py` nor `app.py` reads it; it is **not** written into the JSON export. The CLI
prints a warning only when *all* notes are unplayable (`main.py:519`, `:559`).

```
grep -rn "last_skipped_notes"  →  agents.py (defn+log), tests/, validate.py
                                   main.py: 0 hits    app.py: 0 hits
```

**Verdict: Partial.** The "one bad note doesn't kill the track" fix is real (verified:
mixed playable/unplayable input keeps the playable notes and returns them). The
"reported" half is docstring-only.

### 3.5 Export validity — the formats themselves are sound 【Verified】

Everything that *is* assigned is well-formed:

| Check | Result |
|---|---|
| `.tab` parses, has grid + legend + timestamp | **OK** |
| `.json` schema (`instrument`, `timestamp`, `num_strings`, `tablature[]{string,fret,technique,start_time}`) | **OK** |
| Frets within `0..num_frets`, strings within instrument width | **0 out-of-range entries** |
| `.mid` readable by `pretty_midi` | **OK** — 1 instrument, 16 notes, end 2.974 s |
| Chords column-aligned on 50 ms groups | **OK** |
| Adjacent-cell collision (`1215`) previously fixed | **OK** — `12 15` renders separated |

A caveat worth stating: `render_tab_text` **groups by time only**, so a long sustained
note is never carried into later columns. The grid shows *onsets*, not durations.

---

## 4. REAPER integration

### 4.1 Settings → CLI flags 【Verified】

`tools/check_reaper_settings.py` executes `reaper/TabAgent.lua` against a stubbed REAPER
API and asserts the built argv: **15/15 checks pass**, including `--profile`,
`--instrument`, `--onset`, `--frame`, `--no-midi`/`--no-tab`/`--no-json`, `--output-dir`,
path quoting, and blank-value omission. This genuinely repairs the original P1-3
("settings are write-only").

### 4.2 Stale MIDI is imported when the pipeline **fails** 【P0】

**Evidence (`reaper/TabAgent.lua`, `poll_pipeline`).** On stream end the script computes
`local imported = settings.export_midi and import_midi(basename, output_dir) or 0`
**before** examining `success`, then only shows the error box afterwards. `import_midi`
glob-matches `<basename>_*.mid` in a **shared** output directory it does not own.

**Reproduced** with a shim that makes `main.py` exit 1 while four same-basename MIDI files
from an earlier run already exist in the output dir:

```
pipeline exit: nonzero (sys.exit(1))
MIDI files imported on FAILURE: 4
    song_lead_guitar.mid  song_rhythm_L.mid  song_rhythm_R.mid  song_bass.mid
console: "Done — imported 4 MIDI track(s)."
message boxes: "Transcription failed."
```

The user is told the run failed *and* has four tracks added to the project from a
previous session. `Undo_EndBlock` records them as this run's output.

### 4.3 Command construction allows shell injection 【P0】

**Evidence (`reaper/TabAgent.lua`, `quote()` + `string.format('cd %s && %s %s 2>&1', …)`).**
`quote` escapes `"` only. The command is then handed to `io.popen`, which on POSIX runs
`/bin/sh -c`, where `$(…)`, backticks and `$VAR` are expanded **inside double quotes**.
The audio path comes from `GetMediaSourceFileName` — i.e. a filename the user may not
control (a downloaded or collaborative project).

**Observed on the audited revision:** a local fixture confirmed that the shell
interpreted filename text rather than passing it to the child literally.
The current regression checks that boundary using inert local filenames.

### 4.4 ReaPack pin looked absent in the shallow checkout 【Corrected】

`git cat-file -t 4061696b` failed in the original shallow clone, causing
`check_index.py` and two tests to fail. The initial report inferred that all
ReaPack source URLs would fail. That was incorrect: after fetching history the
pin resolved, and both tests passed. The local failure did not establish that
the source URLs were broken.

### 4.5 Path resolution 【Verified】
`find_install_path` checks `GetExtState`, then five candidate locations, and validates
`<path>/main.py` exists. Missing item / non-file-backed item / no-take all produce
actionable message boxes before any shell call. Confirmed by executing the Lua.

---

## 5. `AUDIT.md` Appendix C — spot-check of "all fixed" claims

| Original finding | App. C claim | Independently observed | Verdict |
|---|---|---|---|
| P0-1 UI button raises | ✅ Fixed | Real HTTP round-trip returns 14-file ZIP in 24.3 s | **Verified** |
| P0-2 Docker cannot transcribe | ✅ Fixed | `mir_eval`/`resampy` pinned and present; `BASIC_PITCH_AVAILABLE=True` on the Docker-matching install | **Verified** |
| P0-3 binds 127.0.0.1 | ✅ Fixed | `serve()` defaults `HOST=0.0.0.0`, honours `PORT`; Dockerfile `ENV HOST/PORT`; served on `0.0.0.0:7860` | **Verified** |
| P0-4 "Zero GPU" false | ✅ Fixed | No `spaces` dependency anywhere; `README`/`CHANGELOG` no longer claim it | **Verified** |
| P1-1 no `/ready` | ✅ Fixed | `/health` 200 + components, `/ready` 200/503 on `basic_pitch`, `/health/metrics` present | **Verified** |
| P1-2 `--profile` inert | ✅ Fixed | `--profile bass_4_string` → 4-string EADG tab, thresholds `0.55/0.35`, config persisted | **Verified** |
| P1-3 REAPER settings inert | ✅ Fixed | 15/15 by executing the Lua | **Verified** |
| P1-4 `index.xml` malformed | ✅ Fixed | Structure is right; the pin resolves after fetching history (§4.4) | **Partial** (no native host) |
| P1-5 one bad note kills track | ✅ Fixed | Playable notes survive; `last_skipped_notes` populated — but "**reported**" is false (§3.4) | **Partial** |
| P1-6 tab cells collapse | ✅ Fixed | `12 15` no longer `1215`; but same-string collisions still render (`15/19h`) | **Partial** |
| P1-7 full-mix only via Demucs | 🟡 Documented | README now says "residual `other` stem"; limitation stated | **Verified (as documented)** |
| P1-8 `validate.py` unusable | ✅ Fixed | Prints PASS/FAIL; **exits 1** on the stale `os.execute` check; CI step would fail | **Partial** |
| P1-9 wheel ships no modules | ✅ Fixed | `pip wheel .` → 7 modules (`agents, app, init_memory, main, monitoring, suno_postprocessor, validate`) | **Verified** |
| P2-1 `.env.example` inert | ✅ Fixed | 10 of 11 read; **`CACHE_DIR` unread**; guard fails | **Partial** |
| P2-2 stale counts | ✅ Fixed | 275 ≠ "290"/"291"; lock 116 ≠ "118"; advisories 66 ≠ "70" | **Failed** |
| P2-3 live Space stale | 🟡 Process | Not re-tested (out of scope / no credentials) | **Unverifiable** |
| N-1 health lies | ✅ Fixed | Forced-no-backend yields `down` + `/ready` 503; status derived from components | **Verified** |
| N-3 YourMT3 checkpoint pin | ✅ Fixed | YourMT3+ path deleted entirely — the finding is moot, not fixed as described | **Verified (moot)** |
| N-7 lockfile + CVE gate | ✅ Fixed | `check_vulns` passes; **but** `check_lockfile --verify` fails (missing `huggingface-hub`) | **Partial** |

Separately: `AUDIT.md` (top) says the original 194-test claim was "honest" and gives
`194 passed, 85 % coverage` at `c503e0b`; that is a historical, self-consistent record
and is not contradicted here.

---

## 6. Claims that could not be verified in this environment

| Claim | Why unverifiable | What *would* settle it |
|---|---|---|
| "Works on anything from a clean DI track to a full mix with drums and vocals" | No Demucs weights (egress blocked); both examples are short mono clips | A labelled multitrack corpus, real Demucs, per-track pitch/onset scoring |
| Stem-separation *quality* | Same | Same. Only the fallback chain was exercised. |
| GPU acceleration on real hardware (`cuda`/`rocm`/`vulkan`/`mps`/`directml`) | No GPU; only CPU-wheels installed | On-device run asserting the ONNX provider actually *executed* |
| "ReaPack install → run" end-to-end | No REAPER host | Real REAPER in a disposable project |
| Docker image builds & healthcheck passes | No Docker/Podman | `docker build` + `/ready` probe |
| Live HF Space behaviour | No Space credentials | — |

**Design note (verified by reading + local state):** the ONNX provider override in
`device.install_onnx_provider_override` patches `ort.InferenceSession` *before* Basic
Pitch constructs its session (`basic_pitch/inference.py:132` calls
`ort.InferenceSession(...)` at instantiation time), so the patch is correctly scoped. The
CPU fallback path was exercised for all seven `DEVICE` values and logs
`device_unavailable` as documented.

---

## 7. Verification Matrix

Legend: **Verified** = executed and matches the claim · **Partial** = mechanism exists but
behaviour falls short of the wording · **Failed** = contradicted by execution ·
**Unverifiable** = no reproducible evidence available here.

### 7.1 Transcription pipeline

| # | Claim (source) | Evidence | Result |
|---|---|---|---|
| 1 | Audio → stems → MIDI → tab → MIDI/Tab/JSON export (README pipeline) | `main.py` run on `examples/guitar_solo.wav` → 9 files, exit 0 | **Verified** |
| 2 | Basic Pitch ONNX transcription, no TensorFlow (README) | `basic_pitch` resolved to `…/icassp_2022/nmp.onnx`; `BASIC_PITCH_AVAILABLE=True`; 16 notes in 0.13 s | **Verified** |
| 3 | Demucs splits mix into vocals/drums/bass/other; `bass`→bass, `other`→guitar (README) | `_separate_with_api` `stem_map = {"other","bass"}`; README wording corrected | **Verified** |
| 4 | **"Full-mix processing" / "works on a full mix"** (README top) | No weights here → passthrough only; shipped examples are 3 s **mono** | **Unverifiable** |
| 5 | Demucs failure degrades gracefully, pipeline completes (README note) | `demucs_raw_fallback` logged after API **and** CLI failure; run still produced tab/MIDI/JSON | **Verified** |
| 6 | Partial Demucs result is handled | Returns nonexistent `bass.wav`, sets health `loaded`, downstream `FileNotFoundError` | **Failed** |
| 7 | Mid-side "splits lead (centre) from rhythm (sides)" (README table) | `mid=(L+R)/2` retains hard-panned content at −6 dB; rhythm keeps 20 % centre | **Partial** |
| 8 | Multi-track output: Lead, Rhythm L/R, bass (README feature) | Files emitted; on mono input L/R are **bit-identical** (same MIDI md5) | **Partial** |
| 9 | **DP Viterbi optimal string/fret assignment** (README table) | Monophonic passages optimal; 37.9 % of 2-note chords collapse to one string; 5-note chord → 1 string | **Failed** |
| 10 | Unplayable notes skipped, not whole track (README table) | Mixed `[playable, 25-invalid, playable]` keeps playables; `last_skipped_notes=[25]` | **Verified** |
| 11 | Skipped notes are *reported* (AUDIT App. C) | No reader in `main.py`/`app.py`; absent from JSON | **Failed** |
| 12 | Technique detection: slides / hammer-ons / pull-offs (README feature) | `s`/`h`/`p` emitted; but separate picked notes 150 ms apart across silence → `slide` | **Partial** |
| 13 | Column-aligned tab, chords grouped, `-` for unused strings (README) | 50 ms grouping and padding verified; same-string collisions still rendered `15/19h` | **Partial** |
| 14 | Legend line present (README example) | `Legend: s=slide, h=hammer-on, p=pull-off` in every `.tab` | **Verified** |
| 15 | README example is "verbatim" output | Content matches line-for-line, trailing spaces trimmed | **Partial** |
| 16 | MIDI output is valid and importable | `pretty_midi` parses: 1 instrument, 16 notes, end 2.974 s, velocities 40–96 | **Verified** |
| 17 | JSON schema (`string`, `fret`, `technique`, `start_time`) | 16/15/15 entries across three tracks, all in range | **Verified** |
| 18 | Correct frets/strings, no silent drops in export | 0 out-of-range entries; **but** dropped notes (see #11) and `\`/`\`` collisions are silent in the files themselves | **Partial** |

### 7.2 Configuration, profiles, devices

| # | Claim | Evidence | Result |
|---|---|---|---|
| 19 | 8 preset profiles (README) | `init_memory.PROFILES` = 8; `--list` renders all | **Verified** |
| 20 | `--profile` sets tuning/thresholds (README) | `bass_4_string` → 4 strings EADG, onset 0.55 / frame 0.35, persisted to `user_preferences.json` | **Verified** |
| 21 | Threshold precedence CLI → profile → default (README) | `resolve_thresholds` with `onset=None` reads profile; explicit flag wins; validated to `[0,1]`, NaN/inf rejected | **Verified** |
| 22 | `--instrument Bass/Guitar/All`, case-insensitive | `_normalize_instrument`; bass run produced only bass artefacts | **Verified** |
| 23 | `--no-midi / --no-tab / --no-json` | `_enabled_formats`; REAPER guard asserts the flags | **Verified** |
| 24 | CPU default; `DEVICE` = auto/cpu/cuda/rocm/vulkan/mps/directml | All 7 resolved; unavailable backends fall back to CPU with `device_unavailable` warning | **Verified** |
| 25 | `DEVICE` GPU backends actually accelerate | No GPU hardware available | **Unverifiable** |
| 26 | `.env` read by `app.py`, `main.py`, `run.sh` | `_load_dotenv` in both modules | **Verified** |
| 27 | Every documented env var is read | `CACHE_DIR` is read by **no** Python or Lua code | **Failed** |
| 28 | `CACHE_DIR` = model cache for Basic Pitch + Demucs | ONNX model lives in `site-packages`; Demucs uses `TORCH_HOME` | **Failed** |

### 7.3 Web UI / service

| # | Claim | Evidence | Result |
|---|---|---|---|
| 29 | `python app.py` serves the UI on `HOST`/`PORT`, default `0.0.0.0:7860` | Bound `0.0.0.0:7860`; `/` served | **Verified** |
| 30 | UI transcribes end-to-end | `gradio_client` `/process_audio` → "Transcription Complete", **14 files**, 24.3 s | **Verified** |
| 31 | UI limits: 50 MB, 5 min, `.wav/.mp3/.flac/.ogg/.m4a/.aiff` | 51 MB → rejected; 400 s → rejected; `.mp4` → rejected | **Verified** |
| 32 | CLI has no such limits | No size/duration logic in `main.py` | **Verified** |
| 33 | Results returned as a ZIP | `guitar_solo_tablature.zip` contiguously containing mid/tab/json + stems | **Verified** |
| 34 | `/health` liveness + component detail, `/ready` 503 until backend loads, `/health/metrics` | 200 `{status:"up", ready:true, components:{…}}`; `/ready` `{ready:true, components:{basic_pitch:true}}`; metrics `{}` → populated after a run | **Verified** |
| 35 | Health status derived, not hardcoded | `effective_status()` from `REQUIRED_FOR_TRANSCRIPTION`; `demucs` degradable only | **Verified** |
| 36 | Bounded queue via `MAX_CONCURRENT_REQUESTS` / `QUEUE_MAX_SIZE` | `demo.queue(default_concurrency_limit=…, max_size=…)` | **Verified** |
| 37 | `UPLOAD_TTL_SEC` pruning | `cleanup_stale_uploads` on each request | **Verified** |
| 38 | Structured JSON logging | Every record has `ts/level/module/event/uptime_s` | **Verified** |
| 39 | Per-request isolated output dirs | `tempfile.mkdtemp(prefix="session_")` | **Verified** |

### 7.4 REAPER

| # | Claim | Evidence | Result |
|---|---|---|---|
| 40 | ReaPack index importable, sources resolve | Guard + 2 tests initially failed in the shallow clone; the pin resolved after fetching history | **Failed** on the initial checkout |
| 41 | Settings script values passed to `main.py` as CLI flags | Lua executed against stub: 15/15 | **Verified** |
| 42 | Thresholds/export toggles reach the CLI | `--onset/--frame/--no-*` asserted from the built argv | **Verified** |
| 43 | Install-path autodetect + actionable errors | 5 candidates + `main.py` validation; MB on missing/non-file item | **Verified** |
| 44 | Progress streamed to console, REAPER stays responsive | `io.popen` + `reaper.defer` polling | **Verified** |
| 45 | MIDI imported to new tracks on success | `InsertMedia` + track naming; import count reported | **Verified** |
| 46 | A failed run must not add tracks | **4 stale MIDI files imported** on exit 1, reported as "Done — imported 4" | **Failed** |
| 47 | Filenames/paths passed literally | Shell interpreted filename text rather than passing it literally | **Failed** |
| 48 | Works inside real REAPER | No host available | **Unverifiable** |

### 7.5 Build, CI, packaging, security

| # | Claim | Evidence | Result |
|---|---|---|---|
| 49 | `pytest tests/` "291 tests, 91 % coverage" | 275 collected; 93 % project-wide, 7 failing | **Failed** |
| 50 | CI gates blocking and green | ruff 8 ✗, black 3 files ✗, mypy 9 ✗, index ✗, env ✗, docker-guard ✗, validate ✗, lockfile --verify ✗ | **Failed** |
| 51 | `validate.py` end-to-end checklist | PASS 100 / FAIL 1 / SKIP 1, **exit 1** | **Partial** |
| 52 | `tools/simulate_failures.py` — 15 checks | 11/14; 3 harnesses call removed APIs | **Failed** |
| 53 | `requirements.lock` reproducible, CPU-only, 118 pins | 116 pins; `--verify` fails (missing `huggingface-hub`); counting the declared install pulls 15 NVIDIA CUDA wheels unless torch is index-pinned | **Partial** |
| 54 | `pip install .` installs the app | wheel contains 7 modules | **Verified** |
| 55 | Docker: non-root, read-only code, `HOST=0.0.0.0`, `/ready` healthcheck | `USER tabagent`, `chmod -R a-w`, `ENV HOST/PORT`, `HEALTHCHECK … /ready`; `RUN python -c "import agents; assert BASIC_PITCH_AVAILABLE"` after `USER` | **Verified** (static) |
| 56 | Docker image builds and serves | No Docker/Podman | **Unverifiable** |
| 57 | `bandit` SAST clean | 0 issues at `-ll` | **Verified** |
| 58 | `pip-audit` gate with expiring acceptances | 167 packages, 4 accepted, none expired | **Verified** |
| 59 | `docs/ACCEPTED_RISKS.md` numbers accurate | "164 distributions", "70 advisories" → 116 / 66 | **Failed** |
| 60 | MIT licence, version 1.1.0 consistent | `LICENSE`, `pyproject`, `--version`, `CHANGELOG`, badge all agree | **Verified** |

**Tally:** 36 Verified · 8 Partial · 12 Failed · 4 Unverifiable (60 original checks, including the shallow-clone failure corrected in §4.4).

---

## 8. Do the artefacts actually hold up?

**`.mid` — yes.** Parses in `pretty_midi`, correct pitches/timings/velocities, importable
into any DAW. Note durations come from Basic Pitch, so sustain is meaningful here even
though the tab grid discards it.

**`.tab` — no, not as playable tablature.** Three independent reasons:

1. **Unplayable chords.** `/`-joined cells put two (or three) frets on one string in one
   column (`15/19h`, `7/2p/9h`). A player cannot fret two positions on one string at once.
2. **Wrong legato labels.** `p`/`h`/`s` are asserted from timing alone; separated picked
   notes get `slide`.
3. **Duplicated tracks.** On mono material `rhythm_L` and `rhythm_R` are identical; on
   stereo they are each 20 % centre + one side, i.e. the same guitar twice.

**`.json` — schema is fine, content inherits both defects**, and it omits any record of
notes dropped as unplayable, so a consumer cannot tell a partial transcription from a
complete one.

**Silent note drops:** yes, two kinds — (a) notes outside the fret range are logged but
never surfaced to the user or the JSON; (b) within a 50 ms group on the same string, extra
notes are joined with `/` rather than being resolved or reported as a conflict.

---

## 9. Recommended remediation order

| # | Action | Files | Severity |
|---|---|---|---|
| 1 | Add an onset-group/chord constraint to the DP: assign one note per string per time group (or an optimal chord voicing), and refuse/report rather than emit `/`-cells | `agents.py` `generate_tab` | **P0** |
| 2 | Never import MIDI on a failed run; delete/ignore pre-existing same-basename files; import only files this run produced | `reaper/TabAgent.lua` | **P0** |
| 3 | Pass argv as an array (or single-quote-escape for POSIX / use `reaper.ExecProcess` with a quoted list); stop shelling through `/bin/sh` | `reaper/TabAgent.lua` | **P0** |
| 4 | Make technique labels pessimistic: require the previous note to still be ringing (`end_time` overlap) and same-string; otherwise `pick`/unknown | `agents.py` `generate_tab` | P1 |
| 5 | Surface `last_skipped_notes` (and same-string conflicts) in CLI output, UI status and JSON (`"warnings": [...]`) | `main.py`, `app.py` | P1 |
| 6 | Validate Demucs output: if a required stem is missing, fail over to passthrough instead of returning a phantom path and marking `loaded` | `agents.py` `_separate_with_api` | P1 |
| 7 | Fix the repo guards: `os.execute`→`io.popen` in `validate.py`; accept `--index-url` torch form in `check_docker_install.py`; drop or wire `CACHE_DIR` | `validate.py`, `tools/*`, `.env.example` | P1 |
| 8 | Retracted: fetching history resolved the existing `index.xml` pin (§4.4) | `index.xml` | — |
| 9 | Refresh `requirements.lock` (`huggingface-hub`, CUDA-free closure) and update every count in README/AUDIT/ACCEPTED_RISKS from a single source | `requirements.lock`, docs | P2 |
| 10 | Repair the three broken simulations in `tools/simulate_failures.py` (`EarAgent.model/_parse_mt3_tokens`, `SplitterAgent.device`) | `tools/simulate_failures.py` | P2 |
| 11 | Ship a stereo example and a genuine multi-instrument mix so "full mix" can be exercised in CI | `examples/` | P2 |
| 12 | Consider defaulting `python main.py` to run **both** instruments, matching the README's "multi-track output" promise (currently guitar-only unless `--instrument All`) | `main.py` | P2 |

---

## Appendix — reproduction

```bash
# environment (the project's own supported install path)
python -m venv /tmp/venv
/tmp/venv/bin/pip install --no-deps "basic-pitch>=0.4.0,<0.5"
grep -v '^basic-pitch' requirements.txt > /tmp/req-nobp.txt
/tmp/venv/bin/pip install -r /tmp/req-nobp.txt
/tmp/venv/bin/pip install pytest pytest-cov lupa "ruff==0.11.5" "black==26.5.1" \
                        "mypy==2.4.0" "bandit==1.9.4" "pip-audit==2.10.1"

# gates
/tmp/venv/bin/ruff check . && /tmp/venv/bin/black --check . && /tmp/venv/bin/mypy .
/tmp/venv/bin/python -m pytest tests/ -q --no-header          # 7 failed, 268 passed
/tmp/venv/bin/python validate.py; echo $?                     # 1
for t in check_index check_env_example check_docker_install; do /tmp/venv/bin/python tools/$t.py; done
/tmp/venv/bin/python tools/simulate_failures.py                # 11/14
/tmp/venv/bin/python tools/check_reaper_settings.py            # 15/15
/tmp/venv/bin/python tools/check_vulns.py                      # pass

# end-to-end CLI
/tmp/venv/bin/python main.py examples/guitar_solo.wav --output-dir /tmp/out1

# web UI
HOST=0.0.0.0 PORT=7860 /tmp/venv/bin/python app.py &
curl -s localhost:7860/health localhost:7860/ready localhost:7860/health/metrics
/tmp/venv/bin/python -c "from gradio_client import Client, handle_file; \
  print(Client('http://127.0.0.1:7860/').predict(handle_file('examples/guitar_solo.wav'), \
  'Guitar', True, True, True, api_name='/process_audio')[0])"

# chord-collapse probe
/tmp/venv/bin/python - <<'PY'
import sys; sys.path.insert(0,'.')
from agents import TabAgent; import note_seq
mk=lambda p: note_seq.NoteSequence.Note(pitch=p,start_time=0.0,end_time=0.5,velocity=80)
print([ (x['string'],x['fret']) for x in TabAgent([40,45,50,55,59,64]).generate_tab(
       [mk(p) for p in (48,52,55,60,64)]) ])   # -> all string 0
PY
```

*The original baseline audit did not modify application code.*
