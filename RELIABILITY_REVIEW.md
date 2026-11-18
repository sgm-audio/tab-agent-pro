# Tab Agent Pro — Reliability Review

**Subject:** `sgm-audio/tab-agent-pro` @ `a4814b2` (`refactor: remove Hugging Face, add local GPU, stream REAPER console`)
— the base of branch `arena/18940913-tab-agent-pro`. Re-verified at `6d1dbd9`
("Audit and strengthen transcription and REAPER safety tests") before merge; §0 lists what
that commit fixed, acknowledged, or left alone.
**Review date:** 2026-11-18
**Focus:** defects that produce unusable tabs, hang the UI, or damage a REAPER project —
plus the build/lockfile/check-tool drift that lets them ship.

**Method:** every claim below was reproduced against the real `agents.py`, `main.py`,
`monitoring.py`, `reaper/*.lua` and the shipped CI tooling. Where a runtime dependency is
too heavy for this environment (`torch`, `gradio`'s own inference stack, `librosa`),
a stub was used only for the third-party module — never for the code under review.
The harnesses are checked in under `audit_poc/` and can be re-run:

```bash
# tablature DP + technique detection + readiness flip
PYTHONPATH=audit_poc/stubs:. python audit_poc/repro.py

# REAPER script driven under a stubbed REAPER API (needs lupa)
python audit_poc/reaper_probe.py

# Demucs empty-result, zero-note run, stale MIDI, memory arithmetic
PYTHONPATH=audit_poc/stubs:. python audit_poc/pipeline_probe.py

# alternative DP that produces playable tabs (used by the T-1 fix)
PYTHONPATH=audit_poc/stubs:. python -c "import audit_poc.playable_tab"
```

All quoted outputs are verbatim from those runs unless marked "(computed)".

---

## 0. Status against the merged `main` (`6d1dbd9`)

This review was produced against `a4814b2`. Before it was merged, `main` advanced to
`6d1dbd9` ("Audit and strengthen transcription and REAPER safety tests", #11), which fixes
two of the build findings and **acknowledges** six product findings — as strict `xfail`
tests plus `docs/TRANSCRIPTION_TEST_AUDIT.md`, which states plainly that "the six strict
xfails remain unresolved product defects".

Every reproduction below and every check tool/lint gate was re-run against the merged
tree. Line numbers are for `6d1dbd9`; the reproductions are unchanged.

| Status on `6d1dbd9` | Findings |
|---|---|
| **Fixed upstream** | **D-2** (`huggingface-hub==2.4.0` is now pinned, gradio bumped to 6.32.0), **D-3** (`ruff check` clean, `CACHE_DIR` removed from `.env.example`/Dockerfile, `check_docker_install` accepts the `TORCH_INDEX_URL` ARG form) |
| **Acknowledged as known defects, still broken** — strict `xfail` tests | **T-1** (`test_simultaneous_playable_chord_uses_distinct_strings`), **T-2** (`test_separate_picked_notes_are_not_automatically_slides`), **M-1** (`test_demucs_missing_api_stem_must_not_report_success`), **M-2/R-7** (`test_failed_pipeline_never_imports_stale_midi`), **R-4** (`test_shell_metacharacters_in_audio_path_remain_literal`), **M-5**'s octave deletion (`test_suno_cleanup_preserves_legitimate_octave_chord`) |
| **Still broken, not acknowledged** | **T-3**, **M-3**, **M-4**, **W-1…W-7**, **R-1**, **R-2**, **R-3**, **R-5**, **R-6**, **R-8**, **D-1**, **D-4**, **D-5**, **D-6**, **D-7** (version pin), **D-8**, **L-1…L-3**, **V-1** |

Two consequences worth stating up front:

* An `xfail(strict=True)` is a *guard against silent regression*, not a fix — and not a
  guarantee of the behaviour, either: the same defect reproduces exactly as before
  (`audit_poc/repro.py`, `reaper_probe.py`). They are honest about this; the risk is a
  reader treating "tested" as "handled".
* One **new** test asserts the M-3 behaviour as correct
  (`tests/test_transcription_safety.py:125` requires
  `health.set_component("basic_pitch", "error: missing mir_eval")` after a *request*
  failure), so fixing readiness will require changing that test, not just the code.

---

## 1. Summary

| # | Severity | Finding | User-visible effect |
|---|---|---|---|
| **T-1** | High | Chord notes are assigned to a single string; positions drift to the 19th fret (*known defect: strict xfail*) | Tab is physically unplayable for most dyads/triads |
| **T-2** | High | Slides / hammer-ons / pull-offs are fabricated from timing alone, order-dependent (*known defect: strict xfail*) | Every fast run is mislabelled; no bend data even exists |
| **T-3** | Medium | Per-note drops are invisible in the exported artifacts | Silent note loss; audit says it is reported |
| **M-1** | High | Demucs "succeeds" with no stem files; documented fallback is skipped (*known defect: strict xfail*) | Generic failure instead of passthrough; health says `loaded` |
| **M-2** | High | Zero-note transcription reports success (CLI exit 0 + "Done"); REAPER imports stale MIDI (*known defect: strict xfail*) | Silent empty/incorrect output |
| **M-3** | High | One failed request writes `error:` into component health → `/ready` 503 | Instance leaves rotation permanently |
| **M-4** | Medium | Any Basic Pitch exception is reported as "no models installed" | Misleading diagnosis |
| **M-5** | Medium | Suno detector reads 30 s; post-processor silently deletes notes/octaves | Garbage-in, "success" out |
| **W-1** | High | Upload endpoint has no size cap; 50 MB rule is enforced after the file is on disk | Unbounded disk write, no auth |
| **W-2** | High | Global upload TTL pruner deletes inputs of jobs still queued | "File not found" under backlog |
| **W-3** | High | No request timeout; one hung job wedges the single worker | Queue fills, service dead until restart |
| **W-4** | Medium | Device choice is process-global and sticky in the ONNX patcher | One user changes/ignores everyone's device |
| **W-5** | Medium | `atexit` wipes the shared output dir; stale-session pruning can delete live work | Multi-instance / long-job corruption |
| **R-1** | High | `stream:read("*l")` inside `reaper.defer` blocks REAPER's main thread | DAW freezes during long silent stages |
| **R-2** | High | `Undo_BeginBlock`/`Undo_EndBlock` are paired across idle cycles | Unrelated edits folded into / excluded from undo |
| **R-3** | High | MIDI is imported into whatever project is active when the poll finishes | Tracks land in the wrong project |
| **R-4** | High | ExtState values are interpolated into a shell command (*known defect: strict xfail*) | `$(…)` execution; broken quoting on Windows |
| **R-5** | Medium | Comma-joined settings dialog misassigns fields | Thresholds/exports silently shifted |
| **R-6** | Medium | REAPER default profile rewrites shared prefs on every run | Web UI device/tuning changes lost |
| **R-7** | Medium | Script and CLI disagree on the default output dir; stale MIDI re-imported | "No Output" or wrong tracks |
| **R-8** | Medium | No confirmation, cancel, run lock, or orphan cleanup | Concurrent runs clobber each other |
| **D-1** | High | `.dockerignore` excludes `tools/` while the Dockerfile `COPY`s it | Image cannot build |
| **D-2** | High → **fixed** | `requirements.lock` omitted `huggingface-hub` (gradio + demucs require it) | Image build failed at the lock check; now pinned on main |
| **D-3** | High → **fixed** | CI was red at `a4814b2` (lint), so every later gate was unenforced | Fixed on main; every guard now runs |
| **D-4** | Medium | The lock verifier cannot see a dependency missing from both lock and environment | The D-2 class of drift reappears if an install step silently skips |
| **D-5** | Medium | `setuptools` is excluded from the lock; the lock's own scope/counts are wrong | "Lock is what ships" is false |
| **D-6** | Medium | `run.sh` reinstalls Basic Pitch with deps → TensorFlow, fails on Python 3.12 | Documented quick start breaks |
| **L-1** | High | No length limit on the CLI; whole-file loads + STFT exceed RAM on long input | OOM on an hour of audio |
| **L-2** | Medium | `_remove_octave_errors` is O(n²) | Minutes-to-hours on dense long files |
| **L-3** | Medium | Fixed staging filenames in a shared output dir | Concurrent CLI/REAPER runs mix stems |
| **V-1** | Medium | The guards that should have caught the above are vacuous or `xfail`; one new test codifies M-3 | Regressions invisible to CI |

The documents this review adjudicates are `AUDIT.md` (Appendix C), `SECURITY.md` and
`docs/ACCEPTED_RISKS.md`. Section 9 is a claim-by-claim ledger, updated for the fixes that
landed in `6d1dbd9` (see §0).

---

## 2. Tablature generation and technique detection

### T-1 (High) — Chord notes collapse onto one string, and voicings drift to fret 19

**Location:** `agents.py:621-752` (`TabAgent.generate_tab`), `agents.py:572-618`
(`calculate_cost`), `agents.py:557-570` (`get_valid_positions`).

**What happens.** `generate_tab` flattens the note list into a linear chain and runs a
pairwise DP over it. Notes that start at the same instant become consecutive DP layers
with `time_delta = 0.0`. `calculate_cost` then applies

```python
if 0 <= time_delta < 0.2:          # Fast transition
    if string_distance == 0:
        cost -= 5.0                # "likely slide/hammer/pull"
    else:
        cost += 5.0
```

so putting two *simultaneous* notes on the *same* string is rewarded, and there is no
constraint anywhere that a chord must use distinct strings. There is also no absolute
hand-position term, and because the per-pair costs are frequently equal (every
same-string move of the same fret distance costs the same), the winner is simply the
first candidate in tuning order — the lowest-index string that can play the pitch.

**Reproduction** (`audit_poc/repro.py`, case 1; the inputs are ordinary guitar dyads):

```
input : [('pitch', 64, 0.0), ('pitch', 67, 0.0)]        # E4 + G4, a beginner dyad
output: [(1, 19, 'pick', 0.0), (1, 22, 'hammer', 0.0)]  # A-string frets 19/22
rendered:
E|-
B|-
G|-
D|-
A|19/22h
```

A three-note chord and a plain ascending scale:

```
triad   64/68/71, same onset  -> [(2,14,'pick'), (2,18,'hammer'), (2,21,'hammer')]   distinct strings: 1
scale   64,66,67,69,71 @125 ms -> [(2,14,'pick'), (2,16,'slide'), (2,17,'slide'), (2,19,'slide'), (2,21,'slide')]
```

The chords are unplayable: one string cannot sound three pitches at once, and the
open-position dyad was rendered 19 frets up the neck. The fast scale is playable but
every note after the first is mislabelled a slide (T-2). This is not an edge case: any
chord whose notes fit within 24 frets of one string — i.e. every power chord and most
triads — is a candidate for collapse.

**Exact fix.** Group by onset and make the DP state a whole *voicing* (one position per
note, distinct strings, bounded fret span), with an absolute hand-position cost and no
negative weights. A working, tested replacement is in
`audit_poc/playable_tab.py` (~120 lines, same inputs/outputs, same `last_skipped_notes`
contract). The core of it:

```python
CHORD_EPS = 0.05

def _chord_groups(notes):
    notes = sorted(notes, key=lambda n: (round(n["start"], 3), n["pitch"]))
    groups = []
    for n in notes:
        if groups and abs(n["start"] - groups[-1][0]["start"]) <= CHORD_EPS:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups

def _voicings(agent, pitches, max_span=5):
    per_note = [agent.get_valid_positions(p) for p in pitches]
    if any(not p for p in per_note):
        return []
    out = []
    for combo in product(*per_note):
        strings = [c["string"] for c in combo]
        if len(set(strings)) != len(strings):
            continue                      # two notes cannot share a string
        fretted = [c["fret"] for c in combo if c["fret"] > 0]
        if fretted and max(fretted) - min(fretted) > max_span:
            continue                      # one hand cannot span this
        out.append(list(combo))
    return out[:MAX_VOICINGS]

# per-chord Viterbi; cost = absolute hand position + hand shift + string crossing
costs = [[hand(v) for v in layers[0]]]
for i in range(1, len(layers)):
    row = []
    for v in layers[i]:
        row.append(min(costs[-1][j] + hand(v) + abs(hand(v) - hand(pv)) * 1.5
                       for j, pv in enumerate(layers[i - 1])))
    costs.append(row)
# backpointers + backtrack as before
```

Result on the same inputs as the broken run:

```
dyad   64/67     -> [(4,5),(5,3)]              B-string fret 5, high-E fret 3
triad  64/68/71  -> [(3,9),(4,9),(5,7)]        one string each, span 2
scale  64..71    -> [(5,0),(5,2),(5,3),(5,5),(5,7)]   open position
16ths  45 -> 47  -> [(1,0),(1,2)]
```

All `pick`. With `annotate_techniques=True`, a genuine legato overlap (64 held to
0.15 s, 65 starting at 0.10 s) yields `(5,1,'hammer')`; the module has no `slide` path
at all, because there is no bend evidence to justify one.

Also delete `cost -= 5.0`: a shortest-path cost must not be negative, and here the
bonus is *only* there to steer the DP — it is the same mechanism that manufactures the
technique labels in T-2.

**Claim check.** `AUDIT.md` P1-9 (“a single unplayable note silently discards an entire
track”) was fixed narrowly — unplayable notes are now skipped (`agents.py:662-675`), and
that part holds. But `AUDIT.md` §4 row 5/6 (“column-aligned chords”, “technique
detection verified”) and `README` (“Assigns notes to strings/frets optimally”) do **not**
hold: the DP assigns chords to impossible positions.

On `6d1dbd9` this is **acknowledged rather than fixed**: `tests/test_transcription_safety.py:65`
is a strict `xfail` named `test_simultaneous_playable_chord_uses_distinct_strings`, reason
"DP assigns chord tones independently to the same string", and
`docs/TRANSCRIPTION_TEST_AUDIT.md` lists it as a P0 known defect. A strict xfail turns an
unexpected *pass* into a failure, so it does guard against a silent regression — but the
shipped behaviour is unchanged. The test's own chord (`45, 50, 55`) is precisely a case
the DP collapses — on the merged tree it yields
`[(0,5,'pick'), (0,10,'hammer'), (0,15,'hammer')]`, three pitches on the low E string and
two invented hammer-ons — which is why the strict xfail reports it as failing, and why the
suite's "6 xfailed" is not a clean bill of health.

---

### T-2 (High) — Techniques are fabricated and depend on input note order

**Location:** `agents.py:725-750` (annotation), `agents.py:351-403`
(`multiple_pitch_bends=False`).

**What happens.** After the DP, any two chronologically adjacent notes on the same string
within `technique_window = 0.05 + sensitivity*0.25` seconds are labelled:

* 1–2 frets apart → `slide`
* ascending → `hammer`
* descending → `pull`

There is no evidence of legato, picking, or pitch movement: Basic Pitch is called with
`multiple_pitch_bends=False`, so **no bend data exists anywhere in the pipeline** — a
slide annotation is invented by construction. Worse, the same-string adjacency the
detector requires is *manufactured* by the `cost -= 5.0` bonus in T-1, so the detector
mostly confirms the DP's own bias.

**Reproduction** (`audit_poc/repro.py`, cases 1–2):

```
two picked 16ths, A2 -> B2, 125 ms apart, same string:
    [(0, 5, 'pick', 0.0), (0, 7, 'slide', 0.125)]   # nothing in the data says "slide"

simultaneous E4 + G4:
    [(1, 19, 'pick', 0.0), (1, 22, 'hammer', 0.0)]  # a "hammer-on" between two
                                                    # notes struck at the same instant
```

and the same chord, submitted in a different order:

```
64,68,71 -> [(2,14,'pick'), (2,18,'hammer'), (2,21,'hammer')]
71,68,64 -> [(2,21,'pick'), (2,18,'pull'), (2,14,'pull')]
```

A Viterbi pass over simultaneous notes must be invariant to input order. This one is not.

**Exact fix.** Stop inferring technique from note times. The default must be `pick`, with
annotation opt-in and evidence-based:

```python
def generate_tab(self, midi_notes, technique_sensitivity=0.7, annotate_techniques=False):
    ...
    technique = "pick"
    if annotate_techniques and single_note_pair and same_string:
        gap = curr["start"] - prev["start"]
        legato = prev["end"] >= curr["start"] - 0.02
        interval = abs(curr["fret"] - prev["fret"])
        if legato and 0 < interval <= 4 and gap < 0.04:
            technique = "hammer" if curr["fret"] > prev["fret"] else "pull"
```

Slides may only be emitted when a note carries bend evidence; enable it explicitly with
`multiple_pitch_bends=True` in `_transcribe_with_basic_pitch` and require a monotonic
bend of ≥ 1 semitone across the interval. Otherwise remove the `slide` label from the
legend and the JSON entirely. Add `"technique_source": "heuristic" | "bend"` so consumers
can tell.

**Claim check.** `AUDIT.md` §4 row 6 ("`slide`/`hammer`/`pull` produced and annotated")
is evidence that the code emits the labels, not that they are correct. The older tests are
vacuous: `tests/test_agents_edge.py:335-343` asserts the pull-off result is *in*
`["slide", "hammer", "pull", "pick"]`, and the hammer test accepts `slide` or `hammer`.

On `6d1dbd9` there is a real regression test, `tests/test_transcription_safety.py:76`
(`test_separate_picked_notes_are_not_automatically_slides`), strict `xfail`, reason "onset
proximity alone is not evidence of a slide" — an acknowledged defect, not a fix. Its
fixture uses a 70 ms gap between notes, while the shipped heuristic fires for any gap under
`0.05 + 0.7·0.25 = 0.225 s`: picked 16ths 125 ms apart are still labelled slides.

---

### T-3 (Medium) — Per-note drops are invisible in every artifact

**Location:** `agents.py:662-675`; `main.py:200-216` (`export_tab_to_json`);
`app.py:466-493` (status message).

**What happens.** `last_skipped_notes` is populated and logged, but:

* `export_tab_to_json` emits only `instrument/timestamp/num_strings/tablature` — no
  `skipped` field;
* the CLI prints nothing at all when some notes were dropped (only when the *whole* track
  produced no tab, `main.py:545-546`);
* the web UI appends “No tablature for: &lt;stem&gt;” only when an entire stem is empty;
* one `TabAgent` instance is reused across stems, so `last_skipped_notes` is overwritten
  by the next stem before anyone could read it.

**Reproduction** (`audit_poc/repro.py`, case 3):

```
input [64, 30] -> output notes: [(0, 24, 'pick')], last_skipped_notes: [30]
JSON keys: ['instrument', 'num_strings', 'tablature', 'timestamp']
JSON mentions the dropped pitch 30? False
JSON has a skipped/ warning field? False
```

**Exact fix.** Return the report instead of stashing it on the agent, and serialise it:

```python
# agents.py
def generate_tab(...) -> tuple[list[dict], list[int]]:
    ...
    return final_tab, skipped_pitches

# main.py
def export_tab_to_json(tab_data, output_path, instrument=..., num_strings=..., skipped=()):
    data = {..., "tablature": tab_data, "skipped_pitches": sorted(set(skipped))}

# app.py status
if skipped_notes:
    status_msg += f"\n⚠️ {len(skipped_notes)} note(s) fell outside the tuning/range: {sorted(set(...))}"
```

(The reference implementation in `audit_poc/playable_tab.py` deliberately keeps the shipped
list-return signature so it can be dropped in; apply this tuple change in the same patch.)

**Claim check.** `AUDIT.md` Appendix C row P1-5 says the notes are “counted in
`last_skipped_notes` and reported”. Counting is true; *reporting* is not. `docs/RUNBOOK.md`
§7 (“the warning names which track and how many notes were skipped”) is false on both
counts — no count is ever shown, and per-note drops within a non-empty track are never
mentioned.

---

## 3. Model backends and "success with empty/garbage output"

### M-1 (High) — Demucs reports success with no stems, so the fallback never runs

**Location:** `agents.py:126-157` (`_separate_with_api`), `agents.py:159-181`
(`_separate_with_subprocess`), `agents.py:87-112` (`separate_stems`).

**What happens.** Both separation paths return a dict of *constructed* paths without
checking that anything was written:

```python
stem_map = {"other": "other", "bass": "bass"}
for stem_key, file_key in stem_map.items():
    if stem_key in separated:
        ... sf.write(...)
return {"guitar": os.path.join(base_path, "other.wav"),
        "bass":   os.path.join(base_path, "bass.wav")}
```

If the model returns different stem keys (or a write fails silently), the function still
returns paths, no exception is raised, and `separate_stems` marks the component
`loaded`. The documented raw-audio passthrough (`_raw_audio_fallback`) is only reached
from the `except` branches, so the pipeline proceeds to `librosa.load` a non-existent
file and dies with a generic error.

**Reproduction** (`audit_poc/pipeline_probe.py`, case A; fake separator returns
`{"drums": ...}`):

```
returned stems: {'guitar': '.../other.wav', 'bass': '.../bass.wav'}
all returned paths exist: False
health says demucs is: loaded
next stage raises: NotImplementedError stub    (FileNotFoundError with the real deps)
=> the documented raw-audio fallback is NOT used: no exception was raised
```

The same defect exists in the stale-file direction: because the stem directory is keyed
by song basename, a failed run over a *previous* good run's directory will happily
transcribe the old stems and report success.

**Exact fix.** Make the contract explicit and fail into the existing fallback:

```python
class SeparationError(RuntimeError):
    pass

def _require_stems(self, base_path):
    paths = {}
    for key in ("other", "bass"):
        p = os.path.join(base_path, f"{key}.wav")
        if not os.path.isfile(p) or os.path.getsize(p) <= 44:   # 44-byte WAV header
            raise SeparationError(f"demucs produced no usable {key} stem at {p}")
        paths[key] = p
    return {"guitar": paths["other"], "bass": paths["bass"]}
```

and `return self._require_stems(base_path)` in both `_separate_with_api` and
`_separate_with_subprocess`. Because `separate_stems` already catches `Exception` around
the API branch and around the CLI branch, a `SeparationError` now correctly triggers the
raw-audio fallback and `health.set_component("demucs", "unavailable")`. Add a regression
test that the fake separator returning `{}` produces two existing files.

**Claim check.** `AUDIT.md` §4 row 7 (“Demucs fallback is graceful”) and
`docs/ACCEPTED_RISKS.md` P1-7 (“degrades to raw audio instead of failing silently”) only
hold for the *exception* path. `tools/simulate_failures.py::simulate_demucs_offline`
patches `DEMUCS_API_AVAILABLE=False` and a raising subprocess — it never exercises an API
that returns successfully but writes nothing, which is the path that breaks.

---

### M-2 (High) — A zero-note run reports success, and REAPER imports stale MIDI

**Location:** `agents.py:298-350` (`transcribe_stem` returns `[]` as success),
`agents.py:504-519` (`export_midi` silently no-ops on empty), `main.py:378-625`,
`reaper/TabAgent.lua:208-231` and `:288-315`.

**What happens.** Basic Pitch "succeeding" with zero notes is a normal outcome for silent,
noisy, or out-of-range audio. The CLI then exports nothing (all three exporters are
guarded by `if tab:` / `if notes:`), and still prints

```
✅ Done: silent
   Formats    : MIDI, Tab, JSON
```

with **exit code 0**. In REAPER, `import_midi()` globs `basename_*.mid` in the output
directory; a previous successful run's files are still there, so the script imports
*yesterday's* MIDI into the project, names the tracks after today's file, and reports
“Done — imported 4 MIDI track(s)”.

**Reproduction** (`audit_poc/pipeline_probe.py`, cases B and C):

```
=== B. zero notes => exit 0, 'Done', no artifacts ===
exit code: 0
files produced: []

=== C. stale MIDI survives the zero-note run ===
stale file present before re-run: True
stale file still present, untouched: True
REAPER import_midi() would import this file and report success
```

and `audit_poc/reaper_probe.py` imports four MIDI files that were created *before* the
script ran, into “PROJECT-B”.

**Exact fix.** Two changes, both small:

```python
# main.py — treat "nothing transcribed" as a failure
total = sum(len(notes) for notes in (lead_notes, rhythm_l_notes, rhythm_r_notes, bass_notes))
if total == 0:
    print("❌ No notes were transcribed (silent audio, wrong instrument range, or a "
          "backend failure).", file=sys.stderr)
    sys.exit(2)
```

```python
# main.py — never leave a previous run's artifacts behind
for stale in Path(output_dir).glob(f"{song_name}_*"):
    if stale.suffix in {".mid", ".tab", ".json"}:
        stale.unlink()
```

```lua
-- reaper/TabAgent.lua — give every run its own directory and import only from it
local run_id = tostring(os.time()) .. "-" .. tostring(math.random(1, 1e6))
local output_dir = settings.output_dir ~= "" and settings.output_dir
                   or (install_path .. "/output")
output_dir = output_dir .. "/" .. basename .. "-" .. run_id
-- pass --output-dir output_dir, then import from output_dir
```

**Claim check.** `AUDIT.md` P1-6 was about `transcribe_stem` raising rather than
returning `[]`; the current code returns `[]` and the *empty* result is never escalated.
`docs/RUNBOOK.md` §7 says an empty ZIP is reported with which track and how many notes
were skipped — it is not. The REAPER script's own completion path treats “no MIDI found”
as a dialog *after* attempting the import, so stale files defeat it entirely.

---

### M-3 (High) — One failed request flips `/ready` to 503, permanently in a traffic-gated deployment

**Location:** `agents.py:385-403` (`set_component("basic_pitch", f"error: …")`),
`monitoring.py:167-180` (`is_ready`), `app.py:726-742` (`/ready`),
`Dockerfile` `HEALTHCHECK`.

**What happens.** `_transcribe_with_basic_pitch` writes a *request-scoped* failure into
*component* health:

```python
except Exception as e:
    health.set_component("basic_pitch", f"error: {type(e).__name__}")
```

`is_ready()` is `all(component_health().values())`, so one bad upload (corrupt decode,
OOM, ONNX error, even a cancelled file) makes `/ready` return 503 until the next
`EarAgent()` construction resets the component. In the web app a new `EarAgent` is built
per request, so the instance recovers on the *next* request — but in a Kubernetes Service
a 503 removes the pod from the endpoint list, so there is no next request: it never
recovers. The Docker healthcheck (and HF Spaces) will also mark it unhealthy after
90 seconds of quiet.

**Reproduction** (`audit_poc/repro.py`, case 4):

```
ready before: True | component: {'basic_pitch': True}
transcribe_stem raised: RuntimeError No transcription models available ...
component after: {'basic_pitch': False}
ready after  : False    <- /ready now returns 503
health status: down
```

**Exact fix.** Never let a request failure mutate component availability. Record it on
the request counters and keep a separate, decaying circuit-breaker if you want one:

```python
# agents.py — _transcribe_with_basic_pitch
except ModuleNotFoundError as e:
    health.set_component("basic_pitch", f"error: missing {e.name}")  # load-time, OK
    raise
except Exception:
    health.record_request(success=False, details=f"basic_pitch: {os.path.basename(audio_path)}")
    raise
```

`health.record_request` already feeds `error_rate` and `last_error`, which is what the
RUNBOOK alert actually wants. Then add a test: run a failing transcription and assert
`health.is_ready()` is still `True` when the backend imported.

**Claim check.** `monitoring.is_ready`'s own docstring says “Independent of request-level
error rate so a single bad upload does not take an otherwise healthy instance out of
rotation.” It is not independent. `AUDIT.md` N-1 and Appendix C P1-1 claim the readiness
split fixed the conflation of liveness and readiness — it fixed the *startup* case only.

`6d1dbd9` goes the other way: `tests/test_transcription_safety.py:125`
(`test_basic_pitch_failure_updates_health_and_does_not_return_empty_success`) *asserts*
`health.set_component("basic_pitch", "error: missing mir_eval")` after a failed
prediction. Fixing M-3 therefore means changing that test as well — the suite currently
locks the defect in.

---

### M-4 (Medium) — Every Basic Pitch failure is reported as "no models installed"

**Location:** `agents.py:298-350`.

`transcribe_stem` catches every exception from Basic Pitch, logs it, then raises a canned
`RuntimeError("No transcription models available … Install Basic Pitch …")`. A corrupt
file, a CUDA OOM, or an ONNX provider error therefore all tell the user to reinstall
Basic Pitch. `audit_poc/repro.py` case 4 shows this with the backend installed.

**Fix:** preserve the cause and only use the “not installed” message when it is true:

```python
except Exception as e:
    log.error("basic_pitch_failed", exc=e, file=file)
    raise RuntimeError(f"Basic Pitch failed on {file}: {type(e).__name__}") from e
...
if not BASIC_PITCH_AVAILABLE:
    raise RuntimeError("Basic Pitch is not installed (pip install basic-pitch mir_eval resampy)")
```

---

### M-5 (Medium) — Suno detection/cleanup silently discards audio and notes

**Location:** `suno_postprocessor.py:52` (`duration=30`),
`:240-274` (`_remove_octave_errors`), `:276-293` (`_remove_spurious_high_notes`),
`:124-170` (`SunoAudioPreprocessor.process`).

Three separate silent behaviours:

1. Detection analyses only the **first 30 seconds** (`librosa.load(..., duration=30)`).
   A track whose first 30 s is an intro or a drum fill is classified on that evidence
   alone; the verdict then decides whether the *whole* file is re-encoded.
2. If `is_suno`, the entire file is replaced by a **22.05 kHz mono** preprocessed WAV, so
   every downstream stage sees band-limited mono audio regardless of the input.
3. `_remove_spurious_high_notes` deletes *all* notes above MIDI 84 when more than 30 % of
   notes are above it, and `_remove_octave_errors` deletes every octave-doubled note —
   both with no count in the output or the JSON.

**Fix:** analyse several windows (e.g. four 15 s windows spread across the file, average
the metrics, and require a majority), log both the analysed window count and the decision;
return counts from the post-processors and include them in the status/JSON; and only
downsample to 22.05 kHz for *Basic Pitch*, keeping the original for Demucs.

**Claim check.** `README` (“AI audio support — detection and cleanup”) is true as a
mechanism. `AUDIT.md` P1-7 documents the Demucs limitation but says nothing about the
30-second evidence window or the silent note deletions; the “graceful” wording in
`ACCEPTED_RISKS.md` does not cover them.

---

## 4. Web UI: uploads, queue, timeouts, health

### W-1 (High) — The upload endpoint is unbounded; the 50 MB rule runs after the bytes are on disk

**Location:** `app.py:77-95` (limits), `app.py:706-753` (`create_app`),
`app.py:136-164` (`_validate_audio`).

**What happens.** The app mounts Gradio with `gr.mount_gradio_app(parent_app, demo,
path="/", theme=..., css=...)` and never passes `max_file_size`, whose default is `None`.
In the locked Gradio 6.31.0 that becomes `math.inf` for the upload parser
(`gradio/routes.py`, `POST /gradio_api/upload`):

```python
output_files, files_to_copy, locations = await upload_fn(
    request, app.uploaded_file_dir,
    blocks.max_file_size if blocks.max_file_size is not None else math.inf, ...)
```

and the only size gate is the per-chunk check in `GradioMultiPartParser.parse`
(`gradio/route_utils.py`):

```python
if (part.file.size or 0) + len(data) > self.max_file_size:   # math.inf here
    raise MultiPartException(f"File size exceeded …")
```

So an unauthenticated client can stream an arbitrarily large body into the container's
`/tmp/gradio` before `_validate_audio` ever sees it. The file is written first, validated
second; the 50 MB/5 min rules only bound what happens *after* the write. (That same
per-chunk check is what makes `max_file_size` a real bound — with a value set, the
over-limit chunk is rejected before it is written.)

**Reproduction** (with the app running; no queue involved):

```bash
head -c 209715200 /dev/urandom > /tmp/big.wav      # 200 MB
curl -sS -X POST http://host:7860/gradio_api/upload -F "files=@/tmp/big.wav"
# accepted; 200 MB now under ${TMPDIR:-/tmp}/gradio/<sha256>/big.wav
# _validate_audio only runs later, when the change event fires
```

`SECURITY.md` says “50 MB and 5 minutes … validated before decode”, and
`ACCEPTED_RISKS.md` R-6 says abuse “degrades to rejection, not memory exhaustion”. Both
are true of the *handler* and false of the *upload*.

**Exact fix.** Set the HTTP-layer cap and delete the file after the request:

```python
mount_kwargs: dict = {}
if _gradio_major() >= 6:                     # 4.x mount_gradio_app has no such argument
    mount_kwargs = {"theme": gr.themes.Soft(), "css": UI_CSS, "max_file_size": "50mb"}
return gr.mount_gradio_app(parent_app, demo, path="/", **mount_kwargs)
```

plus, at the proxy, `client_max_body_size 50m;` (nginx) or the equivalent. Keep
`_validate_audio` as the second line of defence. Note also `app.py:584` uses
`gr.Audio(type="filepath", sources=["upload"])` without `format=None`; on the 4.x side of
the declared `gradio>=4.44,<7` range the default is `format="wav"`, which **transcodes
every upload to WAV in `preprocess`** — before `_validate_audio` runs. That makes the
extension allowlist meaningless (everything arrives as `.wav`) and applies the 50 MB cap
to the *converted* file (a 49 MB MP3 becomes a ~500 MB WAV). Pin `gradio>=6.31,<7` and
pass `format=None` explicitly.

---

### W-2 (High) — The global upload pruner deletes inputs of jobs that are still queued

**Location:** `app.py:793-826` (`cleanup_stale_uploads`), called from
`app.py:259` at the start of every request; `UPLOAD_TTL_SEC` default 900.

**What happens.** Every request deletes everything under `${TMPDIR}/gradio` whose mtime is
older than 15 minutes. Gradio has already received the upload when the event is queued,
so a job waiting in the queue has its input file sitting in that directory. With one
worker and multi-minute jobs, a backlog easily exceeds 15 minutes; when the queued job
finally starts, the first thing it does is delete its own input (`cleanup_stale_uploads`)
and `_validate_audio` returns “File not found”. It also deletes other Gradio
applications' uploads on the same host.

**Reproduction:** no HTTP needed — the sequence is deterministic. Submit N jobs so the
queue wait exceeds `UPLOAD_TTL_SEC`, then observe the UI:

```
❌ File not found
```

with the upload's path logged as removed by `uploads_pruned`.

**Exact fix.** Delete the pruner and let Gradio own its cache, with an age longer than
any plausible queue wait:

```python
with gr.Blocks(title=..., delete_cache=(3600, 7200), **_ui_style_kwargs()) as demo:
```

and remove both `cleanup_stale_uploads()` calls. If you keep a pruner, restrict it to the
app's own cache dir and skip any path present in the queue's pending events.

**Claim check.** `ACCEPTED_RISKS.md` R-9 says the TTL pruner is the mitigation for
uploads being written before validation. The pruner is precisely what breaks queued
requests; it also prunes only on request, so an idle instance never cleans up (the
RUNBOOK admits this).

---

### W-3 (High) — No job timeout: one hung request wedges the service

**Location:** `agents.py:159-181` (`subprocess.run` without `timeout`),
`app.py:713-719` (`demo.queue`, defaults `default_concurrency_limit=1`, `max_size=32`), `app.py:233-556` (no deadline).

`demo.queue(default_concurrency_limit=1, max_size=32)` bounds the *backlog*, not the
*duration*. Gradio's queue has no per-event execution timeout (the `timeout` symbols in
`queueing.py` are the SSE heartbeat). A Demucs weight download with blocked egress, a
CUDA OOM retry loop, or a wedged `subprocess.run(["demucs", ...], capture_output=True)`
(the CLI fallback has no `timeout=` and buffers all output in RAM) holds the single
worker forever. `/ready` stays 200 because the process is alive; every new request is
queued and then rejected once `max_size` is reached. The service is down while reporting
healthy, which contradicts `ACCEPTED_RISKS.md` R-6 (“degrades to rejection”).

**Exact fix.** Add a bounded execution wrapper and a subprocess timeout:

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import os

MAX_JOB_SEC = int(os.getenv("MAX_JOB_SEC", "1800"))
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS, thread_name_prefix="tab")

def _run_with_deadline(fn, *args, **kwargs):
    future = _executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=MAX_JOB_SEC)
    except FutureTimeout:
        future.cancel()
        raise TimeoutError(f"transcription exceeded {MAX_JOB_SEC}s") from None
```

```python
subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=MAX_JOB_SEC)
```

and map the timeout to a user message plus `health.record_request(success=False)`. Note
that `future.cancel()` cannot stop a running thread; for hard cancellation the pipeline
must check a deadline between stages (Suno → Demucs → per stem) and raise.

---

### W-4 (Medium) — Device selection is process-global and the ONNX patch is sticky

**Location:** `app.py:699-704` (`_on_device_change`), `app.py:221-230`
(`_active_device_config`), `device.py:240-278` (`install_onnx_provider_override`).

* `_on_device_change` writes `os.environ["DEVICE"]` — process-wide. Every concurrent user
  now shares the last dropdown selection, and the change persists to a single
  `user_preferences.json` shared by all users.
* `install_onnx_provider_override` sets `ort.InferenceSession._tab_agent_patched` on
  first use and then returns early forever, capturing the provider list from the *first*
  GPU config. Switching cuda → vulkan afterwards silently keeps the CUDA providers.

**Fix:** carry the device in `gr.State` per session and pass it into the pipeline
(`resolve_device(state_value)`), keeping `DEVICE` as an operator-only default; make the
ONNX patch keyed on the provider tuple:

```python
wanted = tuple(config.onnx_providers)
if getattr(ort.InferenceSession, "_tab_agent_providers", None) == wanted:
    return
... install wrapper ...
ort.InferenceSession._tab_agent_providers = wanted
```

---

### W-5 (Medium) — `atexit` and the session pruner can destroy live work

**Location:** `app.py:756-786`.

`atexit.register(cleanup_temp_dirs)` removes the entire `OUTPUT_DIR`
(`${TMPDIR}/tab_agent_outputs`) — every session, including those created by another app
process on the same host (two `python app.py` instances share `/tmp`). `handle_signal`
does the same on SIGTERM (`app.py:761-768`), so a container stop during one user's job
deletes the completed outputs of every other session on that host. Likewise
`cleanup_stale_sessions(max_age_hours=1)` deletes by directory mtime; with
`MAX_CONCURRENT_REQUESTS > 1` and a job longer than an hour, a *different* request's
cleanup deletes a session that is still being written.

**Fix:** track this process's session dirs in a set and delete only those (`atexit` and
pruning both), and use a heartbeat file inside the session (`touch` between stages) so
pruning can distinguish active from abandoned work.

---

### W-6 (Medium) — A corrupt preferences file is flattened by the device save

**Location:** `app.py:189-219` (`_save_device_preference`), `app.py:166-187`.

If `user_preferences.json` fails to parse, `_save_device_preference` catches `ValueError`,
sets `preferences = {}` and writes `{"config": {"device": …}}` — destroying tunings,
thresholds and the active profile. (It is at least an atomic replace.)

**Fix:** refuse to rewrite an unparseable file:

```python
try:
    preferences = json.load(f)
except ValueError:
    shutil.copy2(prefs_path, prefs_path + ".corrupt")
    raise OSError(f"{prefs_path} is not valid JSON; refusing to overwrite")
```

---

### W-7 (Medium) — Config parsing crashes at import; concurrency has no memory bound

`MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv(...) or "1"))` (and the two neighbours)
raise `ValueError` at import time on a non-numeric value, so the container exits with a
traceback instead of a configuration error. And any value above 1 multiplies per-request
peak memory (see L-1) with no cap. Validate with a helper that clamps and logs, and tie
the maximum to an explicit `MAX_MEMORY_MB`-style budget rather than letting the operator
set 8 on a 4 GB box.

---

## 5. REAPER integration

Findings here are reproduced by driving the real `reaper/TabAgent.lua` under a stubbed
REAPER API (`audit_poc/reaper_probe.py`), which records undo calls, the active project at
import time, and the wall-clock duration of one deferred callback.

### R-1 (High) — The deferred poll blocks REAPER's main thread

**Location:** `reaper/TabAgent.lua:164-206`.

`Pipeline:poll()` does `self.stream:read("*l")` inside `reaper.defer`. `read("*l")` blocks
until the child writes a newline or exits, and the deferred callback runs on REAPER's main
thread. The pipeline is silent for long stretches: model load, the Demucs weight download
on first use, and every Basic Pitch inference (no progress output). During those stretches
the `read` blocks, and REAPER's UI freezes — exactly what the comment above the function
claims it avoids (“so REAPER stays responsive instead of blocking on os.execute”).

**Reproduction** (probe B, child sleeps 3 s before its first line):

```
one poll() blocked the main thread for 3.00s while the child was quiet
```

**Exact fix.** Never read a pipe on the idle thread. Redirect the child to a log file and
tail it:

```lua
function Pipeline:start()
  self.log = os.tmpname()
  self.done = self.log .. ".done"
  self.offset = 0
  local args = table.concat(build_args(self.settings, self.audio_path), " ")
  local cmd = string.format(
    'cd %s && %s %s > %s 2>&1; echo $? > %s',
    quote(self.install_path), python, args, quote(self.log), quote(self.done))
  reaper.ExecProcess(cmd, 0)          -- 0 = do not wait
end

function Pipeline:poll()
  local f = io.open(self.log, "rb")
  if f then
    f:seek("set", self.offset)
    local chunk = f:read("*a") or ""
    f:close()
    self.offset = self.offset + #chunk
    self.pending = (self.pending or "") .. chunk
    for line in self.pending:gmatch("(.-)\n") do reaper.ShowConsoleMsg(line .. "\n") end
    self.pending = self.pending:match("[^\n]*$") or ""
  end
  local d = io.open(self.done, "r")
  if d then
    local code = tonumber(d:read("*l")) or -1
    d:close()
    return "done", code == 0
  end
  return "streaming", nil
end
```

This also gives an explicit exit code instead of relying on Lua's `file:close()`
semantics. (`build_args` already returns *pre-quoted* fragments, so the concat above is
safe only after R-4 replaces `quote()`; apply both together.)

---

### R-2 (High) — The undo block spans idle cycles

**Location:** `reaper/TabAgent.lua:236` (`Undo_BeginBlock`), `:303` (`Undo_EndBlock`),
`:272-283` (`reaper.defer`).

`Undo_BeginBlock()` runs in the script body; `Undo_EndBlock()` runs in a *later* deferred
invocation, minutes later. Between them the user is free to edit the project. Depending
on REAPER's handling of an open block across invocations, either those unrelated edits
are folded into the “Tab Agent — Transcribe” undo point, or the block is auto-closed at
the end of the first invocation and the later `InsertMedia` calls land with no undo point
at all. Both are wrong, and neither is acceptable for a script that inserts tracks.

**Reproduction** (probe A): `Undo_BeginBlock` at script body; `Undo_EndBlock` only after
three defer ticks, after the project has been switched to PROJECT-B.

```
script body : ['Undo_BeginBlock']
defer tick 1: ['Undo_BeginBlock']
defer tick 2: ['Undo_BeginBlock']
defer tick 3: ['Undo_BeginBlock', 'InsertMedia_into_PROJECT-B', 'InsertMedia_into_PROJECT-B',
               'InsertMedia_into_PROJECT-B', 'InsertMedia_into_PROJECT-B', 'Undo_EndBlock']
```

**Exact fix.** Begin and end the block inside the same callback, immediately around the
only mutating work:

```lua
local function poll_pipeline()
  ...
  if phase == "streaming" then reaper.defer(poll_pipeline); return end

  reaper.Undo_BeginBlock()                 -- same invocation as the inserts
  local imported = settings.export_midi and import_midi(basename, output_dir) or 0
  reaper.Undo_EndBlock("Tab Agent — Transcribe " .. basename, -1)
  ...
end
```

and delete the outer `Undo_BeginBlock()` at line 236 (the early-return `Undo_EndBlock`
calls at 246/252/284 are no-ops and should go with it).

---

### R-3 (High) — MIDI is imported into whichever project is active at completion

**Location:** `reaper/TabAgent.lua:288-315`.

The script never records the project it started in. `reaper.defer` keeps running across
project-tab switches, and `InsertMedia` targets the *current* project at the moment the
poll finishes. Probe A demonstrates four tracks inserted into PROJECT-B while the
transcription was requested from PROJECT-A.

**Exact fix.**

```lua
local start_project = reaper.EnumProjects(-1, "")

local function poll_pipeline()
  ...
  if reaper.EnumProjects(-1, "") ~= start_project then
    reaper.MB("The active project changed while transcribing.\n\n" ..
              "Files were written to:\n  " .. output_dir ..
              "\n\nRun Tab Agent again from the intended project.",
              "Tab Agent — Project Changed", 0)
    return
  end
  ...
end
```

Also capture the edit cursor at start (`reaper.GetCursorPosition()`) and insert at that
position rather than wherever the cursor is at completion.

---

### R-4 (High) — ExtState values are interpolated into a shell command

**Location:** `reaper/TabAgent.lua:109-136` (`quote`, `build_args`), `:167`.

`quote()` wraps a value in double quotes and escapes `"` as `\"`. Inside POSIX double
quotes, `$(…)` and backticks are still expanded, so a profile or path containing
`$(command)` executes. On Windows, `\"` is not an escape in `cmd.exe`, and a value ending
in a backslash escapes the closing quote. The generated command is:

```
cd "/tmp/fake-install" && python3 main.py "/tmp/song.wav" --profile "$(echo INJECTED > /tmp/tabagent_pwned)" 2>&1
```

and probe C shows the sentinel file created (`sentinel file created by shell substitution:
True`). ExtState is writable by any installed ReaScript, so this is a local injection
primitive, not merely a user typo.

**Exact fix.** Two layers:

1. Validate every setting before it reaches the command. `profile` must be one of the
   `PROFILES` keys, `instrument` ∈ `{Guitar, Bass, All}`, `onset`/`frame` numeric in
   `[0,1]`, `output_dir`/`install_path` absolute and existing. Reject with a clear `MB`.
2. Stop using a shell. Write the argv to a temp file and have a tiny launcher read it:

```lua
local argv = { python, install_path .. "/main.py", audio_path, "--profile", settings.profile, ... }
local f = io.open(arg_file, "w")
for _, a in ipairs(argv) do f:write(a, "\0") end
f:close()
reaper.ExecProcess(string.format('%s %s %s > %s 2>&1; echo $? > %s',
  quote(python), quote(install_path .. "/launch.py"), quote(arg_file),
  quote(log), quote(done)), 0)
```

```python
# launch.py
args = open(sys.argv[1], "rb").read().decode().split("\0")
os.execv(args[0], args)
```

If you must keep a shell string, use POSIX single quotes with `'` → `'\''`, and on
Windows reject `% & | < > ^ "` outright.

---

### R-5 (Medium) — The settings dialog's CSV round-trip silently shifts fields

**Location:** `reaper/Settings.lua:77-90` (`split_csv`), `:104-126`.

`reaper.GetUserInputs` joins the nine fields with commas and cannot escape them. Any
value containing a comma (very common in a path such as `D:\Music, Guitar\out`, and legal
on every OS) adds a field. `#parts < 9` does not fire, so every field after the comma is
shifted: onset becomes the path fragment, frame becomes the old onset, `export_midi`
becomes the old frame, and so on. Because `0.0`/`1.0` are valid threshold values, this
can silently produce a garbage-parameter run rather than an error.

**Exact fix.** Reject commas at the UI boundary and validate each field:

```lua
if input:find(",", 1, true) and select(2, input:gsub(",", "")) ~= 8 then
  reaper.MB("One of the fields contains a comma, which this dialog cannot represent.\n" ..
            "Move the folder or remove the comma and try again.", "Tab Agent — Settings", 0)
  return
end
```

(Simpler and more robust: replace the nine-field `GetUserInputs` with a `gfx` dialog or
nine separate prompts.)

Also validate `parts[3]` against `{Guitar, Bass, All}` before saving; today a typo makes
every subsequent run fail in argparse, with the reason only visible in the console.

---

### R-6 (Medium) — The REAPER default profile rewrites the shared configuration

**Location:** `reaper/Settings.lua:46-74`, `reaper/TabAgent.lua:117-119`,
`main.py:226-247` (`apply_profile`), `init_memory.py:128-170` (`save_profile`).

`load_settings` defaults `profile` to `"rock_standard"` and `show_gui` always saves it on
OK, even when the user only changed the install path. From then on every REAPER run
passes `--profile rock_standard`, and `apply_profile` → `save_profile` **replaces**
`preferences["config"]` wholesale. That file is also read by the web UI
(`app._load_saved_config`) and the CLI, so:

* the web UI's `device` choice is erased;
* any tuning/threshold edits made outside REAPER are overwritten with the profile's;
* `active_profile` is set, which flips `guitar_prefer_low` (`main.py:465-468`).

**Exact fix.** Default `profile` to `""` (“use current settings”) and only pass
`--profile` for an explicit selection; make `save_profile` merge into the existing
`config` instead of replacing it; and never write preferences from a transcription run
(pass the profile through as a transient config dict).

---

### R-7 (Medium) — Output-directory disagreement, and stale MIDI is imported

The script assumes `install_path .. "/output"`; `main.py:395-401` uses `/app/output` if it
exists, else `./output`. Any install under `/app` (including the Docker image layout) makes
the script look in the wrong place and show “No Output” even on success. Combined with
M-2, a failed run leaves the previous MIDI to be imported. **Fix:** always pass
`--output-dir` explicitly, and use the per-run directory from M-2/R-8.

---

### R-8 (Medium) — No confirmation, no cancel, no run lock, no orphan handling

The script starts a multi-minute job immediately, with no confirmation of the selected
item/settings and no way to cancel. Quitting REAPER mid-run leaves the Python child
running (`io.popen` children are not killed), still writing `stems/processed_*.wav` with
*fixed filenames* (`agents.py:183-255`), so a second run — or the CLI — interleaves with
it and produces mixed stems and wrong MIDI. **Fix:** per-run output directories (R-7),
a persisted lock (`SetExtState("TabAgent","running", pid, true)`) that refuses to start a
second run, a confirmation dialog showing the item, project, profile and output path, and
`reaper.atexit` cleanup that terminates the child by PID.

---

## 6. Build, lockfile, and check-tool drift

### D-1 (High) — The Docker image cannot build: `tools/` is excluded but copied

**Location:** `.dockerignore:26` (`tools/`), `Dockerfile:54` (`COPY tools ./tools`),
`Dockerfile:55-57`.

`.dockerignore` excludes `tools/`, so the build context has no `tools` directory. With
BuildKit, `COPY tools ./tools` fails (`failed to compute cache key: "tools" not found`);
with the classic builder it fails with `COPY failed: stat …/tools: no such file or
directory`. If a builder instead creates an empty directory, line 55 fails with
`can't open file '/app/tools/check_lockfile.py'`. Either way the image advertised in
`README`/`SECURITY.md` cannot be produced. (Not executed here — no Docker in the sandbox —
but deterministic from the context rules.)

**Exact fix.** Remove `tools/` from `.dockerignore`, or add a narrow exception:

```
tools/
!tools/check_lockfile.py
```

and copy only what the build needs. Add a `docker build --target builder` smoke job;
`validate.py::check_docker_build` already exists but skips when Docker is absent.

---

### D-2 (High — **fixed in `6d1dbd9`**) — `requirements.lock` omitted `huggingface-hub`

**Status.** Fixed on main: `requirements.lock` now pins `huggingface-hub==2.4.0`
(line 64), with gradio bumped to 6.32.0, pillow to 13.0.0 and tornado to 6.6.1. Verified on
the merged tree:

```
$ python tools/check_lockfile.py --static
requirements.lock covers all 17 declared dependencies (4 exact pins verified).
$ grep -n "huggingface" requirements.lock
64:huggingface-hub==2.4.0
```

The entry below is kept because the *shape* of the defect is what D-4 is about, and because
the same lock still carries stale counts (D-5).

**What it was at `a4814b2`.** The lock had 118 pins and no `huggingface-hub`, while the
three locked packages that require it were these (the first from the locked wheel's own
`METADATA`):

```
$ unzip -p /tmp/gdl/gradio-6.31.0-py3-none-any.whl '*/METADATA' | grep -i huggingface
Requires-Dist: huggingface-hub<3.0,>=1.16.0

gradio-client 2.7.2 : Requires-Dist: huggingface-hub<3.0,>=1.16.0
demucs 4.1.0        : Requires-Dist: huggingface-hub
```

The Dockerfile re-applies the lock with `--no-deps` (which cannot install a package the
lock does not name) and then runs the exact check (`Dockerfile:53-57`). On the state the
builder reached — everything in the lock plus the `huggingface-hub` that the resolver step
installed — that check reported:

```
requirements.lock is out of date with the installed environment.
(If this environment holds lint/test tooling that is not shipped, use --verify instead of exact mode.)
  + huggingface-hub==2.4.0 is installed but not locked

Regenerate with: python tools/check_lockfile.py --write
exit code: 1
```

so the image build stopped at the lock check, and `SECURITY.md`'s "verified in CI and in
the Docker build" did not hold at that commit.

---

### D-3 (High — **fixed in `6d1dbd9`**) — CI was red at `a4814b2`, so every later gate was unenforced

**Status.** Fixed on main. The eight lint errors were cleared (the `torch` import is now
explicitly `# noqa: F401` with the reason that callers patch `agents.torch`; `Optional[X]`
became `X | None`; the unused `note_seq` and `logging` imports and the unsorted import
block are gone), `CACHE_DIR` was removed from `.env.example` and the Dockerfile, and
`check_docker_install.py` now accepts the `TORCH_INDEX_URL` ARG form. Verified on the
merged tree:

```
$ ruff check agents.py app.py device.py main.py tests/
All checks passed!
$ python tools/check_env_example.py
.env.example OK — 6 variables all referenced in code.
$ python tools/check_docker_install.py
Docker checks passed (basic-pitch deps, external bind, non-root, readiness probe, CPU torch, lockfile enforce+verify).
$ python tools/check_reaper_settings.py
All REAPER settings checks passed.
```

**What it was at `a4814b2`.** The last CI run on that commit failed at the lint step:

```
$ gh run list --branch main --limit 1
completed  failure  refactor: remove Hugging Face, add local GPU, stream REAPER console  CI  main  push  43330247605  25s
$ gh run view 43330247605 --json jobs --jq '.jobs[].steps[] | select(.conclusion=="failure") | .name'
Lint with Ruff
```

with, locally:

```
$ ruff check --output-format=concise agents.py app.py device.py main.py tests/
agents.py:25:8: F401 [*] `torch` imported but unused
agents.py:310:20: UP007 [*] Use `X | Y` for type annotations
agents.py:363:20: UP007 [*] Use `X | Y` for type annotations
agents.py:432:20: UP007 [*] Use `X | Y` for type annotations
agents.py:442:16: F401 [*] `note_seq` imported but unused
app.py:17:1: I001 [*] Import block is un-sorted or un-formatted
device.py:30:8: F401 [*] `logging` imported but unused
tests/test_hardening.py:17:8: F401 [*] `os` imported but unused
Found 8 errors.
[*] 8 fixable with the `--fix` option.
```

Because lint is the fourth step of the job, **none** of `check_index`,
`check_env_example`, `check_docker_install`, `check_lockfile`, gitleaks, bandit, the tests,
or `check_vulns` ran on that commit, and two of them were independently failing
(`check_env_example` on `CACHE_DIR`, `check_docker_install` on the literal torch index
string).

---

### D-4 (Medium) — The verifier cannot see a dependency missing from both the lock and the environment

**Location:** `tools/check_lockfile.py:273-303` (`_walk_closure`),
`:304-390` (`check_verify`).

```python
if dep not in seen and dep in dists:      # packages that are *not installed* are skipped
    todo.append(dep)
```

Exact mode (what the Dockerfile runs) catches “installed but not locked” — that is how
D-2 is caught. The gap is the other shape: a transitive dependency that is **not
installed at all**. The closure is built from *installed* metadata, so such a package is
invisible: it is not in `unlocked` (not installed), not in `drifted`, and not in
`absent` (the lock never named it). `--static` only checks the declared names. So if the
resolver install of `huggingface-hub` had failed (or been skipped with `--no-deps`, as a
lock re-apply does), nothing would report it. The tool's own comment — re-adding
`setuptools` as a walk root because it is declared but not shipped — shows the authors hit
this class of gap once already.

**Exact fix.** Compare against the resolver's view, not the environment's:

```python
report = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
     "--report", "-", "-r", "requirements.txt"],
    capture_output=True, text=True, check=True,
)
resolved = {canonical(p["metadata"]["name"]): p["metadata"]["version"]
            for p in json.loads(report.stdout)["install"]}
missing = sorted(set(resolved) - set(locked))
```

(Or query each locked distribution's `requires_dist` on PyPI and assert every
requirement appears in the lock.) Add a test with a fake distribution whose requirement
is absent.

---

### D-5 (Medium) — `setuptools` is outside the lock, and the lock's own scope is misdescribed

`requirements.lock` deliberately excludes `setuptools` (`check_lockfile.py:43-49`), so the
load-bearing `<81` pin documented in `ACCEPTED_RISKS.md` R-2 is enforced only by the
range-based `pip install -r requirements.txt` step. If the base image ever ships ≥ 81,
nothing in the lock prevents `pretty_midi` from failing at import. The self-description is
also wrong: the lock header says “131 packages”, `ACCEPTED_RISKS.md` says “164
distributions”, `AUDIT.md` N-7 says “118 pins”, and the file now contains **119** entries
(`grep -c ==`; it was 118 before `6d1dbd9` added `huggingface-hub`). No two of those four
numbers agree.

**Exact fix:** include `setuptools==80.10.2` in the lock (it is installed and scanned
anyway) or document the exclusion in the header with the reason; correct the three counts
in the docs. (If setuptools is added to the lock, remove it from `EXCLUDED` and keep the
`pkg_resources`-related comment.)

---

### D-6 (Medium) — `run.sh` re-introduces TensorFlow and fails on Python 3.12

**Location:** `run.sh:9-13`.

```bash
pip install --no-deps "basic-pitch>=0.4.0,<0.5"
pip install -r requirements.txt          # includes basic-pitch *with* deps
```

`requirements.txt` lists `basic-pitch`, so the second command resolves its dependencies:
on Linux with Python ≥ 3.11 that is `tensorflow<2.15.1,>=2.4.1`. On 3.12 there is no
such wheel, so the documented one-command start fails with a resolver error; on 3.11 it
succeeds but installs ~500 MB of TensorFlow that the project promises it does not use
(`README`, `requirements.txt` comment, `ACCEPTED_RISKS.md` R-3). No check tool covers
`run.sh`.

**Exact fix:** mirror the Dockerfile:

```bash
pip install --no-deps "basic-pitch>=0.4.0,<0.5" "mir_eval>=0.6" "resampy>=0.2.2,<0.4.3"
grep -v '^basic-pitch' requirements.txt > /tmp/req-nobp.txt
pip install -r /tmp/req-nobp.txt
```

and extend `check_docker_install.py` to run the same assertions over `run.sh`.

---

### D-7 (Low) — The ReaPack pin is valid but stale, and the guard mis-handles shallow checkouts

`index.xml` pins commit `4061696b…` (which does resolve upstream — verified via the GitHub
API, and `tools/check_index.py` now passes) but advertises version **1.0.1** while the
project is **1.1.0**. ReaPack users therefore install the pre-1.1 scripts (no streamed
console, older settings handling) and will never see an update. `check_index.py` only
checks resolvability, so staleness is invisible.

The guard also failed spuriously on this checkout while it was shallow, because
`git cat-file` cannot see a commit outside the truncated history:

```
$ python tools/check_index.py        # in a depth-1 clone
✗ index.xml: commit 4061696b does not exist in this repository — every <source> ...
```

After `git fetch --unshallow` the same command passes, so the failure was environmental,
not a bad pin — but CI or an exported source archive can hit it.

**Fix:** regenerate `index.xml` with `reapack-index` at release time and bump `<version
name>`; treat “not a git repository” and “object not found in a shallow clone”
(`git rev-parse --is-shallow-repository`) as skip conditions rather than pass/fail.

---

### D-8 (Low) — `.pip-audit-ignore` duplicates and a docstring that overstates the gate

`PYSEC-2025-41`, `PYSEC-2025-191` and `PYSEC-2026-1970` are each listed twice, the second
time with a comment saying it is a duplicate ("duplicate advisory identifier reported by
two feeds", "duplicate listing"). The parser keeps the last entry, so this is harmless at
runtime — but it contradicts the file's own "one entry per advisory" format line, defeats
any uniqueness assertion, and doubled entries are exactly what an expiry audit trips over.
`AUDIT.md` still says 70 advisories while the current file has **27** ID lines (61 lines
including comments; `6d1dbd9` pruned the expired ones). The `check_vulns.py` docstring
still says it "fails on HIGH/CRITICAL findings"; the code fails on *any* unaccepted
advisory regardless of severity. **Fix:** keep one line per advisory (fold the reason into
the first entry), assert uniqueness in `load_ignores`, correct the count in `AUDIT.md`,
and either implement the severity filter or fix the docstring.

---

## 7. Long files: memory and time (the CLI has no limit)

`README` documents the absence of a CLI limit, but not its consequence. Measured/derived
for a 60-minute 44.1 kHz stereo input:

| Stage | Working set |
|---|---|
| Demucs mix load (`demucs.api`) — whole file | 1.27 GB + ~5.08 GB for four stems |
| `process_guitars` (`librosa.load(mono=False, sr=None)`) | 1.27 GB in + ~1.91 GB for `mid`/`rhythm_l`/`rhythm_r` |
| `SunoAudioPreprocessor` STFT at 22.05 kHz (1025 × ~155 k bins) | 2.54 GB per complex array; mag + phase + reconstruction ≈ 7.6 GB per pass, twice |

(computed from array shapes and `float32`/`complex128` sizes; `audit_poc/pipeline_probe.py`
prints the same numbers). The STFT path runs **only when the detector flags the track**,
which makes memory use input-dependent. At the web UI's 5-minute cap the same numbers are
1/12 — still ~1 GB transient for the STFT path, which matters because
`MAX_CONCURRENT_REQUESTS` multiplies it.

**Exact fix (L-1).** Bound the input explicitly and stream the DSP:

```python
# main.py — before any load
info = sf.info(audio_path)
if info.duration > MAX_CLI_SECONDS and not args.allow_long:
    raise SystemExit(f"{info.duration/60:.1f} min exceeds --max-minutes={MAX_CLI_MINUTES}; "
                     "pass --allow-long to process anyway")
```

and replace whole-file STFTs with `librosa.stream(...)` (block-wise, with overlap), or
process Demucs stems segment by segment. At minimum, refuse to run the Suno
preprocessor on files above a threshold and log the decision.

**L-2 (Medium) — `_remove_octave_errors` is O(n²).** `suno_postprocessor.py:238-272`
compares every note to every other note. Measured on the real function:

```
 1000 notes ->   0.055s
 2000 notes ->   0.209s
 4000 notes ->   0.843s
 8000 notes ->   3.130s
```

(each doubling is ~4× — quadratic; run-to-run variance is ~10 %.) Extrapolating from the
8000-note point: 60 minutes of dense material at ~3000 notes/min is ~180 000 notes,
`3.13 s × (180000/8000)² ≈ 26 min` of CPU for one stem.

and it runs per stem (up to four) inside the request. **Fix:** bucket by rounded onset and
compare only within each bucket:

```python
buckets = {}
for i, n in enumerate(notes):
    buckets.setdefault(round(n.start_time, 1), []).append(i)
# for each bucket, compare pitches ±12; O(n * k)
```

**L-3 (Medium) — fixed staging filenames.** `process_guitars`/`process_bass` write
`processed_lead.wav`, `processed_rhythm_L.wav`, `processed_rhythm_R.wav`,
`processed_bass_clean.wav` into a shared `output_dir` (`agents.py:199-255`). Two CLI runs
or two REAPER runs share those paths and overwrite each other mid-flight. **Fix:** per-run
subdirectory (`mkdtemp` or the run id from R-7).

---

## 8. Test and guard gaps that let all of this ship

**V-1 (Medium) —** at `a4814b2` every guard that should have caught the findings above was
either vacuous or never executed the code path. `6d1dbd9` adds real tests for six of them,
but marks them strict `xfail`: they *document* the defects and fail on an unexpected pass,
while the shipped behaviour is unchanged (see §0). One test goes further and asserts the
buggy behaviour as correct.

| Guard | Why it does not catch the defect |
|---|---|
| `tests/test_transcription_safety.py:65,76,112` and `tests/test_reaper_safety.py:46,67`, `tests/test_transcription_safety.py:150` | strict `xfail` — they encode T-1, T-2, M-1, R-4, M-2 and the octave deletion as known failures. Useful as regression sentinels; they are not passing safety guarantees, and `docs/TRANSCRIPTION_TEST_AUDIT.md` says so |
| `tests/test_transcription_safety.py:125` | asserts `health.set_component(..., "error: …")` after a *request* failure — codifies M-3 |
| `tests/test_agents_edge.py::test_technique_pull_descending` | asserts the value is in `["slide","hammer","pull","pick"]` — always true |
| `…::test_technique_hammer_ascending` | accepts `slide` **or** `hammer` |
| No chord test that passes | the only chord test is the xfail above; every *passing* `generate_tab` test uses single notes, so T-1 is unguarded |
| `tests/test_hardening.py::test_concurrent_sessions_get_distinct_directories` | calls `tempfile.mkdtemp` twice itself; never enters `_process_audio_impl` |
| `tools/check_reaper_settings.py` | stubs `reaper.defer` to a no-op, so `poll_pipeline`, `import_midi`, undo pairing and project capture never execute; the "no spurious No Output" assertion is in a branch that cannot run |
| `tools/simulate_failures.py::simulate_demucs_offline` | only covers `DEMUCS_API_AVAILABLE=False` + a raising subprocess; not "API returns without writing stems" (M-1) |
| `tests/test_app_integration.py` | mocks `SplitterAgent`/`EarAgent`, so backend contract violations (empty notes, missing stems) are invisible |
| `validate.py --quick` | skips Docker and the LLM-heavy checks; does not run `run.sh` or the lock regeneration |

**Fix:** un-xfail as you fix. For each strict xfail there is a matching fix section above
(T-1 voicing DP, T-2 evidence-based techniques, M-1 stem validation, M-2 run-scoped output,
R-4 argv file, M-5 octave policy); land the fix and the xfail removal in the same commit,
which is what the marker is for. In addition: add a passing chord test that covers narrow
dyads and triads (the current xfail's chord is not the collapsing case — it xfails, but it
does not characterise the bug), assert an exact technique value for a picked run, replace
the M-3 test's `set_component` assertion with `is_ready() is True`, and make the REAPER
harness invoke the deferred callback at least twice with an `EnumProjects` probe.

---

## 9. Does the claimed protection still hold?

### 9.1 `SECURITY.md` — "Built-in controls"

| Claim | Holds? | Why |
|---|---|---|
| Upload limits 50 MB / 5 min, extension allowlist, **validated before decode** | ❌ | `max_file_size` is never set, so the upload is unbounded (W-1). On the 4.x side of `gradio>=4.44,<7` the file is transcoded to WAV before validation. |
| Request isolation: `mkdtemp` per request, unique ZIP | ✅ / 🟡 | Holds for the web path (and is guarded by a source test). The CLI uses shared fixed staging names (L-3), and `atexit` deletes *all* sessions (W-5). |
| No info leak: short message + `error_ref` | ✅ | `app.py:493-516` does this correctly. |
| Non-root, read-only container | 🟡 | The Dockerfile does it (and CI checks the directives), but the image cannot be built at all (D-1), so no running container has ever been verified. |
| Readiness gating: `/ready` 503 unless a required backend loads | 🟡 | Correct at startup; a single failed request flips it (M-3), so it gates on the *last error*, not on load. |
| Secret handling: `.env` excluded from the image | ✅ | `.dockerignore` has `.env` and `.env.local`. |
| Dependency pinning verified in CI **and in the Docker build** | 🟡 | Fixed on main: `huggingface-hub==2.4.0` is pinned and CI's lint gate is green, so the checks run. The Docker claim is still unproven because the image cannot build at all (D-1). |
| Vulnerability gate: fails on any unaccepted/expired advisory | 🟡 | Logic is sound and it now runs (lint is green); the ignore file still carries three duplicate entries and `AUDIT.md`'s advisory count is stale (D-8). |
| Static analysis: bandit + gitleaks, both blocking | ✅ | Unenforced at `a4814b2` (behind the failing lint step); the gate now runs — CI runs 44216256025 / 44217717369 on this code passed. |
| Model provenance: weights from upstream repos, versions pinned | 🟡 | Dependency versions are pinned (the D-2 gap is closed); Demucs weights are fetched at first use with no digest or size check — a truncated download is exactly the M-1 case. |

### 9.2 `docs/ACCEPTED_RISKS.md`

| Entry | Holds? | Why |
|---|---|---|
| R-1 torch 2.5.1 (23 advisories), "no user-supplied model files, no `torch.jit`" | ✅ (claim still true) | `grep -rn "torch.load\|torch.jit\|torch.compile" --include=*.py .` → no matches; the YourMT3+ code that the entry and `AUDIT.md` N-3 describe no longer exists, so N-3's "single `torch.load` verified `weights_only=True`" is stale (harmlessly: there is no `torch.load`). |
| R-2 setuptools `<81`, "runtime never imports setuptools" | 🟡 | The pin is load-bearing — but `pretty_midi` imports `pkg_resources` *at runtime*, which is setuptools. The entry's own proof contradicts its exposure paragraph. Exclusion from the lock also means the pin is not reproducible (D-5). |
| R-3 pillow, "not a declared dependency … absent from the shipped image" | ❌ | `pillow==13.0.0` **is in `requirements.lock`** (gradio requires it). The entry is unchanged on main and now contradicts the lock header that `6d1dbd9` rewrote — “Some dependencies such as pillow and protobuf also arrive through other packages and are shipped”. The exposure argument (“absent”) is factually wrong; the conclusion (“not imported by this code”) may still be acceptable — re-justify it on that basis. |
| R-4 black, dev-only | ✅ | Not in the lock, not in the image. |
| R-5 no authentication | ✅ (by design) | Consistent with the code; but see W-1: no auth plus an unbounded upload endpoint is a stronger exposure than “nothing to protect”. |
| R-6 no rate limiting; "queue is bounded, abuse degrades to rejection" | ❌ | Uploads never touch the queue and are unbounded (W-1); a hung job is never rejected and blocks the only worker (W-3). |
| R-7 soxr LGPL | ✅ | Documented in `THIRD_PARTY_NOTICES.md`. |
| R-8 no error tracker | ✅ | `error_ref` + JSON logs work as described. |
| R-9 uploads written to disk before validation; TTL pruner bounds disk | 🟡/❌ | True that they are written first; the TTL pruner is what deletes queued jobs' inputs (W-2) and does not bound the write *rate* (W-1). |
| "Scope of the scan: exactly what ships (164 distributions)" | ❌ | 119 pins today, and three different counts are in circulation (D-5). The missing-package half is fixed. |

### 9.3 `AUDIT.md` — Appendix C (post-fix verification)

| Row | Claim | Holds? |
|---|---|---|
| P0-1 | UI button works | ✅ (5-arg wrapper + defaulted `progress`; the `.click` wiring is correct) |
| P0-2 | Basic Pitch deps fixed, guarded by `check_docker_install.py` | 🟡 — the deps are right and the guard now passes; it still does not cover `run.sh` (D-6) |
| P0-3 | Binds `0.0.0.0`, honours `PORT` | ✅ |
| P0-4 | Zero-GPU claim corrected | ✅ |
| P1-1 | `/ready` + derived health | 🟡 — M-3 |
| P1-2 | `--profile` applied | ✅ for the CLI; in REAPER it is forced on every run (R-6) |
| P1-3 | REAPER settings reach `main.py` | 🟡 — they are passed, but can be misparsed (R-5), injected (R-4), or destructive (R-6) |
| P1-4 | `index.xml` valid | ✅ format; stale at 1.0.1 (D-7) |
| P1-5 | Unplayable notes skipped and reported | 🟡 — skipped and counted; not reported (T-3) |
| P1-6 | Tablature rendering fixed | ✅ |
| P1-7 | Demucs limitation documented | 🟡 — the *documented* fallback is not reached on an empty success (M-1) |
| P1-8 | `validate.py` usable | ✅ |
| P1-9 | Wheel ships modules | ✅ |
| P2-1 | `.env.example` variables all referenced | ✅ — fixed in `6d1dbd9` (`CACHE_DIR` removed); `check_env_example.py` passes |
| P2-2 | Stale counts/drift fixed | ❌ — the drift was re-fixed but the counts did not follow: 119 pins vs “131”/“164”/“118” (D-5); RUNBOOK §6/§7 still wrong (L-1, T-3) |
| N-1 | health/readiness split | 🟡 — M-3 |
| N-2 | per-request sessions | ✅ code; test vacuous (V-1) |
| N-5 | upload TTL pruner | ❌ — W-2 |
| N-7 | lock + vulnerability gate | 🟡 — the lock is complete and the gate runs now, but the advisory/ignore bookkeeping is stale (D-8) and D-4 remains |
| N-8 | non-root read-only image | 🟡 — image does not build (D-1) |

Also: the index pin in `index.xml` (`4061696b`) is **not** the commit the audit says was
used (`e7cd693f…`), and `AUDIT.md` describes a YourMT3+ loader that this revision has
removed. The document is a snapshot of an earlier tree and is being cited as if it
described this one.

---

## 10. Remediation order

1. **Stop the bleeding in the tablature path** (T-1, T-2): chord-grouped DP with distinct
   strings and a hand-span limit; techniques off by default. This is the product.
2. **Fail loudly instead of succeeding empty** (M-1, M-2): verify Demucs outputs; make a
   zero-note run an error; per-run output directories.
3. **REAPER: never block the main thread and never insert into the wrong project**
   (R-1, R-2, R-3, R-8). These are the only findings that can damage a user's session.
4. **Make the health signal honest** (M-3): request failures must not change readiness.
5. **Make the build reproducible again** (D-1, D-4): fix `.dockerignore`, and teach the
   verifier to compare against the resolver. D-2 and D-3 are already fixed on main; the
   image still cannot build (D-1), so CI's Docker claim remains unproven.
6. **Bound the service** (W-1, W-2, W-3): `max_file_size`, drop the global pruner, add a
   job deadline.
7. **Bound the CLI** (L-1, L-2, L-3): duration guard, streaming DSP, per-run staging.
8. Everything else (T-3, M-4, M-5, W-4…W-7, R-4…R-7, D-5…D-8) as a follow-up batch.

Each of the six strict `xfail` tests in §0/§8 names the fix it is waiting for; remove the
marker in the same commit that fixes the behaviour, so the suite turns from "known defect"
into "guard" without an interval in which it protects nothing.

---

## 11. Boundaries of this review

* **Docker was not available**, so D-1 is proven from the context rules, the
  Dockerfile text, upstream package metadata and the repo's own checker — not from a
  `docker build`. That failure is deterministic; run `docker build --target builder .`
  to confirm.
* **The real inference stack** (`torch`, `onnxruntime`, Demucs weights, `gradio`'s runtime)
  was not installed. All findings in `agents.py`/`main.py`/`monitoring.py`/`reaper/*.lua`
  are from the shipped code with only the third-party imports stubbed. The tablature,
  technique, readiness and empty-result findings are pure-Python and do not depend on
  model output.
* **No REAPER binary.** The script is executed under a stubbed API (`lupa`), which is
  sufficient to observe call order, blocking `read`, and the active project at import
  time; it cannot observe REAPER's internal handling of an undo block left open across
  invocations. R-2 is therefore stated as "either of two wrong behaviours", and the fix
  is correct under both.
* **CI logs** for the failing run were not retrievable from this sandbox (only the job
  summary: “Lint with Ruff — failure”), so D-3's list of downstream gates is inferred from
  step order, which the workflow makes explicit. Locally, every gate in that workflow can
  be reproduced except those needing the full dependency set (mypy, pytest, pip-audit).
* **The checkout was shallow** for most of this review (`git rev-parse
  --is-shallow-repository` → `true`, two entries in `.git/shallow`). That is why the
  ReaPack pin appeared unresolvable (D-7) and why `main` looked like an orphan commit.
  After `git fetch --unshallow`, `6d1dbd9` is a normal child of `a4814b2` and
  `check_index.py` passes; any claim here that depended on history traversal was re-checked
  after the unshallow.
* **`6d1dbd9` was merged during the review.** All findings were re-run against the merged
  tree; §0 lists what changed. The test suite as a whole (287 passed / 6 xfailed per
  `docs/TRANSCRIPTION_TEST_AUDIT.md`) was not executed here — the real dependency set
  (torch, gradio, librosa, soundfile) is not installable in this sandbox, so only the
  dependency-free harnesses and guards were run.
