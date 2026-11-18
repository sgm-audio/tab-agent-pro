# Static quality & pipeline-correctness review — Tab Agent Pro

**Scope:** `agents.py`, `main.py`, `suno_postprocessor.py`, `app.py`, `reaper/*.lua`, `device.py`, `init_memory.py`, plus the gates that are supposed to protect them.
**Method:** read every line of the primary targets; cross-checked upstream sources (`basic-pitch` 0.4.0 `inference.py`, `demucs` 4.1.0 `api.py`, `note_seq` 0.0.5 `midi_io.py`, REAPER ReaScript API docs); executed the pure-Python logic against the real `note_seq`/`librosa`/`soundfile`/`scipy` stack to confirm each numerical claim. Every "verified" line below is something I ran, not something I inferred.
**Severity:** **P0** = produces unplayable or actively misleading tablature, or can damage a REAPER project. **P1** = wrong/misleading output or a control that silently doesn't do what it says. **P2** = gate/claim/maintainability.
**Not in scope:** I did not build the Docker image, run Demucs inference (weights unreachable), or run the 268 tests (deps absent in this sandbox).

---

## Status of this review (updated 2026-11-18, after `main` moved)

This document describes `a4814b2`. `main` has since been remediated by a larger, independent pass, and most of the P0 tablature and REAPER findings here are **fixed there** — by different code, with its own tests (`tests/test_tab_playability.py`, `tests/test_reaper_contract.py`, `audit_poc/`). The solver rewrite in this review's "suggested order of work" should **not** be applied on top of it.

Fixed on `main` (verified against `1bdd494`):
P0-1/P0-2/P0-3/P0-4/P0-5 (fingering, grouping, techniques), P0-7 (tuning labels), **P0-8** (`InsertMedia(path, 1)`), **P0-9** (undo block opened and closed around the import, with a project-changed guard), P1-1, and the `os.execute` → `io.popen` gate in `validate.py`.

Still present on `main`, fixed in the same change that added this note:
**P1-10** (40 Hz high-pass) and **P1-11** (`istft` without `length=`) in `suno_postprocessor.py`, with regression tests in `tests/test_suno.py`.

Still present and **not** fixed: **P0-6** (mid/side emits three stems from one signal; see the corrected text), and the P2 gate items.

Corrections to earlier drafts of this document, kept so the record is honest:
- **P0-6 was overstated.** `rhythm_l` and `rhythm_r` are identical only for mono input. For genuine stereo they differ (`max|Δ| = 0.32`, `corr = −0.58`); what is actually wrong is that neither is a *side* signal. Rewritten below.
- **P1-10's attenuation figures were wrong in the conservative direction.** Measured values for the 4th-order Butterworth at 40 Hz are B0 −9.5 dB, C1 −7.8 dB, E1 −2.5 dB (the draft said −10.5/−5.1); the conclusion and the fix are unchanged, and the cutoff is now 23 Hz, measured at B0 −0.9 dB / E1 −0.1 dB with 20 Hz rumble still down 12.2 dB.
- **`Undo_EndBlock` flags.** An intermediate draft of the fix asserted that `-1` is the value that suppresses an undo point. That is backwards: `-1` means "everything changed" and is what the REAPER docs recommend for scripts that modify project state directly; flags *other than* `-1` are what risk creating no undo point. `main`'s `-1` is correct.

---

## Summary

The pipeline's *plumbing* is in better shape than its *musical model*. Health, metrics, atomic writes, bounded queues and error references are genuinely good. But the thing the project exists to produce — a playable tab — is built on three assumptions that are false:

1. **Notes are fingered one at a time.** There is no simultaneity in the DP, so chords are assigned as if strummed sequentially, which puts multiple notes on the same string and produces stretches that don't exist. The shipped README example demonstrates the bug.
2. **The renderer's "chord grouping" is a chained 50 ms comparison**, so a fast run collapses into a single column with all notes stacked on one string.
3. **"Lead / rhythm L / rhythm R" are three views of one signal** for any mono or centre-panned input, which is every example file and every Suno run.

Around that, the technique heuristics confidently mislabel (hammer-ons on simultaneous notes, "slides" that are one fret), the bass preprocessor removes the low-B fundamental, the REAPER script imports MIDI onto the wrong track while inside an undo block that spans a deferred loop, and several CI gates are currently red or structurally incapable of passing.

Counts: **9 × P0, 22 × P1, 20 × P2**, plus 15 claimed features that the code does not implement.

---

# P0 — unplayable or misleading tablature; REAPER project damage

## P0-1 `TabAgent.generate_tab` has no concept of simultaneity — chords are fingered as arpeggios

**Where:** `agents.py:627-757` (DP at 685-727; `calculate_cost` at `578-624`).
**Why it is bad:** `notes` is a flat time-ordered list (line 661). Each note gets its own DP layer, and for two notes at the same onset `time_delta == 0`, which satisfies `0 <= time_delta < 0.2` (line 607) and therefore awards the **legato bonus** (`cost -= 5.0`, line 610) to simultaneous pairs. There is no constraint that two notes in the same column occupy different strings, and no cost for absolute hand position or hand span. The result is arpeggio fingering printed as a chord.

Verified — three notes at t=0, standard tuning:

```
input (40, 59, 64)  ->  [('string':0,'fret':0), ('string':4,'fret':0), ('string':4,'fret':5)]   # two notes, one string
input (40,45,50,55,59) -> all five assigned to string 0, frets 0/5/10/15/19, four labelled "hammer"
300 random 3-note chords -> 287 placed 2+ notes on the SAME string; 215 exceeded a 5-fret hand span
```

`string:4, fret:0` + `string:4, fret:5` at the same instant is not a chord, it is a typo. And because `render_tab_text` joins same-string cells with `/` (`main.py:156`), the tab *displays* it: `B|-  15/19h ...` — which is exactly the text in the README's showcase (see P1-14).

**Fix:** quantise notes into onset columns (one shared grouping function, see P0-2) and run the DP over *columns*, not notes. Within a column:
- enforce `len(set(strings)) == len(column)` as a hard constraint (drop the column and log if it cannot be satisfied);
- add a hand-span term: `cost += max(0, spread - 5) ** 2` and a fixed-position cost so the whole chord is fingered at one position;
- require `time_delta > 0` before any legato bonus, so a column can never be "legato" with itself.

## P0-2 `render_tab_text` grouping is a chained comparison, not a 50 ms cluster

**Where:** `main.py:145-150`.
```python
if groups and abs(groups[-1][0] - t) < 0.05:
    groups[-1][1].append(pos)
```
The comparison is against the group's **first** timestamp but the group keeps growing, so any run of notes spaced < 50 ms apart collapses into a single column — no upper bound on the group's duration.

Verified:
```
20 distinct notes, 40 ms apart  -> ONE column: E|1/1 1/1 1/1 ... (0.76 s of music as one chord)
same string, 40 ms apart        -> E|12/14        (two sequential notes merged into one cell)
```

The docstring ("Groups simultaneous notes (within 50ms)") and the README ("a note is never merged into its neighbour") are both false. Worse, it interacts with P0-1: the `/` join exists *only* because the DP produces same-string collisions, and the renderer then legitimises them.

**Fix:** bucket on a quantisation grid or against the group's **start**, not the previous member:
```python
group_start = groups[-1][0] if groups else None
if group_start is not None and t - group_start < 0.05 and pos["string"] not in used_strings: ...
```
and make "two positions on one string in one column" an error, not a `/` join. Add a guard: if a column would need a `/`, split it into two columns (they are sequential by definition).

## P0-3 A single unplayable note is skipped note-wise, and the fact is thrown away

**Where:** `agents.py:665-680`; no consumer anywhere (`grep last_skipped_notes` → only tests and `validate.py`).
**Why it is bad:** the fix for the old "one note kills the track" bug went too far in the other direction: a note is dropped, the *rest of its column* is kept, and the DP then treats the survivors as an arpeggio (P0-1). So a three-note voicing with one out-of-range note silently becomes a two-note sequential figure. `main.py:545` only warns when the **entire** track produced no tab (`if notes and not tab`), so partial loss is invisible.
**Verified:** `[40, 10, 45]` → tab of length 2, `last_skipped_notes == [10]`, and the CLI prints nothing.
**Fix:** skip at column granularity, emit per-column diagnostics (`{"time":…, "dropped":[pitch,…]}`), write them into the JSON export, and print a summary count in `main()` (`Skipped N unplayable notes on M columns`). Silence here is the difference between a tab you can trust and one you can't.

## P0-4 First note is free, so the lowest string always wins

**Where:** `agents.py:687` (`prev_costs = [0.0] * len(layers[0])`) and `calculate_cost`'s `if prev is None: return 0.0` (line 593).
**Why it is bad:** every position of the first note has identical cost, so ties are broken by iteration order — `layers[0]` is ordered by increasing string index, and `prev_costs.index(min(prev_costs))` (line 721) returns the first minimum. String index 0 is the lowest-pitched string. There is no cost for absolute fret, so 24th fret is as attractive as open.

Verified:
```
single note 64 (E4) -> string 0 (low E) fret 24     # instead of string 5 fret 0
single note 88 (E6) -> string 5 fret 24             # the only legal position, fine
```
A tab whose very first note is "24th fret on the low E" when the open high E was available is the reviewer's signal that no hand model exists.
**Fix:** seed `prev_costs` with an opening-position preference (penalise `fret > 12`, reward open strings and frets ≤ 4, and prefer the string where the note is *lowest*-fretted), and add a per-position cost to `calculate_cost` rather than only transition costs.

## P0-5 Technique heuristics state things that did not happen

**Where:** `agents.py:733-757`.
**Why it is bad, concretely:**
- `time_delta == 0` passes `0 <= time_delta < technique_window` (line 743), so **simultaneous** notes are labelled hammer/pull. That is how the README's `15/19h` was produced.
- Slide is chosen for a fret difference of **1**, which is not a slide; and because slide is tested first (line 745, comment: "priority over hammer/pull"), a genuine 2-fret hammer-on is always reported as a slide.
- There is no test that the source note is still ringing, that the destination has its **own** onset (the defining property of a hammer-on is that it *doesn't*), or that the two notes are on the same physical string at different frets rather than the same string because of P0-1.
- Only the destination is annotated, so `5h` is emitted with no indication of what it was hammered *from*.

**Fix:** require `time_delta > 0`; require the source note's `end_time >= destination.start_time` (ringing); require the destination to be an onset (Basic Pitch gives you onset events — `_note_events` is currently discarded at `agents.py:374`); classify slide only for `fret_diff >= 2` with monotone directional motion across ≥ 2 consecutive notes; and annotate as a pair (`5h7`), not a suffix.

## P0-6 `process_guitars` mid/side "separation" emits three files from one signal

**Where:** `agents.py:196-220`.
```python
mid = (left + right) / 2
rhythm_l = left  - mid * 0.8
rhythm_r = right - mid * 0.8
```
**Why it is bad:** the two cases behave differently and only one of them is a duplicate.

*Mono input* (which `process_guitars` itself creates at lines 199-201 by duplicating the channel, and which is what the shipped `examples/*.wav` are). Here `left == right`, so:
- `rhythm_l == rhythm_r` exactly (verified: `np.allclose(...) == True`, correlation `1.0`);
- `rhythm_l == 0.2 * left` exactly (verified), i.e. the "rhythm" stem is the **lead at −14 dB**.

*Genuine stereo.* These are **not** duplicates — an earlier draft of this review said they were and that was wrong. Verified on a stereo mix (centred lead + left-panned part): `max|rhythm_l − rhythm_r| = 0.32`, `corr = −0.58`. What is true is that neither is a *side* signal: the code uses `mid` subtraction, not `(L−R)/2`, so `corr(rhythm_l, true_side) = 0.92`, and centre content survives in both at −4.4 dB with inverted polarity on one side — a worst case for an onset detector.

The pipeline then transcribes the guitar stem three times, writes `_lead_guitar.mid`, `_rhythm_L.mid`, `_rhythm_R.mid`, and the UI/README advertise "Multi-track output — Lead guitar, rhythm L/R" as if three parts were found. On the shipped mono examples that is verified arithmetic, not a corner case.
**Fix:** compute a side/mid energy ratio first; if it is below a threshold (or input is mono), return only the lead stem and log `mid_side_skipped: no stereo information`. For genuine stereo, use `side = (left − right) / 2` rather than mid subtraction with a 0.8 kill factor. Never emit three files from one signal.

## P0-7 String labels ignore the tuning; Drop-D tabs say "E"

**Where:** `main.py:107-115` — `_string_labels(num_strings)` derives names from the *count* only, returning `["E","B","G","D","A","E"]` for 6 strings.
**Why it is bad:** `init_memory`'s `rock_drop_d` profile sets `guitar_tuning = [38,45,50,55,59,64]` (low string D2). Verified render:
```
=== Guitar (Drop D, lowest string is D) Tablature ===
E|-        <-- wrong: this string is D
```
A player reading `E|0` plays E2 (82 Hz) instead of D2 (73 Hz). Every alternate tuning (Drop C, DADGAD, 5-string bass in anything but B-E-A-D-G) is mislabelled the same way. Labels are also wrong for 4-string bass in E-A-D-G vs a 5-string in B-E-A-D-G when `num_strings` is inferred (see P1-15).
**Fix:** derive labels from the tuning MIDI values, not the count:
```python
names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
labels = [names[m % 12] for m in tuning]     # high string first after reversing
```
and pass the tuning into `render_tab_text`/`export_tab_to_txt` instead of `num_strings`. Add the invariant test `tuning[string] + fret == pitch` (P2-18) — it would catch this class of bug automatically.

## P0-8 REAPER: MIDI is imported onto the *current* track and an unrelated track is renamed

**Where:** `reaper/TabAgent.lua:219-224`.
```lua
reaper.InsertMedia(path, 0)                                   -- mode 0 = "add to current track"
local track = reaper.GetTrack(0, reaper.GetNumTracks() - 1)   -- then rename... the last track
reaper.GetSetMediaTrackInfo_String(track, "P_NAME", ...)
```
**Why it is bad:** REAPER's API is explicit — `mode: 0=add to current track, 1=add new track`. So the MIDI item is dropped onto whatever track was last touched (typically the user's glued audio, or a folder, or a track they were editing), and then the **last track in the project** gets renamed `song - Lead Guitar` even though it has nothing to do with the import. On a real project this both garbles the arrangement and relabels unrelated tracks; with ripple editing it can shift everything. The comment on line 219 ("InsertMedia creates a new track automatically") is false, and the stub test in `tools/check_reaper_settings.py` (`InsertMedia = function(path, mode) end`) cannot catch it because it never asserts the mode.
**Fix:**
```lua
local before = reaper.CountTracks(0)
if reaper.InsertMedia(path, 1) == 1 and reaper.CountTracks(0) > before then
  local track = reaper.GetTrack(0, reaper.CountTracks(0) - 1)
  reaper.GetSetMediaTrackInfo_String(track, "P_NAME", name, true)
end
```
and add a lupa assertion that the captured mode is `1`.

## P0-9 REAPER: the undo block spans a deferred loop and is not guaranteed to close

**Where:** `reaper/TabAgent.lua:236` (`Undo_BeginBlock`) … `292` (`defer`) … `303` (`Undo_EndBlock`).
**Why it is bad:**
- `Undo_BeginBlock()` is called **before** any modification and the block stays open for the entire (minutes-long) pipeline. Everything the user does in that window — moving items, editing FX — is folded into a single undo point named "Tab Agent — Transcribe …"; one Ctrl+Z then reverts the transcription *and* the user's session work.
- The deferred callback can die (Lua error in `import_midi`, project closed, script re-run via the action), and then `Undo_EndBlock` is never reached. REAPER's own docs warn that this leaves an undo block open and causes undefined undo behaviour.
- `Undo_EndBlock(...)` for the early-return paths at 246/252/284 is fine, but there is a fourth exit (a second run while the first is polling) that isn't guarded at all.

**Fix:** do not open the block until you have something to commit:
```lua
local function committed_import()
  reaper.Undo_BeginBlock()
  local ok, err = pcall(import_midi, basename, output_dir)
  reaper.Undo_EndBlock("Tab Agent — Transcribe " .. basename, -1)
  if not ok then ... end
end
```
and refuse to start when an ExtState `running` flag is set (P1-20).

---

# P1 — wrong output, or a control that silently does not do what it says

## P1-1 `transcribe_stem` converts every failure into "install basic-pitch"

**Where:** `agents.py:340-354`. Any exception from `_transcribe_with_basic_pitch` is caught (`except Exception`), logged as `basic_pitch_failed`, and then re-reported as:
> `No transcription models available for {file}. Install Basic Pitch (pip install basic-pitch) …`

Corrupt audio (`NoBackendError`), a truncated ONNX file (`OnnxRuntimeError`), a provider mismatch (`Failed to load library` for the CUDA EP), and OOM all produce the same instruction — which is wrong for every one of them. `_transcribe_with_basic_pitch` already distinguishes `ModuleNotFoundError` (line 394) and sets a good component status; the outer handler throws that away.
**Fix:** classify and re-raise: `BackendMissing` (ModuleNotFoundError / `BASIC_PITCH_AVAILABLE` False), `AudioUnreadable` (soundfile/audioread), `InferenceFailed` (everything else, with the original exception chained). Only the first message tells the user to install anything.

## P1-2 Silent, empty and corrupt audio are not handled before Demucs

**Where:** `suno_postprocessor.py:52` (`librosa.load(..., duration=30)`), `main.py:431` (no `try`), `agents.py:98`.
Verified behaviour:
| input | Suno detector | pipeline |
|---|---|---|
| digital silence (1 s, 44.1 k) | `hi_freq_ratio 0.0`, `peaks 0`, `is_suno False` | passes straight to Demucs → `notes == []` → **no .tab/.json written** but `✅ Done` printed |
| zero-frame WAV | librosa warns `n_fft=2048 is too large for input signal of length=0` and continues | same |
| 2 kB of junk named `corrupt.wav` | raises `NoBackendError` | **CLI traceback** (`main.py` wraps nothing) |

A REAPER user gets `Transcription finished but no MIDI files found in: …` with no reason; a CLI user gets a stack trace.
**Fix:** a cheap preflight in `main()`/`app.py`: `sf.info()` for frames/samplerate/channels, reject `frames == 0`, reject `peak/rms < 1e-5`, reject duration < 0.1 s with a named error. Wrap each stage (`suno`, `demucs`, `transcribe`) so the CLI exits non-zero with a one-line reason instead of a traceback.

## P1-3 The CLI prints ✅ and lists formats when it wrote nothing

**Where:** `main.py:530-560` (exports are conditional on `tab`), `main.py:622-628` (summary is unconditional).
Every export is inside `if tab:` / `if notes:`, so an all-unplayable or all-empty result produces zero files, yet the summary says `Formats: MIDI, Tab, JSON` and `✅ Done`. Exit code is always 0. `_enabled_formats(args)` reports *flags*, not *files*, which is the misleading part.
**Fix:** build the format list from what was actually written, print `Files written: 0`, and `sys.exit(2)` when nothing was produced.

## P1-4 `init_memory.save_profile` replaces the whole config and destroys sibling settings

**Where:** `init_memory.py:156-166` (`preferences["config"] = config`), called from `main.apply_profile` (`main.py:239`) and by `init_memory.py --profile`.
`app.py:_save_device_preference` merges; `save_profile` does not. Verified round-trip:
```
step 1 (user picks cuda in the web UI): {"device": "cuda", "onset_threshold": 0.6}
step 2 (user runs `main.py --profile classical`):  {...  "active_profile": "classical"}   # "device" is gone
```
So choosing a device in the browser and then applying a profile on the CLI silently reverts the device, and any other hand-added key (custom tuning) is lost. `apply_profile` also accepts a `memory_file` argument (`main.py:218`) that it never uses — the docstring promises persistence to that path.
**Fix:** `preferences.setdefault("config", {}).update(config)`, and honour `memory_file`. Add a JSON-schema validator for `user_preferences.json` so unknown-key loss can't happen silently again.

## P1-5 `num_frets` / `technique_sensitivity` / tuning are unvalidated

**Where:** `main.py:460-467`. Thresholds get `validate_thresholds` (good, both CLI and profile, fails loudly). Tuning, frets and sensitivity get `int()`/`float()` at most:
- `"num_frets": "24 "` → `ValueError` traceback at pipeline start;
- `technique_sensitivity = 5.0` → `technique_window = 0.05 + 5*0.25 = 1.3 s`, so every same-string note within 1.3 s is a slide;
- `guitar_tuning = [64,59,50]` (descending) → negative frets silently rejected and every note "unplayable";
- `num_frets = 0` → every note unplayable, empty tab, `✅ Done` (P1-3).

**Fix:** one `validate_tab_config()` beside `validate_thresholds()`: tuning = non-empty list of ints 0..127, strictly ascending; `1 <= num_frets <= 36`; `0.0 <= sensitivity <= 1.0`. Raise `SystemExit` with the same style as P1-4's threshold path.

## P1-6 The Gradio device dropdown can never enable Vulkan/DirectML

**Where:** `app.py:310-315` resolves a `DeviceConfig`, then passes **`device_config.torch_device`** into `EarAgent(device=…)`, which calls `resolve_device()` again.
Verified with a simulated Vulkan-capable ONNX Runtime:
```
user picks 'vulkan'  -> backend=vulkan  torch_device='cpu'  onnx=('VulkanExecutionProvider','CPUExecutionProvider')
app.py relays 'cpu'  -> EarAgent resolves 'cpu' -> onnx=('CPUExecutionProvider',)     # GPU silently lost
```
Since Vulkan and DirectML are *ONNX-only* backends by design (`device.py:_torch_device_for` returns `"cpu"` for them), the round-trip through `torch_device` throws away exactly the information those backends carry. The advertised feature (`DEVICE=vulkan` / `directml`) does not work from the UI at all. Same for `mps` → the ONNX side never gets `CoreMLExecutionProvider`.
**Fix:** pass the preference string or the `DeviceConfig` itself: `EarAgent(config=device_config)` / `SplitterAgent(config=device_config)`, and delete the double resolution. Add a test that `EarAgent(...).backend == "vulkan"` when the provider is present.

## P1-7 The ONNX provider override is a global, permanent monkeypatch

**Where:** `device.py:247-274`.
- It replaces `ort.InferenceSession` **globally** for the rest of the process; any other library that creates a CPU session afterwards silently gets the GPU EP.
- `_tab_agent_patched` (line 257) makes it install-once, so the "takes effect on next transcription" promise in the UI (`app.py:589`) is false — the providers are frozen at first `EarAgent` construction.
- The replacement is a *function*, so `isinstance(session, ort.InferenceSession)` and subclassing break.
- The trigger `providers == ["CPUExecutionProvider"]` (line 266) is exactly what basic-pitch 0.4.0 passes (verified: `inference.py:132`), but a list-vs-tuple or an added `sess_options` changes the call and the override silently stops applying. There is no log when the GPU provider is requested and the CPU one is actually used.

**Fix:** subclass/replace with a callable object that records which providers were honoured and logs `onnx_provider_fallback` when it isn't the first-choice EP; key the patch to the model path or pass a session factory; make it re-patchable per request, or scope it to the transcription call.

## P1-8 `onnxruntime-vulkan` does not exist on PyPI (and neither does the Rocm pip index)

**Where:** `device.py:22-23`, `requirements.txt:9-10`, `README.md:299`, `Dockerfile:35`, `docs/RUNBOOK.md:40`.
Verified against PyPI: `onnxruntime-gpu` ✔, `onnxruntime-rocm` ✔, `onnxruntime-directml` ✔, **`onnxruntime-vulkan` → HTTPError (does not exist)**. Yet four files instruct the user to install it, and `README.md:299` says "install onnxruntime-vulkan at runtime" as the documented Vulkan path.
**Fix:** remove or replace with the real distribution (Intel's wheel / source build) and say plainly that Vulkan requires a build from source; keep `directml` and `rocm` as the supported non-CUDA paths.

## P1-9 `/ready` goes 503 permanently after one bad upload; `/ready` is also 200 during startup

**Where:** `agents.py:396`/`405` set the component to `"error: …"`; `monitoring.py:is_ready()`; `agents.py:58` `health.set_status("up")`.
Verified:
```
fresh tracker + registered components      -> is_ready True,  status 'starting'   # 200 while starting
after ONE NoBackendError                    -> is_ready False, status 'starting'   # 503 until a success
```
The Dockerfile healthcheck probes `/ready`, so a single malformed upload takes a healthy container out of rotation and keeps it out until some other user's upload succeeds. Separately, `agents.py:58` hard-codes `set_status("up")` at import — the exact anti-pattern `monitoring.py`'s own docstring says was removed.
**Fix:** distinguish *capability* failures (import/load) from *request* failures. Keep per-request errors in `record_request`/`last_error` only; never let a transcription exception rewrite a required component's status. Drop line 58 (the derived status already handles "starting"). Treat `starting` as not-ready.

## P1-10 The "AI audio" preprocessor deletes the bass fundamental (down 10.5 dB at B0)

**Where:** `suno_postprocessor.py:139` + `152-158`: `_highpass_filter(y, sr, cutoff=40)`.
Verified magnitudes through that 4th-order Butterworth:

| note | freq | attenuation |
|---|---|---|
| B0 (5-string bass open) | 30.87 Hz | **−10.5 dB** |
| E1 | 41.2 Hz | −5.1 dB |
| A1 | 55 Hz | −0.2 dB |

`process_suno_audio` runs this on the **whole mix before Demucs** (`main.py:418`), and the `bass_5_string` profile sets `suno_aggressive_mode: True`, so the documented 5-string-bass path actively removes the low B and weakens E1 — and a weakened, phase-shifted fundamental is the classic trigger for octave errors in pitch trackers.
**Fix:** cutoff ≤ 20 Hz for the mix, or apply the high-pass only to the guitar stem after separation, or skip it when the requested instrument is bass. Document the number in the docstring instead of "remove ultra-low rumble".

## P1-11 `_spectral_gate` shortens the audio and desyncs the timeline

**Where:** `suno_postprocessor.py:209` `return librosa.istft(gated)` — no `length=`.
Verified: 22050 in → 22016 out (−1.5 ms). The sibling `_reduce_hi_freq_artifacts` correctly passes `length=len(y)` (line 183), so the code is internally inconsistent and the loss is silent. Basic Pitch windows at 43 ms; a 1.5 ms offset is small but it is gratuitous, and the same bug is a real offset if the STFT parameters ever change.
**Fix:** `return librosa.istft(gated, length=len(y))`.

## P1-12 The Suno detector fires on white noise and on a pure 10 kHz tone

**Where:** `suno_postprocessor.py:99-113`. The `is_pure_tone` override (line 96) only neutralises the *flatness* rule; the `hi_freq_ratio` rule (line 103) is unguarded.
Verified:
```
1 kHz sine     -> is_suno False   (hi_freq_ratio 0.0019)
10 kHz sine    -> is_suno TRUE    (hi_freq_ratio 3.639)  <-- pure tone, guard did not apply
15 kHz sine    -> is_suno False   (0.0216)
white noise    -> is_suno TRUE    (0.9995)
```
So a cymbal-heavy or HF-bright recording, or an actual noise floor test tone, triggers the destructive preprocessor (which downsamples to 22.05 kHz mono 16-bit — P1-13). The class docstring's promise ("The harmonic peak filter prevents pure tones … from being falsely flagged regardless of aggressiveness") is false for exactly the tone it names.
**Fix:** `metrics["hi_freq_ratio"] > thr and not is_pure_tone`, plus an absolute HF-energy floor (a lone 10 kHz sine has no broadband shimmer) and a broadband/flatness co-requirement.

## P1-13 Preprocessing throws away sample rate, channels and bit depth for the whole pipeline

**Where:** `suno_postprocessor.py:37` (`sample_rate = 22050`), `process()` line 135 (`librosa.load(audio_path, sr=22050, mono=True)`), `sf.write(output_path, y, sr)` (line 149, default `PCM_16`), consumed by `main.py:434`/`app.py:279`.
For any audio the detector flags, everything downstream — Demucs (htdemucs is a 44.1 kHz stereo model), mid/side (dead, see P0-6), Basic Pitch — works from **22.05 kHz mono 16-bit**. Content above 11 kHz is gone before separation, so the guitar's upper harmonics/fret noise are removed, and the "8–16 kHz shimmer reduction" (line 179) can only ever touch 8–11 kHz while the detector's docstring talks about 16 kHz. Nothing in the README or docstrings mentions any of this.
**Fix:** preserve `sr`, channel count and `subtype` through the preprocessor (`librosa.load(..., sr=None, mono=False)`, operate per channel, `sf.write(..., subtype="FLOAT")`), and use one sample-rate constant shared with the detector.

## P1-14 The README's showcase tab proves P0-1 rather than demonstrating correctness

**Where:** `README.md:192-201`, presented as "verbatim" CLI output:
```
B|-  -  15/19h 22 24 -      20 -  -
D|-  -  -      -  21 24/17p 21 -  -
```
`render_tab_text` only ever produces `x/y` for **two positions on the same string in the same column** (`main.py:156`), and `h`/`p` on a simultaneous note is the `time_delta == 0` bug (P1-5). So the sample literally shows a string being played at fret 15 and fret 19 at the same instant, and the D string at 24 and 17 simultaneously. It is not a stale sample; it is evidence.
**Fix:** fix the DP (P0-1), regenerate the sample from the fixed code, and add a test that fails if any rendered column contains `/` — because `/` in output means the fingering model was violated.

## P1-15 `num_strings` inference silently invents 2-string instruments

**Where:** `main.py:139-140` `num_strings = max(max_used, num_strings or 0)`.
Verified — `validate.py:349` calls `export_tab_to_txt(tab_data, path, "Test Guitar")` with no `num_strings`:
```
=== Test Guitar Tablature ===
B|3s
A|0
```
A six-string guitar rendered as two strings labelled "B" and "A". `validate.py:352` then asserts all six labels are present, which can never pass (verified `False`) — so `validate.py` is both wrong *and* currently red on that check (see P2-11). Same for `tests/test_main_coverage.py:246`.
**Fix:** make `num_strings` required (or accept `tuning`), and raise `ValueError` when any `pos["string"] >= num_strings` instead of widening the grid.

## P1-16 The bass filter tilts the spectrum against fretted notes (up to −4.7 dB)

**Where:** `agents.py:236-253` — a 200 Hz 4th-order split, reconstructed as `low + 0.5*high`.
Verified composite response: D3 −0.3 dB, G3 −2.1 dB, B3 −3.8 dB, D4 −4.7 dB, G4 −4.6 dB. The docstring claims it "preserves low frequencies (fundamental tones below 200 Hz)" and "reduces high frequencies (fret noise, harmonics)" — in effect it turns the bass into a low-passed instrument and *removes the harmonics Basic Pitch needs to place a note in the right octave*, biasing toward sub-octave errors precisely on the upper strings of a 4-string bass.
**Fix:** delete it. Demucs already gives you an isolated bass stem; a `-5 dB` tilt at 300 Hz is a rig tone, not an accuracy improvement. If a de-noise is wanted, use a shelf that is flat below 1 kHz and document the response.

## P1-17 Every intermediate stem is written as PCM_16 and clips silently

**Where:** `agents.py:216-219` (`sf.write(lead_path, mid, sr)`), `256`, `157`.
Verified: float input at peak 1.2 → `PCM_16`, 822/4410 samples hard-clipped, **no warning**; peak 1.6 → 1258/4410 clipped. `mid` has the same peak as the input, and typical masters are ≥ 0.95, so any inter-sample overshoot or summed content clips before transcription — and clipping is broadband distortion, i.e. spurious high harmonics for Basic Pitch and extra "shimmer" for the Suno detector.
**Fix:** `subtype="FLOAT"` (or peak-normalise with a logged gain) and assert `max(|y|) <= 1.0` before writing.

## P1-18 Config-provided tunings are never cross-checked against the profile the user thinks they applied

**Where:** `main.py:460-461` reads `guitar_tuning`/`bass_tuning` from `user_preferences.json`; `init_memory.save_profile` derives *both* keys from a single profile (`init_memory.py:143-150`).
`save_profile("bass_4_string")` writes `bass_tuning=[28,33,38,43]` and `guitar_tuning=[40,45,50,55,59,64]` — fine — but `save_profile("classical")` writes `bass_tuning=[23,28,33,38,43]` (hard-coded fallback), so applying a classical-guitar profile silently resets a previously chosen 4-string bass tuning to 5-string. The `--profile` flag therefore has side effects on the *other* instrument, which nothing documents.
**Fix:** store a per-instrument profile (`bass_profile`, `guitar_profile`) or only write the keys the profile actually configures, and merge (P1-4).

## P1-19 The web UI's ZIP includes every intermediate, and "Files Generated" counts them

**Where:** `app.py:428-436` — `for file in session_dir.rglob("*")`, `written += 1` for each non-export file.
The session dir holds `stems/htdemucs/<song>/{other,bass}.wav`, `processed_lead.wav`, `processed_rhythm_{L,R}.wav`, `processed_bass_clean.wav`, the Suno `*_processed.wav`, plus the exports. A 50 MB upload therefore becomes a several-hundred-MB ZIP, and the status message's "**Files Generated**: 12" is a count of intermediates, not results.
**Fix:** whitelist `*.mid`, `*.tab`, `*.json` for the ZIP (with an opt-in "include stems" toggle) and count only what the user asked for.

## P1-20 REAPER: no confirmation, no cancellation, no re-entrancy guard, no timeout

**Where:** `reaper/TabAgent.lua:290-327`.
One click starts a minutes-long subprocess that writes into `output_dir` (default `<repo>/output`), with no confirmation and no way to stop it (the `defer` loop cannot be interrupted except by running another action). A second run while the first polls starts a second `demucs`/`basic_pitch` on the same output paths → the first run's `.mid` files can be replaced mid-import, i.e. the wrong song's MIDI lands on the track. There is no wall-clock limit; a hung Demucs holds the pipe and the deferred loop forever.
**Fix:** `reaper.MB(msg, title, 4)` confirm with the resolved command line; set `ExtState("TabAgent","running","1")` and refuse if set (cleared in every exit path); add a deadline (`os.time() - start > 1800`) that kills the stream and reports; add an "Esc to cancel" check in the poll loop.

## P1-21 REAPER: exit-code interpretation is wrong on Lua 5.1

**Where:** `reaper/TabAgent.lua:196-201`.
```lua
local ok, err, code = self.stream:close()
self.success = (ok == true) or (code == 0)
```
Under Lua 5.1 (`io.popen`'s `close` returns one value, a boolean-ish status) `code` is `nil`; under 5.2+ `ok` is `true` **even on failure**, with the exit status in `code`. So on 5.1 a `sys.exit(1)` from `main.py` becomes `success = true`, the "Transcription failed" dialog (line 306) never fires, and the user is told only "no MIDI files found". The comment at line 198 documents 5.2 semantics while REAPER ships 5.4 — but the code must not depend on which.
**Fix:** normalise:
```lua
local a, b, c = self.stream:close()
local status
if type(a) == "number" then status = a else status = c end
self.success = (status == 0)
```
and additionally treat a missing output file as failure regardless of exit code.

## P1-22 `io.popen` line-reading stalls the "streamed" console on progress bars

**Where:** `reaper/TabAgent.lua:186-190` (`stream:read("*l")`), `agents.py:139` (`progress=True` for Demucs).
Demucs' `progress=True` emits `\r`-terminated progress updates to **stderr**, which is merged with `2>&1`. `read("*l")` only returns on `\n`, so the console shows nothing for the whole separation (the longest stage) and then dumps it. `main.py`'s own `_progress` prints are `flush=True` (good), but they are queued behind whatever is in the pipe.
**Fix:** `read(4096)` and forward chunks, or set `progress=False` for the Demucs API call and emit your own `_progress` lines; also `PYTHONUNBUFFERED=1` in the command environment.

---

# P2 — gates, claims, and brittleness

## P2-1 The repo does not pass its own CI gates

Verified in this checkout:

| gate | result |
|---|---|
| `ruff check .` | **8 errors** — `F401` ×4 (`agents.py:25 torch`, `agents.py:442 note_seq`, `device.py:30 logging`, `tests/test_hardening.py:17 os`), `UP007` ×3 (`agents.py:310,363,432`), `I001` (`app.py:17`) |
| `black --check .` | **would reformat** `agents.py`, `app.py`, `suno_postprocessor.py` |
| `mypy .` | **9 errors** — 6 × `app.py` return-type, 3 × `tools/simulate_failures.py` referencing removed `EarAgent.model/.processor/_parse_mt3_tokens` |
| `tools/check_docker_install.py` | **fails** — requires the literal `--index-url https://download.pytorch.org/whl/cpu`, which moved into `ARG TORCH_INDEX_URL` (`Dockerfile:4,36`) |
| `tools/check_env_example.py` | **fails** — `CACHE_DIR` documented in `.env.example:12`, read nowhere |
| `tools/check_index.py` | **fails** here (shallow clone); passes in CI with `fetch-depth: 0` |
| `tools/simulate_failures.py` | **9/11 fail** without the optional deps; exits 1 |
| `python -m pytest` | not run here (deps absent) |

Also `tests/` has 268 `def test_` against README's "291 tests, 91% coverage" (×2) and `requirements.lock` has 116 `==` pins against "118 exact pins" (the README, and `docs/ACCEPTED_RISKS.md` says 164 distributions). Numbers typed into prose always drift; generate them.
**Fix:** run `ruff --fix`/`black` and commit; fix the 9 mypy errors (they are real: the 3-tuple/2-tuple mismatch in P2-2); update `check_docker_install` to assert the `ARG` + `--index-url "$TORCH_INDEX_URL"` pair (i.e. "torch is not resolved from default PyPI", not a literal string); delete or implement `CACHE_DIR`; generate the counts in a docs check.

## P2-2 `app.py`'s return annotation is a 2-tuple while every path returns 3

**Where:** `app.py:242` `-> tuple[str, str | None]` vs `return msg, zip_path, inline_tab_html` (lines 458, 495) and `return msg, None, None` (247, 253, 306, 517). Gradio consumes 3 outputs. mypy cannot check any success path because the annotation is wrong; the next refactor that changes arity will fail at runtime in the browser, not in CI.
**Fix:** `-> tuple[str, str | None, str | None]`.

## P2-3 Shutdown deletes every session, including other users' in-flight ZIPs

**Where:** `app.py:761-764` `shutil.rmtree(OUTPUT_DIR)` from `atexit` and `handle_signal`.
`OUTPUT_DIR` is shared (`/tmp/tab_agent_outputs`); with `MAX_CONCURRENT_REQUESTS > 1` or any second worker, one process exiting removes another's results while Gradio is still serving them. The in-request temp files are already cleaned per session.
**Fix:** delete only the session directory in a `finally:`; leave the root alone on shutdown.

## P2-4 `cleanup_stale_uploads` prunes a *shared* temp root and may delete the current request's own upload

**Where:** `app.py:797-830` — `root = Path(tempfile.gettempdir()) / "gradio"`, called at the top of every request (`app.py:261`) with a 900 s TTL.
That directory belongs to every Gradio app on the machine (and possibly other users). With `QUEUE_MAX_SIZE=32`, a request that waited > 15 min in the queue has its own uploaded file deleted before it is read. `entry.unlink(missing_ok=True)` / `rmtree` follow no ownership check.
**Fix:** point Gradio at a dedicated dir (`GRADIO_TEMP_DIR=<app-specific>`), record the upload paths created for the current session, and prune only that dir, only files this process created.

## P2-5 `SplitterAgent` returns stem paths it never verified exist

**Where:** `agents.py:117-161` (API), `164-181` (CLI).
`_separate_with_api` returns `os.path.join(base_path, "other.wav")` unconditionally — if `separated` lacks `other`/`bass` (a model with different stem names, a partial write), the loop silently writes nothing and the paths 404 later inside `librosa.load`, surfacing as an unrelated error in the *next* stage. `_separate_with_subprocess` guesses between two directory layouts and returns without checking either. `except subprocess.CalledProcessError: raise` (line 175) is a no-op that hides the intent to wrap.
**Fix:** after writing/deriving, assert `os.path.isfile(p) and os.path.getsize(p) > 44` for each stem, and raise a typed `StemSeparationError` naming what was missing.

## P2-6 `_raw_audio_fallback` labels the full mix as "bass" and the pipeline transcribes it

**Where:** `agents.py:121-129` copies the input to both `other.wav` and `bass.wav`.
`--instrument All` (or Bass) then produces a bass tab of vocals, drums and everything else. That is not graceful degradation, it is a wrong tab with a reassuring log line. (Note `AGENTS` correctly keeps Demucs out of `REQUIRED_FOR_TRANSCRIPTION` for this reason — the health decision is right, the output labelling is not.)
**Fix:** return an extra key `{"separated": False}`; have `main.py` refuse bass transcription (and prefix output filenames with `_unseparated`) when it is false, and say so in the console.

## P2-7 `EarAgent.humanize_and_clean` de-dupes on a 2-decimal *bucket*, not a window

**Where:** `agents.py:465-500` — `time_key = round(note.start_time, 2)`.
`0.014` and `0.016` round to `0.01`/`0.02` and are both kept; `0.0149` and `0.0151` round to `0.01`/`0.02`. Two notes 1 ms apart inside one bucket are deduped, two notes 2 ms apart across a boundary are not. Bass upper-limit is hard-coded `> 67` (line 486) rather than derived from the tuning, so a 24-fret 4-string bass loses legitimate notes above G4 (67) with no log.
**Fix:** compare against the last accepted note of that pitch within a real time window (`abs(start - prev[start]) < 0.03`), and derive the range from the tuning + frets.

## P2-8 `note_seq` is imported lazily inside four methods and unused at module level

**Where:** `agents.py:411, 427, 442, 508`. `import note_seq` inside `_filter_by_instrument_range` (442) is unused (ruff F401). The pipeline therefore depends on `note-seq` being importable but never states it at import time, so a missing `note_seq` fails inside a transcription call rather than at startup with a clear message.
**Fix:** import at module scope in the same guarded block as `basic_pitch`, and set a health component if it is missing.

## P2-9 `NONE` of the technique/export constants are shared

**Where:** `main.py:19-23` defines `TECHNIQUE_*`; `agents.py:739-753` writes the raw strings `"pick"/"slide"/"hammer"/"pull"`. A rename in one file silently degrades `_format_position` to printing bare frets (its `else` branch) rather than failing.
**Fix:** a single `techniques.py` (or `enum`) imported by both, and a test that every value `TabAgent` can emit renders with its marker.

## P2-10 `validate.py`'s Lua checks are structurally incapable of passing

**Where:** `validate.py:484-488` asserts `"os.execute(" in content`, but `reaper/TabAgent.lua` was converted to `io.popen` — verified `False`. The remaining checks are `content.count("{") == content.count("}")` and `"reaper." in content`, i.e. comments and string literals count toward brace balance. Nothing verifies comments, globals, or the `InsertMedia` mode. Meanwhile `tools/check_reaper_settings.py` **skips with exit 0** when `lupa` is absent (verified) — so a local `python validate.py` can be fully green with a broken REAPER script.
**Fix:** assert the intent (`"io.popen(" in content or "os.execute(" in content`), add `luacheck reaper/*.lua` to CI (`--no-max-line-length`, `--globals reaper`), and make the lupa check fail when `CI=true` rather than skip.

## P2-11 `check_skip` inverts the meaning of its own argument in half its uses

**Where:** `validate.py:46-52` — `def check_skip(name, condition, detail): if condition: SKIP else PASS`. The caller must remember that `True` means "skip", and the module does not:
- line 408 `check_skip("export_midi()", "basic_pitch" in str(e).lower(), str(e))` → skips when Basic Pitch is *missing from the message*, passes otherwise;
- line 414 `check_skip("Transcription backend available", not basic_pitch_ok, …)` → a missing backend reports **SKIP**, and the summary prints SKIP with a ⏭️ while `FAIL=0`.
So "the whole pipeline cannot transcribe" is not a failure of the validation suite. Combined with P2-10, `validate.py` is a green light that does not test the things it says it tests.
**Fix:** replace with explicit `skip(name, reason)`; never derive a skip from a condition; hard-fail missing backends in CI.

## P2-12 `index.xml` pins a commit whose REAPER script is the *old* one

**Where:** `index.xml:24` pins `4061696b…`; `index.xml:27,38` still say `version name="1.0.1"` while `pyproject.toml`/`--version` say `1.1.0`.
Verified against GitHub: the blob at `reaper/TabAgent.lua` in that pinned commit is **256 lines, uses `os.execute`, has no streaming** — i.e. ReaPack users install the pre-streaming script, not the 327-line `io.popen` version in this tree. The pinned commit is also 5 commits behind `main` (compare API: `ahead_by: 5, behind_by: 0`). And `tools/check_index.py` only asks whether the sha *resolves locally* (`_commit_exists`), which a `fetch-depth: 0` CI job answers "yes" while shipping stale script bytes.
**Fix:** regenerate `index.xml` from the release commit with `reapack-index`; extend `check_index.py` to compare the pinned blob's content hash against the working-tree file (`git cat-file -p <sha>:reaper/TabAgent.lua` vs `reaper/TabAgent.lua`) and to require `<version name>` to equal the project version.

## P2-13 `Settings.lua` writes tuning/frets nowhere, and its "profile" is enforced in Python

**Where:** `reaper/Settings.lua:55-56,141-142` persist `onset`/`frame` as **raw strings** with no numeric validation (the manual entry `abc` is accepted, shown in the confirmation dialog, then ignored by `tonumber` in the Lua and by argparse in Python). The docstring/`index.xml` advertise "tuning preset … frets", but the only tuning path is `--profile <key>`, which calls `init_memory.save_profile` and **rewrites the user's Python-side config** (destroying `device`, see P1-4). So changing a preset in REAPER silently changes the CLI/web configuration, which nothing documents.
**Fix:** validate `tonumber(parts[4])` in the settings dialog and show an error; either add real `tuning`/`frets` ExtState keys passed as CLI flags, or stop claiming it; document the shared-memory coupling (or give REAPER its own profile file).

## P2-14 `--output-dir` default is decided by probing `/app`

**Where:** `main.py:393` (`elif os.path.exists("/app/output")`) and `init_memory.py:123` (`if os.path.exists("/app")`).
Any Linux machine with a top-level `/app` directory (common: AppImage mounts, other containers, a dev's own `/app` project) silently redirects output and user memory there — possibly unwritable, producing an `OSError` inside `os.makedirs` at startup, or a permission-denied on the atomic write at the end. Both probes are unrelated to whether *this* project is containerised.
**Fix:** one env var (`TAB_AGENT_DATA_DIR`, default `./output`, set to `/app/output` in the Dockerfile), no filesystem sniffing. `init_memory.get_memory_dir` has the same problem and additionally uses `sys.platform` to mean "is Docker".

## P2-15 `apply_profile`'s `memory_file` parameter is dead

**Where:** `main.py:218-241`. The signature accepts `memory_file: str | None = None`, the docstring says "path to persist to", and the body never reads it. `ruff`'s `ARG` rules aren't enabled, so it survives.
**Fix:** pass it to `save_profile(profile_key, memory_dir=...)`, or delete it.

## P2-16 `process(notes, is_suno, metrics)` documents a dependency on `metrics` it does not have

**Where:** `suno_postprocessor.py:214-237`. `metrics` is accepted, documented ("Detection metrics from SunoArtifactDetector") and never used. Callers pass it, tests assert around it, and the docstring implies the cleanup is adaptive. Same for `SunoAudioPreprocessor.sample_rate` (line 121) which is set and never read.
**Fix:** delete the parameter, or actually use it (e.g. scale `_remove_octave_errors` by `hi_freq_ratio`).

## P2-17 `_remove_spurious_high_notes` and `_remove_octave_errors` delete real notes

**Where:** `suno_postprocessor.py:279-294` and `240-277`.
Verified:
```
E2+E3 (40,52) simultaneous  -> [40]          # legitimate octave doubling erased
A2+A3 (45,57) simultaneous  -> [45]          # same
power chord + octave        -> [40,47]       # 52 erased
2 of 3 notes > 84           -> all >84 dropped      # a high-register solo loses its top
1 of 4 notes > 84           -> kept
```
Both rules are global and unconditional: a fixed 30 % ratio and "always keep the lower octave". Octave doubling is extremely common in real guitar parts and in AI-generated ones, and the whole cleanup only runs when the (unreliable, P1-12) detector says "AI".
**Fix:** make both aggressiveness-scaled and evidence-based: drop the upper octave only when its velocity is materially lower or its duration materially shorter than the lower note's; apply the high-note rule per time window with the column-uniqueness guard, and never drop the only note in a column.

## P2-18 Missing static checks (the ones that would have caught the P0s)

None of these exist; each maps to a specific defect above.

1. **Tab invariant validator** (would catch P0-1, P0-4, P0-7): for every column, `len({p["string"]}) == len(column)`; `tuning[p["string"]] + p["fret"] == pitch`; `0 <= fret <= num_frets`; a hand span ≤ 5 frets unless explicitly allowed; times non-decreasing. Run it inside `generate_tab` (assert) *and* as a property test.
2. **Round-trip MIDI validator** (would catch silent loss): parse the exported `.mid` with `pretty_midi` and assert the pitch multiset and durations equal the input `note_seq`. Verified working: 2 notes in → 2 notes, correct times. Add the zero-duration/velocity-0 case and overlapping same-pitch (which `note_seq` truncates silently — verified: 0.1 s overlap became 0.1→0.5).
3. **Render-then-reparse** (would catch P0-2, P1-15): parse the emitted ASCII tab back to `(column, string, fret)` and compare to `tab_data`; assert zero columns contain `/`.
4. **luacheck in CI** (would catch P2-10): `luacheck --no-max-line-length --globals reaper reaper/*.lua`, plus a `lupa` stub that asserts `InsertMedia` mode, undo balance on every exit path, and no accidental globals.
5. **Audio preflight asserts** (would catch P1-2, P1-17): processed audio is finite, non-silent (`rms > 1e-5`), same samplerate and length as input, and `max(|y|) <= 1.0`.
6. **Config schema** (`jsonschema` on `user_preferences.json`): thresholds were the only validated field; tuning/frets/device/sensitivity are not (P1-5, P1-4).
7. **Gate self-tests**: each `tools/check_*.py` should have a test that feeds it a deliberately broken fixture and asserts exit 1. `check_docker_install` (P2-1) and `validate.py`'s Lua check (P2-10) are both currently incapable of passing, i.e. they were never exercised against the code they guard.

---

# Claimed features the code does not deliver

| Claim | Where | Reality |
|---|---|---|
| "Column-aligned tablature — Chords grouped vertically" | `README.md` Features | Chords are never grouped for fingering (P0-1); the renderer chains groups (P0-2) |
| "a single unplayable note no longer discards the whole track" | `README.md` | True, but the rest of the chord is silently re-fingered as an arpeggio (P0-3) |
| "Multi-track output — Lead guitar, rhythm L/R" | `README.md` | For mono/centre content all three are one signal (P0-6) |
| "Technique detection — Slides (`s`), hammer-ons (`h`), pull-offs (`p`) annotated" | `README.md` | Fires on simultaneous notes; 1-fret "slides"; slides shadow hammer-ons (P0-5) |
| "Viterbi-style DP for optimal fingering paths" | `agents.py:531` docstring | Optimal per note in time order; no simultaneity constraint, no hand model (P0-1, P0-4) |
| "Lead (centre) / rhythm (panned L/R)" | `agents.py:186-194` docstring | Subtraction with a 14 dB leak, not separation (P0-6) |
| "GPU acceleration … `vulkan` (AMD Vulkan)" | `README.md`, `device.py`, `Dockerfile`, `RUNBOOK.md` | `onnxruntime-vulkan` is not on PyPI (P1-8) and the UI drops the backend anyway (P1-6) |
| "Chords grouped vertically … within 50 ms" | `main.py:122-124` docstring | Chained comparison, unbounded group length (P0-2) |
| "`process()` … Detection metrics from SunoArtifactDetector" | `suno_postprocessor.py:227` | Parameter unused (P2-16) |
| "The harmonic peak filter prevents pure tones … being falsely flagged regardless of aggressiveness" | `suno_postprocessor.py:11-13` | 10 kHz sine is flagged (P1-12) |
| "Status is derived from component health … never hard-coded" | `monitoring.py:97-101` | `agents.py:58` hard-codes `set_status("up")` (P1-9) |
| "Bass processing … preserve low frequencies (fundamental tones below 200 Hz)" | `agents.py:222-230` | Composite is −4.7 dB at D4; a 200 Hz split is not neutral (P1-16) |
| "Detects AI-generated audio artifacts" tuned for Suno/Udio | `suno_postprocessor.py:26-30` | White noise and white-noise-like cymbals are flagged (P1-12) |
| "291 tests, 91% coverage" / "118 exact pins" / "164 distributions" | `README.md` ×2, `ACCEPTED_RISKS.md` | 268 / 116 / — (P2-1) |
| "Verbatim" tablature sample | `README.md:190` | Contains an impossible same-string, same-instant chord — the bug, not an example (P1-14) |
| `--no-tab --no-json` documented as "export JSON only" | `README.md` Quick Start | Correct, but the summary still lists all three formats (P1-3) |

---

# What is actually good (so the above is read in proportion)

Called out because a review that only lists defects misleads:

- `main.py`'s threshold validation is genuinely well done: range + NaN/inf, applied to CLI *and* profile values, failing at start rather than producing an empty transcription.
- Atomic writes (`os.replace` on `.tmp`) in `main.py:615`, `app.py:_save_device_preference` — correct pattern.
- `app.py`'s `tempfile.mkdtemp` per session and the bounded Gradio queue fix two real classes of bug (cross-session file bleed, OOM under load).
- Structured JSON logging with per-stage timing windows, and the *derived* health model in `monitoring.py` are above the bar for a project this size — the only defect is the one line that hard-codes the status.
- `SplitterAgent`'s API → CLI → raw fallback chain is the right shape; the problems are the unverified paths and the mislabelled fallback, not the strategy.
- The 8 profiles in `init_memory.py` and `reaper/Settings.lua` are **in sync** — I diffed every tuning, fret count, threshold and flag; zero drift. That is better than most duplicated config in the wild.
- `tools/check_reaper_settings.py`'s idea (execute the Lua against a stub and assert the emitted CLI) is exactly the right test for that integration; it just needs the `InsertMedia` assertion and a non-skipping failure mode.

---

# Suggested order of work

1. **P0-1 + P0-2 + P0-7** together (column-based fingering, honest grouping, tuning-derived labels) — one change, one new invariant test, and the README sample regenerates itself. Nothing else about tab quality is worth fixing until chords are physical.
2. **P0-8 + P0-9** (REAPER `InsertMedia(path, 1)`, undo scoped to the import) — these are the only findings that can damage a user's project.
3. **P1-2 + P1-3 + P1-1** (preflight, real exit codes, classified errors) — turns silent wrong output into diagnosable failure.
4. **P1-9 + P1-6 + P1-8** (health, device relay, remove the non-existent package) — restores the truth of the deployment story.
5. **P1-10 … P1-13** (Suno processing: 20 Hz HPF, `length=`, purity guard, preserve sr/channels) — cheap, and each one measurably reduces octave errors.
6. **P1-4 + P1-5 + P2-1** (config merge, config validation, make the gates green) — otherwise every future change is unverified.
7. **P2-18** (the missing validators) so none of the above can regress.
