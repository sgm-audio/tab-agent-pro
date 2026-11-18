# Substantiation Log — Tab Agent Pro

**Audited baseline:** `a4814b2588bbfb93819ab50b1742d9fe6d1b3824` · 2026-11-18 UTC

These findings describe that revision, not subsequent fixes. Source line links are pinned to the audited commit. See [Unreleased changes](../CHANGELOG.md#unreleased) for the remediation pass.

**Decision: do not treat the output as reliable tablature.** Use it as a draft to check against the recording. The solver can minimize its own cost and still assign several simultaneous notes to one string. A successful export, a readable grid, or a green test for flag construction is not proof of musical accuracy or a working integration. See [recorded probe outputs](evidence/2026-11-18.txt) and the [reproduction steps](#reproduce-the-probes).

Evidence levels describe **support for the stated claim**, not confidence in a finding against it. **None** = no qualifying evidence; **Weak** = proxy evidence only or contradicted at the claimed scope; **Moderate** = directly reproduced on a narrow slice; **Strong** = direct, independent checks or an enforced invariant at the stated scope.

| Claim | Current evidence level | Required next evidence | Sufficient to trust the tab? |
|---|---|---|---|
| **Playable, rhythmically usable tabs** | **Weak.** Individual frets are in range ([position generation](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L563-L576)); a playable six-note chord nevertheless lands entirely on one string. Different rhythms render identically. The recorded-chord probes also contain exact-onset string conflicts. | Hard constraints for simultaneous and held notes, reachable fret spans, and explicit handling of impossible groups. Zero unflagged string conflicts; blind players must be able to reproduce timing and voicings from the deliverable. | **No.** |
| **Viterbi minimizes the implemented per-note cost** | **Strong, for that objective only.** The [min-cost recurrence and backtrace](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L682-L729) agreed with exhaustive enumeration in 128/128 seeded, non-example sequences across four tunings. | Keep the exhaustive oracle in regression coverage; when adding chord/sustain states, compare with an independent constrained oracle. | **No.** The cost does not model physical playability or identify what the performer played. |
| **High-polyphony assignment is correct** | **Weak.** Six and seven-note counterexamples below; a six-string score-derived clip yields a group of seven and 121 excess events at identical time/string pairs. `test_chord_simultaneous_notes` feeds the renderer *preassigned* strings and only checks labels. | Score against synchronized string/fret truth on strummed chords, overlapping voices, arpeggios, unisons and over-capacity groups. Report position accuracy, conflicts, dropped notes and reason per clip, not only note F1. | **No.** |
| **Reliable slides, hammer-ons and pull-offs** | **Weak.** Onsets, inferred string and fret delta are the only [annotation inputs](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L731-L756). All-picked input with gaps yields all three non-pick labels; reversing the order of simultaneous notes changes hammers to pulls. | Blind, time-aligned technique labels on real performances, including matched pick/legato, re-attacks, rests, bends and chord negatives. Report per-class precision/recall, false positives and abstentions. | **No.** Treat marks as guesses. |
| **Accurate guitar and bass on real full mixes with drums/vocals** | **None for the successful-separation quality claim.** Real and rendered mixes were run, but all used raw fallback. The singer-stem and jazz-trio controls generated purported guitar, and the app returned success on fallback. No note-scored real mix with confirmed Demucs success was found or run. | Pin/verify the model and run it on a fixed set of real multitracks with guitar/bass targets plus drums, voice and other pitched instruments. Score isolated input, separated stems and raw fallback against the same references; include instrument-absent and center-panned controls. Make fallback explicit on each export. | **No.** |
| **Alternate tunings are handled end to end** | **Weak.** Configuration alone has moderate evidence: Drop D and 4/5-string profiles change the accepted pitches. But Drop D prints low `E`, both synthetic B0 events arrive over 200 ms late on five strings, and a fixed upper bound drops valid pitches in a raised tuning. | Real recorded Drop D and 5-string low-B inputs with note/string ground truth; high/low range boundaries, chords and retunings. Verify printed labels and JSON carry the actual tuning and flagged omissions. | **No.** |
| **README example is representative** | **None.** The shown grid **was reproduced**, but it is one track from a three-second synthetic clip and itself contains two frets at one exact onset on the D string. No sampling rule, reference or error distribution supports generalizing from it. | Label it as a format example; select a median case on a frozen test set and show difficult/failure cases as well, with the commands, complete outputs and per-slice scores. | **No.** No claim about the author's selection intent is warranted. |
| **REAPER settings reach the CLI in the documented workflow** | **Weak.** The stock Lua guard passed flag construction, but the same script with the documented source-file API contract never reaches `io.popen`. The ReaPack index guard also accepts relative download targets. | Install from ReaPack and run on a newly glued item in actual REAPER. Capture the process arguments and resulting files for non-default thresholds, instrument/profile, spaced paths, output directory and each export toggle, plus failed launch/no-output behavior. | **No — integration evidence is not musical evidence.** |
| **Lockfile gate guarantees the shipped dependency closure** | **Weak.** The static check passed; `--verify` rejected a real missing lock entry after install. A synthetic missing transitive package passed both verify modes. HEAD CI did not reach these steps. | Fresh Python 3.12 CPU image build from exact wheel identities; compare the image's actual inventory and required transitive metadata, excluding only justified packages; reject missing requirements, wrong variants and unverified pins. | **No — release gate only.** |
| **Vulnerability gate runs and blocks unaccepted/expired findings** | **Moderate.** Actual installed-set scan failed on four unaccepted findings on locked versions. Synthetic tests rejected a returned expired finding but accepted a skipped package and an expired entry not returned by the feed. No successful current-image scan. | Scan the actual release inventory and attest package coverage; prove scanner failures, skipped runtime distributions and expiry policy fail closed. Re-review existing exceptions against the current dependency paths. | **No — release gate only.** |

## Musical evidence and counterexamples

**Runtime.** Linux x86-64, Python 3.11.2, Basic Pitch 0.4.0 bundled `nmp.onnx` (SHA-256 `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`), ONNX Runtime 1.31.1 CPU provider, NumPy 2.4.6, Demucs 4.1.0. Range install followed by the 116 lock pins; PyPI Torch 2.5.1+cu124 ran on **CPU**. This is not the Python 3.12 Docker image. CLI runs had fresh working directories and default thresholds unless noted. The README rerun attempted model downloads with an empty cache and hit TLS/egress failures. Subsequent runs set `HF_HUB_OFFLINE=1`; the legacy weight fetch still failed in this environment. Every audio result below is **real ONNX inference, not a mock, with raw fallback, not successful Demucs**.

### What is optimized, and what is not

- [TabAgent](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L654-L729) drops notes that have no position, then optimizes over a layer per note, in input order. It projects away note ends and has no state for occupied strings or hand span. Fast same-string moves are rewarded, including at `delta = 0`. The oracle in [`tools/probe_tab_evidence.py`](../tools/probe_tab_evidence.py) independently enumerates paths over 2–5 non-overlapping notes, then scores them using the production objective. That checks the optimization, not the choice of objective.
- At one onset, feed MIDI `[40,47,52,56,59,64]` with 600 ms holds. The standard E-major shape is `0-2-2-1-0-0` (low to high); the solver outputs `0/7h/12h/16h/19h/24h` entirely on the low E. Reverse the six input events at the same onset and the path now annotates five pull-offs. Add pitch `67` to the onset: **7 emitted, 0 skipped**. A2 held from 0 to 650 ms plus B2 at 150 ms can use distinct strings, yet the solver uses low E frets 5 and 7 for both.
- Feed four independently picked notes `40,42,46,42` with 150 ms onset spacing, 80 ms holds and 70 ms of silence between them: `pick, slide, hammer, pull`. With only pitch/onset and assigned position, the code cannot tell an actual hammer-on from a picked note sharing those features; it calls all qualifying 1–2 fret changes slides. [`test_technique_detection_slide`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/tests/test_tab.py#L134-L144) and the hammer test restate the rule; [`test_technique_detection_pull_off`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/tests/test_tab.py#L157-L170) accepts **any** non-pick label. Neither the pitch bends nor the audio envelope reach this heuristic ([model conversion](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L374-L387), [note projection](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L409-L425)).
- [ASCII rendering](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/main.py#L118-L183) groups within **<50 ms of the first onset in a column** and joins same-string events with `/`. This preserves text that was previously overwritten; it neither validates a chord nor encodes elapsed time or note duration. Probed 0.5 s and 5 s onset gaps: same grid after stripping the generation timestamp. [JSON](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/main.py#L200-L215) keeps start times but has no durations, open-string pitches or profile identifier. MIDI is a separate timed output, not proof the tab can reproduce it.

### Audio comparisons

For scored rows, the reference is the authored score or the generated fixture, measured against *decoded tab pitches*, not MIDI-only output. Equal MIDI pitches and onsets within ±50 ms are matched one to one without fitted offset, octave correction or threshold tuning; F1 = `2 × matches / (reference + emitted)`. A string match additionally requires the reference string (therefore same fret under the fixed tuning); different feasible voicings are not necessarily mistakes. A collision column has at least two entries on one string. **Exact-onset excess** counts entries beyond one per identical `(time, string)`; collisions within 50 ms alone do not prove simultaneity in the original audio.

| Input and tab measured | Observed result | What it does and does not establish |
|---|---|---|
| Bundled `guitar_solo.wav`, lead; also ran bundled bass with `--instrument Bass` | Guitar grid matches README after removing the runtime timestamp and right padding. Not byte-for-byte; only one of the nine MIDI/tab/JSON files is displayed. Lead: 16 notes / 9 columns, 2 collision columns. `24/17p` is **same string and exact onset** (D, `1.8924263 s`) in JSON. Bass: 21 notes / 13 columns, one exact-onset excess. | Both WAVs are 3 s mono PCM, described as synthetic in [the prior audit](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/AUDIT.md#L511-L526). The displayed lead really exists; it proves neither correctness nor representativeness. No source note/technique truth shipped. |
| Two recorded acoustic chords: first file (`_1`) for each of the source repo's only labels, A and C | A: 27 notes / 19 columns, **4** collision columns, **2** exact-onset excess. C: 20 / 11, **3** collision columns, **2** exact-onset excess. Emitted pitch classes stay in the labeled triads. | Positive evidence for pitch classes on two clean recordings, negative for playable high-polyphony output. Chord labels do not establish exact note times or voicings. The default preprocessor raised A's thresholds to 0.6/0.4; C used 0.5/0.3. |
| AlpacaGuitarSet `starlight`, isolated nylon-guitar render (29.14 s, standard tuning, up to six reference notes at an onset) | **313 / 324** reference events matched; **357** emitted (44 extra, 11 missed), F1 **0.919**. Only **77** matches also on the score's string. **64/93** columns collide, **121** exact-onset excess, largest column **7**. | High pitch/onset F1 can coexist with unusable chord assignments. This is score-rendered material, not a held-out human performance. |
| AlpacaGuitarSet `blur_of_life`, 48 s isolated bass render **vs. mix**, both `--instrument Bass --profile bass_4_string --onset 0.5 --frame 0.3` | Same 96 reference events. Isolated: 238 emitted / 77 matches (F1 **0.461**). Raw mix with other instruments: 538 emitted / 65 matches (F1 **0.205**), 473 extra. | Quantifies the fallback penalty for this rendered arrangement, not Demucs quality. Preprocessing can also differ on mix vs. stem. |
| AlpacaGuitarSet `tec`, distorted-guitar technique render | Five score-marked transitions (two ascending and one descending legato, two slides). At the five arrival pitches, **all seven** emitted events within ±150 ms were `pick`. Elsewhere the lead emitted 28 slides, 2 hammers and 2 pulls. The four-note all-picked synthetic Drop D run also emitted 2 pulls. | A diagnostic, not per-class precision/recall: `tec`'s render clock may be shifted (nearest predictions ~117–128 ms early), and an authored mark is not a human performance annotation. Even the wider window found no correct label at those pitches. |
| Real mixtures: IRMAS `And The Body Will Die-8` (2 s, labels `gel`/`voi`); Jazz Trio Database `All Gods Children` (5 s of piano, bass and drums) | `--instrument All` returned guitar and bass exports on both, including **50** lead tab events from the guitar-absent trio. For IRMAS, `other.wav` and `bass.wav` had identical SHA-256 and were copies of the preprocessed mix. | Actual audio reaches the pipeline; an instrument heading is not evidence the instrument was isolated or even present. These excerpts have no aligned guitar/bass tab truth. |
| Negative controls and no-output behavior | MedleyDB `NightOwl` male-vocal stem: **11** lead and **11** bass tab events. DALI's 1.96 s `B.Y.O.B.` excerpt: four zero-note tracks, **exit 0**, `Done` with enabled format names but no tab/MIDI/JSON. | These outcomes rule out using successful completion as a quality gate. They do not measure model recall on an unannotated mix. |
| New synthetic Drop D and low-B plucks, explicit thresholds 0.5/0.3 | Drop D profile: 4/4 onsets in tolerance, **9 extra** events; standard: 2/4, **9 extra**. Low-B: 2/4 within tolerance with either bass profile; 5-string allowed both MIDI 23 detections, but at about **+214/+230 ms**, 4-string excludes them. | Wiring, not accuracy. Isolated D2 maps to `string=0,fret=0` yet prints `E\|0` ([label selection](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/main.py#L107-L115)). A six-string tuning raised two semitones admits MIDI 90 at fret 24; [EarAgent](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L428-L452) drops it before assignment because guitar `max_pitch=88` is fixed. |

[Full-mix code path](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L92-L162): API, CLI, then raw copies for both targets. `other` is *not* a guitar stem, and the [mid/side formula](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L188-L220) makes the rhythm channels 20% amplitude copies for mono/center content. `TestSplitterAgentSeparateStems.test_separate_stems_api_and_cli_fail_raw_fallback` and `TestSplitterAgentRawAudioFallback.test_raw_audio_fallback_content_preserved` test recovery, not separation quality. [The “end-to-end” synthetic test](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/tests/test_pipeline.py#L76-L120) replaces the transcriber, as does [the benchmark](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/tests/test_benchmark.py#L56-L87). The real IRMAS upload returned `Transcription Complete!` and a 15-entry ZIP, without a separation warning in the status. `/ready` remained true with degraded status. The API also requests `segment=10`; [upstream documents a 7.8 s maximum for Hybrid Transformer](https://github.com/facebookresearch/demucs/blob/e976d93ecc3865e5757426930257e200846a520a/README.md#L202). Test the actual API with cached weights rather than inferring that either API success or failure was proved here.

## REAPER and release-gate evidence

- **REAPER flags, not a working action.** [`Settings.lua:64–74`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/reaper/Settings.lua#L64-L74) writes ExtState; [`TabAgent.lua:54–69, 113–139`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/reaper/TabAgent.lua#L54-L69) reads it and [builds flags](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/reaper/TabAgent.lua#L113-L139). `tools/check_reaper_settings.py` passed all 15 checks with Lupa; it seeds ExtState, makes `GetMediaSourceFileName` return `"", AUDIO_PATH`, intercepts `io.popen` and [never runs the deferred callback](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/tools/check_reaper_settings.py#L38-L73). Running `Settings.lua` first with a nine-field dialog fixture also produced the expected flags (including spaced paths) under that same stub, not in the DAW. The [documented Lua API](https://github.com/ReaTeam/Doc/blob/746c22eec33ecd069b6e8a10fa007833d80453c3/reascripthelp.html#L2822-L2827) returns **one** string. With that contract in the stub, [`local _, fn = ...`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/reaper/TabAgent.lua#L76-L102) leaves `fn=nil`, stops at `file_exists`, and no Python command is launched. This is a contract reproduction, not an in-DAW test. The same filename failure reproduces on the action indexed at `4061696b`. The stock guard cannot check completion. In a separate Lupa probe that deliberately bypassed the filename problem and **executed** deferred callbacks, `export_midi=0` suppressed the no-output dialog, and `export_midi=1` showed it when no MIDI existed. But given a failed child and an old `song_lead_guitar.mid`, it imported the old file *before* reporting the error ([completion handler](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/reaper/TabAgent.lua#L288-L327)). The [pipe read](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/reaper/TabAgent.lua#L182-L201) is blocking despite `defer`; UI responsiveness was not tested in REAPER.
- **ReaPack false positive.** `check_index.py` initially failed because this clone was shallow. After fetching the pinned commit (`4061696b`), the guard passed; GitHub has both blobs there, although its `TabAgent.lua` is an older non-streaming copy. That does **not** make the two relative `<source>` paths in [`index.xml`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/index.xml#L24-L43) installable. The [ReaPack parser](https://github.com/cfillion/reapack/blob/1b5ef48218949ea3bd92034e3a20b28f900006ef/src/index_v1.cpp#L149-L170) takes the source text as the URL; the [installer passes it directly to the downloader](https://github.com/cfillion/reapack/blob/1b5ef48218949ea3bd92034e3a20b28f900006ef/src/install.cpp#L59-L75), not relative to the index or its `commit` attribute. Also, absent `main` leaves the registration section mask at zero. The guard checks the local path, not actual download/registration. No real REAPER install was performed. The two current source bodies are `reaper/TabAgent.lua` and `reaper/Settings.lua`; neither identifies a download host.
- **Current CI is red, not the previous release's CI.** [Run 43330247605 on the exact SHA](https://github.com/sgm-audio/tab-agent-pro/actions/runs/43330247605) failed at Ruff and skipped lock, scan, tests and REAPER; local `ruff==0.11.5` reports eight violations. [Run 42356262183](https://github.com/sgm-audio/tab-agent-pro/actions/runs/42356262183) was green on its parent, not on HEAD. With the index commit fetched and `DEVICE` unset, local `pytest` ran **270 passed, 5 failed** (two Docker guard tests, two `.env` guard tests, one validator test; 93% statement coverage in this environment). Forcing `DEVICE=cpu` adds a test failure in `test_separate_with_api_uses_cuda`; that mocked CUDA test passes with the variable unset. Coverage is not a transcription metric.
- **Lockfile.** `--static` passed (17 declared dependencies, 4 exact declarations checked); the lock has **116** entries, not 118 in README or 131 in its own comment. Its CPU-wheel example points to PyPI, unlike Docker's CPU index default. Dry-run and overlay of the pins succeeded in the diagnostic environment. `--verify --extra pytest --extra pytest-cov` then failed on **`huggingface-hub==2.4.0` in the runtime closure, absent from the lock**. `importlib.metadata` shows both locked Gradio 6.31.0 and Demucs 4.1.0 require it. Demucs 4.1.0's `pretrained.get_model()` also tried `huggingface.co/adefossez/HTDemucs/resolve/main/htdemucs.yaml` before the legacy download in the unmodified README run. Local inference is true; **“no model-hub dependency” is not**. Docker's [builder](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/Dockerfile#L48-L58) overlays pins and checks the builder inventory; CI verifies its own Python environment rather than attesting the final image. The [interpreter check](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/tools/check_lockfile.py#L238-L270) returned 0 with all 116 pins unverified in `--offline` mode (network metadata misses are also counted as unknown). The [closure walk](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/tools/check_lockfile.py#L273-L301) only visits already installed distributions. A root requiring an absent `audit-missing>=1`, with only that root installed and locked, passes `--verify` and exact mode. Both reject a version change to an installed root, and `--static` rejects removing a declared pin. These checks do not certify dependency completeness, wheel identity or variant; `setuptools` and CUDA redistributables are excluded and Torch local version tags are ignored by design.
- **Advisory scan.** `python tools/check_vulns.py` exited **1** against the diagnostic installed set. Four findings on locked versions: Pillow 12.3.0 (`CVE-2026-128066`, `CVE-2026-128067`, `PYSEC-2026-4363`) and Tornado 6.6 (`CVE-2026-126347`). This was a PyPI-feed scan, not a claim about current image contents or exploitability. [The gate](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/tools/check_vulns.py#L57-L132) trusts the returned dependency list rather than checking scan completeness: synthetic results gave exit 0 for an active exception, 1 for a returned expired exception or an unaccepted finding, and 0 for a skipped package or an expired exception not in results. `len(deps)` counts skipped entries as scanned. [R-3's “Pillow absent” rationale](ACCEPTED_RISKS.md#L61-L76) is contradicted by the lock and Gradio's mandatory Pillow metadata. The Demucs 4.1.0 legacy loader uses `check_hash=True, weights_only=False`; don't carry over [the old `weights_only` rationale](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/.pip-audit-ignore#L9-L45) from the removed YourMT3+ path. The ignore parser also silently keeps the last entry for duplicate IDs. Verify the actual model and build inventory before sign-off.

## Reconciliation with `AUDIT.md`

The original audit and [Appendix C](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/AUDIT.md#L604-L652) reuse some P1 IDs for different issues. Rows below identify both when needed. “Closed” means the **original narrow failure** was not reproduced and the fix has direct support, not that the entire feature is validated. The Appendix C phase verification/sign-off JSONs are not tracked at this SHA or present in the tracked phase-6 ZIP; a historic pass is not a current receipt. This review does not re-attest every item in the historical claim inventory (such as licensed distribution, remote deployment, GPU support or load/performance).

| Historical finding | Disposition at `a4814b2` and current evidence |
|---|---|
| **P0-1** — Gradio button arity | **Closed.** `test_process_audio_impl_accepts_five_args` passes; [button binding](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/app.py#L681-L692) points at the wrapper. A real in-process HTTP upload of IRMAS, POST `/gradio_api/call/process_audio` with five inputs, and SSE GET completed with a 15-entry ZIP. That confirms the route, not accuracy. |
| **P0-2** — Basic Pitch deps missing from image | **Narrow fix retained; whole-image claim unverified.** Quoted dependency line still includes `mir_eval`/`resampy` ([Dockerfile](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/Dockerfile#L38-L57)); actual ONNX inference works in the diagnostic venv. Docker/Podman absent. `check_docker_install.py` fails because it expects the literal CPU `--index-url`, even though Docker now defaults `TORCH_INDEX_URL` to that value and interpolates it. Do not mistake this static failure for proof that the default Docker build uses CUDA. The lock drift above is separate. |
| **P0-3** — loopback bind | **Closed in code/tests.** `test_serve_defaults` and `test_port_env_is_used` pass; [`serve()`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/app.py#L832-L853) uses `0.0.0.0` and `PORT`. No new local Docker networking test. |
| **P0-4 / P2-3; P1-5 original / N-3** — ZeroGPU, stale Space, unreachable/mutable YourMT3+ | **Retired in first-party runtime/docs.** [Agents](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L262-L271) use Basic Pitch only; no live Space is now advertised. Local GPU acceleration was not exercised. Demucs' transitive model acquisition is a separate outstanding issue, not proof of cloud inference. |
| **P1-1 original / P1-8 in C** — validator | **Re-opened.** `validate.py --quick`: 100 pass / 1 fail / 1 skip (exit 1), requires [`os.execute(`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/validate.py#L472-L488) while Lua uses `io.popen`. Fixed reporting still works. `simulate_failures.py`: **11/14**, with stale `__new__` stubs missing `log`/`device` and a removed `_parse_mt3_tokens` method. Those are broken probes, not evidence that three production behaviors failed. |
| **P1-2** — CLI profile/instrument/thresholds inert | **Closed for forwarding.** `test_cli_profile_sets_thresholds`, `test_cli_flags_override_profile`, `test_session_log_is_valid_json_after_run` and `test_default_runs_guitar_only` pass; the synthetic low-note results above distinguish tunings. The CLI now reads the Suno and technique knobs ([preprocessing](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/main.py#L414-L418), [technique sensitivity](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/main.py#L458-L468)); the string count comes from the tuning list, not an independent count override. Rendering and tuning accuracy remain open. |
| **P1-3 / P1-4** — REAPER and ReaPack | **Re-opened for the documented workflow.** The Lupa flag guard proves ordinary command construction, not the source-file call, deferred polling or installed action. Contract and source-URL failures above. |
| **P1-6 original** — failure contract mismatch | **Closed narrowly.** [Docstring and `RuntimeError` path](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L303-L354) agree that the backend is required; `test_transcribe_stem_raises_when_no_models` passes. |
| **P1-7** — full-mix claim | **Still open for quality.** README's pipeline table now says `other`, but the [opening promise](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/README.md#L7-L11) is still universal; [UI help](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/app.py#L656-L663) still says isolates guitar. Fallback is logged and functional, can be mislabeled a successful guitar transcription, and has no per-track provenance in tab JSON. |
| **P1-8 original / P1-9 in C** — wheel | **Re-opened.** `pip wheel . --no-deps` produces modules, but [the manifest](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/pyproject.toml#L49-L60) omits `device.py`, now imported by `agents.py`. `pip install --no-deps --target ...` of the wheel and `import agents` from outside the checkout fails `ModuleNotFoundError: No module named 'device'`. |
| **P1-9 original / P1-5 in C** — one out-of-range note discards the track | **Closed narrowly.** [`test_unplayable_interleaved_with_playable`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/tests/test_agents_edge.py#L279-L288) and `test_unplayable_notes_are_recorded` pass; [skip code](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/agents.py#L663-L680) counts missing positions. It does not detect over-capacity chords, and drops are not structured into tab JSON. |
| **P2-1** — `.env` unused keys | **Partly re-opened.** Dotenv/`PORT`/`DEVICE` tests pass. `tools/check_env_example.py` fails on `CACHE_DIR`; no Python or Lua component reads that documented setting. |
| **P2-2 / P2-4 / P2-5** — Gradio, test counts, CI/local and miscellaneous drift | **Mixed.** Gradio bounds remain; real HTTP works. README says 291 tests/15 failure checks, versus 275 collected/14 checks now. Ruff/mypy versions and mypy strictness now align, but local Ruff formatting plus Black differs from CI Black alone and CI pytest tools are still unpinned. Python 3.10+ still appears in [the UI footer](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/app.py#L673-L678) despite the 3.11+ manifest. |
| **P2-5 rendering / P1-6 in C / N-9** — overwritten columns and fictitious README grid | **Closed for formatting and authenticity.** `test_adjacent_columns_do_not_merge` and `test_notes_on_same_string_in_one_group_are_all_kept` pass. README grid reproduces. Those tests do not prove fingering validity; the overwrite fix now *shows* conflicting assignments. |
| **N-1 / P1-1 in C; N-2, N-4, N-5** — readiness, session isolation, metrics, TTL | **Narrow fixes retained.** `test_ready_endpoint_reports_backend_availability`, `test_same_second_sessions_get_distinct_directories`, and `test_retained_samples_are_bounded` pass. [`mkdtemp`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/app.py#L255-L263), [`deque(maxlen=...)`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/monitoring.py#L287-L296) and [request-triggered pruning](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/app.py#L777-L829) remain. Readiness is not an inference or separation quality check, and pruning is not a periodic job. |
| **N-6 / N-8** — hidden failures, root/writable container | **Code protections retained in part.** Observed explicit fallback log events; no blanket claim that all errors are surfaced to the caller (the app success above is a counterexample). Docker still declares [`USER tabagent`](https://github.com/sgm-audio/tab-agent-pro/blob/a4814b2588bbfb93819ab50b1742d9fe6d1b3824/Dockerfile#L83-L107) with root-owned code; no image build/runtime attestation here. |
| **N-7 / N-10** — dependency gates and load-bearing `setuptools` | **N-7 re-opened** for closure and current release sign-off; see actual gate exits above. **N-10 confirmed narrowly:** `pretty_midi` imports under 80.10.2 and fails in an isolated venv at 84.1.0 with missing `pkg_resources`. The distribution is needed at runtime, even though the lock excludes it as “install-time only.” This is not an advisory-exposure determination. |

## Reproduce the probes

Keep downloaded audio, transcriptions and model weights out of Git. Source files were used as published; no third-party audio is redistributed here.

| External input | Provenance at immutable revision |
|---|---|
| A/C acoustic recordings (no note-level reference) | [`regisrfn/CreateDatasetFromChords@5201deb`](https://github.com/regisrfn/CreateDatasetFromChords/tree/5201deb98376172720409fe69e0dabde79ce776b/Dataset/Main); `A/A_1.wav` SHA-256 `11dce13aa530f6f72093da3161880db64325aa2710cd1fe3e98d8232e78444c1`, `C/C_1.wav` `bbc1f2535f16b61148c5741c2b204a48ec63d4d60070e091bed58ad0cc3a7220`. |
| Recorded mixes and vocal control | [`mirdata@96d298b`, test excerpts](https://github.com/mir-dataset-loaders/mirdata/tree/96d298b5ec4b7c5dadcd498a3b50a0204b8c3a85/tests/resources/mir_datasets). IRMAS archive labels `gel` = electric guitar, `voi` = singing voice. The MedleyDB Pitch metadata labels `NightOwl_STEM_08` male singer. Jazz Trio Database metadata names the pianist, bassist and drummer. DALI metadata identifies the song but does not provide guitar/bass notes. The small GuitarSet mic/mix test fixtures are byte-identical and peak at one PCM quantization step; they were **not** used as musical evidence. |
| Score-rendered references | [AlpacaGuitarSet open subset (anonymous submission, AMIW 2026)](https://github.com/Alpaca2025/alpaca-guitarset/tree/e08936b562bf4a8de8a3768444f2682530907ec8/data/open_subset), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Unmodified Guitar Pro WAV renders; paired GM-token score annotations as published. `starlight` is in the open bonus set, `blur_of_life` in the open corpus. |

To reproduce the baseline without changing branches, extract the audited code
into a temporary directory and copy in the probe helper:

```bash
baseline=$(mktemp -d)
git archive a4814b2588bbfb93819ab50b1742d9fe6d1b3824 | tar -x -C "$baseline"
cp tools/probe_tab_evidence.py "$baseline/tools/"
cd "$baseline"
```

Use the extracted sources and requirements for the commands below. For a check
that needs Git history (such as `check_index.py`), use the archive results only
for code inspection and consult the receipt for the original Git check.

**Diagnostic setup used:** from a clean venv, install `basic-pitch==0.4.0` with `--no-deps`, then `grep -v '^basic-pitch' requirements.txt` into a temporary requirements file and install it along with `torch==2.5.1 torchaudio==2.5.1 setuptools==80.10.2 pytest pytest-cov lupa ruff==0.11.5 pip-audit==2.10.1` from PyPI. Then `python -m pip install --no-deps -r requirements.lock`. When the upstream CPU Torch wheel is available, use it for a faithful CPU build; PyPI Torch here is only a diagnostic substitute. Expected `pip check` exception in this configuration: Basic Pitch declares TensorFlow but this audit uses the bundled ONNX model.

### Audio and optimization

From the repo root with the venv active (the absolute script path and fresh cwd isolate saved profiles):

```bash
repo=$PWD
b=$(mktemp -d)
export DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_HOME="$b/torch" HF_HOME="$b/hf"
python -m tools.probe_tab_evidence logic
python -m tools.probe_tab_evidence fixtures "$b/fixtures"

mir=96d298b5ec4b7c5dadcd498a3b50a0204b8c3a85
mir_file() {
  gh api -H 'Accept: application/vnd.github.raw+json' \
    "repos/mir-dataset-loaders/mirdata/contents/tests/resources/mir_datasets/$1?ref=$mir"
}
mir_file 'irmas/IRMAS-TestingData-Part1/Part1/02%20-%20And%20The%20Body%20Will%20Die-8.wav' > "$b/irmas-mix.wav"
mir_file 'medleydb_pitch/audio/AClassicEducation_NightOwl_STEM_08.wav' > "$b/voice.wav"
mir_file 'jtd/raw/barronk-allgodschildren-drummondrrileyb-1990-8b77c067.wav' > "$b/trio.wav"
mir_file 'dali/audio/4b196e6c99574dd49ad00d56e132712b.mp3' > "$b/dali.mp3"
for c in A C; do
  gh api -H 'Accept: application/vnd.github.raw+json' \
    "repos/regisrfn/CreateDatasetFromChords/contents/Dataset/Main/$c/${c}_1.wav?ref=5201deb98376172720409fe69e0dabde79ce776b" \
    > "$b/${c}_1.wav"
done
alpaca=e08936b562bf4a8de8a3768444f2682530907ec8
curl -fsS "https://codeload.github.com/Alpaca2025/alpaca-guitarset/tar.gz/$alpaca" -o "$b/alpaca.tar.gz"
tar -xf "$b/alpaca.tar.gz" -C "$b" --wildcards '*/data/open_subset/*'
a="$b/alpaca-guitarset-$alpaca/data/open_subset"

run() {
  local name=$1 input=$2
  shift 2
  mkdir -p "$b/work/$name" "$b/out/$name"
  (cd "$b/work/$name" && python "$repo/main.py" "$input" --output-dir "$b/out/$name" "$@") \
    > "$b/out/$name/cli.log" 2>&1
}
run demo "$repo/examples/guitar_solo.wav"
# Raw-fallback runs: HF_HUB_OFFLINE still allows the legacy weight lookup;
# use an empty cache with both weight endpoints blocked for this test.
export HF_HUB_OFFLINE=1
run demo-bass "$repo/examples/bass_groove.wav" --instrument Bass
run irmas "$b/irmas-mix.wav" --instrument All
run voice "$b/voice.wav" --instrument All
run trio "$b/trio.wav" --instrument All
run dali "$b/dali.mp3" --instrument All
for c in A C; do run "chord-$c" "$b/${c}_1.wav"; done
run nylon "$a/starlight/audio/starlight_2.3_nylon_guitar.wav"
run tec "$a/tec/audio/tec.wav"
for mode in isolated mix; do
  audio="$a/blur_of_life/audio/blur_of_life.wav"
  [ "$mode" = mix ] || audio="$a/blur_of_life/audio/blur_of_life_4.5_fretless_bass.wav"
  run "$mode" "$audio" --instrument Bass --profile bass_4_string --onset 0.5 --frame 0.3
done
for profile in rock_standard rock_drop_d; do
  run "$profile" "$b/fixtures/drop-d.wav" --profile "$profile" --onset 0.5 --frame 0.3
done
for profile in bass_4_string bass_5_string; do
  run "$profile" "$b/fixtures/low-b.wav" --instrument Bass --profile "$profile" --onset 0.5 --frame 0.3
done
# Verify that each log contains demucs_raw_fallback before comparing to this log's fallback results.
grep -l demucs_raw_fallback "$b"/out/*/cli.log
python -m tools.probe_tab_evidence score "$b/out/nylon/starlight_2.3_nylon_guitar_lead_guitar.json" \
  --tuning guitar --reference "$a/starlight/gmtoken/starlight_n.guit_140.txt"
python -m tools.probe_tab_evidence score "$b/out/mix/blur_of_life_bass.json" \
  --tuning bass4 --reference "$a/blur_of_life/gmtoken/blur_of_life_fl.bs_120.txt"
python -m tools.probe_tab_evidence score "$b/out/rock_drop_d/drop-d_lead_guitar.json" \
  --tuning drop-d --reference "$b/fixtures/drop-d.ref.json"
python -m tools.probe_tab_evidence score "$b/out/bass_5_string/low-b_bass.json" \
  --tuning bass5 --reference "$b/fixtures/low-b.ref.json"
```

For the README comparison, take the fenced block under “Example tablature output” and compare its lines to `demo/guitar_solo_lead_guitar.tab` in the output directory above. Remove only the `Generated:` line first (not equal), then `rstrip()` each line (equal). Inspect the paired JSON for the two `string=2` entries with `start_time=1.8924263038548752`.

For the technique diagnostic, inspect `tec/score/tec.xml` at the legato and slide groups and the adjoining score times in `tec/gmtoken/tec_od.guit_120.txt`. On the receiving notes, use the signed pitch change to distinguish ascending/descending legato, not the token's start/stop label. Compare the tab labels at MIDI/time pairs `(71,14.125)`, `(73,14.625)`, `(72,15.125)`, `(57,16.625)` and `(59,17.625)`, widening the window to ±150 ms only for this diagnostic. The [receipt](evidence/2026-11-18.txt) records the candidate deltas and labels.

For the HTTP smoke check: use `fastapi.testclient.TestClient(app.create_app())`, upload the IRMAS WAV to `/gradio_api/upload` in multipart `files`, then POST `/gradio_api/call/process_audio` with `data = [{"path": <returned path>, "meta": {"_type": "gradio.FileData"}}, "Guitar", true, true, true]`. GET the returned `event_id` under that same call endpoint and inspect the `complete` event, status text and ZIP contents. No public listener or inference stubs are required.

### REAPER and gate failures

```bash
# Fetch only the pinned object if it is missing in a shallow checkout.
git fetch --no-tags --depth=1 origin 4061696bc2438002dbb3b5f1689385d87665e8b1
python tools/check_index.py
python tools/check_reaper_settings.py  # requires lupa; no Lupa is a skip, not proof
ruff check .
env -u DEVICE python -m pytest tests/ -o addopts='' --cov=. --cov-report=term -ra -q
python validate.py --quick
python tools/simulate_failures.py
python tools/check_docker_install.py
python tools/check_env_example.py
python tools/check_lockfile.py --static
python tools/check_lockfile.py --interpreter --offline
python tools/check_lockfile.py --verify --extra pytest --extra pytest-cov
python tools/check_vulns.py
```

For the source-file contract counterexample, run from the repository root:

```python
from pathlib import Path
from lupa import LuaRuntime
from tools import check_reaper_settings as guard

lua = LuaRuntime()
g = lua.globals()
g.EXTSTATE_INIT = lua.table(install_path=guard.INSTALL_PATH)
g.INSTALL_PATH, g.AUDIO_PATH = guard.INSTALL_PATH, guard.AUDIO_PATH
lua.execute(guard.STUB)
lua.execute('''
reaper.GetMediaSourceFileName = function(src, buf) return AUDIO_PATH end
local old = reaper.file_exists
reaper.file_exists = function(path)
  assert(type(path) == "string", "expected string, got " .. type(path))
  return old(path)
end
''')
lua.execute(Path("reaper/TabAgent.lua").read_text())  # line 93: expected string, got nil
```

To isolate the downstream completion check, keep the guard's original two-return stub, queue and execute `reaper.defer` callbacks, and have `io.popen` return one line, then EOF and `(false, "exit", 1)` from `close`. Stub `file_exists` true for an existing `song_lead_guitar.mid` in the output directory; record `InsertMedia` and dialogs to see their order. For the settings-only positive control, stub `GetUserInputs` to accept `5,/tmp/Tab Agent,Bass,0.62,0.31,/tmp/My Output,0,1,0`, run `Settings.lua` before `TabAgent.lua` in the same Lupa runtime, and verify the ExtState/command in the [receipt](evidence/2026-11-18.txt). Neither control launches a process in the DAW.

To repeat the wheel check with the diagnostic dependencies available, run `python -m pip wheel . --no-deps -w "$b/wheels"`, install the built wheel with `python -m pip install --no-deps --target "$b/wheel-env" "$b/wheels/"tab_agent_pro-*.whl`, then run `(cd "$b" && PYTHONPATH="$b/wheel-env" python -c 'import agents')`. Importing from the repository root instead would hide the missing module.

For the dependency negative control, put `Name: audit-root`, `Version: 1.0` and `Requires-Dist: audit-missing>=1` in a temporary `audit_root-1.0.dist-info/METADATA` (one field per line). Patch `check_lockfile.metadata.distributions()` to return only `importlib.metadata.PathDistribution` of that directory, `declared_requirements()` to return `{"audit-root": None}`, and `LOCK` to a temporary file containing `audit-root==1.0`. Both `check_verify()` and `main([])` print OK despite the unsatisfied metadata. Change the installed version to 2.0; both reject. For the advisory control, stub `check_vulns.audit()` to return a single package with `vulns=[{"id": "TEST-ADVISORY"}]`, and `load_ignores()` to return no exception, a 2000 expiry, or a 2099 expiry. Expected exit codes: 1, 1, 0. With the expired entry retained but `audit()` returning no finding (or only a `skip_reason`), the current gate returns 0.

**Sign-off order:** First restore the shipping, DAW and CI gates and surface fallback/no-output to users. Then solve and regression-test string occupancy/timing, and label technique guesses as such until validated. For any reliability claim, preregister slice-specific acceptance thresholds, lock versions/model checksums, and publish per-clip results from a held-out, independently annotated set (isolation, real mixes, high polyphony, and alternate tunings). Do not tune or choose README examples on that set without disclosing the selection rule.
