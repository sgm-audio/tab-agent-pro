# Transcription test audit

## What the baseline actually covered

Re-verified `a4814b2` on Linux/Python 3.11 with the repository's pinned Basic Pitch, ONNX
Runtime and Demucs versions, on CPU. CI targets Python 3.12 with CPU-specific PyTorch
wheels; this was not a Docker/REAPER-host validation. The two bundled WAVs are each
3 seconds, mono, 44.1 kHz. Neither exercises stereo separation or a full mix.

| Check | Baseline result |
|---|---|
| Pytest, clean temp root, `DEVICE` unset | **275 collected: 270 passed, 5 failed.** Failures were the stale `.env.example` key, Docker CPU-index guard, and `validate.py` expecting `os.execute` after the switch to `io.popen`. With ambient `DEVICE=cpu`, the mock CUDA test also failed. |
| Coverage | With `--cov=. --cov-branch`: **93.3% statements, 75.2% branches** (91% combined, counting tests and some tools). The same run on the seven application modules alone: **89.0% statements, 76.4% branches**. This is not model, Lua, or musical correctness coverage. |
| `simulate_failures.py` | **11/14 reached**, exit 1. Two simulations bypassed the constructor and used unset fields; an obsolete MT3 test failed before reaching both assertions. The repaired harness runs all 15 checks. |
| Guards | Static lock check passed; installed closure was missing the `huggingface-hub` pin required by the locked Demucs 4.1.0. ReaPack's pin resolves when the shallow checkout fetches the pinned object. The old Lua guard passed despite never executing the deferred callback and giving `GetMediaSourceFileName` an extra return value. |
| Dependency audit | The local `check_vulns.py` run still reports unaccepted advisories for the existing Pillow 12.3.0 and Tornado 6.6 pins. No exceptions were added, and this is not a clean dependency-audit or CI sign-off. |

### Where the old tests gave false confidence

- **Fingering and playability.** `tests/test_agents_edge.py` does test out-of-range notes,
  partial skips and reset semantics; it is not missing that branch entirely. Its
  backtracking test checks keys/counts, not open-string pitch plus fret or minimum path
  cost. `test_chord_simultaneous_notes` in `tests/test_main_coverage.py` gives the renderer
  *already assigned* string/fret positions. `15/8p` preserves text but does not make a
  simultaneous same-string chord playable. `TabAgent` still optimizes one note per layer,
  not occupied strings across a chord or a sustain.
- **Technique false positives.** Several legacy tests only assert inside `if len(tab) > 1`
  or accept any label, including `pick`. The negative cases primarily cover wide onset
  gaps and string changes, not a rest inside the onset window or a skipped intervening
  note. MIDI times, pitches and duration alone cannot prove a slide or distinguish it
  from two quickly picked notes. Treat these labels as hints, not acoustic detections.
- **Demucs.** `TestSplitterAgentSeparateStems` stubs the methods that return stem paths;
  the raw-copy tests even use `FAKE-WAV`. A zero subprocess exit or non-empty API mapping
  can look successful while a required output is unreadable, empty or from a prior run.
  The original center test only compared peaks for identical L/R, and bass tests mainly
  checked that a file exists. A raw fallback is degraded processing, not separation.
- **Basic Pitch.** `test_ear_agent_transcribes_synthetic` used to replace
  `transcribe_stem` itself. `test_benchmark.py` times DSP plus a fake model returning two
  notes at every duration; it cannot measure model accuracy or real-time factor. The
  old CI smoke and `validate.py` only checked imports. Error injection reached generic
  `RuntimeError`, but did not prove the distinction between missing dependencies,
  missing weights, and a valid clip with no notes.
- **Multi-track and REAPER.** The old web “full_pipeline” tests replace the splitter,
  transcriber and tab agent, and often check only the ZIP path. CLI filtering tests
  usually check spy calls rather than output content. The old Lua guard seeded ExtState,
  searched the command string, and treated `defer` as a no-op. This missed real selection
  failure, path handling, stale imports on a nonzero exit, and mode `0` inserting onto an
  existing track that the script could then rename incorrectly. None of the tests ran
  the Settings dialog, saved values, real CLI parser and deferred import together.

## Regression tests now in the tree

Local verification: **302 passed, 1 skipped, 3 strict xfailed**; **88.7% statements,
78.3% branches** on application code. The standalone REAPER guard passed 14 tests,
the failure harness passed 15/15 checks, and `validate.py --quick` passed 100 checks
with the Docker build intentionally skipped. In a disposable copy, removing
accumulated Viterbi cost, routing left rhythm to right audio, and letting REAPER
import after a failed child each failed the corresponding new regression test.

P0 is playability or project-state safety. The noted expected failures are open bugs,
not coverage credit; a fix must remove the marker. The strict markers only accept the
known occupancy failure, not unrelated missing/wrong-note assertions.

| Priority / test | Must assert | Why the baseline missed it |
|---|---|---|
| P0 `test_viterbi_matches_exhaustive_global_cost` | Recover exact pitch and timing and equal a small exhaustive minimum. The fixture's greedy path is strictly worse. | Counting backtracked entries cannot catch greedy or misindexed backpointers. |
| P0 `test_fret_boundaries_keep_playable_notes_after_a_rejected_event` | Classical 19-fret, Drop D, 4- and 5-string bass hit both inclusive boundaries, skip the pitches one semitone outside, reconstruct tuning exactly, and preserve chronology despite reversed input. | The existing range tests are mostly counts or distant invalid pitches on standard tuning. |
| P0 `test_playable_chord_keeps_all_pitches_on_distinct_strings` **(strict xfail)** | Simultaneous E2/A2/D3 retain all pitches with three distinct strings. | Renderer fixtures assign strings in advance; the real DP puts all three on low E. |
| P0 `test_sustained_string_cannot_be_reused_during_a_later_note` **(strict xfail)** | A sustained A2 is on a different string to the later E2 that must use low E; unrelated voices are not annotated as a pull-off. | Start-time-only transition cost ignores still-ringing notes. |
| P0 `test_impossible_same_string_dyad_reports_the_conflict` **(strict xfail)** | Simultaneous E2/F2 on standard tuning retain one and explicitly report the other as skipped. No invented alternative string or silent loss. | `last_skipped_notes` only reported individually out-of-range notes; both pitches are individually in range. |
| P0 `test_reaper_settings_dialog_reaches_real_cli_and_controls_outputs` | Execute both scripts, resolve a real persisted Bass-4 profile, exercise explicit/blank thresholds, write only enabled formats under the same directory, and import only MIDI reported by that invocation. Pitch 23 must be filtered on the 4-string profile while open E (28) survives. | String searches did not prove persistence, precedence, range filtering, output routing, or import. |
| P0 `test_reaper_never_imports_failed_or_stale_midi` / `test_reaper_no_project_mutation_on_selection_or_start_failure` | No insert, rename or undo changes for nonzero exit (including partial output), unreported old MIDI, changed project, missing selection/take/source file or failed launch. No misleading success. | The old stub never drove the deferred EOF or failure paths. |
| P0 `test_reaper_renames_only_a_successfully_created_track` | New-track insertion mode, name the new track even when it is not last, count no failed import, and close the undo block. | No previous assertion checked the track identity, return status or undo lifecycle. |
| P0 `test_reaper_posix_literal_paths_reach_the_child_unchanged` / `test_reaper_windows_uses_drive_switch_and_rejects_expansion` / `test_reaper_invalid_threshold_is_not_persisted_or_silently_dropped` | Spaces, quotes, Unicode and a literal dollar-containing path reach POSIX argv intact; Windows changes drives and rejects ambiguous shell characters before launch; invalid thresholds do not silently become defaults. | The old test only matched ordinary substrings in a command and treated a no-op callback as success. The Windows case tests command construction, not native `cmd.exe`. |
| P1 `test_no_legato_annotation_without_a_plausible_transition` | Exact `pick` annotations for a rest, simultaneous pair, skipped intermediary, and repeated attack; notes still survive where playable. | Simple positive runs and permissive labels missed those cases. The implementation now uses note ends (20 ms jitter tolerance) and sequence adjacency. |
| P1 `test_cli_all_writes_distinct_lead_rhythm_bass_note_content` | In `--instrument All`, check actual L/R/center PCM and bass filtering, exact role-specific MIDI events, JSON pitches/times, ASCII columns/Drop-D labels and the four MIDI report records. A wide playable dyad must survive on two strings in the same column. | Existing integration tests checked paths and often substituted all the agents with the same note. The new test fakes **only** separation/inference boundaries, not DSP, parser, tuning, or exporters. |
| P1 `test_incomplete_demucs_outputs_fall_back_to_readable_mix` | Missing API stem and zero-exit CLI with missing, corrupt or empty file cannot reuse old stems or report success. Copies match the real stereo input, run through DSP, and record each failure and degradation. | Happy mocked paths and arbitrary-byte copy fixtures could never catch readable-output failures. |
| P1 `test_basic_pitch_cpu_recovers_plucked_chord_and_later_note` / `test_basic_pitch_cpu_unpitched_audio_is_empty_not_backend_failure` | Run the shipped ONNX graph on CPU: match each chord member and a later note within 90 ms, permit at most three extra events, and return no notes or backend error for silence and seeded clipped percussion. | Imports, hand-made MIDI and sine/DSP-only tests did not execute or score inference. |
| P1 `test_basic_pitch_bad_weights_fail_as_runtime_error` / `simulate_basic_pitch_runtime_dependency_missing` | Missing/corrupt weights go through the real loader, report a runtime failure with the cause and count the failed stage; injected missing dependency names the module and degrades health. | A generic `RuntimeError` test accepts the wrong diagnosis or cannot distinguish empty output. |
| P1 `test_cli_with_missing_gpu_and_demucs_runs_real_cpu_inference` | Inject GPU and separator unavailability, not `predict`; decode and transcribe the mix path with actual CPU ONNX and reconstruct the bass events from real MIDI and tab. | Isolated fallback mocks never proved the degraded audio still reached real inference and exports. |
| P2 `test_demucs_cached_cpu_separates_reference_mix` (provisioned lane) | Load trusted cached `htdemucs` weights and use the real API. No download or fallback. Bass/other must be finite, nonempty, distinct from each other and distinct from the input, at expected shape/rate. | Output-existence checks or a passing passthrough can stand in for true separation. This is an execution smoke, not a source-separation quality benchmark. |

## Next gates, in priority order

These are not yet implemented. Do not use the demo files or the current xfails to sign
off on these behaviors.

| Priority / proposed test | What to assert | Existing blind spot |
|---|---|---|
| P0 `test_strummed_chord_and_fast_run_keep_distinct_playable_voices` | Once the solver tracks occupancy, synthesize a strum (e.g. 12 ms attack offsets, overlapping releases), a 40 ms alternating-picked run, and an over-wide all-fretted voicing. Each playable voice survives without string conflicts; errors explicitly identify omitted notes. Rendering must not collapse successive picks into a chord. Decide hand-span/open-string and performed arpeggio policies first. | A hard-coded 50 ms text-rendering window is not an assignment or hand-reach constraint. |
| P0 `test_reaper_native_import_and_abort_preserve_project` | With a disposable project and pre-existing sentinel tracks, test the published ReaPack pair on Linux/macOS and through native Windows `cmd.exe` (cross-drive path/spaces). Verify track names, insertion position, selection and undo, an actual failed child and no stale-file import. Verify a slow/no-output child does not freeze editing or, until reads are non-blocking, document that limitation. | Lupa stubs cannot prove OS shell, REAPER track placement/undo semantics, or UI responsiveness. `defer` schedules callbacks but `io.popen:read` can block. The current `index.xml` intentionally still pins the published revision; update it at release. |
| P1 `test_picked_pairs_octaves_and_bends_do_not_become_legato_or_false_hits` | Use a licensed annotated direct recording of fast alternate picks, a real slide/hammer/pull, bends and deliberately played octave double-stops. Report technique precision/recall separately; require no chord tone removal on the clean path and flag octave loss on the Suno cleanup path. | The heuristic does not use acoustic attack/pitch-contour evidence; Suno cleanup can drop genuine octaves by design. |
| P1 `test_invalid_decode_is_not_empty_success_and_retry_is_healthy` | Run corrupt/truncated input (including a valid container with truncated samples) and then a valid file through the real decoder/CLI or web handler. No success/export for invalid input, no model-install diagnosis for a bad clip, and readiness recovers/remains usable. | The old invalid-WAV test stopped at upload validation; runtime load failure and bad input can still affect shared health similarly. |
| P1 `test_full_mix_real_stem_and_tab_accuracy` | On a small licensed multitrack set with source stems and note/technique labels, mix vocals/drums under panned guitars and bass; include mono fold-down, polarity inversion and clipped or masked cases. Score each stem separately (note-level pitch/onset F1, chord recall, cross-stem leakage and unplayable/omitted rate). Set tolerances from a reviewed baseline; report raw-mix metrics separately and fail any silent substitution. | Stub DSP routing and a cached model smoke don't establish real full-mix accuracy or acceptable leakage. |
| P1 `test_device_switch_does_not_leak_onnx_provider_between_requests` | Run GPU → CPU and CPU → GPU selections in a clean provider test harness, then confirm parity on actual hardware. Verify Vulkan/DirectML still select an ONNX provider when Torch remains on CPU. | The app currently passes the Torch device to EarAgent and the process-global override captures the first config; mocked CUDA availability is not actual provider execution. |

## Keep execution, coverage and quality separate

- **Default PR lane:** `python -m pytest -ra`. No GPU, no downloads, no Demucs weights
  required. Basic Pitch's 0.4.0 wheel includes the ONNX weights; missing wheels or weights
  *fail* `basic_pitch_model` tests rather than relabeling them as skips. Use the ONNX-only
  install instructions and `pytest`, `pytest-cov`, `lupa`. `python -m pytest -m basic_pitch_model`
  reruns the required CPU inference cases. Coverage now counts only the seven application
  modules, not exercised tests/tools. Report statement and branch coverage separately and
  don't use either as a substitute for content assertions.
- **Deliberately reduced local run:** `python -m pytest -m 'not basic_pitch_model and not demucs_model' -ra`.
  This is not the PR gate and must not be described as an inference pass. The REAPER tests
  still run; `python tools/check_reaper_settings.py` runs them on their own.
- **Provisioned Demucs lane:** stage the trusted upstream `955717e8-8726e21a.th` in
  `$(python -c 'import torch; print(torch.hub.get_dir())')/checkpoints` (or set `TORCH_HOME`).
  Store and verify the full artifact digest in the cache setup; the test also checks the
  hash prefix from the upstream filename. Run `TAB_AGENT_TEST_DEMUCS=1 python -m pytest -m demucs_model -ra`.
  Not opting in is one explicit skip. Opting in with missing, wrong or unloadable weights
  is a failure, not a fallback. Use a dedicated runner and an offline test step. No weights
  were available for a successful run of this lane in the audit sandbox.
- **GPU lane:** add a scheduled/dispatch job per supported provider after hardware and
  driver preflight. Run it in a fresh process, require an initialized *requested* provider
  and (where supported) profiling evidence of GPU execution, then compare note-match
  metrics with the CPU reference within a tolerance, not byte-for-byte MIDI. A provisioned
  job that silently falls back is red; absence of optional hardware never counts as a GPU
  pass. Keep CUDA/ROCm Torch, ONNX Runtime builds, weights digests, and observed providers in
  the test report. Do performance timing against real inference on named hardware, not the
  existing mock benchmark.
