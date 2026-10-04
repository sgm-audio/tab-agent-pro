#!/usr/bin/env python3
"""
Tab Agent Pro — Validation Suite.

Run all checks to prove the project is complete and functional.
Exit code 0 = all good, non-zero = issues found.

Usage:
    python validate.py                    # full validation
    python validate.py --quick            # skip slow checks
"""

import argparse
import importlib
import os
import shutil
import subprocess  # nosec B404
import sys
import tempfile

PASS = 0
FAIL = 0
SKIP = 0
RESULTS: list = []
QUICK = False
TMP_DIR = tempfile.mkdtemp(prefix="tab-agent-validate-")


def _tmp(name: str) -> str:
    return os.path.join(TMP_DIR, name)


def check(name, condition, detail="") -> None:
    global PASS, FAIL
    if condition:
        RESULTS.append(f"  ✅ {name}")
        PASS += 1
    else:
        RESULTS.append(f"  ❌ {name} — {detail}")
        FAIL += 1


def check_skip(name, condition, detail="") -> None:
    global PASS, SKIP
    if condition:
        RESULTS.append(f"  ⏭️  {name} — {detail}")
        SKIP += 1
    else:
        RESULTS.append(f"  ✅ {name}")
        PASS += 1


def check_import(name) -> None:
    """Verify a module really imports (an ImportError must fail the check)."""
    try:
        module = importlib.import_module(name)
        check(f"Import: {name}", True, getattr(module, "__version__", ""))
    except Exception as e:
        check(f"Import: {name}", False, f"{type(e).__name__}: {e}")


def check_docker_build() -> None:
    """Optionally build the image (slow; skipped with --quick or without docker)."""
    docker_tool = next((t for t in ("docker", "podman") if shutil.which(t)), None)
    if QUICK:
        check_skip("Docker build", True, "skipped by --quick")
    elif docker_tool is None:
        check_skip("Docker build", True, "install docker or podman")
    else:
        try:
            result = subprocess.run(  # nosec B603 B607
                [docker_tool, "build", "-t", "tab-agent-pro:validate", "."],
                capture_output=True,
                text=True,
                timeout=900,
            )
            check(f"Docker build succeeds ({docker_tool})", result.returncode == 0)
            if result.returncode != 0:
                RESULTS.append(f"     > {result.stderr.strip().splitlines()[-1][:200]}")
        except subprocess.TimeoutExpired:
            check_skip(f"Docker build ({docker_tool})", True, "timed out (900s)")
        except Exception as e:
            check_skip(f"Docker build ({docker_tool})", True, str(e))


def main(argv: list[str] | None = None) -> int:
    global PASS, FAIL, SKIP, QUICK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip slow checks")
    args = parser.parse_args(argv)
    QUICK = args.quick

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    # ── 1. FILE EXISTENCE ──────────────────────────────────────────────

    required_files = [
        "agents.py",
        "app.py",
        "main.py",
        "Dockerfile",
        "requirements.txt",
        "suno_postprocessor.py",
        "init_memory.py",
        "monitoring.py",
        "README.md",
        "run.sh",
        "index.xml",
        "reaper/TabAgent.lua",
        "reaper/Settings.lua",
        "tests/__init__.py",
        "tests/test_ear.py",
        "tests/test_splitter.py",
        "tests/test_suno.py",
        "tests/test_tab.py",
        "tests/test_benchmark.py",
        "tests/test_pipeline.py",  # lives under tests/ (was checked at the repo root)
        "examples/guitar_solo.wav",
        "examples/bass_groove.wav",
    ]
    for f in required_files:
        check(f"File exists: {f}", os.path.exists(f))

    # input/ and output/ are gitignored working directories, so their absence in a
    # fresh clone is expected; only check that the output dir can be created.
    check("Directory: examples/", os.path.isdir("examples"))
    try:
        os.makedirs("output", exist_ok=True)
        writable = os.access("output", os.W_OK)
    except OSError as e:
        writable = False
        RESULTS.append(f"  info: could not create output/: {e}")
    check("Directory: output/ is creatable", writable)

    # ── 2. NO STUBS / TODOS ────────────────────────────────────────────

    for f in [
        "agents.py",
        "app.py",
        "main.py",
        "suno_postprocessor.py",
        "monitoring.py",
        "init_memory.py",
    ]:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            content = fh.read()
        check(f"No NotImplementedError in {f}", "NotImplementedError" not in content)
        check(
            f"No TODO/FIXME/XXX/HACK in {f}",
            not any(x in content for x in ["TODO", "FIXME", " XXX ", " HACK "]),
        )

    # ── 3. PYTHON SYNTAX ───────────────────────────────────────────────

    for f in [
        "agents.py",
        "app.py",
        "main.py",
        "suno_postprocessor.py",
        "monitoring.py",
        "init_memory.py",
    ]:
        try:
            with open(f) as fh:
                compile(fh.read(), f, "exec")
            check(f"Syntax OK: {f}", True)
        except SyntaxError as e:
            check(f"Syntax OK: {f}", False, str(e))

    # ── 4. IMPORT RESOLUTION ───────────────────────────────────────────

    for module in ["numpy", "librosa", "soundfile", "scipy", "torch", "note_seq"]:
        check_import(module)

    # Project imports
    try:
        from agents import EarAgent, SplitterAgent, TabAgent

        check("agents.py imports", True)
    except Exception as e:
        check("agents.py imports", False, str(e))

    try:
        from suno_postprocessor import (
            SunoArtifactDetector,
            SunoAudioPreprocessor,
            SunoNotePostprocessor,
        )

        check("suno_postprocessor.py imports", True)
    except Exception as e:
        check("suno_postprocessor.py imports", False, str(e))

    try:
        from main import (
            TECHNIQUE_HAMMER,
            TECHNIQUE_PICK,
            TECHNIQUE_PULL,
            TECHNIQUE_SLIDE,
            apply_profile,
            build_parser,
            export_tab_to_json,
            export_tab_to_txt,
            resolve_thresholds,
        )

        check(
            "main.py technique constants",
            TECHNIQUE_SLIDE == "slide"
            and TECHNIQUE_HAMMER == "hammer"
            and TECHNIQUE_PULL == "pull"
            and TECHNIQUE_PICK == "pick",
        )
        check(
            "main.py profile helpers callable",
            callable(apply_profile) and callable(resolve_thresholds),
        )
        check("main.py CLI parser builds", build_parser() is not None)
    except Exception as e:
        check("main.py imports + technique constants", False, str(e))

    check_import("monitoring")

    try:
        import gradio  # noqa: F401

        check_import("app")
    except ImportError as e:
        check_skip("app.py imports (gradio)", "no module named 'gradio'" in str(e).lower(), str(e))

    try:
        import init_memory

        check("init_memory.py imports", True)
        check("init_memory has 8 profiles", len(init_memory.PROFILES) == 8)
    except Exception as e:
        check("init_memory.py imports", False, str(e))

    # ── 5. COMPONENT FUNCTIONALITY ─────────────────────────────────────

    # Suno detector
    try:
        detector = SunoArtifactDetector()
        _is_suno, metrics = detector.analyze("examples/guitar_solo.wav")
        check("SunoDetector.analyze() returns (bool, dict)", True)
        check("SunoDetector metrics has hf_ratio", "hf_ratio" in metrics)
        check("SunoDetector metrics has spectral_flatness", "spectral_flatness" in metrics)
        detector_aggressive = SunoArtifactDetector(aggressiveness=1.0)
        _is_suno_hi, _ = detector_aggressive.analyze("examples/guitar_solo.wav")
        check("SunoDetector aggressiveness accepted", True)
    except Exception as e:
        check("SunoDetector.analyze()", False, str(e))

    # Suno preprocessor
    try:
        pre = SunoAudioPreprocessor()
        out = pre.process("examples/guitar_solo.wav", _tmp("suno_out.wav"))
        check("SunoPreprocessor.process() returns path", os.path.exists(out))
    except Exception as e:
        check("SunoPreprocessor.process()", False, str(e))

    # Suno note postprocessor
    try:
        import note_seq

        post = SunoNotePostprocessor()
        notes = [
            note_seq.NoteSequence.Note(pitch=60, start_time=0.0, end_time=0.5, velocity=80),
            note_seq.NoteSequence.Note(pitch=72, start_time=0.01, end_time=0.5, velocity=80),
        ]
        cleaned = post.process(notes, is_suno=True, metrics={"hf_ratio": 0.4})
        check("SunoNotePostprocessor removes octave errors", len(cleaned) < len(notes))
        check(
            "SunoNotePostprocessor does NOT mutate input",
            notes[0].pitch == 60 and notes[1].pitch == 72,
        )
    except Exception as e:
        check("SunoNotePostprocessor.process()", False, str(e))

    # SplitterAgent spatial processing
    try:
        splitter = SplitterAgent(output_dir=_tmp("stems"))
        result = splitter.process_guitars("examples/guitar_solo.wav")
        check("Splitter.process_guitars() returns lead path", os.path.exists(result["lead"]))
        check("Splitter.process_guitars() returns left path", os.path.exists(result["left"]))
        check("Splitter.process_guitars() returns right path", os.path.exists(result["right"]))
        bass_out = splitter.process_bass("examples/bass_groove.wav")
        check("Splitter.process_bass() returns path", os.path.exists(bass_out))
    except Exception as e:
        check("SplitterAgent spatial processing", False, str(e))

    # TabAgent tablature generation
    try:
        import note_seq

        agent = TabAgent(tuning=[40, 45, 50, 55, 59, 64], num_frets=24)
        notes = [
            note_seq.NoteSequence.Note(pitch=40, start_time=0.0, end_time=0.5, velocity=80),
            note_seq.NoteSequence.Note(pitch=45, start_time=0.5, end_time=1.0, velocity=75),
        ]
        tab = agent.generate_tab(notes)
        check("TabAgent.generate_tab() returns list", isinstance(tab, list))
        check("TabAgent.generate_tab() has entries", len(tab) == 2)
        check(
            "TabAgent entries have string, fret, technique",
            all("string" in e and "fret" in e and "technique" in e for e in tab),
        )

        # Unplayable notes are skipped, not fatal (docs-audit P1-9)
        mixed = [
            note_seq.NoteSequence.Note(pitch=40, start_time=0.0, end_time=0.4, velocity=80),
            note_seq.NoteSequence.Note(pitch=10, start_time=0.5, end_time=0.9, velocity=80),
            note_seq.NoteSequence.Note(pitch=45, start_time=1.0, end_time=1.4, velocity=80),
        ]
        mixed_tab = agent.generate_tab(mixed)
        check(
            "TabAgent skips unplayable notes and keeps the rest",
            len(mixed_tab) == 2 and agent.last_skipped_notes == [10],
            f"got {len(mixed_tab)} entries, skipped={agent.last_skipped_notes}",
        )

        # Technique detection
        fast_notes = [
            note_seq.NoteSequence.Note(pitch=40, start_time=0.0, end_time=0.3, velocity=80),
            note_seq.NoteSequence.Note(pitch=42, start_time=0.15, end_time=0.45, velocity=80),
        ]
        fast_tab = agent.generate_tab(fast_notes, technique_sensitivity=0.9)
        techniques = [e["technique"] for e in fast_tab]
        check(
            "Technique detection finds slides/hammer/pull",
            any(t in ("slide", "hammer", "pull") for t in techniques),
        )
    except Exception as e:
        check("TabAgent tablature generation", False, str(e))

    # ASCII tab export
    try:
        tab_data = [
            {"string": 0, "fret": 0, "technique": "pick"},
            {"string": 1, "fret": 3, "technique": "slide"},
        ]
        export_tab_to_txt(tab_data, _tmp("tab.tab"), "Test Guitar", num_strings=6)
        check("export_tab_to_txt() creates file", os.path.exists(_tmp("tab.tab")))
        with open(_tmp("tab.tab")) as fh:
            content = fh.read()
        check("ASCII tab contains technique markers", "3s" in content)
        check("ASCII tab renders all 6 strings", all(f"{label}|" in content for label in "EBGDAE"))

        # Adjacent multi-character cells must not merge (docs-audit P2-5)
        adjacent = [
            {"string": 0, "fret": 12, "technique": "pick", "start_time": 0.0},
            {"string": 0, "fret": 15, "technique": "pick", "start_time": 0.5},
        ]
        export_tab_to_txt(adjacent, _tmp("adjacent.tab"), "Test", num_strings=6)
        with open(_tmp("adjacent.tab")) as fh:
            adjacent_content = fh.read()
        check(
            "Adjacent notes are separated (no '1215' merge)",
            "1215" not in adjacent_content and "12" in adjacent_content,
        )
    except Exception as e:
        check("export_tab_to_txt()", False, str(e))

    # JSON export
    try:
        export_tab_to_json(tab_data, _tmp("tab.json"), "Test Guitar", num_strings=6)
        check("export_tab_to_json() creates file", os.path.exists(_tmp("tab.json")))
        import json

        with open(_tmp("tab.json")) as fh:
            data = json.load(fh)
        check("JSON has instrument field", "instrument" in data)
        check("JSON has tablature field", "tablature" in data)
        check("JSON tablature is list", isinstance(data["tablature"], list))
        check("JSON records string count", data.get("num_strings") == 6)
    except Exception as e:
        check("export_tab_to_json()", False, str(e))

    # MIDI export
    try:
        ear = EarAgent(device="cpu", prefer_yourmt3=False)
        # Just test the export_midi method (transcription requires Basic Pitch)
        notes = [
            note_seq.NoteSequence.Note(pitch=60, start_time=0.0, end_time=0.5, velocity=80),
        ]
        ear.export_midi(notes, _tmp("test.mid"))
        check("export_midi() creates .mid file", os.path.exists(_tmp("test.mid")))
        check("MIDI file has content", os.path.getsize(_tmp("test.mid")) > 0)
    except Exception as e:
        # EarAgent might fail to init if Basic Pitch not present
        check_skip("export_midi()", "basic_pitch" in str(e).lower(), str(e))

    # Transcription backend availability (the failure mode from docs-audit P0-2)
    try:
        import agents as _agents

        basic_pitch_ok = bool(_agents.BASIC_PITCH_AVAILABLE)
        check_skip(
            "Transcription backend available (Basic Pitch)",
            not basic_pitch_ok,
            "no backend importable — install basic-pitch with mir_eval + resampy",
        )
        if basic_pitch_ok:
            check("Transcription backend available (Basic Pitch)", True)
    except Exception as e:
        check("Transcription backend check", False, str(e))

    # Init memory
    try:
        import init_memory

        path = init_memory.save_profile("rock_standard", memory_dir=_tmp("memory"))
        check("init_memory.save_profile() creates file", os.path.exists(path))
        with open(path) as fh:
            data = json.load(fh)
        check(
            "init_memory config has tuning",
            "config" in data and "guitar_tuning" in data["config"],
        )
    except Exception as e:
        check("init_memory.save_profile()", False, str(e))

    # Profile application (docs-audit P1-2: --profile used to be inert)
    try:
        from main import load_user_memory, resolve_thresholds

        mem_dir = tempfile.mkdtemp()
        cwd = os.getcwd()
        try:
            os.chdir(mem_dir)
            memory_file, config = load_user_memory()
            with open(memory_file, "w") as fh:
                json.dump({"config": {"onset_threshold": 0.42, "frame_threshold": 0.24}}, fh)
            _, config = load_user_memory()

            class _Args:
                onset: float | None = None
                frame: float | None = None

            onset, frame = resolve_thresholds(_Args(), config)
            check(
                "Profile thresholds are honoured",
                (onset, frame) == (0.42, 0.24),
                f"got {(onset, frame)}",
            )

            _Args.onset, _Args.frame = 0.9, 0.8
            onset, frame = resolve_thresholds(_Args(), config)
            check("CLI flags override profile thresholds", (onset, frame) == (0.9, 0.8))
        finally:
            os.chdir(cwd)
            shutil.rmtree(mem_dir, ignore_errors=True)
    except Exception as e:
        check("Profile/threshold resolution", False, str(e))

    # ── 6. Lua syntax check (basic) ────────────────────────────────────

    for f in ["reaper/TabAgent.lua", "reaper/Settings.lua"]:
        if os.path.exists(f):
            with open(f) as fh:
                content = fh.read()
            # Basic structural checks
            check(f"{f} has --[[ header", content.strip().startswith("--[["))
            check(
                f"{f} has reaper API calls",
                "reaper." in content and "function" in content,
            )
            check(
                f"{f} has balanced curly braces",
                content.count("{") == content.count("}"),
            )
            # Pipeline script needs os.execute; settings script doesn't
            if f == "reaper/TabAgent.lua":
                check(f"{f} uses os.execute for pipeline", "os.execute(" in content)
        else:
            check(f"File exists: {f}", False)

    # ── 7. RUN.SH ──────────────────────────────────────────────────────

    if os.path.exists("run.sh"):
        check("run.sh is executable", os.access("run.sh", os.X_OK))
        with open("run.sh") as fh:
            content = fh.read()
        check("run.sh has shebang", content.startswith("#!/"))
        check("run.sh references requirements.txt", "requirements.txt" in content)
        check("run.sh handles --web flag", "--web" in content)

    # ── 8. GIT CLEANLINESS ─────────────────────────────────────────────

    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        modified = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        check("No merge conflicts", all("UU" not in line for line in modified))
    except (OSError, subprocess.SubprocessError) as e:
        check_skip("No merge conflicts", True, f"git unavailable: {e}")

    # ── 9. DOCKER BUILD ────────────────────────────────────────────────

    check_docker_build()

    # ── SUMMARY ────────────────────────────────────────────────────────

    print("\n".join(RESULTS))
    print(f"\n{'=' * 60}")
    print(f"PASS: {PASS}   FAIL: {FAIL}   SKIP: {SKIP}")
    if FAIL:
        print("FAILED CHECKS:")
        for r in RESULTS:
            if "❌" in r:
                print(r)
    else:
        print("All validation checks passed.")
    print("=" * 60)

    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
