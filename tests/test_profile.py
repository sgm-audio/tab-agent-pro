"""
Tests for preset profile handling (docs-audit P1-2).

Before the fix, ``--profile`` was parsed and then ignored, and the thresholds /
technique settings stored in user memory never reached the pipeline.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import init_memory
from main import apply_profile, resolve_thresholds


class _Args:
    def __init__(self, onset=None, frame=None):
        self.onset = onset
        self.frame = frame


class TestResolveThresholds(unittest.TestCase):
    """Precedence: CLI flag > profile value > built-in default."""

    def test_defaults_when_nothing_set(self) -> None:
        onset, frame = resolve_thresholds(_Args(), {})
        assert (onset, frame) == (0.5, 0.3)

    def test_profile_values_used(self) -> None:
        config = {"onset_threshold": 0.6, "frame_threshold": 0.4}
        assert resolve_thresholds(_Args(), config) == (0.6, 0.4)

    def test_cli_flags_win(self) -> None:
        config = {"onset_threshold": 0.6, "frame_threshold": 0.4}
        assert resolve_thresholds(_Args(0.2, 0.1), config) == (0.2, 0.1)

    def test_partial_cli_override(self) -> None:
        config = {"onset_threshold": 0.6, "frame_threshold": 0.4}
        assert resolve_thresholds(_Args(onset=0.2), config) == (0.2, 0.4)

    def test_zero_is_a_valid_threshold(self) -> None:
        assert resolve_thresholds(_Args(0.0, 0.0), {}) == (0.0, 0.0)


class TestApplyProfile(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self._cwd = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_profile_is_persisted_and_returned(self) -> None:
        config = apply_profile("suno_aggressive", {})

        assert config["active_profile"] == "suno_aggressive"
        assert (
            config["onset_threshold"] == init_memory.PROFILES["suno_aggressive"]["onset_threshold"]
        )

        with open(os.path.join(self.tmp, "user_memory", "user_preferences.json")) as f:
            saved = json.load(f)
        assert saved["config"]["active_profile"] == "suno_aggressive"

    def test_every_profile_can_be_applied(self) -> None:
        for key in init_memory.PROFILES:
            config = apply_profile(key, {})
            assert config["active_profile"] == key
            assert "onset_threshold" in config and "frame_threshold" in config

    def test_unknown_profile_exits_with_message(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            apply_profile("does_not_exist", {})
        assert "Unknown profile" in str(ctx.exception)

    def test_applied_profile_feeds_thresholds(self) -> None:
        config = apply_profile("live_band", {})
        onset, frame = resolve_thresholds(_Args(), config)
        assert (onset, frame) == (
            init_memory.PROFILES["live_band"]["onset_threshold"],
            init_memory.PROFILES["live_band"]["frame_threshold"],
        )


class TestCliProfileWiring(unittest.TestCase):
    """`main.py --profile X` must reach transcribe_stem() and the session log."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        self.audio = os.path.join(self.tmp, "song.wav")
        with open(self.audio, "wb") as f:
            f.write(b"RIFF....WAVE")
        self.out = os.path.join(self.tmp, "out")

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, extra_args):
        import main

        with (
            patch("main.process_suno_audio", return_value=(self.audio, False, {})),
            patch("main.SplitterAgent") as splitter_cls,
            patch("main.EarAgent") as ear_cls,
            patch("main.TabAgent") as tab_cls,
            patch("main.SunoNotePostprocessor"),
        ):
            splitter_cls.return_value.separate_stems.return_value = {
                "guitar": "g.wav",
                "bass": "b.wav",
            }
            splitter_cls.return_value.process_guitars.return_value = {
                "lead": "l.wav",
                "left": "lf.wav",
                "right": "r.wav",
            }
            ear_cls.return_value.transcribe_stem.return_value = []
            ear_cls.return_value.humanize_and_clean.return_value = []
            tab_cls.return_value.generate_tab.return_value = []

            main.main([self.audio, "--output-dir", self.out, *extra_args])

        return ear_cls.return_value, tab_cls.return_value

    def test_cli_profile_sets_thresholds(self) -> None:
        ear, _tab = self._run(["--profile", "suno_aggressive"])
        profile = init_memory.PROFILES["suno_aggressive"]
        _args, kwargs = ear.transcribe_stem.call_args
        assert kwargs["onset_threshold"] == profile["onset_threshold"]
        assert kwargs["frame_threshold"] == profile["frame_threshold"]

    def test_cli_flags_override_profile(self) -> None:
        ear, _tab = self._run(
            ["--profile", "suno_aggressive", "--onset", "0.11", "--frame", "0.07"]
        )
        _args, kwargs = ear.transcribe_stem.call_args
        assert kwargs["onset_threshold"] == 0.11
        assert kwargs["frame_threshold"] == 0.07

    def test_session_log_records_profile(self) -> None:
        self._run(["--profile", "bass_5_string"])
        with open(os.path.join(self.tmp, "user_memory", "user_preferences.json")) as f:
            saved = json.load(f)
        sessions = saved["sessions"]
        assert sessions[-1]["profile"] == "bass_5_string"
        assert "onset_threshold" in sessions[-1]

    def test_no_profile_uses_defaults(self) -> None:
        ear, _tab = self._run([])
        _args, kwargs = ear.transcribe_stem.call_args
        assert kwargs["onset_threshold"] == 0.5
        assert kwargs["frame_threshold"] == 0.3

    def test_unknown_profile_fails_loudly(self) -> None:
        with self.assertRaises(SystemExit):
            self._run(["--profile", "nope"])

    def test_corrupt_memory_file_does_not_crash(self) -> None:
        """A truncated preferences file must not break the run (atomic writes added)."""
        memory_dir = os.path.join(self.tmp, "user_memory")
        os.makedirs(memory_dir, exist_ok=True)
        with open(os.path.join(memory_dir, "user_preferences.json"), "w") as f:
            f.write('{"config": {"onset_threshold": 0.6, "is_suno": ')

        self._run([])  # must not raise

    def test_session_log_is_valid_json_after_run(self) -> None:
        self._run(["--profile", "rock_drop_d"])
        path = os.path.join(self.tmp, "user_memory", "user_preferences.json")
        with open(path) as f:
            saved = json.load(f)
        assert saved["sessions"], "session was not logged"


class TestCliErgonomics(unittest.TestCase):
    """Small CLI fixes from the docs audit (P2-5)."""

    def test_missing_audio_prints_help(self) -> None:
        import main

        with self.assertRaises(SystemExit) as ctx:
            main.main([])
        assert ctx.exception.code == 1

    def test_missing_file_is_reported(self) -> None:
        import main

        with self.assertRaises(SystemExit) as ctx:
            main.main(["/nonexistent/audio.wav"])
        assert ctx.exception.code == 1

    def test_parser_has_prog_name_and_flags(self) -> None:
        import main

        parser = main.build_parser()
        assert parser.prog == "tab-agent"
        opts = {a.dest for a in parser._actions}
        for expected in ("audio", "instrument", "onset", "frame", "profile", "output_dir"):
            assert expected in opts

    def test_version_string(self) -> None:
        import main

        with self.assertRaises(SystemExit) as ctx:
            main.parse_args(["--version"])
        assert ctx.exception.code == 0


if __name__ == "__main__":
    unittest.main()
