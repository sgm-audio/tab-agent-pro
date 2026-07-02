import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    TECHNIQUE_HAMMER,
    TECHNIQUE_PICK,
    TECHNIQUE_PULL,
    TECHNIQUE_SLIDE,
    export_tab_to_json,
    export_tab_to_txt,
    load_user_memory,
)


class TestLoadUserMemory(unittest.TestCase):
    """Tests for load_user_memory() — Docker path, local path, file handling."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def _local_memory_file(self):
        return os.path.join(self.tmp, "user_memory", "user_preferences.json")

    def test_local_path_no_file_uses_defaults(self) -> None:
        orig_exists = os.path.exists

        def fake_exists(path):
            if path == "/app":
                return False
            return orig_exists(path)

        with mock.patch("main.sys.platform", "win32"):
            _memory_file, config = load_user_memory()
        assert config["bass_num_strings"] == 5
        assert config["guitar_num_strings"] == 6
        assert config["num_frets"] == 24
        assert config["prefer_low_strings"]

    def test_local_path_with_valid_preferences(self) -> None:
        old_cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            memory_dir = os.path.join(self.tmp, "user_memory")
            os.makedirs(memory_dir)
            with open(os.path.join(memory_dir, "user_preferences.json"), "w") as f:
                json.dump(
                    {
                        "config": {
                            "bass_num_strings": 4,
                            "guitar_num_strings": 7,
                            "num_frets": 22,
                            "prefer_low_strings": False,
                        }
                    },
                    f,
                )
            with mock.patch("main.sys.platform", "win32"):
                _, config = load_user_memory()
        finally:
            os.chdir(old_cwd)
        assert config["bass_num_strings"] == 4
        assert config["guitar_num_strings"] == 7
        assert config["num_frets"] == 22
        assert not config["prefer_low_strings"]

    def test_local_path_with_corrupt_json(self) -> None:
        old_cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            memory_dir = os.path.join(self.tmp, "user_memory")
            os.makedirs(memory_dir)
            with open(os.path.join(memory_dir, "user_preferences.json"), "w") as f:
                f.write("not valid json")
            with mock.patch("main.sys.platform", "win32"):
                _, config = load_user_memory()
        finally:
            os.chdir(old_cwd)
        assert config["guitar_tuning"] == [40, 45, 50, 55, 59, 64]

    def test_docker_path(self) -> None:
        with mock.patch("main.os.path.exists", return_value=True):
            with mock.patch("main.os.makedirs"):
                with mock.patch("main.sys.platform", "linux"):
                    memory_file, config = load_user_memory()
        assert memory_file.startswith("/app/user_memory")
        assert config["bass_num_strings"] == 5

    def test_partial_config_override(self) -> None:
        old_cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            memory_dir = os.path.join(self.tmp, "user_memory")
            os.makedirs(memory_dir)
            with open(os.path.join(memory_dir, "user_preferences.json"), "w") as f:
                json.dump({"config": {"num_frets": 21}}, f)
            with mock.patch("main.sys.platform", "win32"):
                _, config = load_user_memory()
        finally:
            os.chdir(old_cwd)
        assert config["num_frets"] == 21
        assert config["bass_num_strings"] == 5
        assert config["guitar_num_strings"] == 6

    def test_empty_preferences_file(self) -> None:
        old_cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            memory_dir = os.path.join(self.tmp, "user_memory")
            os.makedirs(memory_dir)
            with open(os.path.join(memory_dir, "user_preferences.json"), "w") as f:
                json.dump({}, f)
            with mock.patch("main.sys.platform", "win32"):
                _, config = load_user_memory()
        finally:
            os.chdir(old_cwd)
        assert config["bass_num_strings"] == 5
        assert config["guitar_num_strings"] == 6

    def test_preferences_without_config_key(self) -> None:
        old_cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            memory_dir = os.path.join(self.tmp, "user_memory")
            os.makedirs(memory_dir)
            with open(os.path.join(memory_dir, "user_preferences.json"), "w") as f:
                json.dump({"sessions": [{"song": "test"}]}, f)
            with mock.patch("main.sys.platform", "win32"):
                _, config = load_user_memory()
        finally:
            os.chdir(old_cwd)
        assert config["guitar_num_strings"] == 6


class TestExportTabToTxtCoverage(unittest.TestCase):
    """Coverage gaps for export_tab_to_txt."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def _path(self, name="test.tab"):
        return os.path.join(self.tmp, name)

    def _read(self, name="test.tab"):
        with open(self._path(name)) as f:
            return f.read()

    def test_4_string_output(self) -> None:
        tab_data = [
            {"string": 3, "fret": 0},
            {"string": 2, "fret": 3},
            {"string": 1, "fret": 5},
            {"string": 0, "fret": 7},
        ]
        export_tab_to_txt(tab_data, self._path("4string.tab"), "Bass")
        content = self._read("4string.tab")
        assert "=== Bass Tablature ===" in content
        assert "G|" in content
        assert "D|" in content
        assert "A|" in content
        assert "E|" in content

    def test_5_string_output(self) -> None:
        tab_data = [{"string": i, "fret": i} for i in range(5)]
        export_tab_to_txt(tab_data, self._path("5string.tab"), "Bass")
        content = self._read("5string.tab")
        assert "G|" in content
        assert "D|" in content
        assert "A|" in content
        assert "E|" in content
        assert "B|" in content

    def test_generic_label_fallback(self) -> None:
        tab_data = [{"string": i, "fret": i} for i in range(3)]
        export_tab_to_txt(tab_data, self._path("gen.tab"), "Custom")
        content = self._read("gen.tab")
        assert "C|" in content
        assert "B|" in content
        assert "A|" in content

    def test_single_note(self) -> None:
        tab_data = [{"string": 5, "fret": 0}]
        export_tab_to_txt(tab_data, self._path("single.tab"), "Guitar")
        content = self._read("single.tab")
        assert "E|0|" in content

    def test_chord_simultaneous_notes(self) -> None:
        tab_data = [
            {"string": 5, "fret": 0},
            {"string": 4, "fret": 2},
            {"string": 3, "fret": 2},
            {"string": 2, "fret": 0},
            {"string": 1, "fret": 1},
            {"string": 0, "fret": 0},
        ]
        export_tab_to_txt(tab_data, self._path("chord.tab"), "Guitar")
        content = self._read("chord.tab")
        for label in ["E|", "B|", "G|", "D|", "A|", "E|"]:
            assert label in content

    def test_all_four_techniques_in_one_export(self) -> None:
        tab_data = [
            {"string": 5, "fret": 0, "technique": TECHNIQUE_PICK},
            {"string": 4, "fret": 3, "technique": TECHNIQUE_SLIDE},
            {"string": 3, "fret": 5, "technique": TECHNIQUE_HAMMER},
            {"string": 2, "fret": 7, "technique": TECHNIQUE_PULL},
        ]
        export_tab_to_txt(tab_data, self._path("alltech.tab"), "Guitar")
        content = self._read("alltech.tab")
        assert "0" in content
        assert "3s" in content
        assert "5h" in content
        assert "7p" in content

    def test_time_grouping_within_50ms(self) -> None:
        tab_data = [
            {"string": 5, "fret": 0, "start_time": 0.0},
            {"string": 4, "fret": 2, "start_time": 0.03},
            {"string": 3, "fret": 3, "start_time": 1.0},
        ]
        export_tab_to_txt(tab_data, self._path("time.tab"), "Guitar")
        content = self._read("time.tab")
        assert "E|" in content


class TestExportTabToJsonCoverage(unittest.TestCase):
    """Coverage gaps for export_tab_to_json."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def _path(self, name="test.json"):
        return os.path.join(self.tmp, name)

    def _load(self, name="test.json"):
        with open(self._path(name)) as f:
            return json.load(f)

    def test_instrument_field_correct(self) -> None:
        tab_data = [{"string": 5, "fret": 0}]
        export_tab_to_json(tab_data, self._path("inst.json"), "Bass")
        data = self._load("inst.json")
        assert data["instrument"] == "Bass"

    def test_timestamp_is_iso_format(self) -> None:
        tab_data = [{"string": 5, "fret": 0}]
        export_tab_to_json(tab_data, self._path("ts.json"), "Guitar")
        data = self._load("ts.json")
        assert "timestamp" in data
        assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", data["timestamp"])

    def test_multiple_instruments_different_files(self) -> None:
        export_tab_to_json([{"string": 5, "fret": 0}], self._path("guitar.json"), "Lead Guitar")
        export_tab_to_json([{"string": 4, "fret": 0}], self._path("bass.json"), "5-String Bass")
        g = self._load("guitar.json")
        b = self._load("bass.json")
        assert g["instrument"] == "Lead Guitar"
        assert b["instrument"] == "5-String Bass"


if __name__ == "__main__":
    unittest.main()
