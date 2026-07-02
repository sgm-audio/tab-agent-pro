import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    TECHNIQUE_HAMMER,
    TECHNIQUE_PICK,
    TECHNIQUE_PULL,
    TECHNIQUE_SLIDE,
    export_tab_to_json,
    export_tab_to_txt,
)


class TestExportTabToTxt(unittest.TestCase):
    """Tests for export_tab_to_txt ASCII tab export."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def _path(self, name="test.tab"):
        return os.path.join(self.tmp, name)

    def _read(self, name="test.tab"):
        with open(self._path(name)) as f:
            return f.read()

    def test_export_tab_to_txt_creates_file(self) -> None:
        tab_data = [
            {"string": 5, "fret": 0},
            {"string": 3, "fret": 2},
        ]
        export_tab_to_txt(tab_data, self._path(), "Guitar")
        assert os.path.exists(self._path())

    def test_export_tab_to_txt_content_guitar(self) -> None:
        tab_data = [
            {"string": 5, "fret": 0},
            {"string": 3, "fret": 2},
            {"string": 0, "fret": 3},
        ]
        export_tab_to_txt(tab_data, self._path(), "Guitar")
        content = self._read()
        assert "=== Guitar Tablature ===" in content
        assert "E|" in content
        assert "B|" in content
        assert "G|" in content
        assert "D|" in content
        assert "A|" in content
        assert "Legend: s=slide, h=hammer-on, p=pull-off" in content

    def test_export_tab_to_txt_content_bass(self) -> None:
        tab_data = [
            {"string": 4, "fret": 0},
            {"string": 2, "fret": 3},
        ]
        export_tab_to_txt(tab_data, self._path("bass.tab"), "5-String Bass")
        content = self._read("bass.tab")
        assert "=== 5-String Bass Tablature ===" in content
        assert "B|" in content
        assert "E|" in content
        assert "A|" in content
        assert "D|" in content
        assert "G|" in content
        assert "=== Guitar" not in content

    def test_export_tab_to_txt_technique_markers(self) -> None:
        tab_data = [
            {"string": 5, "fret": 0, "technique": TECHNIQUE_PICK},
            {"string": 4, "fret": 3, "technique": TECHNIQUE_SLIDE},
            {"string": 3, "fret": 5, "technique": TECHNIQUE_HAMMER},
            {"string": 2, "fret": 7, "technique": TECHNIQUE_PULL},
        ]
        export_tab_to_txt(tab_data, self._path("tech.tab"), "Guitar")
        content = self._read("tech.tab")
        assert "3s" in content
        assert "5h" in content
        assert "7p" in content

    def test_export_tab_to_txt_empty_data(self) -> None:
        export_tab_to_txt([], self._path("empty.tab"), "Guitar")
        assert not os.path.exists(self._path("empty.tab"))


class TestExportTabToJson(unittest.TestCase):
    """Tests for export_tab_to_json programmatic export."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def _path(self, name="test.json"):
        return os.path.join(self.tmp, name)

    def _load(self, name="test.json"):
        with open(self._path(name)) as f:
            return json.load(f)

    def test_export_tab_to_json_creates_file(self) -> None:
        tab_data = [
            {"string": 5, "fret": 0},
        ]
        export_tab_to_json(tab_data, self._path())
        assert os.path.exists(self._path())

    def test_export_tab_to_json_content(self) -> None:
        tab_data = [
            {"string": 5, "fret": 0, "technique": "pick"},
        ]
        export_tab_to_json(tab_data, self._path(), "Lead Guitar")
        data = self._load()
        assert data["instrument"] == "Lead Guitar"
        assert "timestamp" in data
        assert isinstance(data["tablature"], list)
        assert len(data["tablature"]) == 1
        assert data["tablature"][0]["string"] == 5
        assert data["tablature"][0]["fret"] == 0

    def test_export_tab_to_json_empty_data(self) -> None:
        export_tab_to_json([], self._path("empty.json"), "Guitar")
        data = self._load("empty.json")
        assert data["instrument"] == "Guitar"
        assert "timestamp" in data
        assert data["tablature"] == []

    def test_export_tab_to_json_multiple_entries(self) -> None:
        tab_data = [
            {"string": 0, "fret": 0},
            {"string": 1, "fret": 2},
            {"string": 2, "fret": 4},
            {"string": 3, "fret": 5},
            {"string": 4, "fret": 7},
            {"string": 5, "fret": 9},
        ]
        export_tab_to_json(tab_data, self._path("multi.json"), "Guitar")
        data = self._load("multi.json")
        assert len(data["tablature"]) == 6
        frets = [p["fret"] for p in data["tablature"]]
        assert frets == [0, 2, 4, 5, 7, 9]


class TestTechniqueConstants(unittest.TestCase):
    """Technique constants exist and have expected values."""

    def test_technique_constants(self) -> None:
        assert TECHNIQUE_SLIDE == "slide"
        assert TECHNIQUE_HAMMER == "hammer"
        assert TECHNIQUE_PULL == "pull"
        assert TECHNIQUE_PICK == "pick"


if __name__ == "__main__":
    unittest.main()
