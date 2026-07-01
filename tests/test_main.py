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

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _path(self, name="test.tab"):
        return os.path.join(self.tmp, name)

    def _read(self, name="test.tab"):
        with open(self._path(name)) as f:
            return f.read()

    def test_export_tab_to_txt_creates_file(self):
        tab_data = [
            {"string": 5, "fret": 0},
            {"string": 3, "fret": 2},
        ]
        export_tab_to_txt(tab_data, self._path(), "Guitar")
        self.assertTrue(os.path.exists(self._path()))

    def test_export_tab_to_txt_content_guitar(self):
        tab_data = [
            {"string": 5, "fret": 0},
            {"string": 3, "fret": 2},
            {"string": 0, "fret": 3},
        ]
        export_tab_to_txt(tab_data, self._path(), "Guitar")
        content = self._read()
        self.assertIn("=== Guitar Tablature ===", content)
        self.assertIn("E|", content)
        self.assertIn("B|", content)
        self.assertIn("G|", content)
        self.assertIn("D|", content)
        self.assertIn("A|", content)
        self.assertIn("Legend: s=slide, h=hammer-on, p=pull-off", content)

    def test_export_tab_to_txt_content_bass(self):
        tab_data = [
            {"string": 4, "fret": 0},
            {"string": 2, "fret": 3},
        ]
        export_tab_to_txt(tab_data, self._path("bass.tab"), "5-String Bass")
        content = self._read("bass.tab")
        self.assertIn("=== 5-String Bass Tablature ===", content)
        self.assertIn("B|", content)
        self.assertIn("E|", content)
        self.assertIn("A|", content)
        self.assertIn("D|", content)
        self.assertIn("G|", content)
        self.assertNotIn("=== Guitar", content)

    def test_export_tab_to_txt_technique_markers(self):
        tab_data = [
            {"string": 5, "fret": 0, "technique": TECHNIQUE_PICK},
            {"string": 4, "fret": 3, "technique": TECHNIQUE_SLIDE},
            {"string": 3, "fret": 5, "technique": TECHNIQUE_HAMMER},
            {"string": 2, "fret": 7, "technique": TECHNIQUE_PULL},
        ]
        export_tab_to_txt(tab_data, self._path("tech.tab"), "Guitar")
        content = self._read("tech.tab")
        self.assertIn("3s", content)
        self.assertIn("5h", content)
        self.assertIn("7p", content)

    def test_export_tab_to_txt_empty_data(self):
        export_tab_to_txt([], self._path("empty.tab"), "Guitar")
        self.assertFalse(os.path.exists(self._path("empty.tab")))


class TestExportTabToJson(unittest.TestCase):
    """Tests for export_tab_to_json programmatic export."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _path(self, name="test.json"):
        return os.path.join(self.tmp, name)

    def _load(self, name="test.json"):
        with open(self._path(name)) as f:
            return json.load(f)

    def test_export_tab_to_json_creates_file(self):
        tab_data = [
            {"string": 5, "fret": 0},
        ]
        export_tab_to_json(tab_data, self._path())
        self.assertTrue(os.path.exists(self._path()))

    def test_export_tab_to_json_content(self):
        tab_data = [
            {"string": 5, "fret": 0, "technique": "pick"},
        ]
        export_tab_to_json(tab_data, self._path(), "Lead Guitar")
        data = self._load()
        self.assertEqual(data["instrument"], "Lead Guitar")
        self.assertIn("timestamp", data)
        self.assertIsInstance(data["tablature"], list)
        self.assertEqual(len(data["tablature"]), 1)
        self.assertEqual(data["tablature"][0]["string"], 5)
        self.assertEqual(data["tablature"][0]["fret"], 0)

    def test_export_tab_to_json_empty_data(self):
        export_tab_to_json([], self._path("empty.json"), "Guitar")
        data = self._load("empty.json")
        self.assertEqual(data["instrument"], "Guitar")
        self.assertIn("timestamp", data)
        self.assertEqual(data["tablature"], [])

    def test_export_tab_to_json_multiple_entries(self):
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
        self.assertEqual(len(data["tablature"]), 6)
        frets = [p["fret"] for p in data["tablature"]]
        self.assertEqual(frets, [0, 2, 4, 5, 7, 9])


class TestTechniqueConstants(unittest.TestCase):
    """Technique constants exist and have expected values."""

    def test_technique_constants(self):
        self.assertEqual(TECHNIQUE_SLIDE, "slide")
        self.assertEqual(TECHNIQUE_HAMMER, "hammer")
        self.assertEqual(TECHNIQUE_PULL, "pull")
        self.assertEqual(TECHNIQUE_PICK, "pick")


if __name__ == "__main__":
    unittest.main()
