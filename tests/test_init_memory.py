import json

import pytest

from init_memory import PROFILES, get_memory_dir, list_profiles, save_profile

REQUIRED_KEYS = {
    "name",
    "description",
    "tuning",
    "num_strings",
    "num_frets",
    "onset_threshold",
    "frame_threshold",
    "suno_aggressive_mode",
    "technique_sensitivity",
    "prefer_low_strings",
}


def test_profiles_have_correct_keys():
    assert len(PROFILES) == 8
    for key, profile in PROFILES.items():
        missing = REQUIRED_KEYS - set(profile.keys())
        assert not missing, f"{key} is missing keys: {missing}"


def test_save_profile_creates_file(tmp_path):
    path = save_profile("rock_standard", memory_dir=str(tmp_path))
    assert path == str(tmp_path / "user_preferences.json")
    assert (tmp_path / "user_preferences.json").exists()


def test_save_profile_content(tmp_path):
    save_profile("rock_standard", memory_dir=str(tmp_path))
    data = json.loads((tmp_path / "user_preferences.json").read_text())
    config = data["config"]
    assert config["onset_threshold"] == 0.5
    assert config["frame_threshold"] == 0.3
    assert config["guitar_tuning"] == [40, 45, 50, 55, 59, 64]


def test_save_profile_guitar_profile(tmp_path):
    save_profile("rock_standard", memory_dir=str(tmp_path))
    data = json.loads((tmp_path / "user_preferences.json").read_text())
    assert data["config"]["guitar_num_strings"] == 6
    assert data["config"]["guitar_tuning"] == [40, 45, 50, 55, 59, 64]


def test_save_profile_bass_profile(tmp_path):
    save_profile("bass_5_string", memory_dir=str(tmp_path))
    data = json.loads((tmp_path / "user_preferences.json").read_text())
    assert data["config"]["bass_num_strings"] == 5
    assert data["config"]["bass_tuning"] == [23, 28, 33, 38, 43]


def test_save_profile_invalid_key_raises():
    with pytest.raises(KeyError):
        save_profile("nonexistent_profile")


def test_get_memory_dir():
    path = get_memory_dir()
    assert isinstance(path, str)
    assert len(path) > 0


def test_list_profiles(capsys):
    list_profiles()
    captured = capsys.readouterr()
    assert captured.out.startswith("\nAvailable")
    assert "rock_standard" in captured.out
    assert "bass_5_string" in captured.out


def test_profile_tunings_are_valid_midi():
    for key, profile in PROFILES.items():
        for note in profile["tuning"]:
            assert 0 <= note <= 127, f"{key} has invalid MIDI note {note}"


def test_all_profiles_have_unique_names():
    names = [p["name"] for p in PROFILES.values()]
    assert len(names) == len(set(names)), "Duplicate profile names found"
