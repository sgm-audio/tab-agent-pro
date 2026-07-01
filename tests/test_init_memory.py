import json
import os
import sys

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


# ── save_profile: existing preferences merge ──────────────────────────────


def test_save_profile_merges_existing_preferences(tmp_path):
    pre_existing = {"user_name": "TestUser", "theme": "dark"}
    pre_path = tmp_path / "user_preferences.json"
    pre_path.write_text(json.dumps(pre_existing))

    save_profile("classical", memory_dir=str(tmp_path))

    data = json.loads(pre_path.read_text())
    assert data["user_name"] == "TestUser"
    assert data["theme"] == "dark"
    assert data["config"]["active_profile"] == "classical"


# ── save_profile: Docker path ─────────────────────────────────────────────


def test_save_profile_docker_resolution(tmp_path, monkeypatch):
    """save_profile works with Docker memory_dir (mocked)."""
    mock_dir = str(tmp_path)
    monkeypatch.setattr("init_memory.get_memory_dir", lambda: mock_dir)
    save_profile("rock_drop_d")
    assert (
        json.loads(tmp_path.joinpath("user_preferences.json").read_text())["config"][
            "active_profile"
        ]
        == "rock_drop_d"
    )


# ── get_memory_dir: Docker path ────────────────────────────────────────────


def test_get_memory_dir_docker(monkeypatch):
    orig_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: p == "/app")
    monkeypatch.setattr(sys, "platform", "linux")
    assert get_memory_dir() == "/app/user_memory"
    monkeypatch.setattr(os.path, "exists", orig_exists)


def test_get_memory_dir_local(monkeypatch):
    orig_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    assert get_memory_dir() == "./user_memory"
    monkeypatch.setattr(os.path, "exists", orig_exists)


# ── interactive_select ─────────────────────────────────────────────────────


def test_interactive_select_valid_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _="": "1")
    from init_memory import interactive_select

    assert interactive_select() == "rock_standard"


def test_interactive_select_out_of_range_retry(monkeypatch):
    """Out-of-range number prints error and retries, then valid input succeeds."""
    inputs = iter(["99", "0", "-1", "1"])
    monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
    from init_memory import interactive_select

    assert interactive_select() == "rock_standard"


def test_interactive_select_invalid_input_exits(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _="": "abc")
    with pytest.raises(SystemExit):
        from init_memory import interactive_select

        interactive_select()


def test_interactive_select_keyboard_interrupt_exits(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _="": (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(SystemExit):
        from init_memory import interactive_select

        interactive_select()


# ── main ───────────────────────────────────────────────────────────────────


def test_main_list_flag(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["init_memory.py", "--list"])
    from init_memory import main

    main()
    captured = capsys.readouterr()
    assert "Available Profiles" in captured.out
    assert "rock_standard" in captured.out


def test_main_profile_valid(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["init_memory.py", "--profile", "bass_4_string"])
    monkeypatch.setattr("init_memory.get_memory_dir", lambda: str(tmp_path))
    from init_memory import main

    main()
    captured = capsys.readouterr()
    assert "4-String Bass" in captured.out
    assert "saved" in captured.out


def test_main_profile_invalid(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["init_memory.py", "--profile", "nonexistent"])
    with pytest.raises(SystemExit):
        from init_memory import main

        main()
    captured = capsys.readouterr()
    assert "Unknown profile" in captured.out


def test_main_interactive_fallback(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["init_memory.py"])
    monkeypatch.setattr("builtins.input", lambda _="": "3")
    monkeypatch.setattr("init_memory.get_memory_dir", lambda: str(tmp_path))
    from init_memory import main

    main()
    captured = capsys.readouterr()
    assert "Classical" in captured.out
    assert "saved" in captured.out


def test_main_module_run(monkeypatch):
    """Cover the `if __name__ == '__main__'` block via runpy."""
    import runpy

    monkeypatch.setattr(sys, "argv", ["init_memory.py", "--list"])
    runpy.run_module("init_memory", run_name="__main__")
