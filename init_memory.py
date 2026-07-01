#!/usr/bin/env python3
"""
Tab Agent - Memory Initialization with Preset Profiles

Run this script to select a preset profile and save it to user memory.
Profiles configure tuning, thresholds, and processing behaviour for the
transcription pipeline.

Usage:
    python init_memory.py              # Interactive selection
    python init_memory.py --list       # List available profiles
    python init_memory.py --profile rock_standard  # Apply a profile directly
"""

import json
import os
import sys

# ── Preset Profiles ─────────────────────────────────────────────────────────

PROFILES = {
    "rock_standard": {
        "name": "Rock Guitar (Standard Tuning)",
        "description": "Standard EADGBE tuning for rock/metal, moderate processing",
        "tuning": [40, 45, 50, 55, 59, 64],  # E2-A2-D3-G3-B3-E4
        "num_strings": 6,
        "num_frets": 24,
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "suno_aggressive_mode": True,
        "technique_sensitivity": 0.7,
        "prefer_low_strings": False,
    },
    "rock_drop_d": {
        "name": "Rock Guitar (Drop D)",
        "description": "Drop D tuning (DADGBE) for heavy rock/metal",
        "tuning": [38, 45, 50, 55, 59, 64],  # D2-A2-D3-G3-B3-E4
        "num_strings": 6,
        "num_frets": 24,
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "suno_aggressive_mode": True,
        "technique_sensitivity": 0.7,
        "prefer_low_strings": False,
    },
    "classical": {
        "name": "Classical / Clean Guitar",
        "description": "Light processing for nylon-string/clean recordings, preserve dynamics",
        "tuning": [40, 45, 50, 55, 59, 64],
        "num_strings": 6,
        "num_frets": 19,
        "onset_threshold": 0.4,
        "frame_threshold": 0.25,
        "suno_aggressive_mode": False,
        "technique_sensitivity": 0.5,
        "prefer_low_strings": False,
    },
    "bass_5_string": {
        "name": "5-String Bass (B-E-A-D-G)",
        "description": "Standard 5-string bass tuning, low-string preference",
        "tuning": [23, 28, 33, 38, 43],  # B0-E1-A1-D2-G2
        "num_strings": 5,
        "num_frets": 24,
        "onset_threshold": 0.55,
        "frame_threshold": 0.35,
        "suno_aggressive_mode": True,
        "technique_sensitivity": 0.6,
        "prefer_low_strings": True,
    },
    "bass_4_string": {
        "name": "4-String Bass (E-A-D-G)",
        "description": "Standard 4-string bass tuning, low-string preference",
        "tuning": [28, 33, 38, 43],  # E1-A1-D2-G2
        "num_strings": 4,
        "num_frets": 24,
        "onset_threshold": 0.55,
        "frame_threshold": 0.35,
        "suno_aggressive_mode": True,
        "technique_sensitivity": 0.6,
        "prefer_low_strings": True,
    },
    "suno_aggressive": {
        "name": "Suno AI Audio (Aggressive Cleanup)",
        "description": "Maximum artifact removal for AI-generated audio",
        "tuning": [40, 45, 50, 55, 59, 64],
        "num_strings": 6,
        "num_frets": 24,
        "onset_threshold": 0.6,
        "frame_threshold": 0.4,
        "suno_aggressive_mode": True,
        "technique_sensitivity": 0.8,
        "prefer_low_strings": False,
    },
    "suno_conservative": {
        "name": "Suno AI Audio (Light Cleanup)",
        "description": "Light artifact removal, preserve more original content",
        "tuning": [40, 45, 50, 55, 59, 64],
        "num_strings": 6,
        "num_frets": 24,
        "onset_threshold": 0.55,
        "frame_threshold": 0.35,
        "suno_aggressive_mode": False,
        "technique_sensitivity": 0.7,
        "prefer_low_strings": False,
    },
    "live_band": {
        "name": "Live Band / Natural Recording",
        "description": "Minimal processing, preserve dynamics, natural feel",
        "tuning": [40, 45, 50, 55, 59, 64],
        "num_strings": 6,
        "num_frets": 24,
        "onset_threshold": 0.45,
        "frame_threshold": 0.25,
        "suno_aggressive_mode": False,
        "technique_sensitivity": 0.9,
        "prefer_low_strings": False,
    },
}


def get_memory_dir() -> str:
    """Resolve user_memory directory (Docker vs local)."""
    if os.path.exists("/app") and not sys.platform.startswith("win"):
        return "/app/user_memory"
    return "./user_memory"


def save_profile(profile_key: str, memory_dir: str | None = None) -> str:
    """Save a preset profile to user_preferences.json."""
    if memory_dir is None:
        memory_dir = get_memory_dir()
    os.makedirs(memory_dir, exist_ok=True)

    profile = PROFILES[profile_key]
    # Map profile keys to the config format expected by main.py
    config = {
        "bass_tuning": (
            profile["tuning"] if profile["prefer_low_strings"] else [23, 28, 33, 38, 43]
        ),
        "guitar_tuning": (
            profile["tuning"] if not profile["prefer_low_strings"] else [40, 45, 50, 55, 59, 64]
        ),
        "bass_num_strings": (profile["num_strings"] if profile["prefer_low_strings"] else 5),
        "guitar_num_strings": (profile["num_strings"] if not profile["prefer_low_strings"] else 6),
        "num_frets": profile["num_frets"],
        "onset_threshold": profile["onset_threshold"],
        "frame_threshold": profile["frame_threshold"],
        "suno_aggressive_mode": profile["suno_aggressive_mode"],
        "technique_sensitivity": profile["technique_sensitivity"],
        "prefer_low_strings": profile["prefer_low_strings"],
        "active_profile": profile_key,
    }

    preferences_path = os.path.join(memory_dir, "user_preferences.json")

    # Load existing preferences if any
    preferences = {}
    if os.path.exists(preferences_path):
        with open(preferences_path) as f:
            preferences = json.load(f)

    preferences["config"] = config

    with open(preferences_path, "w") as f:
        json.dump(preferences, f, indent=2)

    return preferences_path


def list_profiles():
    """Print available profiles to stdout."""
    print("\nAvailable Profiles:\n")
    for key, prof in PROFILES.items():
        strings = f"{prof['num_strings']}-string"
        label = "🎸" if not prof["prefer_low_strings"] else "🎵"
        print(f"  {label} {key:22s} | {prof['name']}")
        print(f"    {'':22s} | {prof['description']}")
        print(
            f"    {'':22s} | {strings}, onset={prof['onset_threshold']}, "
            f"suno={'aggressive' if prof['suno_aggressive_mode'] else 'light'}",
        )
        print()


def interactive_select() -> str:
    """Prompt the user to select a profile interactively."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🎸 Tab Agent - Profile Initialization              ║
╚══════════════════════════════════════════════════════════════╝
""")

    keys = list(PROFILES.keys())
    for idx, key in enumerate(keys, 1):
        prof = PROFILES[key]
        strings = f"{prof['num_strings']}-string"
        label = "🎸" if not prof["prefer_low_strings"] else "🎵"
        print(f"  [{idx}] {label} {prof['name']}")
        print(f"      {prof['description']}")
        print(
            f"      {strings}, onset={prof['onset_threshold']}, "
            f"suno={'aggressive' if prof['suno_aggressive_mode'] else 'light'}",
        )
        print()

    while True:
        try:
            choice = input(f"Select profile [1-{len(keys)}]: ").strip()
            idx = int(choice)
            if 1 <= idx <= len(keys):
                return keys[idx - 1]
            print(f"  Please enter a number between 1 and {len(keys)}")
        except (ValueError, KeyboardInterrupt):
            print("\n  Exiting.")
            sys.exit(1)


def main():
    if "--list" in sys.argv:
        list_profiles()
        return

    profile_key = None

    # Check for --profile flag
    for i, arg in enumerate(sys.argv):
        if arg == "--profile" and i + 1 < len(sys.argv):
            candidate = sys.argv[i + 1]
            if candidate in PROFILES:
                profile_key = candidate
            else:
                print(f"❌ Unknown profile: {candidate}")
                print("   Use --list to see available profiles")
                sys.exit(1)

    if profile_key is None:
        profile_key = interactive_select()

    memory_path = save_profile(profile_key)
    prof = PROFILES[profile_key]

    print(f"\n✅ Profile '{prof['name']}' saved!")
    print(f"   Config written to: {memory_path}")
    print("   Ready — run 'python main.py <audio_file>' to transcribe.\n")


if __name__ == "__main__":
    main()
