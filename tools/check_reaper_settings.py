#!/usr/bin/env python3
"""Run the executable Lua → CLI/REAPER contract tests (requires pytest and lupa)."""

import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    tests = Path(__file__).resolve().parents[1] / "tests" / "test_reaper_contract.py"
    sys.exit(pytest.main([str(tests), "--no-cov", "-q"]))
