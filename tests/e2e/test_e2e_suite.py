#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest runner for E2E shell-script tests.

Discovers all tests/e2e/test_*.sh scripts and runs them via subprocess,
asserting that each exits with code 0.

Usage:
    pytest tests/e2e/test_e2e_suite.py -v
    pytest tests/e2e/ -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Working directory for all E2E scripts (they use WORKDIR env var default)
WORKDIR = Path("/Users/guillaume/P/GdB1")
E2E_DIR = WORKDIR / "tests" / "e2e"


def discover_scripts() -> list[Path]:
    """Find all test_*.sh scripts in tests/e2e/."""
    if not E2E_DIR.is_dir():
        return []
    return sorted(E2E_DIR.glob("test_*.sh"))


@pytest.mark.parametrize(
    "script",
    discover_scripts(),
    ids=lambda p: p.name,
)
def test_e2e_script(script: Path) -> None:
    """
    Run a single E2E shell script and assert it exits cleanly (code 0).

    On failure the script's stdout+stderr is printed so pytest's verbose
    output shows exactly what went wrong.
    """
    rel = script.relative_to(WORKDIR)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=str(WORKDIR),
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        msg = (
            f"\n{'=' * 60}\n"
            f"E2E script FAILED: {rel}\n"
            f"Exit code: {result.returncode}\n"
            f"{'-' * 60}\n"
            f"[STDOUT]\n{result.stdout}\n"
            f"{'-' * 60}\n"
            f"[STDERR]\n{result.stderr}\n"
            f"{'=' * 60}"
        )
        pytest.fail(msg, pytrace=False)

    # Even on success, print output with -v so devs can audit it
    if result.stdout:
        print(f"\n[{rel}] stdout:\n{result.stdout}", file=sys.stderr)
    if result.stderr:
        print(f"\n[{rel}] stderr:\n{result.stderr}", file=sys.stderr)


def test_discovery() -> None:
    """Smoke test: confirm at least one E2E script was found."""
    scripts = discover_scripts()
    assert len(scripts) >= 1, f"No test_*.sh scripts found in {E2E_DIR}"
    print(f"\nDiscovered {len(scripts)} E2E script(s): {[s.name for s in scripts]}")
