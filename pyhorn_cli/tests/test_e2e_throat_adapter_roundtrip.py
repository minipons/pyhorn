"""E2E roundtrip test: throat-adapter YAML validity.

Verifies that the throat-adapter command produces valid, parseable YAML with
the required fields (ap1, lpt, type) that can be used in a horn project.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml as _yaml

# The pyhorn CLI is installed as a standalone entry-point script (not as a
# module). Use it directly so we pick up the correct Python interpreter.
PYHORN_CLI = "/opt/homebrew/bin/pyhorn"

WORKDIR = Path("/Users/guillaume/pyhorn")


class TestThroatAdapterRoundtrip:
    """throat-adapter output must be valid parseable YAML with required fields."""

    def test_throat_adapter_produces_valid_yaml_with_required_fields(
        self,
    ) -> None:
        """
        Run throat-adapter; verify:
          1. Exit code 0
          2. stdout contains a parseable YAML block with throat_adapter: section
          3. The block contains ap1, lpt, type fields
          4. ap1 > 0 and lpt >= 0 (cylindrical adapters can have lpt=0)
          5. type is a recognised profile type
        """
        result = subprocess.run(
            [
                PYHORN_CLI, "throat-adapter",
                "--d1", "50",
                "--d2", "100",
                "--a1", "30",
                "--a2", "30",
                "--type", "conical",
            ],
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"throat-adapter failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        output = result.stdout

        # Extract the YAML block (before the separator comment)
        sep = "# ────────────────────────────────────────────────────────────────────"
        assert sep in output, (
            f"Could not find YAML block separator in output:\n{output}"
        )
        yaml_text = output.split(sep)[0].strip()

        # Skip header lines and find the throat_adapter section
        lines = yaml_text.splitlines()
        yaml_start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("throat_adapter:"):
                yaml_start_idx = i
                break

        assert yaml_start_idx is not None, (
            f"Could not find 'throat_adapter:' in YAML block:\n{yaml_text}"
        )
        yaml_block = "\n".join(lines[yaml_start_idx:])

        # Parse YAML
        try:
            data = _yaml.safe_load(yaml_block)
        except Exception as exc:
            pytest.fail(
                f"Failed to parse YAML from throat-adapter output: {exc}\n"
                f"YAML block:\n{yaml_block}"
            )

        assert isinstance(data, dict), (
            f"Parsed YAML is not a dict: {type(data)}"
        )
        assert "throat_adapter" in data, (
            f"Output missing 'throat_adapter' key. Keys: {list(data.keys())}"
        )

        ta = data["throat_adapter"]
        assert "ap1" in ta, "throat_adapter missing 'ap1' (throat-side area)"
        assert "lpt" in ta, "throat_adapter missing 'lpt' (length)"
        assert "type" in ta, "throat_adapter missing 'type'"

        # Physical sanity checks
        assert ta["ap1"] > 0, (
            f"ap1 must be positive (throat-side area), got {ta['ap1']}"
        )
        assert ta["lpt"] >= 0, (
            f"lpt must be non-negative (adapter length), got {ta['lpt']}"
        )
        assert ta["type"] in (
            "conical", "exponential", "parabolic", "cylindrical"
        ), f"type must be a recognised profile, got '{ta['type']}'"

    @pytest.mark.parametrize("profile_type", [
        "conical", "exponential", "parabolic", "cylindrical"
    ])
    def test_throat_adapter_all_profile_types_produce_valid_yaml(
        self, profile_type: str
    ) -> None:
        """All profile types should produce YAML with ap1 > 0 and lpt >= 0."""
        result = subprocess.run(
            [
                PYHORN_CLI, "throat-adapter",
                "--d1", "50",
                "--d2", "100",
                "--type", profile_type,
            ],
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"throat-adapter --type {profile_type} failed: {result.stderr}"
        )

        sep = "# ────────────────────────────────────────────────────────────────────"
        yaml_text = result.stdout.split(sep)[0].strip()
        lines = yaml_text.splitlines()
        yaml_start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("throat_adapter:"):
                yaml_start_idx = i
                break

        assert yaml_start_idx is not None
        yaml_block = "\n".join(lines[yaml_start_idx:])
        data = _yaml.safe_load(yaml_block)

        ta = data["throat_adapter"]
        assert ta["ap1"] > 0, (
            f"[{profile_type}] ap1 must be positive, got {ta['ap1']}"
        )
        assert ta["lpt"] >= 0, (
            f"[{profile_type}] lpt must be non-negative, got {ta['lpt']}"
        )
        assert ta["type"] == profile_type, (
            f"[{profile_type}] expected type='{profile_type}', "
            f"got '{ta['type']}'"
        )
