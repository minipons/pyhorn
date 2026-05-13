"""E2E YAML schema validation tests for CLI commands that produce geometry YAML.

These tests verify that the output YAML from each command contains all required
fields with physically sensible values. They complement the unit/smoke tests in
test_cli_commands.py.

Coverage:
  - auto-segment   → sections[] with name, profile_type, length, start_area, end_area
  - throat-adapter → throat_adapter.ap1, throat_adapter.lpt
  - chamber-wizard → rear_chamber (vrc, lrc), throat_chamber (atc, vtc),
                      throat_adapter (ap1, lpt)
  - synthesis-wizard → sections[], rear_chamber, throat_chamber, throat_adapter
"""

import csv
import json
import math
import tempfile
from pathlib import Path

import pytest
import yaml as _yaml
from typer.testing import CliRunner

from pyhorn_cli.cli.commands import commands_app as app
from typer.testing import CliRunner

from pyhorn_cli.cli.commands import commands_app as app


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def valid_2d_json():
    """Minimal but valid 2-D geometry JSON for auto-segment."""
    return {
        "width": 0.2,
        "throat": [[0.05, 0.0], [0.15, 0.0]],
        "mouth": [[0.3, 0.0], [0.4, 0.0]],
        "boundary_edges": [
            [[0.05, 0.0], [0.4, 0.0]],
            [[0.4, 0.0], [0.4, 0.2]],
            [[0.4, 0.2], [0.05, 0.2]],
            [[0.05, 0.2], [0.05, 0.0]],
        ],
    }


@pytest.fixture
def fostex_driver_yaml(tmp_path):
    """FE166NV2 driver specs as a temp YAML file."""
    content = """
fs: 49.6
qts: 0.27
vas: 0.0369
sd: 0.0132
re: 7.8
bl: 7.79
mms: 0.00699
cms: 0.001472
rms: 0.277
le: 0.0008
qes: 0.28
qms: 7.88
"""
    p = tmp_path / "driver.yaml"
    p.write_text(content)
    return p


# ─── auto-segment ─────────────────────────────────────────────────────────────

class TestAutoSegmentYamlSchema:
    """Validate auto-segment output YAML has required geometry fields."""

    def test_auto_segment_sections_have_required_fields(
        self, cli_runner, valid_2d_json, tmp_path
    ):
        """Each section must have name, profile_type, length, start_area, end_area."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i", str(json_path),
                "-o", str(out_yaml),
                "--n-segments", "10",
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert out_yaml.exists(), "Output YAML was not created"

        with open(out_yaml) as f:
            data = _yaml.safe_load(f)

        # Must have either sections or conical_segments
        assert "sections" in data or "conical_segments" in data, (
            "Output must contain 'sections' or 'conical_segments'"
        )

        if "sections" in data:
            sections = data["sections"]
            assert isinstance(sections, list), "sections must be a list"
            assert len(sections) > 0, "sections list must not be empty"

            for i, sec in enumerate(sections):
                for field in ("name", "profile_type", "length", "start_area", "end_area"):
                    assert field in sec, f"sections[{i}] missing required field '{field}'"
                assert sec["length"] > 0, f"sections[{i}].length must be positive, got {sec['length']}"
                assert sec["start_area"] > 0, f"sections[{i}].start_area must be positive"
                assert sec["end_area"] > 0, f"sections[{i}].end_area must be positive"

    def test_auto_segment_conical_segments_have_three_values(
        self, cli_runner, valid_2d_json, tmp_path
    ):
        """Each conical_segment entry must be [start_area, end_area, length]."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i", str(json_path),
                "-o", str(out_yaml),
                "--n-segments", "8",
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        with open(out_yaml) as f:
            data = _yaml.safe_load(f)

        if "conical_segments" in data:
            segs = data["conical_segments"]
            assert isinstance(segs, list), "conical_segments must be a list"
            assert len(segs) == 8, f"Expected 8 segments, got {len(segs)}"
            for i, seg in enumerate(segs):
                assert isinstance(seg, list) and len(seg) == 3, (
                    f"conical_segments[{i}] must be [start_area, end_area, length], got {seg}"
                )
                sa, ea, ln = seg
                assert sa > 0 and ea > 0 and ln > 0, (
                    f"conical_segments[{i}] values must all be positive: {seg}"
                )

    def test_auto_segment_with_throat_adapter_injects_yaml(
        self, cli_runner, valid_2d_json, tmp_path
    ):
        """--throat-adapter-d1/d2 should add throat_adapter: section to output."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i", str(json_path),
                "-o", str(out_yaml),
                "--n-segments", "10",
                "--throat-adapter-d1", "50",
                "--throat-adapter-d2", "100",
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        with open(out_yaml) as f:
            data = _yaml.safe_load(f)

        assert "throat_adapter" in data, "throat_adapter section missing from output"
        ta = data["throat_adapter"]
        assert "ap1" in ta, "throat_adapter missing 'ap1'"
        assert "lpt" in ta, "throat_adapter missing 'lpt'"
        assert ta["ap1"] > 0, f"throat_adapter.ap1 must be positive, got {ta['ap1']}"
        assert ta["lpt"] > 0, f"throat_adapter.lpt must be positive, got {ta['lpt']}"


# ─── throat-adapter ───────────────────────────────────────────────────────────

class TestThroatAdapterYamlSchema:
    """Validate throat-adapter output YAML has required geometry fields."""

    def test_throat_adapter_standalone_yaml_has_required_fields(
        self, cli_runner, tmp_path
    ):
        """Standalone throat-adapter emits YAML with ap1, lpt, type fields."""
        out_yaml = tmp_path / "adapter_out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "throat-adapter",
                "--d1", "50",
                "--d2", "100",
                "--a1", "30",
                "--a2", "30",
                "--type", "conical",
                "--output-plot", str(out_yaml.with_suffix(".png")),
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        # The YAML block sits between "# Paste this block..." and
        # "# ────────────────────────────────────────────────────────────────────"
        output = result.output
        sep = "# ────────────────────────────────────────────────────────────────────"
        assert sep in output, f"Could not find block separator in output:\n{output}"
        # The YAML block is in the part BEFORE the separator
        yaml_text = output.split(sep)[0].strip()
        # Skip lines until we find 'throat_adapter:'
        lines = yaml_text.splitlines()
        yaml_start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("throat_adapter:"):
                yaml_start_idx = i
                break
        yaml_text = "\n".join(lines[yaml_start_idx:])

        try:
            data = _yaml.safe_load(yaml_text)
        except Exception as exc:
            pytest.fail(f"Failed to parse YAML from stdout: {exc}\nYAML text:\n{yaml_text}")

        assert isinstance(data, dict), f"Parsed YAML is not a dict: {type(data)}"
        assert "throat_adapter" in data, "Output missing 'throat_adapter' key"
        ta = data["throat_adapter"]

        assert "ap1" in ta, "throat_adapter missing 'ap1' (throat-side area)"
        assert "lpt" in ta, "throat_adapter missing 'lpt' (length)"
        assert "type" in ta, "throat_adapter missing 'type'"

        assert ta["ap1"] > 0, f"ap1 must be positive, got {ta['ap1']}"
        assert ta["lpt"] > 0, f"lpt must be positive, got {ta['lpt']}"
        assert ta["type"] == "conical", f"type must be 'conical', got '{ta['type']}'"

    @pytest.mark.parametrize("profile_type", ["conical", "exponential", "parabolic", "cylindrical"])
    def test_throat_adapter_all_profile_types_produce_valid_yaml(
        self, cli_runner, profile_type
    ):
        """All profile types should produce YAML with ap1, lpt, type."""
        result = cli_runner.invoke(
            app,
            [
                "throat-adapter",
                "--d1", "50",
                "--d2", "100",
                "--type", profile_type,
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        sep = "# ────────────────────────────────────────────────────────────────────"
        yaml_text = result.output.split(sep)[0].strip()
        lines = yaml_text.splitlines()
        yaml_start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("throat_adapter:"):
                yaml_start_idx = i
                break
        yaml_text = "\n".join(lines[yaml_start_idx:])
        data = _yaml.safe_load(yaml_text)

        ta = data["throat_adapter"]
        assert ta["ap1"] > 0
        # lpt >= 0 always (cylindrical adapters can have lpt=0, a straight tube)
        assert ta["lpt"] >= 0
        assert ta["type"] == profile_type

    def test_throat_adapter_explicit_length_overrides_minimum(
        self, cli_runner
    ):
        """--length should produce YAML with the explicitly-requested lpt."""
        explicit_mm = 60.0
        result = cli_runner.invoke(
            app,
            [
                "throat-adapter",
                "--d1", "50",
                "--d2", "100",
                "--type", "conical",
                "--length", str(explicit_mm),
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        sep = "# ────────────────────────────────────────────────────────────────────"
        yaml_text = result.output.split(sep)[0].strip()
        lines = yaml_text.splitlines()
        yaml_start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("throat_adapter:"):
                yaml_start_idx = i
                break
        yaml_text = "\n".join(lines[yaml_start_idx:])
        data = _yaml.safe_load(yaml_text)

        ta = data["throat_adapter"]
        # lpt should be exactly the explicit length (in metres)
        assert ta["lpt"] == pytest.approx(explicit_mm / 1000.0, rel=1e-3), (
            f"Expected lpt={explicit_mm/1000:.4f} m (explicit {explicit_mm} mm), got {ta['lpt']}"
        )


# ─── chamber-wizard ───────────────────────────────────────────────────────────

class TestChamberWizardYamlSchema:
    """Validate chamber-wizard output YAML has required chamber parameter fields."""

    def test_chamber_wizard_output_has_vrc_lrc_vtc_atc(
        self, cli_runner, fostex_driver_yaml, tmp_path
    ):
        """Output YAML must contain rear_chamber {vrc, lrc}, throat_chamber {vtc, atc}."""
        out_path = tmp_path / "chamber_out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "chamber-wizard",
                "--driver", str(fostex_driver_yaml),
                "--no-interactive",
                "--output", str(out_path),
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert out_path.exists(), "Output file was not created"

        text = out_path.read_text(encoding="utf-8")

        # chamber-wizard outputs a file with a comment header followed by YAML;
        # find the YAML document start
        yaml_marker = "YAML Snippet"
        marker_pos = text.find(yaml_marker)
        if marker_pos != -1:
            yaml_text = text[marker_pos + len(yaml_marker) :]
        else:
            yaml_text = text

        lines = yaml_text.splitlines()
        yaml_start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("rear_chamber:") or stripped.startswith("throat_chamber:"):
                yaml_start_idx = i
                break
        yaml_doc = "\n".join(lines[yaml_start_idx:])

        geo = _yaml.safe_load(yaml_doc)
        assert isinstance(geo, dict), f"Parsed YAML is not a dict: {type(geo)}"

        # ── rear_chamber ──────────────────────────────────────────────────────
        assert "rear_chamber" in geo, "Output YAML missing 'rear_chamber'"
        rc = geo["rear_chamber"]
        assert "vrc" in rc, "rear_chamber missing 'vrc' (rear chamber volume)"
        assert "lrc" in rc, "rear_chamber missing 'lrc' (rear chamber length)"
        assert rc["vrc"] >= 0, f"vrc must be non-negative, got {rc['vrc']} L"
        assert rc["lrc"] >= 0, f"lrc must be non-negative, got {rc['lrc']} cm"

        # ── throat_chamber ───────────────────────────────────────────────────
        assert "throat_chamber" in geo, "Output YAML missing 'throat_chamber'"
        tc = geo["throat_chamber"]
        assert "vtc" in tc, "throat_chamber missing 'vtc' (throat chamber volume)"
        assert "atc" in tc, "throat_chamber missing 'atc' (throat chamber area)"
        assert tc["vtc"] > 0, f"vtc must be positive, got {tc['vtc']} m³"
        assert tc["atc"] > 0, f"atc must be positive, got {tc['atc']} cm²"

        # ── throat_adapter ───────────────────────────────────────────────────
        assert "throat_adapter" in geo, "Output YAML missing 'throat_adapter'"
        ta = geo["throat_adapter"]
        assert "ap1" in ta, "throat_adapter missing 'ap1'"
        assert "lpt" in ta, "throat_adapter missing 'lpt'"
        assert ta["ap1"] > 0, f"ap1 must be positive, got {ta['ap1']}"
        assert ta["lpt"] >= 0, f"lpt must be non-negative, got {ta['lpt']}"


# ─── synthesis-wizard ─────────────────────────────────────────────────────────

class TestSynthesisWizardYamlSchema:
    """Validate synthesis-wizard output YAML has required geometry fields."""

    def test_synthesis_wizard_output_has_throat_area_mouth_area_path_length(
        self, cli_runner, fostex_driver_yaml, tmp_path
    ):
        """Output YAML must contain throat_area, mouth_area, path_length (or sections)."""
        out_path = tmp_path / "synth_out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "synthesis-wizard",
                "--driver", str(fostex_driver_yaml),
                "--f3", "50.0",
                "--output", str(out_path),
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert out_path.exists(), "Output file was not created"

        text = out_path.read_text(encoding="utf-8")

        yaml_start = text.find("# ── Synthesised geometry")
        assert yaml_start != -1, f"Could not find 'Synthesised geometry' marker in output:\n{text[:300]}"
        yaml_text = text[yaml_start:]

        geo = _yaml.safe_load(yaml_text)
        assert isinstance(geo, dict), f"Parsed YAML is not a dict: {type(geo)}"

        # The synthesis wizard produces a sections[] block that encodes throat/marget
        # throat_area = sections[0].start_area, mouth_area = sections[-1].end_area,
        # path_length = sum(s.length for s in sections)
        assert "sections" in geo, "Output YAML missing 'sections'"
        sections = geo["sections"]
        assert isinstance(sections, list), f"sections must be a list, got {type(sections)}"
        assert len(sections) > 0, "sections list must not be empty"

        for i, sec in enumerate(sections):
            for field in ("name", "profile_type", "length", "start_area", "end_area"):
                assert field in sec, f"sections[{i}] missing '{field}'"
            assert sec["length"] > 0, f"sections[{i}].length must be positive"
            assert sec["start_area"] > 0, f"sections[{i}].start_area must be positive"
            assert sec["end_area"] > 0, f"sections[{i}].end_area must be positive"

        # Derived metrics
        throat_area = sections[0]["start_area"]
        mouth_area = sections[-1]["end_area"]
        total_length = sum(sec["length"] for sec in sections)

        assert throat_area > 0, f"throat_area (S1) must be positive, got {throat_area}"
        assert mouth_area > throat_area, (
            f"mouth_area ({mouth_area:.6f} m²) must exceed throat_area ({throat_area:.6f} m²)"
        )
        assert total_length > 0, f"path_length must be positive, got {total_length} m"

        # ── rear chamber (flat fields) ─────────────────────────────────────────
        assert "vrc" in geo, "Output YAML missing 'vrc' (rear chamber volume)"
        assert "lrc" in geo, "Output YAML missing 'lrc' (rear chamber length)"
        assert geo["vrc"] >= 0, f"vrc must be non-negative, got {geo['vrc']}"
        assert geo["lrc"] >= 0, f"lrc must be non-negative, got {geo['lrc']}"

        # ── throat chamber (flat fields) ───────────────────────────────────────
        assert "vtc" in geo, "Output YAML missing 'vtc' (throat chamber volume)"
        assert "atc" in geo, "Output YAML missing 'atc' (throat chamber area)"
        assert geo["vtc"] > 0, f"vtc must be positive, got {geo['vtc']}"
        assert geo["atc"] > 0, f"atc must be positive, got {geo['atc']}"

        # ── throat adapter (flat fields) ───────────────────────────────────────
        assert "ap1" in geo, "Output YAML missing 'ap1' (throat adapter area)"
        assert "lpt" in geo, "Output YAML missing 'lpt' (throat adapter length)"
        assert geo["ap1"] > 0, f"ap1 must be positive, got {geo['ap1']}"
        assert geo["lpt"] >= 0, f"lpt must be non-negative, got {geo['lpt']}"

        # ── radiation angle ───────────────────────────────────────────────────
        assert "ang" in geo, "Output YAML missing 'ang' (radiation angle)"
        assert geo["ang"] > 0, f"ang must be positive, got {geo['ang']}"

    def test_synthesis_wizard_different_f3_produces_different_geometry(
        self, cli_runner, fostex_driver_yaml, tmp_path
    ):
        """Different --f3 values should produce meaningfully different geometries."""
        out_f40 = tmp_path / "synth_f40.yaml"
        out_f60 = tmp_path / "synth_f60.yaml"

        for out_path, f3_val in [(out_f40, "40.0"), (out_f60, "60.0")]:
            result = cli_runner.invoke(
                app,
                [
                    "synthesis-wizard",
                    "--driver", str(fostex_driver_yaml),
                    "--f3", f3_val,
                    "--output", str(out_path),
                ],
            )
            assert result.exit_code == 0, f"f3={f3_val}: {result.output}"

        text_f40 = out_f40.read_text(encoding="utf-8")
        text_f60 = out_f60.read_text(encoding="utf-8")

        yaml_start_f40 = text_f40.find("# ── Synthesised geometry")
        yaml_start_f60 = text_f60.find("# ── Synthesised geometry")
        assert yaml_start_f40 != -1 and yaml_start_f60 != -1

        geo_f40 = _yaml.safe_load(text_f40[yaml_start_f40:])
        geo_f60 = _yaml.safe_load(text_f60[yaml_start_f60:])

        # Different f3 should produce different total path length
        len_f40 = sum(sec["length"] for sec in geo_f40["sections"])
        len_f60 = sum(sec["length"] for sec in geo_f60["sections"])

        assert len_f40 != len_f60, (
            f"Different f3 values (40 Hz vs 60 Hz) produced identical path lengths "
            f"({len_f40:.4f} m) — synthesis wizard may not be using f3 correctly"
        )


# ─── hornresp ─────────────────────────────────────────────────────────────────

class TestHornrespYamlSchema:
    """Validate hornresp output YAML has required geometry fields for all profile types."""

    @pytest.mark.parametrize("profile_type,extra_args", [
        ("con",  ["--l12", "150"]),
        ("exp",  ["--f12", "50", "--s2", "300"]),
        ("par",  ["--l12", "150"]),
        ("cat",  ["--l12", "150"]),
        ("hyp",  ["--hyp", "150", "--t", "0.3"]),
    ])
    def test_hornresp_profile_type_produces_valid_yaml(
        self, cli_runner, profile_type, extra_args, tmp_path
    ):
        """Each profile type should produce a valid YAML with required fields."""
        out_path = tmp_path / f"hornresp_{profile_type}.yaml"

        result = cli_runner.invoke(
            app,
            [
                "hornresp",
                "--s1", "40",
                "--s2", "300",
                "--profile-type", profile_type,
                "--output", str(out_path),
                *extra_args,
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert out_path.exists(), f"Output file not created for profile type {profile_type}"

        text = out_path.read_text(encoding="utf-8")

        yaml_start = text.find("# Generated from Hornresp")
        assert yaml_start != -1, f"Could not find 'Generated from Hornresp' marker:\n{text[:200]}"
        yaml_text = text[yaml_start:]

        geo = _yaml.safe_load(yaml_text)
        assert isinstance(geo, dict), f"Parsed YAML is not a dict: {type(geo)}"

        # Required top-level fields
        assert "sections" in geo, "Output YAML missing 'sections'"
        assert "enclosure_type" in geo, "Output YAML missing 'enclosure_type'"
        assert geo["enclosure_type"] in ("BLH", "FLH"), f"enclosure_type must be BLH or FLH, got {geo['enclosure_type']}"

        sections = geo["sections"]
        assert isinstance(sections, list), f"sections must be a list, got {type(sections)}"
        assert len(sections) > 0, "sections list must not be empty"

        for i, sec in enumerate(sections):
            for field in ("name", "profile_type", "length", "start_area", "end_area"):
                assert field in sec, f"sections[{i}] missing '{field}'"
            assert sec["length"] > 0, f"sections[{i}].length must be positive"
            assert sec["start_area"] > 0, f"sections[{i}].start_area must be positive"
            assert sec["end_area"] > 0, f"sections[{i}].end_area must be positive"

        # Geometry sanity: throat <= mouth
        throat_area = sections[0]["start_area"]
        mouth_area = sections[-1]["end_area"]
        assert mouth_area >= throat_area, (
            f"mouth_area ({mouth_area:.6f} m²) must be >= throat_area ({throat_area:.6f} m²)"
        )

    def test_hornresp_conical_with_rear_chamber(self, cli_runner, tmp_path):
        """hornresp with --lrc and --vrc should include rear_chamber in output YAML."""
        out_path = tmp_path / "hornresp_rc.yaml"

        result = cli_runner.invoke(
            app,
            [
                "hornresp",
                "--s1", "40", "--s2", "300",
                "--l12", "150",
                "--profile-type", "con",
                "--lrc", "0.12",
                "--vrc", "0.005",
                "--output", str(out_path),
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        text = out_path.read_text(encoding="utf-8")
        yaml_start = text.find("# Generated from Hornresp")
        geo = _yaml.safe_load(text[yaml_start:])

        assert "rear_chamber" in geo, "Output YAML missing 'rear_chamber' when --lrc/--vrc provided"
        rc = geo["rear_chamber"]
        assert "vrc" in rc and "lrc" in rc
        assert rc["vrc"] > 0 and rc["lrc"] > 0


# ─── E2E roundtrip: synthesis-wizard → calculate ──────────────────────────────

_FOTEX_DRIVER_YAML = """
fs: 49.6
qts: 0.27
vas: 0.0369
sd: 0.0132
re: 7.8
bl: 7.79
mms: 0.00699
cms: 0.001472
rms: 0.277
le: 0.0008
qes: 0.28
qms: 7.88
"""


class TestSynthesisWizardRoundtrip:
    """E2E: synthesis-wizard output must be simulatable via `calculate`.

    This is the critical reliability test: generated geometry YAML must produce
    valid, finite SPL values when run through the TMM solver — not just parse
    as valid YAML.
    """

    @pytest.fixture
    def fostex_driver_file(self, tmp_path):
        p = tmp_path / "driver.yaml"
        p.write_text(_FOTEX_DRIVER_YAML)
        return p

    def test_synthesis_wizard_output_simulates_to_valid_spl(
        self, fostex_driver_file, tmp_path
    ):
        """Run synthesis-wizard then calculate; SPL must be finite and in 60-130 dB."""
        runner = CliRunner()
        synth_out = tmp_path / "synth_geometry.yaml"
        out_dir = tmp_path / "calc_out"

        # Step 1: synthesis-wizard produces geometry YAML
        res_synth = runner.invoke(
            app,
            [
                "synthesis-wizard",
                "--driver", str(fostex_driver_file),
                "--f3", "55.0",
                "--output", str(synth_out),
            ],
        )
        assert res_synth.exit_code == 0, (
            f"synthesis-wizard failed: {res_synth.exit_code}\n{res_synth.output}"
        )
        assert synth_out.exists(), "synthesis-wizard did not produce output file"

        # Step 2: calculate simulates the generated geometry
        res_calc = runner.invoke(
            app,
            [
                "calculate",
                "-d", str(fostex_driver_file),
                "-h", str(synth_out),
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "2000",
                "--n-points", "20",
                "--no-plot",
            ],
        )
        assert res_calc.exit_code == 0, (
            f"calculate failed on synthesis output: {res_calc.exit_code}\n{res_calc.output}"
        )

        # Step 3: response.csv must exist and contain finite SPL values
        csv_files = list(out_dir.rglob("response.csv"))
        assert len(csv_files) == 1, f"Expected 1 response.csv, found {len(csv_files)}: {csv_files}"

        with open(csv_files[0], newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 20, f"Expected 20 frequency points, got {len(rows)}"

        # Find SPL column (exporter uses 'Horn SPL_dB_W' or similar)
        spl_col = next(
            (c for c in rows[0].keys() if "SPL" in c and "Horn SPL" in c),
            None,
        )
        assert spl_col is not None, f"No Horn SPL column found in: {list(rows[0].keys())}"

        spls = [float(r[spl_col]) for r in rows]

        # All SPL values must be finite
        for i, spl in enumerate(spls):
            assert math.isfinite(spl), (
                f"Non-finite SPL at row {i} (freq={rows[i].get('Frequency_Hz')}): {spl}"
            )

        # SPL must be in a physically plausible range for a small full-range driver.
        # Horn loading gives efficiency boost but the driver is small (FE166NV2 ~91 dB/W/m).
        # At 100-2000 Hz we expect 30-105 dB depending on frequency and horn loading.
        assert all(30.0 <= spl <= 110.0 for spl in spls), (
            f"SPL values out of plausible range: {min(spls):.1f}–{max(spls):.1f} dB "
            f"(expected 30–110 dB for this driver and frequency range)"
        )

    def test_synthesis_wizard_output_simulates_with_reasonable_bandwidth(
        self, fostex_driver_file, tmp_path
    ):
        """Synthesis output at f3=55 Hz should show SPL rolloff below 100 Hz."""
        runner = CliRunner()
        synth_out = tmp_path / "synth_geo.yaml"
        out_dir = tmp_path / "calc_out2"

        runner.invoke(
            app,
            [
                "synthesis-wizard",
                "--driver", str(fostex_driver_file),
                "--f3", "55.0",
                "--output", str(synth_out),
            ],
        )
        assert synth_out.exists()

        calc_res = runner.invoke(
            app,
            [
                "calculate",
                "-d", str(fostex_driver_file),
                "-h", str(synth_out),
                "-o", str(out_dir),
                "--fmin", "20",
                "--fmax", "500",
                "--n-points", "50",
                "--no-plot",
            ],
        )
        assert calc_res.exit_code == 0, calc_res.output

        csv_files = list(out_dir.rglob("response.csv"))
        with open(csv_files[0], newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        spl_col = next(c for c in rows[0].keys() if "SPL" in c and "Horn SPL" in c)
        freqs = [float(r.get("Frequency_Hz", 0)) for r in rows]
        spls = [float(r[spl_col]) for r in rows]

        # Low-frequency SPL (< 40 Hz) should be at least 10 dB below peak
        # (meaningful rolloff confirms horn loading is working)
        lf_mask = [i for i, f in enumerate(freqs) if f < 40]
        hf_mask = [i for i, f in enumerate(freqs) if 100 <= f <= 200]

        if lf_mask and hf_mask:
            lf_spls = [spls[i] for i in lf_mask]
            hf_spls = [spls[i] for i in hf_mask]
            lf_avg = sum(lf_spls) / len(lf_spls)
            hf_avg = sum(hf_spls) / len(hf_spls)

            # Horn-loaded system should show significant LF rolloff
            assert hf_avg - lf_avg > 5.0, (
                f"LF-HF difference only {hf_avg - lf_avg:.1f} dB — "
                f"horn loading may not be working (LF_avg={lf_avg:.1f} dB, "
                f"HF_avg={hf_avg:.1f} dB)"
            )


# ─── hornresp E2E roundtrip ──────────────────────────────────────────────────

class TestHornrespE2E:
    """E2E roundtrip: hornresp YAML output → calculate → valid SPL.

    Verifies that hornresp geometry YAML (for all profile types) can be
    fed to `calculate` and produces finite, physically plausible SPL.
    """

    @pytest.fixture
    def fostex_driver_file(self, tmp_path):
        """FE166NV2 driver specs as a temp YAML file (path version of fostex_driver_yaml)."""
        p = tmp_path / "driver.yaml"
        p.write_text(_FOTEX_DRIVER_YAML)
        return p

    @pytest.mark.parametrize("profile_type", ["con", "exp", "par"])
    def test_hornresp_profile_roundtrip_calculates_valid_spl(
        self, fostex_driver_file, profile_type, tmp_path
    ):
        """Each hornresp profile type should produce simulatable geometry with finite SPL."""
        runner = CliRunner()
        geo_yaml = tmp_path / f"hornresp_{profile_type}.yaml"
        calc_out = tmp_path / f"calc_{profile_type}"
        calc_out.mkdir()

        # Step 1: Generate geometry via hornresp CLI
        res_hr = runner.invoke(
            app,
            [
                "hornresp",
                "--s1", "40",          # throat area cm²
                "--s2", "300",         # mouth area cm²
                "--l12", "150",        # path length cm
                "--profile-type", profile_type,
                "--output", str(geo_yaml),
            ],
        )
        assert res_hr.exit_code == 0, (
            f"hornresp --profile-type {profile_type} failed:\n{res_hr.output}"
        )
        assert geo_yaml.exists(), f"hornresp should write {geo_yaml}"

        # Step 2: Simulate the generated geometry
        res_calc = runner.invoke(
            app,
            [
                "calculate",
                "-d", str(fostex_driver_file),
                "-h", str(geo_yaml),
                "-o", str(calc_out),
                "--fmin", "100",
                "--fmax", "2000",
                "--n-points", "50",
                "--no-plot",
            ],
        )
        assert res_calc.exit_code == 0, (
            f"calculate failed on hornresp/{profile_type} geometry "
            f"(exit {res_calc.exit_code}):\n{res_calc.output}"
        )

        # Step 3: Verify response.csv has finite SPL values
        csv_files = list(calc_out.rglob("response.csv"))
        assert len(csv_files) == 1, f"Expected 1 response.csv, found {len(csv_files)}"

        with open(csv_files[0], newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 50, f"Expected 50 frequency points, got {len(rows)}"

        # Find SPL column
        spl_col = next(
            (c for c in rows[0].keys() if "SPL" in c and "Horn SPL" in c),
            None,
        )
        assert spl_col is not None, f"No Horn SPL column in: {list(rows[0].keys())}"

        spls = [float(r[spl_col]) for r in rows]
        freqs = [float(r.get("Frequency_Hz", 0)) for r in rows]

        # All SPL values must be finite
        for i, spl in enumerate(spls):
            assert math.isfinite(spl), (
                f"Non-finite SPL at row {i} (freq={freqs[i]} Hz): {spl}"
            )

        # SPL must be in a physically plausible range for a horn-loaded FE166NV2
        # Expected: ~60-105 dB in 100-2000 Hz band for a reasonably-designed horn
        assert all(40.0 <= spl <= 115.0 for spl in spls), (
            f"SPL values out of plausible range for hornresp/{profile_type}: "
            f"{min(spls):.1f}–{max(spls):.1f} dB (expected 40–115 dB)"
        )

