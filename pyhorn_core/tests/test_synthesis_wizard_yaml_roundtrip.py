"""Synthesis-wizard YAML roundtrip tests.

Verifies that the synthesis-wizard CLI (`pyhorn synthesis-wizard --f3 N`)
produces a geometry YAML that can be parsed back and re-serialised
without loss of data.

Run:
    pytest pyhorn_core/tests/test_synthesis_wizard_yaml_roundtrip.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from pyhorn_cli.cli.commands import commands_app as app


# ─────────────────────────────────────────────────────────────────────────────
# Paths & fixtures
# ─────────────────────────────────────────────────────────────────────────────

TESTS_DIR = Path(__file__).parent
ROOT_DIR  = TESTS_DIR.parent.parent
DRIVER_YAML = ROOT_DIR / "drivers" / "FE166NV2.yaml"


@pytest.fixture
def cli_runner():
    return CliRunner()


def _run_synthesis_wizard(f3_hz: float, qts_alignment: float = 0.55) -> str:
    """Invoke synthesis-wizard and return the stdout text."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "synthesis-wizard",
            "--driver",  str(DRIVER_YAML),
            "--f3",      str(f3_hz),
            "--qts-alignment", str(qts_alignment),
        ],
    )
    assert result.exit_code == 0, f"CLI exited {result.exit_code}:\n{result.output}\n{result.exception}"
    return result.output


def _extract_geometry_yaml(output: str) -> str:
    """Extract the geometry YAML block from synthesis-wizard stdout.

    The geometry block starts at ``sections:`` and ends before the
    ``✓ Synthesis complete`` completion line.
    """
    start = output.find("sections:")
    end   = output.find("✓", start)
    assert start != -1, "Could not find 'sections:' in synthesis-wizard output"
    assert end   != -1, "Could not find completion marker in synthesis-wizard output"
    return output[start:end].rstrip()


def _yaml_to_horn_geometry_compatible(yaml_text: str) -> dict:
    """Convert synthesis-wizard geometry YAML to a format compatible with
    ``parse_horn_geometry()``.

    The synthesis wizard outputs nested dicts for chambers/adapter and omits
    top-level scalar geometry fields.  This helper:

      - Flattens ``rear_chamber``   → ``vrc``, ``lrc``
      - Flattens ``throat_chamber`` → ``vtc``, ``atc``
      - Flattens ``throat_adapter`` → ``ap1``, ``lpt``
      - Derives ``throat_area``   from the first section's ``start_area``
      - Derives ``mouth_area``    from the last  section's ``end_area``
      - Derives ``path_length``   as the sum of all section lengths
      - Derives ``n_segments``    (default 100)

    Returns the transformed dict ready for ``HornGeometry(**)``.
    """
    data = yaml.safe_load(yaml_text)

    # Flatten rear_chamber
    if "rear_chamber" in data:
        rc = data.pop("rear_chamber")
        data["vrc"] = rc.get("vrc", 0.0)
        data["lrc"] = rc.get("lrc", 0.0) / 100.0   # cm → m

    # Flatten throat_chamber
    if "throat_chamber" in data:
        tc = data.pop("throat_chamber")
        data["vtc"] = tc.get("vtc", 0.0)
        data["atc"] = tc.get("atc", 0.0) / 1e4     # cm² → m²

    # Flatten throat_adapter
    if "throat_adapter" in data:
        ta = data.pop("throat_adapter")
        data["ap1"] = ta.get("ap1", 0.0) / 1e4     # cm² → m²
        data["lpt"] = ta.get("lpt", 0.0) / 1000.0   # mm → m

    # Derive throat/mouth/path from sections
    sections = data.get("sections") or []
    if sections:
        data["throat_area"]  = sections[0].get("start_area", 0.0)
        data["mouth_area"]   = sections[-1].get("end_area",   0.0)
        data["path_length"]  = sum(s.get("length", 0.0) for s in sections)
        data["n_segments"]   = 100
    else:
        data["throat_area"]  = 0.0
        data["mouth_area"]   = 0.0
        data["path_length"]  = 0.0
        data["n_segments"]   = 100

    return data


def _parse_via_horn_geometry(yaml_text: str):
    """Parse geometry YAML through HornGeometry, returning the instance."""
    from pyhorn_core.config.parser import parse_horn_geometry

    transformed = _yaml_to_horn_geometry_compatible(yaml_text)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(transformed, f)
        tmp_path = Path(f.name)

    try:
        return parse_horn_geometry(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Required fields present and physically valid
# ─────────────────────────────────────────────────────────────────────────────

class TestSynthesisWizardRequiredFields:
    """Assert all required geometry fields are present and physically valid."""

    @pytest.mark.parametrize("f3_hz", [50.0, 80.0, 100.0])
    def test_throat_area_is_positive(self, f3_hz: float):
        """throat_area (S1) must be strictly positive (driver piston area)."""
        output = _run_synthesis_wizard(f3_hz)
        yaml_text = _extract_geometry_yaml(output)
        geom = _parse_via_horn_geometry(yaml_text)
        assert geom.throat_area > 0, f"throat_area must be > 0, got {geom.throat_area}"

    @pytest.mark.parametrize("f3_hz", [50.0, 80.0, 100.0])
    def test_mouth_area_greater_than_throat_area(self, f3_hz: float):
        """mouth_area (S2) must exceed throat_area (S1) — horn must expand."""
        output = _run_synthesis_wizard(f3_hz)
        yaml_text = _extract_geometry_yaml(output)
        geom = _parse_via_horn_geometry(yaml_text)
        assert geom.mouth_area > geom.throat_area, (
            f"mouth_area ({geom.mouth_area}) must exceed throat_area ({geom.throat_area})"
        )

    @pytest.mark.parametrize("f3_hz", [50.0, 80.0, 100.0])
    def test_path_length_is_positive(self, f3_hz: float):
        """Total path length must be strictly positive."""
        output = _run_synthesis_wizard(f3_hz)
        yaml_text = _extract_geometry_yaml(output)
        geom = _parse_via_horn_geometry(yaml_text)
        assert geom.path_length > 0, f"path_length must be > 0, got {geom.path_length}"

    @pytest.mark.parametrize("f3_hz", [50.0, 80.0, 100.0])
    def test_sections_list_is_non_empty(self, f3_hz: float):
        """Horn must have at least one section."""
        output = _run_synthesis_wizard(f3_hz)
        yaml_text = _extract_geometry_yaml(output)
        geom = _parse_via_horn_geometry(yaml_text)
        assert geom.sections is not None and len(geom.sections) > 0, (
            "sections list must be non-empty"
        )

    @pytest.mark.parametrize("f3_hz", [50.0, 80.0, 100.0])
    def test_section_areas_are_positive(self, f3_hz: float):
        """Every section must have positive start_area, end_area, and length."""
        output = _run_synthesis_wizard(f3_hz)
        yaml_text = _extract_geometry_yaml(output)
        geom = _parse_via_horn_geometry(yaml_text)
        for i, sec in enumerate(geom.sections):
            assert sec.start_area > 0, f"sections[{i}].start_area must be > 0"
            assert sec.end_area   > 0, f"sections[{i}].end_area must be > 0"
            assert sec.length     > 0, f"sections[{i}].length must be > 0"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Different f3 values produce different geometries
# ─────────────────────────────────────────────────────────────────────────────

class TestSynthesisWizardDifferentF3:
    """Verify that two distinct f3 targets produce distinct horn geometries."""

    def test_f3_50_vs_f3_80_differ_in_path_length(self):
        """Different f3 targets must produce different path lengths (or sections)."""
        out_50 = _run_synthesis_wizard(50.0)
        out_80 = _run_synthesis_wizard(80.0)

        yaml_50 = _extract_geometry_yaml(out_50)
        yaml_80 = _extract_geometry_yaml(out_80)

        geom_50 = _parse_via_horn_geometry(yaml_50)
        geom_80 = _parse_via_horn_geometry(yaml_80)

        # At least one geometric property must differ
        different = (
            geom_50.path_length != geom_80.path_length
            or geom_50.mouth_area  != geom_80.mouth_area
            or geom_50.throat_area  != geom_80.throat_area
            or len(geom_50.sections) != len(geom_80.sections)
        )
        assert different, (
            "f3=50 and f3=80 must produce different geometries; "
            "got identical values for path_length, mouth_area, throat_area, and section count"
        )

    def test_f3_50_vs_f3_100_differ(self):
        """f3=50 and f3=100 must produce detectably different geometries."""
        out_50 = _run_synthesis_wizard(50.0)
        out_100 = _run_synthesis_wizard(100.0)

        yaml_50  = _extract_geometry_yaml(out_50)
        yaml_100 = _extract_geometry_yaml(out_100)

        geom_50  = _parse_via_horn_geometry(yaml_50)
        geom_100 = _parse_via_horn_geometry(yaml_100)

        different = (
            geom_50.path_length != geom_100.path_length
            or geom_50.mouth_area  != geom_100.mouth_area
            or geom_50.throat_area  != geom_100.throat_area
            or len(geom_50.sections) != len(geom_100.sections)
        )
        assert different, (
            "f3=50 and f3=100 must produce different geometries"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Re-serialise parsed geometry to YAML and re-parse → identical fields
# ─────────────────────────────────────────────────────────────────────────────

class TestSynthesisWizardYamlRoundtrip:
    """Verify parsed geometry survives YAML → parse → YAML → parse roundtrip."""

    @pytest.mark.parametrize("f3_hz", [50.0, 80.0, 100.0])
    def test_yaml_roundtrip_produces_identical_fields(self, f3_hz: float):
        """Re-serialising parsed geometry and re-parsing must yield identical field values."""
        from pyhorn_core.config.parser import parse_horn_geometry

        output = _run_synthesis_wizard(f3_hz)
        yaml_text = _extract_geometry_yaml(output)

        # ── First parse ────────────────────────────────────────────────────
        geom1 = _parse_via_horn_geometry(yaml_text)

        # Collect key scalar fields
        n_sections1 = len(geom1.sections) if geom1.sections else 0
        fields1 = {
            "throat_area":  geom1.throat_area,
            "mouth_area":   geom1.mouth_area,
            "path_length":  geom1.path_length,
            "n_segments":   geom1.n_segments,
            "vrc":          geom1.vrc,
            "lrc":          geom1.lrc,
            "vtc":          geom1.vtc,
            "atc":          geom1.atc,
            "ap1":          geom1.ap1,
            "lpt":          geom1.lpt,
            "ang":          geom1.ang,
        }
        section_fields1 = []
        if geom1.sections:
            for sec in geom1.sections:
                section_fields1.append({
                    "name":          sec.name,
                    "profile_type":  sec.profile_type,
                    "length":        sec.length,
                    "start_area":    sec.start_area,
                    "end_area":      sec.end_area,
                })

        # ── Re-serialise to YAML ────────────────────────────────────────────
        roundtrip_data = {
            "throat_area":  geom1.throat_area,
            "mouth_area":   geom1.mouth_area,
            "path_length":  geom1.path_length,
            "n_segments":   geom1.n_segments,
            "vrc":          geom1.vrc,
            "lrc":          geom1.lrc,
            "vtc":          geom1.vtc,
            "atc":          geom1.atc,
            "ap1":          geom1.ap1,
            "lpt":          geom1.lpt,
            "ang":          geom1.ang,
            "sections":      section_fields1,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(roundtrip_data, f)
            tmp_path = Path(f.name)

        try:
            # ── Re-parse ────────────────────────────────────────────────────
            geom2 = parse_horn_geometry(tmp_path)

            # ── Compare scalar fields ──────────────────────────────────────
            for field, val1 in fields1.items():
                val2 = getattr(geom2, field)
                assert val2 == pytest.approx(val1), (
                    f"Roundtrip mismatch for '{field}': {val1} → {val2}"
                )

            # ── Compare section count ───────────────────────────────────────
            n_sections2 = len(geom2.sections) if geom2.sections else 0
            assert n_sections2 == n_sections1, (
                f"Roundtrip mismatch for n_sections: {n_sections1} → {n_sections2}"
            )

            # ── Compare sections ──────────────────────────────────────────
            for i, (s1, s2) in enumerate(zip(geom1.sections, geom2.sections)):
                assert s2.name         == s1.name,        f"sections[{i}].name"
                assert s2.profile_type == s1.profile_type, f"sections[{i}].profile_type"
                assert s2.length       == pytest.approx(s1.length),    f"sections[{i}].length"
                assert s2.start_area   == pytest.approx(s1.start_area), f"sections[{i}].start_area"
                assert s2.end_area     == pytest.approx(s1.end_area),   f"sections[{i}].end_area"

        finally:
            tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Direct parse (without preprocessing) validates the raw YAML structure
# ─────────────────────────────────────────────────────────────────────────────

class TestSynthesisWizardYamlStructure:
    """Validate the raw YAML structure produced by synthesis-wizard."""

    @pytest.mark.parametrize("f3_hz", [50.0, 80.0, 100.0])
    def test_yaml_contains_sections_key(self, f3_hz: float):
        """Raw YAML must contain a top-level ``sections`` key."""
        output = _run_synthesis_wizard(f3_hz)
        yaml_text = _extract_geometry_yaml(output)
        data = yaml.safe_load(yaml_text)
        assert "sections" in data, "YAML must contain 'sections' key"

    @pytest.mark.parametrize("f3_hz", [50.0, 80.0, 100.0])
    def test_yaml_sections_have_name_and_profile_type(self, f3_hz: float):
        """Each section dict must have ``name`` and ``profile_type`` keys."""
        output = _run_synthesis_wizard(f3_hz)
        yaml_text = _extract_geometry_yaml(output)
        data = yaml.safe_load(yaml_text)
        for i, sec in enumerate(data.get("sections") or []):
            assert "name"          in sec, f"sections[{i}] missing 'name'"
            assert "profile_type"  in sec, f"sections[{i}] missing 'profile_type'"
            assert "length"        in sec, f"sections[{i}] missing 'length'"
            assert "start_area"   in sec, f"sections[{i}] missing 'start_area'"
            assert "end_area"     in sec, f"sections[{i}] missing 'end_area'"

    @pytest.mark.parametrize("f3_hz", [50.0, 80.0, 100.0])
    def test_yaml_contains_rear_chamber_fields(self, f3_hz: float):
        """Raw YAML must contain flat ``vrc`` and ``lrc`` fields (rear chamber)."""
        output = _run_synthesis_wizard(f3_hz)
        yaml_text = _extract_geometry_yaml(output)
        data = yaml.safe_load(yaml_text)
        assert "vrc" in data, "YAML must contain 'vrc' (rear chamber volume)"
        assert "lrc" in data, "YAML must contain 'lrc' (rear chamber length)"

    @pytest.mark.parametrize("f3_hz", [50.0, 80.0, 100.0])
    def test_yaml_contains_throat_chamber_fields(self, f3_hz: float):
        """Raw YAML must contain flat ``vtc`` and ``atc`` fields (throat chamber)."""
        output = _run_synthesis_wizard(f3_hz)
        yaml_text = _extract_geometry_yaml(output)
        data = yaml.safe_load(yaml_text)
        assert "vtc" in data, "YAML must contain 'vtc' (throat chamber volume)"
        assert "atc" in data, "YAML must contain 'atc' (throat chamber area)"

    @pytest.mark.parametrize("f3_hz", [50.0, 80.0, 100.0])
    def test_yaml_contains_throat_adapter_fields(self, f3_hz: float):
        """Raw YAML must contain flat ``ap1`` and ``lpt`` fields (throat adapter)."""
        output = _run_synthesis_wizard(f3_hz)
        yaml_text = _extract_geometry_yaml(output)
        data = yaml.safe_load(yaml_text)
        assert "ap1" in data, "YAML must contain 'ap1' (throat adapter area)"
        assert "lpt" in data, "YAML must contain 'lpt' (throat adapter length)"
