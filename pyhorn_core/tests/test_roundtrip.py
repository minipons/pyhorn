"""Regression tests: Import/Export roundtrip validation.

These tests verify that pyhorn can round-trip data through its export formats
without losing or corrupting information. Covers:

1. Driver YAML roundtrip   — parse → serialize → re-parse → all fields preserved
2. Horn geometry roundtrip — parse → serialize → re-parse → key fields preserved
3. FRD export roundtrip   — simulate → export .frd → re-parse → freq points match
4. JSON export roundtrip   — simulate → export JSON → re-parse → arrays match

Run:
    pytest pyhorn_core/tests/test_roundtrip.py -v
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from pyhorn_core.config.models import DriverSpecs, HornGeometry
from pyhorn_core.config.parser import (
    parse_driver_specs,
    parse_horn_geometry,
    parse_horn_project,
)
from pyhorn_core.output.exporter import export_to_csv, export_to_frd, export_to_json
from pyhorn_core.solver.models import horn_response

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

TESTS_DIR = Path(__file__).parent
DRIVER_YAML = TESTS_DIR.parent.parent / "drivers" / "FE166NV2.yaml"
GEOM_HIROB = (
    TESTS_DIR.parent.parent
    / "tests"
    / "benchmarks"
    / "hornresp"
    / "hirob"
    / "fixture"
    / "horn.yaml"
)
PROJECT_HIROB = TESTS_DIR.parent.parent / "projects" / "hirob.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _frd_freqs_and_spl(frd_path: Path):
    """Parse a .frd file. Returns (freqs_ndarray, spl_ndarray, phase_ndarray)."""
    freqs, spls, phases = [], [], []
    with open(frd_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip comment/header lines (!FRD1.0, column headers, etc.)
            if line.startswith("!") or not line[0].isdigit():
                continue
            parts = line.split()
            freqs.append(float(parts[0]))
            spls.append(float(parts[1]))
            phases.append(float(parts[2]) if len(parts) > 2 else 0.0)
    return np.array(freqs), np.array(spls), np.array(phases)


def _asdict_filtered(obj) -> dict:
    """Like dataclasses.asdict but excludes fields with init=False (private/computed fields)."""

    def _serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return [_serializable(item) for item in value]
        if isinstance(value, list):
            return [_serializable(item) for item in value]
        if isinstance(value, dict):
            return {key: _serializable(item) for key, item in value.items()}
        return value

    result = {}
    for f in obj.__class__.__dataclass_fields__.values():
        if f.init:
            result[f.name] = _serializable(getattr(obj, f.name))
    return result


def _run_small_simulation():
    """Run a minimal simulation for exporter roundtrip tests. Returns SimulationResult."""
    driver = parse_driver_specs(DRIVER_YAML)
    horn = parse_horn_geometry(GEOM_HIROB)
    freqs = np.array([100.0, 500.0, 1000.0, 3000.0])
    return horn_response(
        freqs=freqs, driver=driver, horn=horn, compute_distortion=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Driver YAML roundtrip
# ─────────────────────────────────────────────────────────────────────────────


class TestDriverYamlRoundtrip:
    """Verify DriverSpecs survive a YAML parse → serialize → re-parse cycle."""

    @pytest.fixture
    def driver_specs(self):
        if not DRIVER_YAML.exists():
            pytest.skip(f"Driver YAML not found: {DRIVER_YAML}")
        return parse_driver_specs(DRIVER_YAML)

    def test_all_scalar_fields_preserved(self, driver_specs):
        """Every numeric / string field should round-trip unchanged."""
        d = _asdict_filtered(driver_specs)

        # Re-serialize via YAML
        yaml_str = yaml.safe_dump(d, sort_keys=False)
        d2 = yaml.safe_load(yaml_str)

        # Re-create the dataclass (DriverSpecs accepts plain dicts)
        rebuilt = DriverSpecs(**d2)

        for field in DriverSpecs.__dataclass_fields__:
            original = getattr(driver_specs, field)
            result = getattr(rebuilt, field)
            if isinstance(original, np.ndarray):
                np.testing.assert_allclose(
                    result, original, rtol=1e-12, err_msg=f"Field '{field}' mismatch"
                )
            else:
                assert (
                    result == original
                ), f"Field '{field}' mismatch: {result!r} != {original!r}"

    def test_no_extra_keys_introduced(self, driver_specs):
        """Round-tripped YAML should not contain fields not in DriverSpecs."""
        d = _asdict_filtered(driver_specs)
        yaml_str = yaml.safe_dump(d, sort_keys=False)
        d2 = yaml.safe_load(yaml_str)
        rebuilt = DriverSpecs(**d2)

        original_fields = set(DriverSpecs.__dataclass_fields__)
        rebuilt_fields = set(_asdict_filtered(rebuilt))
        assert rebuilt_fields == original_fields, (
            f"Field set changed after round-trip: "
            f"added={rebuilt_fields - original_fields}, "
            f"removed={original_fields - rebuilt_fields}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Horn geometry YAML roundtrip
# ─────────────────────────────────────────────────────────────────────────────


class TestHornGeometryRoundtrip:
    """Verify HornGeometry survives a YAML parse → serialize → re-parse cycle."""

    @pytest.fixture(params=["hirob_geometry", "hirob_project"])
    def geometry(self, request):
        name = request.param
        if name == "hirob_geometry":
            yaml_path = GEOM_HIROB
            if not yaml_path.exists():
                pytest.skip(f"Geometry YAML not found: {yaml_path}")
            horn = parse_horn_geometry(yaml_path)
        else:
            # hirob is a project file — use parse_horn_project to get HornGeometry
            project_path = PROJECT_HIROB
            if not project_path.exists():
                pytest.skip(f"Project YAML not found: {project_path}")
            _, horn = parse_horn_project(project_path)
        return name, horn

    def test_key_geometry_fields_preserved(self, geometry):
        """Throat area, mouth area, path length, and profile type should round-trip."""
        _name, horn = geometry
        h = _asdict_filtered(horn)

        yaml_str = yaml.safe_dump(h, sort_keys=False)
        h2 = yaml.safe_load(yaml_str)
        rebuilt = HornGeometry(**h2)

        assert rebuilt.throat_area == horn.throat_area
        assert rebuilt.mouth_area == horn.mouth_area
        assert rebuilt.path_length == horn.path_length
        assert rebuilt.profile_type == horn.profile_type

    def test_conical_segments_preserved(self, geometry):
        """Conical segment arrays should round-trip intact."""
        _name, horn = geometry
        if not horn.conical_segments:
            pytest.skip("No conical_segments in this geometry")

        h = _asdict_filtered(horn)
        yaml_str = yaml.safe_dump(h, sort_keys=False)
        h2 = yaml.safe_load(yaml_str)
        rebuilt = HornGeometry(**h2)

        assert rebuilt.conical_segments is not None
        assert len(rebuilt.conical_segments) == len(horn.conical_segments)
        for orig, new in zip(horn.conical_segments, rebuilt.conical_segments):
            np.testing.assert_allclose(new, orig, rtol=1e-12)

    def test_rectangular_segments_preserved(self, geometry):
        """Rectangular segments should round-trip intact."""
        _name, horn = geometry
        if not horn.rectangular_segments:
            pytest.skip("No rectangular_segments in this geometry")

        h = _asdict_filtered(horn)
        yaml_str = yaml.safe_dump(h, sort_keys=False)
        h2 = yaml.safe_load(yaml_str)
        rebuilt = HornGeometry(**h2)

        assert rebuilt.rectangular_segments is not None
        assert len(rebuilt.rectangular_segments) == len(horn.rectangular_segments)

    def test_sections_preserved(self, geometry):
        """Profile sections should round-trip intact."""
        _name, horn = geometry
        if not horn.sections:
            pytest.skip("No sections in this geometry")

        h = _asdict_filtered(horn)
        yaml_str = yaml.safe_dump(h, sort_keys=False)
        h2 = yaml.safe_load(yaml_str)
        rebuilt = HornGeometry(**h2)

        assert rebuilt.sections is not None
        assert len(rebuilt.sections) == len(horn.sections)
        for orig_sec, new_sec in zip(horn.sections, rebuilt.sections):
            assert new_sec.name == orig_sec.name
            assert new_sec.profile_type == orig_sec.profile_type
            assert new_sec.length == orig_sec.length
            np.testing.assert_allclose(
                new_sec.start_area, orig_sec.start_area, rtol=1e-12
            )
            np.testing.assert_allclose(new_sec.end_area, orig_sec.end_area, rtol=1e-12)

    def test_coordinates_preserved(self, geometry):
        """Coordinate list should round-trip intact."""
        _name, horn = geometry
        if not horn.coordinates:
            pytest.skip("No coordinates in this geometry")

        h = _asdict_filtered(horn)
        yaml_str = yaml.safe_dump(h, sort_keys=False)
        h2 = yaml.safe_load(yaml_str)
        rebuilt = HornGeometry(**h2)

        assert rebuilt.coordinates is not None
        assert len(rebuilt.coordinates) == len(horn.coordinates)
        for orig, new in zip(horn.coordinates, rebuilt.coordinates):
            np.testing.assert_allclose(new, orig, rtol=1e-12)

    def test_driver_coord_preserved(self, geometry):
        """Driver coordinate should round-trip intact."""
        _name, horn = geometry
        h = _asdict_filtered(horn)
        yaml_str = yaml.safe_dump(h, sort_keys=False)
        h2 = yaml.safe_load(yaml_str)
        rebuilt = HornGeometry(**h2)

        if horn.driver_coord is None:
            assert rebuilt.driver_coord is None
        else:
            np.testing.assert_allclose(
                rebuilt.driver_coord, horn.driver_coord, rtol=1e-12
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. FRD export roundtrip
# ─────────────────────────────────────────────────────────────────────────────


class TestFrdExportRoundtrip:
    """Verify .frd export → re-import preserves frequency points and SPL values."""

    def test_frd_frequencies_preserved(self, tmp_path):
        """Frequency array written to .frd should read back byte-for-byte."""
        result = _run_small_simulation()
        phase_deg = np.rad2deg(result.phase)

        frd_path = tmp_path / "response.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=phase_deg,
            output_path=frd_path,
        )

        reimport_freqs, reimport_spl, reimport_phase = _frd_freqs_and_spl(frd_path)

        np.testing.assert_allclose(reimport_freqs, result.freqs, rtol=1e-6)
        np.testing.assert_allclose(reimport_spl, result.spl, rtol=1e-6)
        # FRD format stores 4 decimal places — roundtrip introduces ~1e-4 error
        np.testing.assert_allclose(reimport_phase, phase_deg, atol=1e-3)

    def test_frd_spl_values_preserved(self, tmp_path):
        """SPL values written to .frd should read back within floating-point tolerance."""
        result = _run_small_simulation()
        phase_deg = np.rad2deg(result.phase)

        frd_path = tmp_path / "response.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=phase_deg,
            output_path=frd_path,
        )

        _reimport_freqs, reimport_spl, _ = _frd_freqs_and_spl(frd_path)

        np.testing.assert_allclose(reimport_spl, result.spl, atol=1e-4)

    def test_frd_phase_values_preserved(self, tmp_path):
        """Phase values written to .frd should read back within floating-point tolerance."""
        result = _run_small_simulation()
        phase_deg = np.rad2deg(result.phase)

        frd_path = tmp_path / "response.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=phase_deg,
            output_path=frd_path,
        )

        _reimport_freqs, _reimport_spl, reimport_phase = _frd_freqs_and_spl(frd_path)

        np.testing.assert_allclose(reimport_phase, phase_deg, atol=1e-3)

    def test_frd_format_header_present(self, tmp_path):
        """FRD file should start with the standard !FRD1.0 header."""
        result = _run_small_simulation()

        frd_path = tmp_path / "response.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=np.rad2deg(result.phase),
            output_path=frd_path,
        )

        with open(frd_path, encoding="utf-8") as f:
            content = f.read()

        assert content.startswith(
            "!FRD1.0"
        ), "FRD file must start with '!FRD1.0' header"

    def test_frd_point_count_matches(self, tmp_path):
        """FRD file should have exactly one data line per frequency point."""
        result = _run_small_simulation()

        frd_path = tmp_path / "response.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=np.rad2deg(result.phase),
            output_path=frd_path,
        )

        reimport_freqs, _, _ = _frd_freqs_and_spl(frd_path)

        assert len(reimport_freqs) == len(result.freqs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. JSON export roundtrip
# ─────────────────────────────────────────────────────────────────────────────


class TestJsonExportRoundtrip:
    """Verify JSON export → re-import preserves all arrays and metadata."""

    def test_frequencies_array_preserved(self, tmp_path):
        """Frequency array written to JSON should read back exactly."""
        result = _run_small_simulation()

        json_path = tmp_path / "response.json"
        export_to_json(
            freqs=result.freqs,
            spl_responses={"total": result.spl},
            output_path=json_path,
        )

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        reimport_freqs = np.array(data["frequencies"], dtype=float)
        np.testing.assert_array_equal(reimport_freqs, result.freqs)

    def test_spl_response_arrays_preserved(self, tmp_path):
        """SPL response arrays should read back exactly after JSON roundtrip."""
        result = _run_small_simulation()

        json_path = tmp_path / "response.json"
        export_to_json(
            freqs=result.freqs,
            spl_responses={"total": result.spl, "horn": result.horn_spl},
            output_path=json_path,
        )

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        for label in ("total", "horn"):
            reimport = np.array(data["responses"][label], dtype=float)
            original = result.spl if label == "total" else getattr(result, "horn_spl")
            np.testing.assert_array_equal(reimport, original)

    def test_cone_acceleration_preserved(self, tmp_path):
        """Cone acceleration array should survive JSON roundtrip."""
        result = _run_small_simulation()

        json_path = tmp_path / "response.json"
        export_to_json(
            freqs=result.freqs,
            spl_responses={"total": result.spl},
            cone_acceleration=result.cone_acceleration,
            output_path=json_path,
        )

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "cone_acceleration" in data
        reimport = np.array(data["cone_acceleration"], dtype=float)
        np.testing.assert_array_equal(reimport, result.cone_acceleration)

    def test_particle_velocity_arrays_preserved(self, tmp_path):
        """Particle velocity arrays (throat/mouth/port) should survive JSON roundtrip."""
        result = _run_small_simulation()

        json_path = tmp_path / "response.json"
        export_to_json(
            freqs=result.freqs,
            spl_responses={"total": result.spl},
            particle_velocity_throat=result.particle_velocity_throat,
            particle_velocity_mouth=result.particle_velocity_mouth,
            particle_velocity_port=result.particle_velocity_port,
            output_path=json_path,
        )

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        for key in (
            "particle_velocity_throat",
            "particle_velocity_mouth",
            "particle_velocity_port",
        ):
            assert key in data, f"{key} should be present in JSON output"
            reimport = np.array(data[key], dtype=float)
            original = getattr(result, key)
            np.testing.assert_array_equal(reimport, original)

    def test_metadata_preserved(self, tmp_path):
        """Custom metadata dict should survive JSON roundtrip."""
        result = _run_small_simulation()

        metadata = {
            "driver": "FE166NV2",
            "geometry": "hirob",
            "simulation_version": "1.0",
        }

        json_path = tmp_path / "response.json"
        export_to_json(
            freqs=result.freqs,
            spl_responses={"total": result.spl},
            metadata=metadata,
            output_path=json_path,
        )

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert data["metadata"]["driver"] == "FE166NV2"
        assert data["metadata"]["geometry"] == "hirob"

    def test_empty_responses_roundtrips(self, tmp_path):
        """Empty responses dict should produce valid JSON and roundtrip cleanly."""
        freqs = np.array([100.0, 200.0, 300.0])

        json_path = tmp_path / "response.json"
        export_to_json(
            freqs=freqs,
            spl_responses={},
            output_path=json_path,
        )

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["responses"] == {}
        reimport_freqs = np.array(data["frequencies"], dtype=float)
        np.testing.assert_array_equal(reimport_freqs, freqs)

    def test_futtrup_gdlimit_preserved(self, tmp_path):
        """futtrup_gdlimit_ms array should survive JSON roundtrip."""
        result = _run_small_simulation()

        json_path = tmp_path / "response.json"
        export_to_json(
            freqs=result.freqs,
            spl_responses={"total": result.spl},
            futtrup_gdlimit_ms=result.futtrup_gdlimit,
            output_path=json_path,
        )

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "futtrup_gdlimit_ms" in data
        reimport = np.array(data["futtrup_gdlimit_ms"], dtype=float)
        np.testing.assert_array_equal(reimport, result.futtrup_gdlimit)
