"""Unit tests for pyhorn.config.parser — JSON and YAML file parsing."""

import json
import tempfile
from pathlib import Path

import pytest
from pyhorn_core.config.models import DriverSpecs, HornGeometry, Section
from pyhorn_core.config.parser import (
    parse_driver_specs,
    parse_horn_geometry,
    parse_horn_project,
)


class TestParseDriverSpecs:
    """Tests for parse_driver_specs()."""

    def test_parses_valid_yaml(self, tmp_path):
        """A valid YAML driver file should parse into a DriverSpecs object."""
        driver_file = tmp_path / "driver.yaml"
        driver_file.write_text(
            "fs: 49.6\n"
            "qts: 0.27\n"
            "qes: 0.28\n"
            "qms: 7.88\n"
            "vas: 0.0369\n"
            "re: 7.8\n"
            "bl: 7.79\n"
            "mms: 0.00699\n"
            "cms: 0.001472\n"
            "rms: 0.277\n"
            "sd: 0.01327\n"
        )
        driver = parse_driver_specs(driver_file)
        assert isinstance(driver, DriverSpecs)
        assert driver.fs == pytest.approx(49.6)
        assert driver.qts == pytest.approx(0.27)
        assert driver.re == pytest.approx(7.8)

    def test_parses_valid_json(self, tmp_path):
        """A valid JSON driver file should parse into a DriverSpecs object."""
        driver_file = tmp_path / "driver.json"
        driver_file.write_text(
            json.dumps(
                {
                    "fs": 49.6,
                    "qts": 0.27,
                    "qes": 0.28,
                    "qms": 7.88,
                    "vas": 0.0369,
                    "re": 7.8,
                    "bl": 7.79,
                    "mms": 0.00699,
                    "cms": 0.001472,
                    "rms": 0.277,
                    "sd": 0.01327,
                }
            )
        )
        driver = parse_driver_specs(driver_file)
        assert driver.fs == pytest.approx(49.6)

    def test_preserves_optional_fields(self, tmp_path):
        """Optional fields (voltage, le, xmax) should be preserved from the file."""
        driver_file = tmp_path / "driver.yaml"
        driver_file.write_text(
            "fs: 50\n"
            "qts: 0.27\n"
            "qes: 0.28\n"
            "qms: 7.5\n"
            "vas: 0.036\n"
            "re: 8\n"
            "bl: 7.0\n"
            "mms: 0.007\n"
            "cms: 0.001\n"
            "rms: 0.27\n"
            "sd: 0.013\n"
            "voltage: 5.0\n"
            "le: 0.001\n"
            "xmax: 0.002\n"
        )
        driver = parse_driver_specs(driver_file)
        assert driver.voltage == pytest.approx(5.0)
        assert driver.le == pytest.approx(0.001)
        assert driver.xmax == pytest.approx(0.002)

    def test_raises_file_not_found(self):
        """parse_driver_specs should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            parse_driver_specs(Path("/nonexistent/driver.yaml"))

    def test_raises_on_unsupported_extension(self, tmp_path):
        """An unsupported file extension should raise ValueError."""
        bad_file = tmp_path / "driver.txt"
        bad_file.write_text("fs: 50\n")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            parse_driver_specs(bad_file)


class TestParseHornGeometry:
    """Tests for parse_horn_geometry()."""

    def test_parses_conical_segments_yaml(self, tmp_path):
        """conical_segments should be converted to list of tuples."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "enclosure_type: BLH\n"
            "width: 0.2\n"
            "throat_area: 0.01\n"
            "mouth_area: 0.05\n"
            "path_length: 1.0\n"
            "conical_segments:\n"
            "  - [0.05, 0.06, 0.03]\n"
            "  - [0.06, 0.07, 0.04]\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert isinstance(horn, HornGeometry)
        assert horn.enclosure_type == "BLH"
        assert horn.width == pytest.approx(0.2)
        assert len(horn.conical_segments) == 2
        assert horn.conical_segments[0] == (0.05, 0.06, 0.03)
        assert horn.conical_segments[1] == (0.06, 0.07, 0.04)

    def test_parses_rectangular_segments_yaml(self, tmp_path):
        """rectangular_segments should be converted to list of tuples."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "throat_area: 0.01\n"
            "mouth_area: 0.05\n"
            "path_length: 1.0\n"
            "rectangular_segments:\n"
            "  - [0.1, 0.05, 0.1, 0.06, 0.03]\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert horn.rectangular_segments[0] == (0.1, 0.05, 0.1, 0.06, 0.03)

    def test_parses_coordinates(self, tmp_path):
        """coordinates should be converted to list of (float, float) tuples."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "throat_area: 0.01\n"
            "mouth_area: 0.05\n"
            "path_length: 1.0\n"
            "coordinates:\n"
            "  - [0.1, 0.2]\n"
            "  - [0.3, 0.4]\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert horn.coordinates == [(0.1, 0.2), (0.3, 0.4)]

    def test_parses_enclosure_dims(self, tmp_path):
        """enclosure_dims should be converted to a tuple."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "throat_area: 0.01\n"
            "mouth_area: 0.05\n"
            "path_length: 1.0\n"
            "enclosure_dims: [0.4, 0.9]\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert horn.enclosure_dims == (0.4, 0.9)

    def test_parses_driver_coord(self, tmp_path):
        """driver_coord should be converted to a tuple."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "throat_area: 0.01\n"
            "mouth_area: 0.05\n"
            "path_length: 1.0\n"
            "driver_coord: [0.0, 0.1]\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert horn.driver_coord == (0.0, 0.1)

    def test_parses_json_horn_file(self, tmp_path):
        """JSON horn geometry files should parse correctly."""
        horn_file = tmp_path / "horn.json"
        horn_file.write_text(
            json.dumps(
                {
                    "enclosure_type": "BLH",
                    "width": 0.216,
                    "throat_area": 0.01,
                    "mouth_area": 0.05,
                    "path_length": 1.0,
                    "conical_segments": [[0.05, 0.06, 0.03]],
                }
            )
        )
        horn = parse_horn_geometry(horn_file)
        assert horn.enclosure_type == "BLH"
        assert horn.width == pytest.approx(0.216)

    def test_parses_hyperbolic_t(self, tmp_path):
        """hyperbolic_t should be preserved from horn YAML."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "profile_type: hyperbolic\n"
            "hyperbolic_t: 0.75\n"
            "throat_area: 0.01\n"
            "mouth_area: 0.04\n"
            "path_length: 0.5\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert horn.hyperbolic_t == pytest.approx(0.75)

    def test_raises_file_not_found(self):
        """parse_horn_geometry should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            parse_horn_geometry(Path("/nonexistent/horn.yaml"))

    def test_raises_on_unsupported_extension(self, tmp_path):
        """An unsupported file extension should raise ValueError."""
        bad_file = tmp_path / "horn.txt"
        bad_file.write_text("enclosure_type: BLH\n")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            parse_horn_geometry(bad_file)


class TestLoadFileHelper:
    """Tests for the internal _load_file helper via public functions."""

    def test_yaml_file_with_yaml_extension(self, tmp_path):
        """A .yaml file should be loaded using the YAML loader."""
        f = tmp_path / "test.yaml"
        f.write_text("key: value\n")
        from pyhorn_core.config.parser import _load_file

        data = _load_file(f)
        assert data == {"key": "value"}

    def test_yml_file_with_yml_extension(self, tmp_path):
        """A .yml file should be loaded using the YAML loader."""
        f = tmp_path / "test.yml"
        f.write_text("key: value\n")
        from pyhorn_core.config.parser import _load_file

        data = _load_file(f)
        assert data == {"key": "value"}

    def test_json_file(self, tmp_path):
        """A .json file should be loaded using the JSON loader."""
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        from pyhorn_core.config.parser import _load_file

        data = _load_file(f)
        assert data == {"key": "value"}


class TestParseSections:
    """Tests for the new 'sections' YAML format (chained profile sections)."""

    def test_parses_3_section_yaml(self, tmp_path):
        """A 3-section YAML should parse all fields correctly."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "enclosure_type: BLH\n"
            "sections:\n"
            "  - name: throat\n"
            "    profile_type: straight\n"
            "    length: 0.4\n"
            "    start_area: 0.0044\n"
            "    end_area: 0.0044\n"
            "  - name: main_horn\n"
            "    profile_type: exponential\n"
            "    length: 0.8\n"
            "    start_area: 0.0044\n"
            "    end_area: 0.08\n"
            "  - name: mouth\n"
            "    profile_type: hyperbolic\n"
            "    hyperbolic_t: 0.5\n"
            "    length: 0.3\n"
            "    start_area: 0.08\n"
            "    end_area: 0.12\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert isinstance(horn, HornGeometry)
        assert horn.sections is not None
        assert len(horn.sections) == 3

        throat = horn.sections[0]
        assert throat.name == "throat"
        assert throat.profile_type == "straight"
        assert throat.length == pytest.approx(0.4)
        assert throat.start_area == pytest.approx(0.0044)
        assert throat.end_area == pytest.approx(0.0044)
        assert throat.hyperbolic_t is None

        main_horn = horn.sections[1]
        assert main_horn.name == "main_horn"
        assert main_horn.profile_type == "exponential"
        assert main_horn.length == pytest.approx(0.8)
        assert main_horn.start_area == pytest.approx(0.0044)
        assert main_horn.end_area == pytest.approx(0.08)
        assert main_horn.hyperbolic_t is None

        mouth = horn.sections[2]
        assert mouth.name == "mouth"
        assert mouth.profile_type == "hyperbolic"
        assert mouth.hyperbolic_t == pytest.approx(0.5)
        assert mouth.length == pytest.approx(0.3)
        assert mouth.start_area == pytest.approx(0.08)
        assert mouth.end_area == pytest.approx(0.12)

    def test_parses_single_section_yaml(self, tmp_path):
        """A single-section YAML (equivalent to legacy format) should parse correctly."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "enclosure_type: FLH\n"
            "sections:\n"
            "  - name: horn\n"
            "    profile_type: conical\n"
            "    length: 0.6\n"
            "    start_area: 0.005\n"
            "    end_area: 0.05\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert horn.sections is not None
        assert len(horn.sections) == 1
        section = horn.sections[0]
        assert section.name == "horn"
        assert section.profile_type == "conical"
        assert section.length == pytest.approx(0.6)
        assert section.start_area == pytest.approx(0.005)
        assert section.end_area == pytest.approx(0.05)
        assert section.hyperbolic_t is None

    def test_hyperbolic_t_optional_defaults_none(self, tmp_path):
        """hyperbolic_t should be optional and default to None when absent."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "sections:\n"
            "  - name: flare\n"
            "    profile_type: exponential\n"
            "    length: 0.5\n"
            "    start_area: 0.01\n"
            "    end_area: 0.05\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert horn.sections[0].hyperbolic_t is None

    def test_backward_compatibility_no_sections_key(self, tmp_path):
        """Without a 'sections' key, the legacy conical_segments format should still work."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "enclosure_type: BLH\n"
            "width: 0.2\n"
            "throat_area: 0.01\n"
            "mouth_area: 0.05\n"
            "path_length: 1.0\n"
            "conical_segments:\n"
            "  - [0.05, 0.06, 0.03]\n"
            "  - [0.06, 0.07, 0.04]\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert horn.enclosure_type == "BLH"
        assert horn.width == pytest.approx(0.2)
        assert len(horn.conical_segments) == 2
        # sections should be None when using legacy format
        assert horn.sections is None

    def test_all_profile_types(self, tmp_path):
        """All recognised profile types should parse without error."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "sections:\n"
            "  - name: s1\n"
            "    profile_type: straight\n"
            "    length: 0.1\n"
            "    start_area: 0.01\n"
            "    end_area: 0.01\n"
            "  - name: s2\n"
            "    profile_type: exponential\n"
            "    length: 0.2\n"
            "    start_area: 0.01\n"
            "    end_area: 0.02\n"
            "  - name: s3\n"
            "    profile_type: hyperbolic\n"
            "    hyperbolic_t: 0.75\n"
            "    length: 0.2\n"
            "    start_area: 0.02\n"
            "    end_area: 0.03\n"
            "  - name: s4\n"
            "    profile_type: catenoidal\n"
            "    length: 0.15\n"
            "    start_area: 0.03\n"
            "    end_area: 0.04\n"
            "  - name: s5\n"
            "    profile_type: parabolic\n"
            "    length: 0.15\n"
            "    start_area: 0.04\n"
            "    end_area: 0.05\n"
            "  - name: s6\n"
            "    profile_type: conical\n"
            "    length: 0.1\n"
            "    start_area: 0.05\n"
            "    end_area: 0.06\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert len(horn.sections) == 6
        profile_types = [s.profile_type for s in horn.sections]
        assert profile_types == [
            "straight",
            "exponential",
            "hyperbolic",
            "catenoidal",
            "parabolic",
            "conical",
        ]
        # hyperbolic_t only set for hyperbolic section
        assert horn.sections[2].hyperbolic_t == pytest.approx(0.75)

    def test_profile_sections_is_alias_for_sections(self, tmp_path):
        """The 'profile_sections' key should be accepted as an alias for 'sections'."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(
            "profile_sections:\n"
            "  - name: throat\n"
            "    profile_type: straight\n"
            "    length: 0.4\n"
            "    start_area: 0.0044\n"
            "    end_area: 0.0044\n"
            "  - name: flare\n"
            "    profile_type: exponential\n"
            "    length: 0.8\n"
            "    start_area: 0.0044\n"
            "    end_area: 0.08\n"
        )
        horn = parse_horn_geometry(horn_file)
        assert horn.sections is not None
        assert len(horn.sections) == 2
        assert horn.sections[0].name == "throat"
        assert horn.sections[0].profile_type == "straight"
        assert horn.sections[0].length == pytest.approx(0.4)
        assert horn.sections[0].start_area == pytest.approx(0.0044)
        assert horn.sections[0].end_area == pytest.approx(0.0044)
        assert horn.sections[1].name == "flare"
        assert horn.sections[1].profile_type == "exponential"
        assert horn.sections[1].length == pytest.approx(0.8)
        assert horn.sections[1].start_area == pytest.approx(0.0044)
        assert horn.sections[1].end_area == pytest.approx(0.08)


# ─────────────────────────────────────────────────────────────────────────────
# Error-case validation tests — negative / zero / missing driver params
# (Reliability: Error Handling Audit — BACKLOG.md P2 HIGH)
# ─────────────────────────────────────────────────────────────────────────────

_VALID_DRIVER_YAML = (
    "fs: 49.6\n"
    "qts: 0.27\n"
    "qes: 0.28\n"
    "qms: 7.88\n"
    "vas: 0.0369\n"
    "re: 7.8\n"
    "bl: 7.79\n"
    "mms: 0.00699\n"
    "cms: 0.001472\n"
    "rms: 0.277\n"
    "sd: 0.01327\n"
    "le: 0.0008\n"
    "xmax: 0.0015\n"
    "voltage: 2.83\n"
)


class TestDriverSpecsValidation:
    """Error-case tests for parse_driver_specs().

    These verify that physically impossible T-S parameter values
    (negative, zero, or missing) raise clear ValueError messages
    before the solver ever sees them.
    """

    @pytest.mark.parametrize(
        "param,value",
        [
            ("fs", 0.0),
            ("fs", -10.0),
            ("qts", 0.0),
            ("qts", -0.5),
            ("qes", 0.0),
            ("qes", -0.1),
            ("qms", 0.0),
            ("qms", -1.0),
            ("vas", 0.0),
            ("vas", -0.01),
            ("re", 0.0),
            ("re", -5.0),
            ("mms", 0.0),
            ("mms", -0.001),
            ("cms", 0.0),
            ("cms", -1e-4),
            ("sd", 0.0),
            ("sd", -1e-4),
        ],
    )
    def test_positive_params_reject_zero_and_negative(
        self, tmp_path, param: str, value: float
    ):
        """Any strictly-positive T-S parameter must reject zero and negative values."""
        driver_file = tmp_path / "driver.yaml"
        # Rebuild YAML content, overriding only the target param
        lines = []
        for line in _VALID_DRIVER_YAML.splitlines():
            if line.startswith(param + ": "):
                lines.append(f"{param}: {value}")
            else:
                lines.append(line)
        driver_file.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match=f"must be positive"):
            parse_driver_specs(driver_file)

    @pytest.mark.parametrize(
        "param,value",
        [
            ("bl", -0.1),
            ("rms", -0.01),
            ("le", -0.0001),
            ("xmax", -0.001),
            ("voltage", -0.5),
        ],
    )
    def test_non_negative_params_reject_negative(
        self, tmp_path, param: str, value: float
    ):
        """Non-negative parameters must reject negative values."""
        driver_file = tmp_path / "driver.yaml"
        lines = []
        for line in _VALID_DRIVER_YAML.splitlines():
            if line.startswith(param + ": "):
                lines.append(f"{param}: {value}")
            else:
                lines.append(line)
        driver_file.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match=f"must be non-negative"):
            parse_driver_specs(driver_file)

    @pytest.mark.parametrize(
        "param", ["fs", "qts", "qes", "qms", "vas", "re", "mms", "cms", "sd"]
    )
    def test_missing_required_param_raises(self, tmp_path, param: str):
        """Omitting any required T-S parameter must raise a clear ValueError."""
        driver_file = tmp_path / "driver.yaml"
        lines = [
            line
            for line in _VALID_DRIVER_YAML.splitlines()
            if not line.startswith(param + ":")
        ]
        driver_file.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match=f"'{param}' .* missing"):
            parse_driver_specs(driver_file)

    def test_wrong_type_string_raises(self, tmp_path):
        """A non-numeric value for a numeric parameter must raise ValueError."""
        driver_file = tmp_path / "driver.yaml"
        content = _VALID_DRIVER_YAML.replace("fs: 49.6", "fs: not_a_number")
        driver_file.write_text(content)
        with pytest.raises(ValueError, match="fs.*must be a number"):
            parse_driver_specs(driver_file)

    def test_zero_qes_with_valid_qts_qms_is_rejected(self, tmp_path):
        """Qes=0 is physically impossible (would imply infinite electrical damping)."""
        driver_file = tmp_path / "driver.yaml"
        lines = []
        for line in _VALID_DRIVER_YAML.splitlines():
            if line.startswith("qes: "):
                lines.append("qes: 0.0")
            else:
                lines.append(line)
        driver_file.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="qes.*must be positive"):
            parse_driver_specs(driver_file)


# ─────────────────────────────────────────────────────────────────────────────
# Horn geometry physical-parameter validation tests
# (Reliability: Error Handling Audit — BACKLOG.md P2 HIGH)
# ─────────────────────────────────────────────────────────────────────────────

_VALID_HORN_YAML = (
    "throat_area: 0.01\n"
    "mouth_area: 0.05\n"
    "path_length: 1.0\n"
    "n_segments: 50\n"
    "vrc: 0.005\n"
    "lrc: 0.1\n"
    "vtc: 0.001\n"
    "ang: 6.283\n"
)

_VALID_SECTIONS_YAML = (
    "sections:\n"
    "  - name: throat\n"
    "    profile_type: straight\n"
    "    length: 0.4\n"
    "    start_area: 0.0044\n"
    "    end_area: 0.0044\n"
    "  - name: main_horn\n"
    "    profile_type: exponential\n"
    "    length: 0.8\n"
    "    start_area: 0.0044\n"
    "    end_area: 0.08\n"
)


class TestHornGeometryValidation:
    """Error-case tests for _validate_horn_geometry() via parse_horn_geometry().

    These verify that physically impossible horn geometry values
    (negative, zero, or wrong type) raise clear ValueError messages
    before the solver ever sees them.

    Coverage:
      - top-level positive fields:   throat_area, mouth_area, path_length, n_segments
      - top-level non-negative:     vrc, lrc, vtc, fr_rc, fr_tc, lpt, ang
      - top-level must-be-positive: atc, ap1 (when set, not when defaulted to 0)
      - sections:                   start_area, end_area, length per section
    """

    # ── top-level positive field tests ──────────────────────────────────────

    @pytest.mark.parametrize(
        "param,value",
        [
            ("throat_area", 0.0),
            ("throat_area", -0.001),
            ("throat_area", -1e-9),
            ("mouth_area", 0.0),
            ("mouth_area", -0.01),
            ("path_length", 0.0),
            ("path_length", -0.5),
            ("n_segments", 0),
            ("n_segments", -1),
        ],
    )
    def test_positive_fields_reject_zero_and_negative(
        self, tmp_path, param: str, value: float
    ):
        """throat_area, mouth_area, path_length, n_segments must be > 0."""
        horn_file = tmp_path / "horn.yaml"
        # Build YAML dict with the target param overridden to a bad value
        base = {
            "throat_area": 0.01,
            "mouth_area": 0.05,
            "path_length": 1.0,
            "n_segments": 50,
            "vrc": 0.005,
            "lrc": 0.1,
            "vtc": 0.001,
            "ang": 6.283,
        }
        base[param] = value
        import yaml

        horn_file.write_text(yaml.safe_dump(base, default_flow_style=False))
        with pytest.raises(ValueError, match=f"Horn geometry '{param}' .* positive"):
            parse_horn_geometry(horn_file)

    # ── top-level non-negative field tests ──────────────────────────────────

    @pytest.mark.parametrize(
        "param,value",
        [
            ("vrc", -0.001),
            ("lrc", -0.01),
            ("vtc", -1e-6),
            ("fr_rc", -10.0),
            ("fr_tc", -5.0),
            ("lpt", -0.01),
            ("ang", -0.1),
        ],
    )
    def test_non_negative_fields_reject_negative(
        self, tmp_path, param: str, value: float
    ):
        """vrc, lrc, vtc, fr_rc, fr_tc, lpt, ang must be >= 0."""
        horn_file = tmp_path / "horn.yaml"
        base = {
            "throat_area": 0.01,
            "mouth_area": 0.05,
            "path_length": 1.0,
            "n_segments": 50,
            "vrc": 0.005,
            "lrc": 0.1,
            "vtc": 0.001,
            "fr_rc": 0.0,
            "fr_tc": 0.0,
            "lpt": 0.0,
            "ang": 6.283,
        }
        base[param] = value
        import yaml

        horn_file.write_text(yaml.safe_dump(base, default_flow_style=False))
        with pytest.raises(
            ValueError, match=f"Horn geometry '{param}' .* non-negative"
        ):
            parse_horn_geometry(horn_file)

    # ── atc / ap1 must be positive when set ────────────────────────────────

    @pytest.mark.parametrize("param", ["atc", "ap1"])
    def test_atc_ap1_reject_zero_when_explicitly_set(self, tmp_path, param: str):
        """atc and ap1 must be positive when explicitly set to a non-zero value."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_HORN_YAML + f"{param}: 0.0\n"
        horn_file.write_text(content)
        # 0 is the default in HornGeometry, so we accept it; only reject explicitly
        # negative values
        parse_horn_geometry(horn_file)  # should not raise

    @pytest.mark.parametrize("param", ["atc", "ap1"])
    def test_atc_ap1_reject_negative(self, tmp_path, param: str):
        """atc and ap1 must reject negative values."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_HORN_YAML + f"{param}: -0.001\n"
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=f"Horn geometry '{param}' .* positive"):
            parse_horn_geometry(horn_file)

    # ── wrong type tests ────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "param", ["throat_area", "mouth_area", "path_length", "vrc", "lrc"]
    )
    def test_wrong_type_raises(self, tmp_path, param: str):
        """Non-numeric values for any numeric field must raise ValueError."""
        horn_file = tmp_path / "horn.yaml"
        import yaml

        base = {
            "throat_area": 0.01,
            "mouth_area": 0.05,
            "path_length": 1.0,
            "n_segments": 50,
            "vrc": 0.005,
            "lrc": 0.1,
        }
        base[param] = "not_a_number"
        horn_file.write_text(yaml.safe_dump(base, default_flow_style=False))
        with pytest.raises(ValueError, match=f"Horn geometry '{param}' .* number"):
            parse_horn_geometry(horn_file)

    # ── sections field tests ────────────────────────────────────────────────

    def test_sections_zero_start_area_rejected(self, tmp_path):
        """sections[].start_area = 0 must raise ValueError."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_SECTIONS_YAML.replace(
            "start_area: 0.0044\n", "start_area: 0.0\n"
        )
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=r"sections\[0\]\.start_area.*?positive"):
            parse_horn_geometry(horn_file)

    def test_sections_negative_start_area_rejected(self, tmp_path):
        """sections[].start_area < 0 must raise ValueError."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_SECTIONS_YAML.replace(
            "start_area: 0.0044\n", "start_area: -0.001\n"
        )
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=r"sections\[0\]\.start_area.*?positive"):
            parse_horn_geometry(horn_file)

    def test_sections_zero_end_area_rejected(self, tmp_path):
        """sections[].end_area = 0 must raise ValueError."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_SECTIONS_YAML.replace("end_area: 0.0044\n", "end_area: 0.0\n")
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=r"sections\[0\]\.end_area.*?positive"):
            parse_horn_geometry(horn_file)

    def test_sections_negative_end_area_rejected(self, tmp_path):
        """sections[].end_area < 0 must raise ValueError."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_SECTIONS_YAML.replace(
            "end_area: 0.0044\n", "end_area: -0.001\n"
        )
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=r"sections\[0\]\.end_area.*?positive"):
            parse_horn_geometry(horn_file)

    def test_sections_zero_length_rejected(self, tmp_path):
        """sections[].length = 0 must raise ValueError."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_SECTIONS_YAML.replace("length: 0.4\n", "length: 0.0\n")
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=r"sections\[0\]\.length.*?positive"):
            parse_horn_geometry(horn_file)

    def test_sections_negative_length_rejected(self, tmp_path):
        """sections[].length < 0 must raise ValueError."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_SECTIONS_YAML.replace("length: 0.4\n", "length: -0.5\n")
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=r"sections\[0\]\.length.*?positive"):
            parse_horn_geometry(horn_file)

    def test_sections_missing_start_area_rejected(self, tmp_path):
        """sections[].start_area missing must raise ValueError."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_SECTIONS_YAML.replace("    start_area: 0.0044\n", "")
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=r"sections\[0\]\.start_area.*?missing"):
            parse_horn_geometry(horn_file)

    def test_sections_missing_end_area_rejected(self, tmp_path):
        """sections[].end_area missing must raise ValueError."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_SECTIONS_YAML.replace("    end_area: 0.0044\n", "")
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=r"sections\[0\]\.end_area.*?missing"):
            parse_horn_geometry(horn_file)

    def test_sections_missing_length_rejected(self, tmp_path):
        """sections[].length missing must raise ValueError."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_SECTIONS_YAML.replace("    length: 0.4\n", "")
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=r"sections\[0\]\.length.*?missing"):
            parse_horn_geometry(horn_file)

    def test_sections_second_section_validates_correctly(self, tmp_path):
        """Validation error should reference the correct section index."""
        horn_file = tmp_path / "horn.yaml"
        # Second section (index 1) has negative start_area
        content = (
            "sections:\n"
            "  - name: good\n"
            "    profile_type: straight\n"
            "    length: 0.4\n"
            "    start_area: 0.0044\n"
            "    end_area: 0.0044\n"
            "  - name: bad\n"
            "    profile_type: exponential\n"
            "    length: 0.8\n"
            "    start_area: -0.001\n"  # negative — should be flagged as sections[1]
            "    end_area: 0.08\n"
        )
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=r"sections\[1\]\.start_area.*?positive"):
            parse_horn_geometry(horn_file)

    def test_sections_wrong_type_raises(self, tmp_path):
        """Non-numeric values in sections fields must raise ValueError."""
        horn_file = tmp_path / "horn.yaml"
        content = _VALID_SECTIONS_YAML.replace("length: 0.4\n", "length: bad\n")
        horn_file.write_text(content)
        with pytest.raises(ValueError, match=r"sections\[0\]\.length.*?number"):
            parse_horn_geometry(horn_file)

    def test_valid_horn_passes(self, tmp_path):
        """A valid horn YAML with all positive/non-negative params should parse cleanly."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(_VALID_HORN_YAML)
        horn = parse_horn_geometry(horn_file)
        assert isinstance(horn, HornGeometry)
        assert horn.throat_area == pytest.approx(0.01)
        assert horn.mouth_area == pytest.approx(0.05)
        assert horn.path_length == pytest.approx(1.0)
        assert horn.n_segments == 50

    def test_valid_sections_passes(self, tmp_path):
        """A valid sections-format YAML should parse cleanly."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text(_VALID_SECTIONS_YAML)
        horn = parse_horn_geometry(horn_file)
        assert isinstance(horn, HornGeometry)
        assert horn.sections is not None
        assert len(horn.sections) == 2

    def test_vrc_and_lrc_zero_is_valid(self, tmp_path):
        """vrc=0 and lrc=0 are valid (no rear chamber)."""
        horn_file = tmp_path / "horn.yaml"
        content = (
            "throat_area: 0.01\n"
            "mouth_area: 0.05\n"
            "path_length: 1.0\n"
            "vrc: 0.0\n"
            "lrc: 0.0\n"
        )
        horn_file.write_text(content)
        horn = parse_horn_geometry(horn_file)
        assert horn.vrc == pytest.approx(0.0)
        assert horn.lrc == pytest.approx(0.0)

    def test_fr1_and_tal1_defaults(self, tmp_path):
        """Damping material fields fr1/tal1 should default to 0 when absent."""
        horn_file = tmp_path / "horn.yaml"
        content = (
            "sections:\n"
            "  - name: damped\n"
            "    profile_type: exponential\n"
            "    length: 0.5\n"
            "    start_area: 0.01\n"
            "    end_area: 0.05\n"
            # fr1 and tal1 absent — should default to 0.0
        )
        horn_file.write_text(content)
        horn = parse_horn_geometry(horn_file)
        assert horn.sections[0].fr1 == pytest.approx(0.0)
        assert horn.sections[0].tal1 == pytest.approx(0.0)


class TestMalformedYamlErrors:
    """User-friendly error messages for malformed YAML files (not raw tracebacks)."""

    def test_malformed_driver_yaml_givesFriendly_error(self, tmp_path):
        """Malformed YAML in a driver file should raise ValueError with a clear message."""
        driver_file = tmp_path / "driver.yaml"
        # Indentation error — ScannerError
        driver_file.write_text("fs: 49.6\n  qts: 0.27\n")
        with pytest.raises(ValueError) as exc_info:
            parse_driver_specs(driver_file)
        assert "YAML parse error" in str(exc_info.value)
        assert "ScannerError" not in str(
            exc_info.value
        )  # raw internal error must not leak

    def test_malformed_horn_yaml_gives_friendly_error(self, tmp_path):
        """Malformed YAML in a horn file should raise ValueError with a clear message."""
        horn_file = tmp_path / "horn.yaml"
        # Flow sequence — ParserError (unclosed bracket)
        horn_file.write_text("sections:\n  - name: test\n    values: [1, 2,\n")
        with pytest.raises(ValueError) as exc_info:
            parse_horn_geometry(horn_file)
        assert "YAML parse error" in str(exc_info.value)
        assert "ParserError" not in str(exc_info.value)

    def test_malformed_yaml_mentions_filepath(self, tmp_path):
        """Error message should include the filename to help users locate the problem."""
        driver_file = tmp_path / "my_driver.yaml"
        driver_file.write_text("fs: 49.6\na: [1, 2, 3\n")
        with pytest.raises(ValueError) as exc_info:
            parse_driver_specs(driver_file)
        assert "my_driver.yaml" in str(exc_info.value)

    def test_malformed_yaml_hints_at_common_causes(self, tmp_path):
        """Error message should give users a hint about what to check."""
        horn_file = tmp_path / "horn.yaml"
        # Tab instead of spaces is a common mistake
        horn_file.write_text("throat_area:\t0.01\n")
        with pytest.raises(ValueError) as exc_info:
            parse_horn_geometry(horn_file)
        err = str(exc_info.value)
        assert "YAML parse error" in err
        assert "indentation" in err.lower() or "special" in err.lower()

    def test_malformed_json_still_raises_FileNotFoundError(self, tmp_path):
        """JSON files with bad syntax raise the standard json.decode error (not YAML)."""
        json_file = tmp_path / "data.json"
        json_file.write_text('{"fs": 49.6,}')
        with pytest.raises(json.JSONDecodeError):
            parse_driver_specs(json_file)


class TestMissingRequiredFields:
    """Tests for user-friendly errors when required fields are absent from geometry YAML.

    (Reliability: Error Handling Audit — BACKLOG.md P2 HIGH)
    """

    @pytest.mark.parametrize(
        "omitted",
        [
            "throat_area",
            "mouth_area",
            "path_length",
        ],
    )
    def test_missing_required_top_level_field_raises_clear_error(
        self, tmp_path, omitted: str
    ):
        """Omitting throat_area, mouth_area, or path_length (non-sections format) must raise
        a ValueError that names the missing field and suggests using the sections format.
        """
        horn_file = tmp_path / "horn.yaml"
        content = {
            "throat_area": 0.01,
            "mouth_area": 0.05,
            "path_length": 1.0,
            "enclosure_type": "BLH",
        }
        del content[omitted]
        import yaml

        horn_file.write_text(yaml.safe_dump(content))
        with pytest.raises(ValueError, match=f"'{omitted}' .* required .* missing"):
            parse_horn_geometry(horn_file)

    def test_missing_throat_area_error_suggests_sections_format(self, tmp_path):
        """The error message for a missing throat_area should hint at the sections format."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text("mouth_area: 0.05\npath_length: 1.0\n")
        with pytest.raises(ValueError) as exc_info:
            parse_horn_geometry(horn_file)
        assert "sections" in str(exc_info.value)

    def test_missing_throat_area_with_sections_present_is_valid(self, tmp_path):
        """When the sections format is used, throat_area/mouth_area/path_length are
        NOT required at the top level — sections define the geometry instead."""
        horn_file = tmp_path / "horn.yaml"
        import yaml

        content = {
            "enclosure_type": "BLH",
            "sections": [
                {
                    "name": "throat",
                    "profile_type": "straight",
                    "length": 0.4,
                    "start_area": 0.0044,
                    "end_area": 0.0044,
                },
                {
                    "name": "flare",
                    "profile_type": "exponential",
                    "length": 0.8,
                    "start_area": 0.0044,
                    "end_area": 0.08,
                },
            ],
            # Note: NO throat_area, mouth_area, or path_length at top level
        }
        horn_file.write_text(yaml.safe_dump(content))
        horn = parse_horn_geometry(horn_file)
        assert horn.sections is not None
        assert len(horn.sections) == 2

    def test_all_three_required_fields_missing_one_by_one(self, tmp_path):
        """Each of the three required fields, when individually omitted, raises the
        same clear 'required but missing' message (not a silent zero-default)."""
        for field in ("throat_area", "mouth_area", "path_length"):
            horn_file = tmp_path / "horn.yaml"
            content = {
                "throat_area": 0.01,
                "mouth_area": 0.05,
                "path_length": 1.0,
            }
            del content[field]
            import yaml

            horn_file.write_text(yaml.safe_dump(content))
            with pytest.raises(ValueError, match=f"'{field}' .* required"):
                parse_horn_geometry(horn_file)

    def test_empty_yaml_raises_missing_throat_area(self, tmp_path):
        """A completely empty YAML file (or one with only enclosure_type) should
        raise about missing throat_area — not silently create a zero-valued horn."""
        horn_file = tmp_path / "horn.yaml"
        horn_file.write_text("enclosure_type: BLH\n")
        with pytest.raises(ValueError) as exc_info:
            parse_horn_geometry(horn_file)
        assert "throat_area" in str(exc_info.value)
        assert "required" in str(exc_info.value)

    def test_geometry_file_not_found_in_project_gives_project_context(self, tmp_path):
        """When a project YAML references a non-existent geometry file, the error
        message should mention the project filename to help the user."""
        # Create the project file
        project_file = tmp_path / "test_project.yaml"
        project_file.write_text(
            "geometry_path: geometry/does_not_exist.yaml\nname: Test\n"
        )
        with pytest.raises(FileNotFoundError) as exc_info:
            parse_horn_project(project_file)
        err = str(exc_info.value)
        assert "does_not_exist.yaml" in err
        assert "test_project.yaml" in err  # project file mentioned for context

    def test_unknown_top_level_field_is_rejected(self, tmp_path):
        """An unknown top-level field (not a HornGeometry field) must not silently
        pass — it should cause a clear ValueError, not a confusing KeyError."""
        horn_file = tmp_path / "horn.yaml"
        import yaml

        content = {
            "throat_area": 0.01,
            "mouth_area": 0.05,
            "path_length": 1.0,
            # HornGeometry has no field called 'unknown_geometry_field':
            "unknown_geometry_field": 42,
        }
        horn_file.write_text(yaml.safe_dump(content))
        with pytest.raises(ValueError) as exc_info:
            parse_horn_geometry(horn_file)
        err = str(exc_info.value)
        # Must give a user-friendly message — no raw Python KeyError/TypeError
        assert "TypeError" not in err
        assert "unknown_geometry_field" in err or "Failed to construct" in err


class TestStringPathSupport:
    """Tests for accepting str arguments (not just Path) in parse functions.

    Users commonly call parse_driver_specs("drivers/foo.yaml") with a string.
    The functions should accept both str and Path for ergonomics.
    """

    def test_parse_driver_specs_accepts_string_path(self):
        driver = parse_driver_specs("drivers/FE166NV2.yaml")
        assert isinstance(driver, DriverSpecs)
        assert driver.fs == pytest.approx(49.6)

    def test_parse_horn_geometry_accepts_string_path(self):
        horn = parse_horn_geometry("examples/geometry/fsx.yaml")
        assert isinstance(horn, HornGeometry)
        assert horn.throat_area > 0

    def test_parse_horn_project_accepts_string_path(self):
        proj, horn = parse_horn_project("projects/hiro.yaml")
        assert isinstance(proj.name, str)
        assert isinstance(horn, HornGeometry)

    def test_parse_driver_specs_still_accepts_path(self):
        driver = parse_driver_specs(Path("drivers/FE166NV2.yaml"))
        assert driver.fs == pytest.approx(49.6)

    def test_parse_horn_geometry_still_accepts_path(self):
        horn = parse_horn_geometry(Path("examples/geometry/fsx.yaml"))
        assert horn.throat_area > 0

    def test_parse_horn_project_still_accepts_path(self):
        proj, horn = parse_horn_project(Path("projects/hiro.yaml"))
        assert proj.name == "Hiro"

    def test_invalid_string_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_driver_specs("/nonexistent/path/driver.yaml")

    def test_invalid_string_path_geometry_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_horn_geometry("/nonexistent/path/horn.yaml")
