"""Tests for pyhorn_core/solver/hornresp_parser.py — Hornresp file format parser.

Covers: parse_key_value_line, parseHornrespProject, parseHornrespDriver,
hornresp_driver_to_specs, hornresp_project_to_driver_specs,
hornresp_project_to_geometry, load_hornresp_project.
"""

import math
import tempfile
from pathlib import Path

import pytest

from pyhorn_core.solver.hornresp_parser import (
    parse_key_value_line,
    parseHornrespProject,
    parseHornrespDriver,
    hornresp_driver_to_specs,
    hornresp_project_to_driver_specs,
    hornresp_project_to_geometry,
    load_hornresp_project,
)


# ─── parse_key_value_line ─────────────────────────────────────────────────────

class TestParseKeyValueLine:
    def test_standard_key_value(self):
        key, val = parse_key_value_line("S1 = 40 cm^2")
        assert key == "S1"
        assert val == "40 cm^2"

    def test_key_value_no_unit(self):
        key, val = parse_key_value_line("Fs = 49.6")
        assert key == "Fs"
        assert val == "49.6"

    def test_strips_whitespace(self):
        key, val = parse_key_value_line("  Re  =   7.80  ")
        assert key == "Re"
        assert val == "7.80"

    def test_returns_none_for_empty_line(self):
        assert parse_key_value_line("") is None
        assert parse_key_value_line("   ") is None

    def test_returns_none_for_no_equals(self):
        assert parse_key_value_line("S1 40") is None
        assert parse_key_value_line("key:value") is None


# ─── parseHornrespProject ─────────────────────────────────────────────────────

class TestParseHornrespProject:
    def test_parses_key_value_lines(self, tmp_path):
        txt = tmp_path / "project.txt"
        txt.write_text(
            "S1 = 40 cm^2\n"
            "S2 = 300 cm^2\n"
            "L12 = 150 cm\n"
        )
        params = parseHornrespProject(txt)
        assert params["S1"] == "40 cm^2"
        assert params["S2"] == "300 cm^2"
        assert params["L12"] == "150 cm"

    def test_ignores_pipe_headers(self, tmp_path):
        txt = tmp_path / "project.txt"
        txt.write_text(
            "|DRIVER|\n"
            "S1 = 40 cm^2\n"
            "|GEOMETRY|\n"
            "S2 = 300 cm^2\n"
        )
        params = parseHornrespProject(txt)
        assert "S1" in params
        assert "S2" in params

    def test_ignores_blank_lines(self, tmp_path):
        txt = tmp_path / "project.txt"
        txt.write_text("\n\n\nS1 = 40\n\n\n")
        params = parseHornrespProject(txt)
        assert params["S1"] == "40"

    def test_duplicate_keys_kept_first(self, tmp_path):
        txt = tmp_path / "project.txt"
        txt.write_text("S2 = 100 cm^2\nS2 = 200 cm^2\nS2 = 300 cm^2\n")
        params = parseHornrespProject(txt)
        assert params["S2"] == "100 cm^2"
        assert params["S2__all"] == ["100 cm^2", "200 cm^2", "300 cm^2"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parseHornrespProject("/nonexistent/path/project.txt")


# ─── parseHornrespDriver ──────────────────────────────────────────────────────

class TestParseHornrespDriver:
    def test_parses_key_value_lines(self, tmp_path):
        txt = tmp_path / "driver.txt"
        txt.write_text(
            "Re = 7.80 ohms\n"
            "Fs = 49.6 Hz\n"
            "Sd = 132.70 cm^2\n"
        )
        params = parseHornrespDriver(txt)
        assert params["Re"] == "7.80 ohms"
        assert params["Fs"] == "49.6 Hz"
        assert params["Sd"] == "132.70 cm^2"

    def test_ignores_blank_lines(self, tmp_path):
        txt = tmp_path / "driver.txt"
        txt.write_text("\n\nRe = 7.80\n\n\n")
        params = parseHornrespDriver(txt)
        assert params["Re"] == "7.80"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parseHornrespDriver("/nonexistent/path/driver.txt")


# ─── hornresp_driver_to_specs ──────────────────────────────────────────────────

class TestHornrespDriverToSpecs:
    def test_minimal_valid_driver(self):
        # Hornresp stores Cms as "1.49E-03" (N/m in scientific notation, no unit suffix)
        # Mmd is in grams. Sd is in cm^2.
        params = {
            "Re": "7.80 ohms",
            "Fs": "49.6 Hz",
            "Qes": "0.28",
            "Qms": "7.88",
            "Vas": "36.9 l",
            "Sd": "132.70 cm^2",
            "Le": "0.80 mH",
            "Bl": "7.79 N/A",
            "Rms": "0.277 kg/s",
            "Cms": "1.49E-03",
            "Mmd": "6.99 g",
        }
        specs = hornresp_driver_to_specs(params)
        assert specs.re == pytest.approx(7.80, rel=1e-3)
        # sd (cm² → m²), le (mH → H), mms (g → kg) conversions are applied by the parser
        assert specs.sd == pytest.approx(0.01327, rel=1e-3)  # cm² → m²
        assert specs.le == pytest.approx(0.0008, rel=1e-3)   # mH → H
        assert specs.mms > 0  # mass should be positive

    def test_missing_required_field_raises(self):
        params = {"Re": "7.80"}  # missing everything else
        with pytest.raises(KeyError):
            hornresp_driver_to_specs(params)


# ─── hornresp_project_to_driver_specs ─────────────────────────────────────────

class TestHornrespProjectToDriverSpecs:
    def test_extracts_driver_fields(self):
        # Hornresp project files store driver params with units; these are
        # read by hornresp_project_to_driver_specs which filters to driver keys
        # and passes to hornresp_driver_to_specs.
        params = {
            "Re": "7.80 ohms",
            "Sd": "132.70 cm^2",
            "Bl": "7.79 N/A",
            "Le": "0.80 mH",
            "Rms": "0.277 kg/s",
            "Cms": "1.472 mm/N",
            "Mmd": "6.99 g",
            "Nd": "1",  # number of drivers
            "Xmax": "1.5 mm",
        }
        specs = hornresp_project_to_driver_specs(params)
        assert specs.re == pytest.approx(7.80, rel=1e-3)
        assert specs.sd == pytest.approx(0.01327, rel=1e-3)  # cm² → m²


# ─── hornresp_project_to_geometry ─────────────────────────────────────────────

class TestHornrespProjectToGeometry:
    def test_basic_horn_geometry(self):
        # Hornresp uses "Hyp" for path length (hyperbolic path, cm)
        params = {
            "S1": "40 cm^2",
            "S2": "300 cm^2",
            "Hyp": "150 cm",  # path length in cm
            "扁率": "0",  # unused
        }
        geo = hornresp_project_to_geometry(params)
        assert geo.throat_area == pytest.approx(0.0040, rel=1e-3)  # m^2
        assert geo.mouth_area == pytest.approx(0.0300, rel=1e-3)  # m^2
        assert geo.path_length == pytest.approx(1.50, rel=1e-3)  # m

    def test_missing_s1_returns_zero_throat(self):
        # Missing S1 defaults to 0 via _f() on "0", no KeyError raised
        params = {"S2": "300 cm^2", "Hyp": "150 cm"}
        geo = hornresp_project_to_geometry(params)
        assert geo.throat_area == 0.0


# ─── load_hornresp_project ─────────────────────────────────────────────────────

class TestLoadHornrespProject:
    def test_loads_driver_and_geometry(self, tmp_path):
        txt = tmp_path / "project.txt"
        txt.write_text(
            "|DRIVER|\n"
            "Re = 7.80 ohms\n"
            "Fs = 49.6 Hz\n"
            "Qes = 0.28\n"
            "Qms = 7.88\n"
            "Vas = 36.9 l\n"
            "Sd = 132.70 cm^2\n"
            "Le = 0.80 mH\n"
            "Bl = 7.79 N/A\n"
            "Rms = 0.277 kg/s\n"
            "Cms = 1.49E-03\n"
            "Mmd = 6.99\n"
            "Nd = 1\n"
            "Xmax = 1.5 mm\n"
            "|GEOMETRY|\n"
            "S1 = 40 cm^2\n"
            "S2 = 300 cm^2\n"
            "Hyp = 150 cm\n"
        )
        geo, driver = load_hornresp_project(txt)
        assert driver.re == pytest.approx(7.80, rel=1e-3)
        assert geo.throat_area == pytest.approx(0.0040, rel=1e-3)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_hornresp_project("/nonexistent/project.txt")
