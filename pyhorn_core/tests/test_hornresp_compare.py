"""Unit tests for the Hornresp text-format parser (hornresp_parser.py)."""

import pytest

from pyhorn_core.solver.hornresp_parser import (
    parseHornrespProject,
    parseHornrespDriver,
    hornresp_project_to_geometry,
    hornresp_driver_to_specs,
    hornresp_project_to_driver_specs,
    load_hornresp_project,
)


# ─── parseHornrespProject ─────────────────────────────────────────────────────

class TestParseHornrespProject:
    def test_parses_basic_key_value(self, tmp_path):
        path = tmp_path / "p.txt"
        path.write_text("ID = 55.30\n|RAD|\nAng = 0.5 x Pi\n")
        params = parseHornrespProject(path)
        assert params["ID"] == "55.30"
        assert params["Ang"] == "0.5 x Pi"

    def test_skips_pipe_section_headers(self, tmp_path):
        path = tmp_path / "p.txt"
        path.write_text("|SECTION|\nS1 = 40.00\n|SECTION 2|\nS2 = 300.00\n")
        params = parseHornrespProject(path)
        assert "S1" in params
        assert "S2" in params

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "p.txt"
        path.write_text("\n  \nID = 42.00\n\n")
        params = parseHornrespProject(path)
        assert params["ID"] == "42.00"

    def test_scientific_notation(self, tmp_path):
        path = tmp_path / "p.txt"
        path.write_text("Cms = 1.47E-03\nMmd = 6.12\n")
        params = parseHornrespProject(path)
        assert params["Cms"] == "1.47E-03"

    def test_first_occurrence_kept(self, tmp_path):
        """For duplicate keys, params[key] keeps the first value; __all has all."""
        path = tmp_path / "p.txt"
        path.write_text("S2 = 300.00\nS2 = 0.00\n")
        params = parseHornrespProject(path)
        assert params["S2"] == "300.00"
        assert params["S2__all"] == ["300.00", "0.00"]

    def test_no_false_duplication(self, tmp_path):
        """Keys that appear once should NOT have a __all entry."""
        path = tmp_path / "p.txt"
        path.write_text("S1 = 40.00\n")
        params = parseHornrespProject(path)
        assert "S1__all" not in params


# ─── parseHornrespDriver ──────────────────────────────────────────────────────

class TestParseHornrespDriver:
    def test_parses_driver_file(self, tmp_path):
        path = tmp_path / "d.txt"
        path.write_text("Sd = 132.70\nBl = 7.80\nCms = 1500.0E-06\n")
        params = parseHornrespDriver(path)
        assert params["Sd"] == "132.70"
        assert params["Bl"] == "7.80"
        assert params["Cms"] == "1500.0E-06"


# ─── hornresp_project_to_geometry ────────────────────────────────────────────

class TestHornrespProjectToGeometry:
    def _params(self) -> dict:
        return {
            "S1": "40.0", "S2": "300.0", "Hyp": "152.7",
            "Vrc": "3.24", "Lrc": "18.0", "Fr1": "2000",
            "Vtc": "88.0", "Atc": "250.18",
        }

    def test_throat_area_cm2_to_m2(self):
        params = self._params()
        geo = hornresp_project_to_geometry(params)
        assert geo.throat_area == pytest.approx(40.0 / 10000.0)

    def test_mouth_area_cm2_to_m2(self):
        params = self._params()
        geo = hornresp_project_to_geometry(params)
        assert geo.mouth_area == pytest.approx(300.0 / 10000.0)

    def test_path_length_cm_to_m(self):
        params = self._params()
        geo = hornresp_project_to_geometry(params)
        assert geo.path_length == pytest.approx(1.527)

    def test_vrc_litres_to_m3(self):
        params = self._params()
        geo = hornresp_project_to_geometry(params)
        assert geo.vrc == pytest.approx(3.24 / 1000.0)

    def test_vtc_cm3_to_m3(self):
        params = self._params()
        geo = hornresp_project_to_geometry(params)
        assert geo.vtc == pytest.approx(88.0 / 1_000_000.0)

    def test_atc_cm2_to_m2(self):
        params = self._params()
        geo = hornresp_project_to_geometry(params)
        assert geo.atc == pytest.approx(250.18 / 10000.0)

    def test_blh_when_vrc_positive(self):
        params = self._params()
        geo = hornresp_project_to_geometry(params)
        assert geo.enclosure_type.upper() == "BLH"

    def test_flh_when_vrc_zero(self):
        params = {**self._params(), "Vrc": "0"}
        geo = hornresp_project_to_geometry(params)
        assert geo.enclosure_type.upper() == "FLH"

    def test_hyperbolic_profile(self):
        params = {**self._params(), "T": "0.30"}
        geo = hornresp_project_to_geometry(params)
        assert geo.profile_type == "Hyperbolic"
        assert geo.hyperbolic_t == pytest.approx(0.30)


# ─── hornresp_driver_to_specs ────────────────────────────────────────────────

class TestHornrespDriverToSpecs:
    def _params(self) -> dict:
        return {
            "Sd": "132.70", "Bl": "7.80", "Cms": "1500.0E-06",
            "Rms": "0.28", "Mmd": "5.90", "Le": "0.80",
            "Re": "7.80",
        }

    def test_sd_cm2_to_m2(self):
        params = self._params()
        specs = hornresp_driver_to_specs(params)
        assert specs.sd == pytest.approx(132.70 / 10000.0)

    def test_mmd_grams_to_kg(self):
        params = self._params()
        specs = hornresp_driver_to_specs(params)
        assert specs.mms == pytest.approx(5.90 / 1000.0)

    def test_le_mh_to_h(self):
        params = self._params()
        specs = hornresp_driver_to_specs(params)
        assert specs.le == pytest.approx(0.0008)

    def test_fs_positive(self):
        params = self._params()
        specs = hornresp_driver_to_specs(params)
        assert specs.fs > 0

    def test_qts_positive(self):
        params = self._params()
        specs = hornresp_driver_to_specs(params)
        assert specs.qts > 0

    def test_vas_positive(self):
        params = self._params()
        specs = hornresp_driver_to_specs(params)
        assert specs.vas > 0


# ─── hornresp_project_to_driver_specs ─────────────────────────────────────────

class TestHornrespProjectToDriverSpecs:
    def test_extracts_driver_keys_from_project(self, tmp_path):
        path = tmp_path / "p.txt"
        path.write_text(
            "S1 = 40.00\n"
            "Sd = 132.70\nBl = 7.80\nCms = 1500.0E-06\n"
            "Rms = 0.28\nMmd = 5.90\nLe = 0.80\nRe = 7.80\n"
        )
        params = parseHornrespProject(path)
        specs = hornresp_project_to_driver_specs(params)
        assert specs.cms > 0
        assert specs.mms > 0
        assert specs.re == 7.80

    def test_returns_zeroed_if_no_driver_keys(self, tmp_path):
        path = tmp_path / "p.txt"
        path.write_text("S1 = 40.00\nS2 = 300.00\n")
        params = parseHornrespProject(path)
        specs = hornresp_project_to_driver_specs(params)
        assert specs.fs == 0.0


# ─── load_hornresp_project ───────────────────────────────────────────────────

class TestLoadHornrespProject:
    def test_loads_geometry_and_specs(self, tmp_path):
        proj = tmp_path / "p.txt"
        proj.write_text("S1=40.0\nS2=300.0\nHyp=152.7\nVrc=3.24\n")
        geo, specs = load_hornresp_project(proj)
        assert geo.throat_area == pytest.approx(40.0 / 10000.0)
        assert specs.fs == 0.0  # no driver params in project file

    def test_driver_override(self, tmp_path):
        proj = tmp_path / "p.txt"
        proj.write_text("S1=40.0\nS2=300.0\nHyp=152.7\n")
        drv = tmp_path / "d.txt"
        drv.write_text("Sd=132.70\nBl=7.80\nCms=1500.0E-06\nRms=0.28\nMmd=5.90\nLe=0.80\nRe=7.80\n")
        geo, specs = load_hornresp_project(proj, drv)
        assert specs.sd > 0
        assert specs.fs > 0
