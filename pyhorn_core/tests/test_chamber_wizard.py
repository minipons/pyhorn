"""Tests for pyhorn_core/solver/chamber_wizard.py — Chamber Design Wizard."""

import math

import pytest

from pyhorn_core.solver.chamber_wizard import (
    QTS_TARGET,
    parse_tsp,
    compute_chamber_params,
    validate_chamber,
    build_yaml_snippet,
    TSPParams,
    ComputedChamberParams,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fe166nv2_yaml() -> str:
    """Fostex FE166NV2 driver YAML text (sd in m² as per pyhorn convention)."""
    return """
fs: 43.0
qts: 0.27
vas: 36.9
sd: 0.01327
"""


@pytest.fixture
def fe166nv2_tsp() -> TSPParams:
    """Fostex FE166NV2 parsed TSP."""
    sd_m2 = 132.7 / 1e4
    return TSPParams(
        fs=43.0,
        qts=0.27,
        vas=36.9 / 1000.0,  # litres → m³
        sd=sd_m2,
        sd_cm2=132.7,
    )


@pytest.fixture
def generic_driver_tsp() -> TSPParams:
    """Generic mid-Qts driver TSP."""
    sd_m2 = 200.0 / 1e4
    return TSPParams(
        fs=60.0,
        qts=0.65,
        vas=50.0 / 1000.0,
        sd=sd_m2,
        sd_cm2=200.0,
    )


# ── parse_tsp ─────────────────────────────────────────────────────────────────

class TestParseTSP:
    def test_parses_required_fields(self, fe166nv2_yaml):
        tsp = parse_tsp(fe166nv2_yaml)
        assert tsp.fs == pytest.approx(43.0)
        assert tsp.qts == pytest.approx(0.27)
        assert tsp.vas == pytest.approx(36.9)
        assert tsp.sd_cm2 == pytest.approx(132.7)
        assert tsp.sd == pytest.approx(132.7 / 1e4)

    def test_missing_optional_sd(self):
        yaml_text = "fs: 50.0\nqts: 0.5\nvas: 20.0\n"
        tsp = parse_tsp(yaml_text)
        assert tsp.sd is None
        assert tsp.sd_cm2 is None

    def test_missing_required_returns_none(self):
        yaml_text = "fs: 50.0\n"
        tsp = parse_tsp(yaml_text)
        assert tsp.qts is None
        assert tsp.vas is None


# ── compute_chamber_params ────────────────────────────────────────────────────

class TestComputeChamberParams:
    def test_vrc_zero_when_qts_below_target(self, fe166nv2_tsp):
        """Low-Qts driver (Qts=0.27 < 0.6 target) → no rear chamber needed."""
        p = compute_chamber_params(fe166nv2_tsp)
        assert p.vrc_L == pytest.approx(0.0)
        assert p.lrc_m == pytest.approx(0.0)

    def test_vrc_positive_when_qts_above_target(self, generic_driver_tsp):
        """High-Qts driver (Qts=0.65 > 0.6 target) → rear chamber volume positive."""
        p = compute_chamber_params(generic_driver_tsp)
        # Vrc = Vas × (Qts² / Qts_target² − 1)
        expected_ratio = (0.65 ** 2) / (QTS_TARGET ** 2) - 1.0
        expected_vrc_m3 = (50.0 / 1000.0) * expected_ratio
        assert p.vrc_L == pytest.approx(expected_vrc_m3 * 1000.0, rel=1e-6)

    def test_vrc_zero_when_qts_equals_target(self):
        """Qts exactly at target → Vrc = 0 (no correction needed)."""
        tsp = TSPParams(fs=60.0, qts=0.6, vas=50.0 / 1000.0, sd=200.0 / 1e4, sd_cm2=200.0)
        p = compute_chamber_params(tsp)
        assert p.vrc_L == pytest.approx(0.0)

    def test_vrc_zero_when_vas_none(self):
        """Missing Vas → Vrc = 0."""
        tsp = TSPParams(fs=60.0, qts=0.65, vas=None, sd=200.0 / 1e4, sd_cm2=200.0)
        p = compute_chamber_params(tsp)
        assert p.vrc_L == pytest.approx(0.0)

    def test_atc_uses_sd_when_available(self, fe166nv2_tsp):
        """Atc should equal driver piston area (Sd)."""
        p = compute_chamber_params(fe166nv2_tsp)
        assert p.atc_cm2 == pytest.approx(132.7, rel=1e-6)
        assert p.atc_m2 == pytest.approx(132.7 / 1e4, rel=1e-6)

    def test_atc_fallback_to_default_diameter(self):
        """No Sd in YAML → use 58 mm default diameter."""
        tsp = TSPParams(fs=60.0, qts=0.5, vas=20.0 / 1000.0, sd=None, sd_cm2=None)
        p = compute_chamber_params(tsp)
        expected_atc = (math.pi / 4.0) * (0.058 ** 2)
        assert p.atc_m2 == pytest.approx(expected_atc, rel=1e-8)

    def test_vtc_fraction_of_vas(self, fe166nv2_tsp):
        """Vtc = 0.002 × Vas."""
        p = compute_chamber_params(fe166nv2_tsp)
        expected_vtc_m3 = 0.002 * (36.9 / 1000.0)
        assert p.vtc_m3 == pytest.approx(expected_vtc_m3, rel=1e-8)
        assert p.vtc_cm3 == pytest.approx(expected_vtc_m3 * 1e6, rel=1e-8)

    def test_lrc_from_vrc_over_atc(self, generic_driver_tsp):
        """Lrc = Vrc / Atc."""
        p = compute_chamber_params(generic_driver_tsp)
        expected_lrc = (p.vrc_L / 1000.0) / (generic_driver_tsp.sd)
        assert p.lrc_m == pytest.approx(expected_lrc, rel=1e-6)

    def test_ap1_fixed_50mm_aperture(self, fe166nv2_tsp):
        """Ap1 is a fixed 50 mm diameter hole."""
        p = compute_chamber_params(fe166nv2_tsp)
        expected_ap1 = (math.pi / 4.0) * (0.050 ** 2)
        assert p.ap1_m2 == pytest.approx(expected_ap1, rel=1e-8)
        assert p.ap1_cm2 == pytest.approx(expected_ap1 * 1e4, rel=1e-8)

    def test_lpt_fixed_12mm(self, fe166nv2_tsp):
        """Lpt is a fixed 12 mm baffle thickness."""
        p = compute_chamber_params(fe166nv2_tsp)
        assert p.lpt_m == pytest.approx(0.012, rel=1e-8)
        assert p.lpt_cm == pytest.approx(1.2, rel=1e-8)

    def test_fe166nv2_expected_values(self, fe166nv2_tsp):
        """Known values for FE166NV2 (Qts=0.27 < 0.6 → Vrc=0)."""
        p = compute_chamber_params(fe166nv2_tsp)
        # Vrc = 0 (low Qts)
        assert p.vrc_L == pytest.approx(0.0)
        # Lrc = 0 (Vrc is 0)
        assert p.lrc_m == pytest.approx(0.0)
        # Vtc = 0.002 × 36.9L = 73.8 cm³
        assert p.vtc_cm3 == pytest.approx(73.8, rel=1e-4)
        # Atc = Sd = 132.7 cm²
        assert p.atc_cm2 == pytest.approx(132.7, rel=1e-4)


# ── validate_chamber ──────────────────────────────────────────────────────────

class TestValidateChamber:
    def test_vrc_lrc_no_warning_for_vrc_zero(self, fe166nv2_tsp):
        """Vrc=0 (low-Qts driver) should not produce a Vrc 'too small' warning.

        Vrc=0 means no rear chamber is needed; that is a valid design choice,
        not a warning-worthy condition. The Lrc warning is also suppressed.
        """
        p = compute_chamber_params(fe166nv2_tsp)
        v = validate_chamber(p, fe166nv2_tsp)
        # Vrc = 0 is intentional for low-Qts drivers — no warning
        assert v.vrc_warning is None
        assert v.lrc_warning is None

    def test_vrc_too_small_warning(self):
        """Vrc < 0.5 L gets a warning."""
        p = ComputedChamberParams(
            vrc_L=0.1, lrc_m=0.001, vtc_m3=1e-4, vtc_cm3=100.0,
            atc_m2=0.01, atc_cm2=100.0, ap1_m2=0.002, ap1_cm2=20.0,
            lpt_m=0.012, lpt_cm=1.2,
        )
        tsp = TSPParams(fs=60.0, qts=0.65, vas=0.05, sd=0.01, sd_cm2=100.0)
        v = validate_chamber(p, tsp)
        assert "too small" in v.vrc_warning

    def test_vrc_too_large_warning(self):
        """Vrc > 30 L gets a warning."""
        p = ComputedChamberParams(
            vrc_L=50.0, lrc_m=0.5, vtc_m3=1e-4, vtc_cm3=100.0,
            atc_m2=0.01, atc_cm2=100.0, ap1_m2=0.002, ap1_cm2=20.0,
            lpt_m=0.012, lpt_cm=1.2,
        )
        tsp = TSPParams(fs=60.0, qts=0.65, vas=0.05, sd=0.01, sd_cm2=100.0)
        v = validate_chamber(p, tsp)
        assert "very large" in v.vrc_warning

    def test_lrc_out_of_range_warning(self):
        """Lrc outside 3–80 cm range gets a warning."""
        p = ComputedChamberParams(
            vrc_L=5.0, lrc_m=0.9, vtc_m3=1e-4, vtc_cm3=100.0,
            atc_m2=0.01, atc_cm2=100.0, ap1_m2=0.002, ap1_cm2=20.0,
            lpt_m=0.012, lpt_cm=1.2,
        )
        tsp = TSPParams(fs=60.0, qts=0.65, vas=0.05, sd=0.01, sd_cm2=100.0)
        v = validate_chamber(p, tsp)
        assert "unrealistic" in v.lrc_warning

    def test_atc_greater_than_2x_sd_warning(self):
        """Atc > 2× Sd gets a warning."""
        p = ComputedChamberParams(
            vrc_L=5.0, lrc_m=0.1, vtc_m3=1e-4, vtc_cm3=100.0,
            atc_m2=0.03, atc_cm2=300.0, ap1_m2=0.002, ap1_cm2=20.0,
            lpt_m=0.012, lpt_cm=1.2,
        )
        tsp = TSPParams(fs=60.0, qts=0.65, vas=0.05, sd=0.01, sd_cm2=100.0)
        v = validate_chamber(p, tsp)
        assert "Atc > 2× Sd" in v.atc_warning

    def test_atc_less_than_half_sd_warning(self):
        """Atc < 0.5× Sd gets a warning."""
        p = ComputedChamberParams(
            vrc_L=5.0, lrc_m=0.1, vtc_m3=1e-4, vtc_cm3=100.0,
            atc_m2=0.0025, atc_cm2=25.0, ap1_m2=0.002, ap1_cm2=20.0,
            lpt_m=0.012, lpt_cm=1.2,
        )
        tsp = TSPParams(fs=60.0, qts=0.65, vas=0.05, sd=0.01, sd_cm2=100.0)
        v = validate_chamber(p, tsp)
        assert "Atc < 0.5× Sd" in v.atc_warning

    def test_ap1_less_than_half_sd_warning(self):
        """Ap1 < 0.5× Sd gets a warning."""
        p = ComputedChamberParams(
            vrc_L=5.0, lrc_m=0.1, vtc_m3=1e-4, vtc_cm3=100.0,
            atc_m2=0.01, atc_cm2=100.0, ap1_m2=0.001, ap1_cm2=10.0,
            lpt_m=0.012, lpt_cm=1.2,
        )
        tsp = TSPParams(fs=60.0, qts=0.65, vas=0.05, sd=0.01, sd_cm2=100.0)
        v = validate_chamber(p, tsp)
        assert "Ap1 < 0.5× Sd" in v.ap1_warning


# ── build_yaml_snippet ────────────────────────────────────────────────────────

class TestBuildYamlSnippet:
    def test_contains_all_keys(self, fe166nv2_tsp):
        p = compute_chamber_params(fe166nv2_tsp)
        yaml_str = build_yaml_snippet(p)
        assert "rear_chamber:" in yaml_str
        assert "throat_chamber:" in yaml_str
        assert "throat_adapter:" in yaml_str
        assert "vrc:" in yaml_str
        assert "lrc:" in yaml_str
        assert "vtc:" in yaml_str
        assert "atc:" in yaml_str
        assert "ap1:" in yaml_str
        assert "lpt:" in yaml_str

    def test_vrc_zero_case(self, fe166nv2_tsp):
        """Vrc=0 should appear as 0.0000 in the YAML."""
        p = compute_chamber_params(fe166nv2_tsp)
        yaml_str = build_yaml_snippet(p)
        assert "vrc: 0.0000" in yaml_str

    def test_vrc_positive_case(self, generic_driver_tsp):
        """Positive Vrc should be a non-zero number."""
        p = compute_chamber_params(generic_driver_tsp)
        yaml_str = build_yaml_snippet(p)
        # Vrc should be positive for high-Qts driver
        lines = [l for l in yaml_str.splitlines() if l.strip().startswith("vrc:")]
        assert len(lines) == 1
        val = float(lines[0].split(":")[1].strip())
        assert val > 0.0
