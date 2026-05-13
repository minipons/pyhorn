"""Tests for pyhorn_core/solver/synthesis_wizard.py"""

import pytest
from pyhorn_core.solver.synthesis_wizard import (
    SynthesisInput,
    SynthesisOutput,
    HornSpec,
    ChamberSpec,
    ValidationWarning,
    synthesize_horn_system,
    synthesis_to_horn_geometry_yaml,
    synthesis_to_driver_yaml,
    _solve_horn_path_length,
    _area_ratio_from_f12_l12,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

FE166NV2 = dict(
    fs=38.0, qts=0.39, vas=25e-3, sd=132.7e-4, re=7.8, bl=7.79,
    mms=6.99e-3, cms=1.472e-3, rms=0.277, qes=0.43, qms=2.64, le=0.8e-3,
)


# ─── SynthesisInput tests ────────────────────────────────────────────────────

class TestSynthesisInput:
    def test_f12_property(self):
        inp = SynthesisInput(**FE166NV2, f3_target_hz=60.0)
        assert inp.f12_hz == pytest.approx(50.0)  # 60/1.2

    def test_f12_property_fraction(self):
        inp = SynthesisInput(**FE166NV2, f3_target_hz=48.0)
        assert inp.f12_hz == pytest.approx(40.0)

    def test_enclosure_type_normalised(self):
        inp = SynthesisInput(**FE166NV2, system_type="FLH", enclosure_type="FLH")
        assert inp.system_type == "FLH"
        assert inp.enclosure_type == "FLH"

    def test_invalid_system_type_defaults_to_blh(self):
        inp = SynthesisInput(**FE166NV2, system_type="invalid")
        assert inp.system_type == "BLH"


# ─── Helper function tests ────────────────────────────────────────────────────

class TestSolveHornPathLength:
    def test_s1_equals_s2_raises(self):
        with pytest.raises(ValueError, match="must be > S1"):
            _solve_horn_path_length(0.01, 0.009, 50.0)

    def test_roundtrip_f12(self):
        """Verify F12 formula is self-consistent."""
        s1 = 50e-4  # 50 cm²
        s2 = 300e-4  # 300 cm²
        l12, _ = _solve_horn_path_length(s1, s2, f12_hz=50.0, max_l_m=5.0)
        ratio = _area_ratio_from_f12_l12(50.0, l12, s1)
        assert ratio == pytest.approx(s2 / s1, rel=1e-6)

    def test_max_length_constraint_returns_max(self):
        s1 = 50e-4
        s2 = 1000e-4  # large mouth
        l12, f12 = _solve_horn_path_length(s1, s2, f12_hz=50.0, max_l_m=0.5)
        assert l12 == 0.5  # capped at max_l_m
        assert f12 > 50.0  # actual F12 is higher due to shorter path


# ─── Core synthesis tests ────────────────────────────────────────────────────

class TestSynthesizeHornSystem:
    def test_fe166nv2_f3_50_blh(self):
        """FE166NV2 at f3=50 Hz — hits MDF path-length constraint, issues warnings."""
        inp = SynthesisInput(**FE166NV2, f3_target_hz=50.0, system_type="BLH")
        result = synthesize_horn_system(inp)

        # Path length capped at max_path_length_m=2.0
        assert result.horn.path_length_m == pytest.approx(2.0, abs=1e-4)
        # F12 is higher than f3/1.2 because of the path-length constraint
        assert 60 < result.horn.f12_computed_hz < 100
        # S2 is at the MDF mouth budget
        assert result.horn.mouth_area_m2 * 1e4 == pytest.approx(1200.0, abs=0.01)
        # Two sections: straight throat + catenoidal main
        assert len(result.horn.sections) == 2
        assert result.horn.sections[0].name == "throat"
        assert result.horn.sections[0].profile_type == "straight"
        assert result.horn.sections[1].name == "main_horn"
        assert result.horn.sections[1].profile_type == "catenoidal"
        # Vrc=0 (Qts=0.39 < Qts_alignment=0.55)
        assert result.chambers.vrc_L == pytest.approx(0.0, abs=1e-6)
        # Warnings issued about path-length constraint
        warn_msgs = [w.message for w in result.warnings]
        assert any("exceeds 1.5 m" in m for m in warn_msgs)
        assert any("F12=" in m and "MDF path-length" in m for m in warn_msgs)

    def test_high_q_driver_gets_rear_chamber(self):
        """High-Q driver (Qts=0.70 > Qts_alignment=0.55) → Vrc > 0."""
        high_q = dict(fs=35.0, qts=0.70, vas=50e-3, sd=200e-4, re=6.0, bl=8.0,
                      mms=10e-3, cms=2e-3, rms=0.3, qes=0.8, qms=3.5, le=1e-3)
        inp = SynthesisInput(**high_q, f3_target_hz=40.0, system_type="BLH",
                             qts_alignment=0.55)
        result = synthesize_horn_system(inp)
        assert result.chambers.vrc_L > 0
        assert result.chambers.lrc_m > 0
        # INFO warning suppressed (Qts > alignment)
        info_msgs = [w.message for w in result.warnings if w.severity == "INFO"]
        assert not any("no rear chamber" in m for m in info_msgs)

    def test_low_q_driver_no_rear_chamber(self):
        """Low-Q driver (Qts=0.30 < Qts_alignment) → Vrc ≈ 0, INFO issued."""
        low_q = dict(fs=45.0, qts=0.30, vas=20e-3, sd=100e-4, re=5.0, bl=6.0,
                     mms=5e-3, cms=1e-3, rms=0.2, qes=0.35, qms=2.0, le=0.5e-3)
        inp = SynthesisInput(**low_q, f3_target_hz=60.0, system_type="FLH")
        result = synthesize_horn_system(inp)
        assert result.chambers.vrc_L < 0.01
        info_msgs = [w.message for w in result.warnings if w.severity == "INFO"]
        assert any("no rear chamber" in m for m in info_msgs)

    def test_relaxed_path_limit_avoids_constraint(self):
        """With max_path_length_m=3.0, f3=30 Hz gives a lower F12."""
        inp = SynthesisInput(**FE166NV2, f3_target_hz=30.0,
                             max_path_length_m=3.0, max_mouth_area_m2=0.12)
        result = synthesize_horn_system(inp)
        assert 0 < result.horn.f12_computed_hz < 100
        assert result.horn.path_length_m <= 3.0 + 1e-6

    def test_throat_adapter_ap1_not_exceeds_s1(self):
        inp = SynthesisInput(**FE166NV2, f3_target_hz=50.0)
        result = synthesize_horn_system(inp)
        assert result.chambers.ap1_m2 <= 1.1 * FE166NV2["sd"]

    def test_vtc_is_fraction_of_vas(self):
        inp = SynthesisInput(**FE166NV2, f3_target_hz=50.0)
        result = synthesize_horn_system(inp)
        # Vtc = 0.002 × Vas
        assert result.chambers.vtc_m3 == pytest.approx(0.002 * FE166NV2["vas"], rel=1e-9)


# ─── YAML serialisation tests ────────────────────────────────────────────────

class TestSynthesisYamlOutput:
    def test_horn_geometry_yaml_has_all_sections(self):
        inp = SynthesisInput(**FE166NV2, f3_target_hz=50.0, system_type="BLH")
        result = synthesize_horn_system(inp)
        yaml_str = synthesis_to_horn_geometry_yaml(result)
        # Check for sections block and required chamber/adapter fields (flat format)
        for key in ["sections:", "vrc:", "lrc:", "vtc:", "atc:", "ap1:", "lpt:",
                    "ang:", "main_horn", "straight", "catenoidal"]:
            assert key in yaml_str, f"Missing: {key}"
        # Numeric values present
        assert "132.70" in yaml_str  # S1
        assert "1200" in yaml_str     # S2

    def test_driver_yaml_has_tsp_fields(self):
        inp = SynthesisInput(**FE166NV2, f3_target_hz=50.0)
        result = synthesize_horn_system(inp)
        yaml_str = synthesis_to_driver_yaml(result, inp)
        assert "fs:" in yaml_str
        assert "qts:" in yaml_str
        assert "vas:" in yaml_str
        assert "sd:" in yaml_str
        assert "#  Synthesised Driver Specs" in yaml_str


# ─── ValidationWarning structure tests ───────────────────────────────────────

class TestValidationWarnings:
    def test_warnings_have_valid_severity(self):
        inp = SynthesisInput(**FE166NV2, f3_target_hz=50.0, system_type="BLH")
        result = synthesize_horn_system(inp)
        for w in result.warnings:
            assert w.severity in ("INFO", "WARN", "ERROR")
            assert len(w.field) > 0
            assert len(w.message) > 0

    def test_large_vrc_warns(self):
        large_vc = dict(fs=30.0, qts=0.90, vas=200e-3, sd=300e-4, re=6.0, bl=8.0,
                         mms=15e-3, cms=3e-3, rms=0.3, qes=1.0, qms=4.0, le=1e-3)
        inp = SynthesisInput(**large_vc, f3_target_hz=35.0, system_type="BLH",
                             qts_alignment=0.50)
        result = synthesize_horn_system(inp)
        warn_msgs = [w.message for w in result.warnings]
        assert any("Vrc=" in m and ("large" in m or "L is" in m) for m in warn_msgs), \
            f"Expected large-Vrc warning, got: {warn_msgs}"

    def test_f7_low_frequencies_warning(self):
        inp = SynthesisInput(**FE166NV2, f3_target_hz=50.0, f7_target_hz=2000.0)
        result = synthesize_horn_system(inp)
        warn_msgs = [w.message for w in result.warnings]
        assert any("f7=" in m for m in warn_msgs)

    def test_qts_alignment_zero_does_not_crash(self):
        """qts_alignment=0 must not raise ZeroDivisionError — treat as no rear chamber."""
        inp = SynthesisInput(**FE166NV2, f3_target_hz=50.0, qts_alignment=0.0)
        # Must not raise ZeroDivisionError
        result = synthesize_horn_system(inp)
        # Should produce Vrc=0 with a warning about qts_alignment
        assert result.chambers.vrc_m3 == 0.0
        warn_msgs = [w.message for w in result.warnings]
        assert any("qts_alignment" in m.lower() or "0" in m for m in warn_msgs), \
            f"Expected qts_alignment warning, got: {warn_msgs}"
