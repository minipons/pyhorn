"""Unit tests for PassiveFilter — electrical R/L/C network simulation."""

import math
import numpy as np
import pytest
from pyhorn_core.solver.filter_schematic import (
    PassiveComponent,
    PassiveFilter,
    compute_filter_schematic,
)


# ─── TestPassiveComponent ──────────────────────────────────────────────────────

class TestPassiveComponent:
    def test_resistor(self):
        r = PassiveComponent("R", 8.0)
        assert r.component_type == "R"
        assert r.value == 8.0
        assert r.position == "series"

    def test_inductor_shunt(self):
        l = PassiveComponent("L", 0.010, position="shunt")
        assert l.component_type == "L"
        assert l.value == 0.010
        assert l.position == "shunt"

    def test_capacitor(self):
        c = PassiveComponent("C", 1e-4)
        assert c.component_type == "C"
        assert c.value == 1e-4
        assert c.position == "series"


# ─── TestPassiveFilterSeriesR ─────────────────────────────────────────────────

class TestPassiveFilterSeriesR:
    """Series resistor alone: fixed 6 dB attenuator at all frequencies."""

    @pytest.fixture
    def series_r_filter(self):
        return PassiveFilter("series", [PassiveComponent("R", 4.0)])

    def test_transfer_function_constant(self, series_r_filter):
        for f in [10.0, 100.0, 1e3, 1e4]:
            H = series_r_filter.transfer_function(f, r_load=4.0)
            # H = R_load / (R + R_load) = 4/(4+4) = 0.5
            assert abs(H - 0.5) < 1e-12

    def test_magnitude_db(self, series_r_filter):
        mag_db = series_r_filter.magnitude_db(100.0, r_load=4.0)
        assert abs(mag_db - (-6.02)) < 0.01

    def test_phase_near_zero(self, series_r_filter):
        phase = series_r_filter.phase_deg(100.0, r_load=4.0)
        assert abs(phase) < 1e-6


# ─── TestPassiveFilterSeriesL ─────────────────────────────────────────────────

class TestPassiveFilterSeriesL:
    """Series inductor alone is a LOW-PASS filter.

    At LF: inductor is short (Z_L ≈ 0) → H → R_load/(R_load+0) → 1 (passes).
    At HF: inductor is open (Z_L → ∞) → H → R_load/(∞) → 0 (blocked).
    """

    @pytest.fixture
    def series_l_filter(self):
        return PassiveFilter("series", [PassiveComponent("L", 0.010)])

    def test_at_dc_passes(self, series_l_filter):
        H = series_l_filter.transfer_function(0.0, r_load=4.0)
        assert abs(H) > 0.99

    def test_at_hf_blocked(self, series_l_filter):
        H = series_l_filter.transfer_function(1e6, r_load=4.0)
        assert abs(H) < 0.001

    def test_magnitude_db_lowpass_shape(self, series_l_filter):
        # LF passes, HF blocked
        mag_lf = series_l_filter.magnitude_db(1.0, r_load=4.0)
        mag_hf = series_l_filter.magnitude_db(10e3, r_load=4.0)
        assert mag_lf > mag_hf


# ─── TestPassiveFilterSeriesC ─────────────────────────────────────────────────

class TestPassiveFilterSeriesC:
    """Series capacitor alone is a HIGH-PASS filter.

    At LF: capacitor is open (Z_C → ∞) → H → 0 (blocked).
    At HF: capacitor is short (Z_C ≈ 0) → H → 1 (passes).
    """

    @pytest.fixture
    def series_c_filter(self):
        return PassiveFilter("series", [PassiveComponent("C", 1e-4)])

    def test_at_dc_blocked(self, series_c_filter):
        H = series_c_filter.transfer_function(0.0, r_load=4.0)
        assert abs(H) < 1e-6

    def test_at_hf_passes(self, series_c_filter):
        H = series_c_filter.transfer_function(1e6, r_load=4.0)
        assert abs(H) > 0.99

    def test_magnitude_db_highpass_shape(self, series_c_filter):
        # LF blocked, HF passes
        mag_lf = series_c_filter.magnitude_db(10.0, r_load=4.0)
        mag_hf = series_c_filter.magnitude_db(10e3, r_load=4.0)
        assert mag_lf < mag_hf


# ─── TestPassiveFilterLeCleach ─────────────────────────────────────────────────

class TestPassiveFilterLeCleach:
    """Le Cléac'h topology: series C → junction → shunt (R_load || L) — 2nd-order HP.

    At low frequencies the series capacitor is open → no current → output ≈ 0.
    At high frequencies the capacitor shorts, inductor passes → output ≈ input.

    Component formulas (Butterworth, Q = 1/√2):
        L = R_load / (√2 · π · fc)   (from LR2 HP alignment)
        C = 1 / (√2 · π · fc · R_load)

    For R_load=4 Ω, fc=80 Hz:
        L ≈ 11.3 mH,  C ≈ 278 µF
    fc = 1/(2π·√(LC)) ≈ 80 Hz  ✓
    """

    @pytest.fixture
    def le_cleach_filter(self):
        R_load = 4.0
        fc = 80.0
        # LR2 Butterworth component values
        L = R_load / (math.sqrt(2) * math.pi * fc)   # ≈ 0.0113 H
        C = 1.0 / (math.sqrt(2) * math.pi * fc * R_load)  # ≈ 278 µF
        return PassiveFilter(
            topology="le_cleach",
            components=[
                PassiveComponent("C", C),
                PassiveComponent("L", L),
            ],
        ), R_load, L, C

    def test_at_dc_zero(self, le_cleach_filter):
        filt, r_load, L, C = le_cleach_filter
        H = filt.transfer_function(0.0, r_load)
        assert abs(H) < 1e-6

    def test_at_hf_near_unity(self, le_cleach_filter):
        filt, r_load, L, C = le_cleach_filter
        H = filt.transfer_function(5e3, r_load)
        assert abs(H) > 0.95

    def test_at_lf_significant_attenuation(self, le_cleach_filter):
        filt, r_load, L, C = le_cleach_filter
        # At 10 Hz (well below fc=80 Hz), expect strong attenuation
        H = filt.transfer_function(10.0, r_load)
        assert abs(H) < 0.1

    def test_magnitude_db_highpass_shape(self, le_cleach_filter):
        filt, r_load, L, C = le_cleach_filter
        mag_lf = filt.magnitude_db(10.0, r_load)
        mag_hf = filt.magnitude_db(5e3, r_load)
        assert mag_lf < mag_hf   # LF more attenuated than HF

    def test_phase_leads_at_hf(self, le_cleach_filter):
        filt, r_load, L, C = le_cleach_filter
        # At HF, series C is short → phase ≈ 0 (in-phase)
        phase = filt.phase_deg(5e3, r_load)
        assert abs(phase) < 30.0

    def test_magnitude_db_unity_at_hf(self, le_cleach_filter):
        filt, r_load, L, C = le_cleach_filter
        mag_db = filt.magnitude_db(5e3, r_load)
        assert mag_db > -1.0  # near 0 dB at HF

    def test_z_at_returns_junction_impedance(self, le_cleach_filter):
        filt, r_load, L, C = le_cleach_filter
        # z_at for le_cleach returns the shunt/junction branch impedance = C || L
        z = filt.z_at(100.0)
        # C || L at 100 Hz: Z_C = -j5.72 Ω, Z_L = j7.1 Ω
        # Z_parallel = Z_C * Z_L / (Z_C + Z_L)
        omega = 2 * math.pi * 100
        z_c = complex(0, -1.0 / (omega * C))
        z_l = complex(0, omega * L)
        z_expected = (z_c * z_l) / (z_c + z_l)
        assert abs(z - z_expected) < 1e-9


# ─── TestPassiveFilterParallelRC ───────────────────────────────────────────────

class TestPassiveFilterParallelRC:
    """Parallel RC shunt to ground — LP: C shorts HF to GND, R passes LF.

    At low frequency, C is open → signal passes through R to output.
    At high frequency, C shorts → output shunted to GND → attenuation.
    """

    @pytest.fixture
    def rc_parallel_filter(self):
        return PassiveFilter(
            "parallel",
            [
                PassiveComponent("R", 4.0, position="shunt"),
                PassiveComponent("C", 1e-4, position="shunt"),
            ],
        )

    def test_z_at_dc_is_r_load(self, rc_parallel_filter):
        z = rc_parallel_filter.z_at(0.0)
        assert abs(z.real - 4.0) < 1e-6

    def test_z_at_hf_low(self, rc_parallel_filter):
        z = rc_parallel_filter.z_at(10e3)
        assert abs(z) < 1.0  # HF shorted by C

    def test_transfer_function_lowpass_shape(self, rc_parallel_filter):
        mag_lf = rc_parallel_filter.magnitude_db(10.0, r_load=4.0)
        mag_hf = rc_parallel_filter.magnitude_db(10e3, r_load=4.0)
        assert mag_lf > mag_hf   # LF passes, HF attenuated

    def test_magnitude_never_exceeds_0db(self, rc_parallel_filter):
        for f in [1.0, 10.0, 100.0, 1e3, 10e3]:
            mag = rc_parallel_filter.magnitude_db(f, r_load=4.0)
            assert mag <= 0.1   # passive filter can't amplify


# ─── TestPassiveFilterTransferFunction ────────────────────────────────────────

class TestPassiveFilterTransferFunction:
    """Verify transfer function physics at specific frequencies."""

    def test_series_r_only_transfer_function(self):
        filt = PassiveFilter("series", [PassiveComponent("R", 4.0)])
        for f in [100.0, 1e3, 10e3]:
            H = filt.transfer_function(f, r_load=4.0)
            assert abs(H - 0.5) < 1e-12

    def test_parallel_r_only_transfer_function(self):
        # Parallel R only: H = R/(R+R_load) = 4/(4+4) = 0.5
        filt = PassiveFilter(
            "parallel",
            [PassiveComponent("R", 4.0, position="shunt")],
        )
        H = filt.transfer_function(100.0, r_load=4.0)
        assert abs(H - 0.5) < 1e-12

    def test_passive_cannot_amplify_series_l(self):
        filt = PassiveFilter("series", [PassiveComponent("L", 0.010)])
        for f in [1e-6, 1.0, 100.0, 1e6]:
            H = filt.transfer_function(f, r_load=4.0)
            assert abs(H) <= 1.0 + 1e-12

    def test_passive_cannot_amplify_parallel_rc(self):
        filt = PassiveFilter(
            "parallel",
            [PassiveComponent("R", 4.0, position="shunt"), PassiveComponent("C", 1e-4, position="shunt")],
        )
        for f in [1e-6, 1.0, 100.0, 1e6]:
            H = filt.transfer_function(f, r_load=4.0)
            assert abs(H) <= 1.0 + 1e-12


# ─── TestPassiveFilterApplyToResponse ─────────────────────────────────────────

class TestPassiveFilterApplyToResponse:
    """Verify that apply_to_response modifies the SPL correctly."""

    def test_flat_response_reduced_by_fixed_attenuator(self):
        # Series R: H = 0.5 → -6 dB at ALL frequencies
        filt = PassiveFilter("series", [PassiveComponent("R", 4.0)])
        freqs = np.array([100.0, 1e3, 5e3])
        spl = np.array([90.0, 90.0, 90.0])
        imp = np.array([4.0, 4.0, 4.0])
        phase = np.array([0.0, 0.0, 0.0])

        fspl, fimp, fphase = filt.apply_to_response(spl, freqs, imp, phase, r_load=4.0)

        np.testing.assert_allclose(fspl, 90.0 - 6.02, rtol=0.01)
        np.testing.assert_allclose(fimp, 2.0, rtol=0.01)
        np.testing.assert_allclose(fphase, 0.0, atol=1e-6)

    def test_le_cleach_filter_highpass_behavior(self):
        # Le Cléac'h HP: LF attenuated, HF passes
        R_load = 4.0
        fc = 80.0
        L = R_load / (math.sqrt(2) * math.pi * fc)
        C = 1.0 / (math.sqrt(2) * math.pi * fc * R_load)
        filt = PassiveFilter(
            "le_cleach",
            [PassiveComponent("C", C), PassiveComponent("L", L)],
        )
        freqs = np.array([10.0, 80.0, 5e3])  # LF, fc, HF
        spl = np.array([80.0, 80.0, 80.0])
        imp = np.array([4.0, 4.0, 4.0])
        phase = np.array([0.0, 0.0, 0.0])

        fspl, _, _ = filt.apply_to_response(spl, freqs, imp, phase, r_load=R_load)

        assert fspl[0] < fspl[2]   # LF attenuated more than HF
        assert fspl[2] > fspl[0] + 20.0  # big difference

    def test_parallel_rc_filter_lowpass_behavior(self):
        filt = PassiveFilter(
            "parallel",
            [PassiveComponent("R", 4.0, position="shunt"), PassiveComponent("C", 1e-4, position="shunt")],
        )
        freqs = np.array([10.0, 1e3, 10e3])
        spl = np.array([85.0, 85.0, 85.0])
        imp = np.array([4.0, 4.0, 4.0])
        phase = np.array([0.0, 0.0, 0.0])

        fspl, _, _ = filt.apply_to_response(spl, freqs, imp, phase, r_load=4.0)

        assert fspl[2] < fspl[0]   # HF more attenuated than LF (LP)

    def test_array_length_preserved(self):
        filt = PassiveFilter("series", [PassiveComponent("R", 4.0)])
        freqs = np.linspace(20.0, 5000.0, 200)
        spl = np.ones_like(freqs) * 85.0
        imp = np.ones_like(freqs) * 4.0
        phase = np.zeros_like(freqs)

        fspl, fimp, fphase = filt.apply_to_response(spl, freqs, imp, phase, r_load=4.0)

        assert len(fspl) == len(freqs)
        assert len(fimp) == len(freqs)
        assert len(fphase) == len(freqs)


# ─── Validation tests ───────────────────────────────────────────────────────────

class TestPassiveFilterValidation:
    def test_empty_components_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            PassiveFilter("series", [])

    def test_invalid_topology_rejected(self):
        with pytest.raises(ValueError, match="topology must be"):
            PassiveFilter("star", [PassiveComponent("R", 4.0)])

    def test_le_cleach_is_valid_topology(self):
        filt = PassiveFilter(
            "le_cleach",
            [PassiveComponent("C", 1e-4), PassiveComponent("L", 0.010)],
        )
        assert filt.topology == "le_cleach"
        H = filt.transfer_function(100.0, r_load=4.0)
        assert 0.0 <= abs(H) <= 1.0


# ─── TestComputeFilterSchematicWithPassiveFilter ───────────────────────────────

class TestComputeFilterSchematicWithPassiveFilter:
    """compute_filter_schematic() dispatches to PassiveFilter.schematic()."""

    def test_series_passive_schematic(self):
        filt = PassiveFilter(
            "series",
            [PassiveComponent("R", 4.0), PassiveComponent("L", 0.010)],
        )
        result = compute_filter_schematic(filt)
        assert isinstance(result, str)
        assert "Passive R/L/C Filter" in result
        assert "Series" in result

    def test_parallel_passive_schematic(self):
        filt = PassiveFilter(
            "parallel",
            [
                PassiveComponent("R", 4.0, position="shunt"),
                PassiveComponent("C", 1e-4, position="shunt"),
            ],
        )
        result = compute_filter_schematic(filt)
        assert "Parallel" in result

    def test_le_cleach_schematic(self):
        filt = PassiveFilter(
            "le_cleach",
            [PassiveComponent("C", 1e-4), PassiveComponent("L", 0.010)],
        )
        result = compute_filter_schematic(filt, r_load=4.0)
        assert "output" in result.lower()

    def test_r_load_passed_through(self):
        filt = PassiveFilter("series", [PassiveComponent("R", 8.0)])
        result = compute_filter_schematic(filt, r_load=8.0)
        assert "8.0" in result
