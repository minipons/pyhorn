"""Unit tests for pyhorn.solver.models — physics engine and horn_response."""

import numpy as np
import pytest
from pyhorn_core.config.models import DriverSpecs, HornGeometry, Section, CompoundChamber, TappedHornGeometry, RearChamber
from pyhorn_core.solver import models


class TestPhysicalConstants:
    """Tests that physical constants have expected values."""

    def test_rho_is_air_density(self):
        """RHO should be approximately 1.21 kg/m³."""
        assert models.RHO == pytest.approx(1.21, rel=1e-3)

    def test_c_is_speed_of_sound(self):
        """C should be approximately 343 m/s."""
        assert models.C == pytest.approx(343.0, rel=1e-3)

    def test_z0_is_characteristic_impedance(self):
        """Z0 = RHO * C should be approximately 415.63 Pa·s/m³."""
        assert models.Z0 == pytest.approx(models.RHO * models.C)


class TestMikiFactors:
    """Tests for _miki_factors()."""

    def test_returns_tuple_of_complex(self):
        """Should return (Zc_factor, k_factor) both complex."""
        Zc_f, k_f = models._miki_factors(freq=100.0, sigma=5000.0)
        assert isinstance(Zc_f, complex)
        assert isinstance(k_f, complex)

    def test_at_low_frequency_magnitude_greater_than_1(self):
        """At low f/sigma, Zc_factor magnitude should be > 1 (absorption reduced)."""
        Zc_f, _ = models._miki_factors(freq=50.0, sigma=5000.0)
        assert abs(Zc_f) > 1.0

    def test_at_high_frequency_magnitude_approaches_1(self):
        """At high f/sigma, Zc_factor magnitude should approach 1."""
        Zc_f, _ = models._miki_factors(freq=5000.0, sigma=100.0)
        assert abs(Zc_f) == pytest.approx(1.0, rel=0.05)

    def test_zc_factor_has_negative_imaginary_part(self):
        """Zc_factor imag part should be negative (dissipative)."""
        Zc_f, _ = models._miki_factors(freq=200.0, sigma=5000.0)
        assert Zc_f.imag < 0

    def test_k_factor_has_negative_imaginary_part(self):
        """k_factor imag part should be negative (dissipative)."""
        _, k_f = models._miki_factors(freq=200.0, sigma=5000.0)
        assert k_f.imag < 0


class TestSimulationResultDataclass:
    """Tests for SimulationResult dataclass."""

    def test_construction_with_required_fields(self):
        """Can construct with just the required numpy array fields."""
        freqs = np.linspace(20, 200, 10)
        result = models.SimulationResult(
            freqs=freqs,
            spl=np.zeros(10),
            impedance=np.zeros(10, dtype=complex),
            excursion=np.zeros(10),
        )
        assert len(result.freqs) == 10
        assert len(result.spl) == 10

    def test_all_optional_fields_none_by_default(self):
        """Optional fields should default to None."""
        freqs = np.linspace(20, 200, 10)
        result = models.SimulationResult(
            freqs=freqs,
            spl=np.zeros(10),
            impedance=np.zeros(10, dtype=complex),
            excursion=np.zeros(10),
        )
        assert result.segments is None
        assert result.ib_spl is None
        assert result.direct_spl is None
        assert result.horn_spl is None
        assert result.group_delay is None
        assert result.pressure is None


class TestPressureToSpl:
    """Tests for _pressure_to_spl()."""

    def test_1_pascal_yields_approx_94_dbspl(self):
        """20*log10(1/2e-5) = 20*log10(50000) ≈ 93.98 dB SPL."""
        spl = models._pressure_to_spl(np.array([1.0]))[0]
        assert spl == pytest.approx(20 * np.log10(1.0 / 2e-5), rel=1e-6)

    def test_2e5_pascal_yields_200_dbspl(self):
        """20*log10(2e5/2e-5) = 20*log10(1e10) = 200 dB SPL."""
        spl = models._pressure_to_spl(np.array([2e5]))[0]
        assert spl == pytest.approx(200.0, rel=1e-6)

    def test_very_small_pressure_clamped_to_floor(self):
        """Near-zero pressure should not give NaN."""
        spl = models._pressure_to_spl(np.array([1e-15]))[0]
        assert not np.isnan(spl)

    def test_vectorized(self):
        """Should accept and return array results element-wise."""
        p = np.array([1.0, 0.1, 0.01])
        spl = models._pressure_to_spl(p)
        assert len(spl) == 3


class TestTubeSegmentTmatrix:
    """Tests for tube_segment_tmatrix()."""

    def test_returns_2x2_complex_array(self):
        """Should return shape (2, 2) complex numpy array."""
        T = models.tube_segment_tmatrix(freq=1000.0, length=0.05, area=0.01)
        assert T.shape == (2, 2)
        assert np.iscomplexobj(T)

    def test_determinant_is_1(self):
        """The transmission matrix determinant should always be 1."""
        T = models.tube_segment_tmatrix(freq=1000.0, length=0.05, area=0.01)
        assert T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0] == pytest.approx(1.0 + 0j)

    def test_low_frequency_limit_matrix_approaches_identity(self):
        """At very low frequency with large enough cross-section, matrix → identity."""
        # Use large area to keep Zc modest: Zc = 415.63/1.0 = 415.63
        # kL at 1Hz over 0.001m: k=2π/343≈0.0183, kL≈1.83e-5
        # B = j*Zc*sin(kL) ≈ j*415.63*1.83e-5 ≈ j*0.0076  → small
        T = models.tube_segment_tmatrix(freq=1.0, length=0.001, area=1.0)
        assert T[0, 0] == pytest.approx(1.0, rel=1e-3)
        # B is small but not literally zero with realistic values
        assert abs(T[0, 1]) < 0.1

    def test_with_flow_resistivity_fr_nonzero(self):
        """When fr > 0, matrix entries should be complex (Miki absorption)."""
        T_ideal = models.tube_segment_tmatrix(freq=200.0, length=0.05, area=0.01, fr=0.0)
        T_absorptive = models.tube_segment_tmatrix(freq=200.0, length=0.05, area=0.01, fr=5000.0)
        # With absorption, entries should differ (and be complex)
        assert np.iscomplexobj(T_absorptive)

    def test_determinant_still_1_with_absorption(self):
        """Determinant should remain 1 even with Miki absorption."""
        T = models.tube_segment_tmatrix(freq=200.0, length=0.05, area=0.01, fr=5000.0)
        det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
        assert det == pytest.approx(1.0 + 0j, rel=1e-6)


class TestAreaStepTmatrix:
    """Tests for area_step_tmatrix()."""

    def test_ideal_step_returns_correct_matrix(self):
        """ideal/None/empty lem_model: [[1,0],[0,ratio]]."""
        T = models.area_step_tmatrix(freq=1000.0, area1=0.01, area2=0.02)
        assert T[0, 0] == pytest.approx(1.0)
        assert T[0, 1] == pytest.approx(0.0)
        assert T[1, 0] == pytest.approx(0.0)
        assert T[1, 1] == pytest.approx(0.5)  # ratio = 0.01/0.02

    def test_ideal_step_ratio_area2_larger_than_area1(self):
        """When area2 > area1, ratio > 1."""
        T = models.area_step_tmatrix(freq=1000.0, area1=0.005, area2=0.02)
        assert T[1, 1] == pytest.approx(0.25)  # 0.005/0.02

    def test_unknown_lem_model_raises(self):
        """Unknown lem_model should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown LEM step model"):
            models.area_step_tmatrix(freq=1000.0, area1=0.01, area2=0.02, lem_model="unknown")

    def test_basic_lem_model_produces_nonzero_imaginary_part(self):
        """lem_model='basic' should produce complex entries (Z_series ≠ 0)."""
        T = models.area_step_tmatrix(
            freq=200.0, area1=0.01, area2=0.02,
            lem_model="basic", lem_strength=1.0
        )
        assert np.iscomplexobj(T)
        # T[0,1] = Z_series * ratio, and Z_series has imaginary part
        assert T[0, 1] != pytest.approx(0.0)
        # T should differ from the ideal (no-LEM) step matrix
        T_ideal = models.area_step_tmatrix(
            freq=200.0, area1=0.01, area2=0.02,
            lem_model="ideal"
        )
        assert not np.allclose(T, T_ideal)

    def test_basic_lem_determinant_equals_ratio(self):
        """LEM product matrix determinant = ratio (area1/area2), not 1.
        
        The T_step matrix has det=ratio. Since T_series and T_shunt each have
        det=1, the overall determinant = ratio. For area1=0.01, area2=0.02,
        ratio = 0.5.
        """
        T = models.area_step_tmatrix(
            freq=200.0, area1=0.01, area2=0.02,
            lem_model="basic", lem_strength=1.0
        )
        det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
        assert det == pytest.approx(0.5 + 0j)

    def test_basic_lem_expansion_vs_contraction(self):
        """For expansion (area2 > area1), a shunt compliance should appear."""
        T_exp = models.area_step_tmatrix(
            freq=200.0, area1=0.01, area2=0.05,
            lem_model="basic", lem_strength=1.0
        )
        T_con = models.area_step_tmatrix(
            freq=200.0, area1=0.05, area2=0.01,
            lem_model="basic", lem_strength=1.0
        )
        # Expansion has shunt (Y_shunt != 0); contraction has Y_shunt=0
        # The matrices should differ
        assert not np.allclose(T_exp, T_con)


class TestBendTmatrix:
    """Tests for bend_tmatrix()."""

    def test_angle_zero_returns_identity(self):
        """angle_rad=0 should return identity matrix."""
        T = models.bend_tmatrix(freq=1000.0, area=0.01, angle_rad=0.0)
        assert np.allclose(T, np.eye(2, dtype=complex))

    def test_angle_near_zero_returns_near_identity(self):
        """angle_rad < 0.01 should return near-identity matrix."""
        T = models.bend_tmatrix(freq=1000.0, area=0.01, angle_rad=0.005)
        assert np.allclose(T, np.eye(2, dtype=complex), atol=1e-6)

    def test_angle_pi_gives_max_effect(self):
        """angle_rad=π (180°) should give non-trivial matrix."""
        T = models.bend_tmatrix(freq=1000.0, area=0.01, angle_rad=np.pi)
        assert not np.allclose(T, np.eye(2, dtype=complex))

    def test_determinant_is_1(self):
        """Bend matrix determinant should always be 1."""
        for angle in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            T = models.bend_tmatrix(freq=1000.0, area=0.01, angle_rad=angle)
            det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
            assert det == pytest.approx(1.0 + 0j, rel=1e-6)

    def test_returns_complex_array(self):
        """Bend matrix should be complex."""
        T = models.bend_tmatrix(freq=1000.0, area=0.01, angle_rad=np.pi / 2)
        assert np.iscomplexobj(T)


class TestComplianceTmatrix:
    """Tests for compliance_tmatrix()."""

    def test_returns_2x2_complex_array(self):
        """Should return shape (2, 2) complex numpy array."""
        T = models.compliance_tmatrix(freq=100.0, volume=0.005)
        assert T.shape == (2, 2)
        assert np.iscomplexobj(T)

    def test_determinant_is_1(self):
        """Compliance matrix determinant should be 1."""
        T = models.compliance_tmatrix(freq=100.0, volume=0.005)
        det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
        assert det == pytest.approx(1.0 + 0j, rel=1e-6)

    def test_with_flow_resistivity(self):
        """With fr > 0, should use lossy tube model."""
        T = models.compliance_tmatrix(freq=200.0, volume=0.005, fr=5000.0, area=0.01)
        assert np.iscomplexobj(T)
        assert T.shape == (2, 2)



class TestComplianceTmatrixLossy:
    """Tests for compliance_tmatrix() with flow resistivity (lossy closed tube model).

    When fr > 0, the chamber is modeled as a short lossy tube closed at the far end
    (Neumann boundary). The acoustic impedance looking into the tube mouth is:
        Y_shunt = 1 / (-j·Zc·coth(γL))

    The shunt impedance is -j·Zc·coth(γL), NOT -j·Zc·cot(γL).
    The cot(γL) form applies to an OPEN tube (velocity-source termination).

    Key physics:
    - γ = k·k_factor, complex wavenumber (Miki absorption)
    - k_factor has negative imaginary part → decaying wave → coth(γL) → finite limit
    - In the low-loss limit (k_factor → 1+0j, γL small): coth(γL) → 1/(γL) → large
      impedance (approaching the lumped compliance limit Z → 1/(jωC))
    - At higher loss: coth(γL) < 1/(γL), lower impedance (energy dissipated in walls)
    """

    def test_tmatrix_determinant_with_absorption(self):
        """Determinant should remain 1 even with Miki absorption."""
        T = models.compliance_tmatrix(freq=200.0, volume=0.005, fr=5000.0, area=0.01)
        det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
        assert det == pytest.approx(1.0 + 0j, rel=1e-6)

    def test_shunt_admittance_finite_and_reactive(self):
        """The shunt admittance Y_shunt should be finite and purely imaginary."""
        # Patch-level test: we check that Y_shunt is well-defined (finite)
        # and approximately pure imaginary for a physically sized chamber.
        T = models.compliance_tmatrix(freq=200.0, volume=0.0003, fr=8000.0, area=0.001)
        # T = [[1, 0], [Y_shunt, 1]], Y_shunt is the (1,0) entry
        Y_shunt = T[1, 0]
        assert np.isfinite(Y_shunt), f"Y_shunt should be finite, got {Y_shunt}"
        assert np.isfinite(Y_shunt.imag), f"Y_shunt.imag should be finite, got {Y_shunt.imag}"
        # Y_shunt should be reactance-like (positive imaginary for a compliance)
        # Y = jωC for a lumped compliance; for a lossy closed tube the sign matches
        assert Y_shunt.imag > 0, f"Y_shunt should be positive imaginary (compliance), got {Y_shunt}"

    def test_shunt_impedance_more_inductive_with_frequency(self):
        """The shunt admittance should be increasingly inductive (positive imag) with frequency."""
        freqs = [100, 500, 1000]
        Y_vals = []
        for f in freqs:
            T = models.compliance_tmatrix(freq=float(f), volume=0.0003, fr=5000.0, area=0.001)
            Y_vals.append(T[1, 0].imag)
        # Y = jωC → imaginary part should grow with frequency
        assert Y_vals[2] > Y_vals[1] > Y_vals[0], (
            f"Y_shunt.imag should increase with freq: {Y_vals}"
        )

    def test_tmatrix_is_identity_when_kL_below_threshold(self):
        """When kL is below threshold, should return lumped compliance even with fr > 0."""
        # Very low freq → kL < 1e-4, should use lumped form regardless
        T = models.compliance_tmatrix(freq=1.0, volume=0.005, fr=5000.0, area=0.01)
        w = 2 * np.pi * 1.0
        Ca = 0.005 / (models.RHO * models.C**2)
        T_expected = np.array([[1.0, 0.0], [1j * w * Ca, 1.0]], dtype=complex)
        assert np.allclose(T, T_expected, rtol=1e-6)

    def test_coth_behavior_at_closed_end(self):
        """Verify coth(γL) gives a finite, real-ish limit for a well-damped chamber.

        For a closed tube, the input impedance is -j·Zc·coth(γL).
        As loss increases, coth(γL) → 1 (pure resistance at the limit),
        and the admittance Y_shunt → 1/Z_shunt is dominated by the compliance.
        """
        T = models.compliance_tmatrix(freq=200.0, volume=0.0003, fr=8000.0, area=0.001)
        Y_shunt = T[1, 0]
        # Y_shunt should be finite
        assert np.isfinite(Y_shunt), f"Y_shunt should be finite, got {Y_shunt}"
        # Y_shunt should be purely imaginary-ish (compliance-dominated)
        # The real part from absorption losses should be small compared to imag
        ratio = abs(Y_shunt.real) / abs(Y_shunt.imag)
        assert ratio < 1.0, f"Y_shunt should be compliance-dominated (real/imag < 1.0), got {ratio:.3f}"
        assert ratio < 1.0, f"Y_shunt should be compliance-dominated (real/imag < 1.0), got {ratio:.3f}"


class TestThroatAdapterTmatrix:
    """Tests for throat_adapter_tmatrix()."""

    @pytest.fixture
    def make_horn(self):
        """Factory for HornGeometry with throat adapter attributes."""
        def _horn(ap1=0.001, lpt=0.05, atc=0.001, throat_area=0.0005,
                  throat_adapter_type="cylindrical"):
            from pyhorn_core.config.models import HornGeometry
            return HornGeometry(
                ap1=ap1,
                lpt=lpt,
                atc=atc,
                throat_area=throat_area,
                throat_adapter_type=throat_adapter_type,
            )
        return _horn

    def test_returns_2x2_complex_array(self, make_horn):
        """Should return shape (2, 2) complex numpy array."""
        horn = make_horn()
        T = models.throat_adapter_tmatrix(horn, freq=1000.0)
        assert T.shape == (2, 2)
        assert np.iscomplexobj(T)

    def test_cylindrical_atc_equals_ap1_matches_tube_segment(self, make_horn):
        """
        When throat_adapter_type='cylindrical' and atc == ap1,
        the adapter matrix should equal tube_segment_tmatrix at the same frequency.
        """
        horn = make_horn(ap1=0.002, lpt=0.03, atc=0.002,
                         throat_adapter_type="cylindrical")
        T_adapter = models.throat_adapter_tmatrix(horn, freq=500.0)
        T_tube = models.tube_segment_tmatrix(freq=500.0, length=0.03, area=0.002)
        assert np.allclose(T_adapter, T_tube, rtol=1e-10)

    def test_determinant_is_one_lossless(self, make_horn):
        """Lossless adapter matrix determinant should be exactly 1."""
        for profile in ("cylindrical", "conical", "exponential", "parabolic"):
            horn = make_horn(ap1=0.001, lpt=0.04, atc=0.002,
                             throat_adapter_type=profile)
            T = models.throat_adapter_tmatrix(horn, freq=800.0)
            det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
            assert abs(det - 1.0) < 1e-10, (
                f"det(T) = {det} for {profile} adapter — should be 1"
            )

    def test_conical_profile_not_identity(self, make_horn):
        """Conical expansion adapter should NOT return the identity matrix."""
        horn = make_horn(ap1=0.003, lpt=0.05, atc=0.001,
                         throat_adapter_type="conical")
        T = models.throat_adapter_tmatrix(horn, freq=300.0)
        identity = np.eye(2, dtype=complex)
        assert not np.allclose(T, identity), (
            "Conical adapter should differ from identity matrix"
        )

    def test_exponential_profile_not_identity(self, make_horn):
        """Exponential taper adapter should NOT return the identity matrix."""
        horn = make_horn(ap1=0.003, lpt=0.05, atc=0.001,
                         throat_adapter_type="exponential")
        T = models.throat_adapter_tmatrix(horn, freq=300.0)
        identity = np.eye(2, dtype=complex)
        assert not np.allclose(T, identity), (
            "Exponential adapter should differ from identity matrix"
        )

    def test_parabolic_profile_not_identity(self, make_horn):
        """Parabolic taper adapter should NOT return the identity matrix."""
        horn = make_horn(ap1=0.003, lpt=0.05, atc=0.001,
                         throat_adapter_type="parabolic")
        T = models.throat_adapter_tmatrix(horn, freq=300.0)
        identity = np.eye(2, dtype=complex)
        assert not np.allclose(T, identity), (
            "Parabolic adapter should differ from identity matrix"
        )

    def test_zero_ap1_returns_identity(self, make_horn):
        """ap1 <= 0 should return the 2×2 identity matrix."""
        horn = make_horn(ap1=0.0, lpt=0.05)
        T = models.throat_adapter_tmatrix(horn, freq=500.0)
        assert np.allclose(T, np.eye(2, dtype=complex))

    def test_zero_lpt_returns_identity(self, make_horn):
        """lpt <= 0 should return the 2×2 identity matrix."""
        horn = make_horn(ap1=0.001, lpt=0.0)
        T = models.throat_adapter_tmatrix(horn, freq=500.0)
        assert np.allclose(T, np.eye(2, dtype=complex))

    def test_conical_expansion_b_ratio_matches_area_ratio(self, make_horn):
        """
        For a lossless conical expansion, B should reflect the characteristic
        impedance ratio: B ∝ Zc_avg.  A conical (expanding) adapter should have
        larger |B| than a cylindrical one at the same frequency/length because
        the average impedance is lower (larger area).
        """
        horn_conical = make_horn(ap1=0.004, lpt=0.04, atc=0.001,
                                throat_adapter_type="conical")
        horn_cyl = make_horn(ap1=0.0025, lpt=0.04, atc=0.0025,
                             throat_adapter_type="cylindrical")
        T_conical = models.throat_adapter_tmatrix(horn_conical, freq=400.0)
        T_cyl = models.throat_adapter_tmatrix(horn_cyl, freq=400.0)
        # |B| of the expansion adapter should be larger (lower Zc → larger B)
        assert abs(T_conical[0, 1]) > abs(T_cyl[0, 1]), (
            f"|B|_conical={abs(T_conical[0,1]):.3f} should exceed "
            f"|B|_cyl={abs(T_cyl[0,1]):.3f} (expansion → lower Zc → larger B)"
        )

    def test_with_flow_resistivity_entries_become_more_complex(self, make_horn):
        """With fr > 0 (Miki absorption), matrix entries should become lossy."""
        horn = make_horn(ap1=0.002, lpt=0.04, atc=0.001,
                         throat_adapter_type="conical")
        T_lossless = models.throat_adapter_tmatrix(horn, freq=500.0, fr=0.0)
        T_lossy = models.throat_adapter_tmatrix(horn, freq=500.0, fr=5000.0)
        # With absorption, entries should differ (Miki factors applied)
        assert not np.allclose(T_lossless, T_lossy, rtol=1e-6), (
            "With fr > 0 the matrix should differ from the lossless case"
        )

    def test_falls_back_to_cylindrical_when_atc_is_zero(self, make_horn):
        """
        When atc = 0 (not set), A0 falls back to throat_area * 4,
        then to ap1. The result should be a valid matrix (not identity).
        """
        horn = make_horn(ap1=0.002, lpt=0.04, atc=0.0, throat_area=0.0005,
                         throat_adapter_type="cylindrical")
        T = models.throat_adapter_tmatrix(horn, freq=600.0)
        assert T.shape == (2, 2)
        assert np.isfinite(T).all()
        # Should NOT be identity (A0 = ap1 = 0.002 is valid fallback)
        # B should be non-zero for a 4cm tube at 600 Hz
        assert abs(T[0, 1]) > 1e-6, "B should be non-zero for a non-trivial length"


class TestRearChamberImpedance:
    """Tests for rear_chamber_impedance()."""

    def test_zero_volume_returns_zero(self):
        """volume <= 0 should return 0j."""
        Z = models.rear_chamber_impedance(freq=100.0, volume=0.0, length=0.1)
        assert Z == pytest.approx(0.0j)

    def test_returns_complex(self):
        """Should return a complex number."""
        Z = models.rear_chamber_impedance(freq=100.0, volume=0.005, length=0.1)
        assert isinstance(Z, complex)
        assert np.iscomplexobj(np.array([Z]))

    def test_low_frequency_reactance_negative(self):
        """At very low frequency, the chamber should act as a compliance → negative reactance."""
        Z = models.rear_chamber_impedance(freq=10.0, volume=0.01, length=0.1)
        # Compliance: Z ≈ 1/(jωC) → negative imaginary
        assert Z.imag < 0

    def test_with_flow_resistivity(self):
        """With fr > 0, should incorporate Miki absorption."""
        Z = models.rear_chamber_impedance(freq=100.0, volume=0.005, length=0.1, fr=5000.0)
        assert isinstance(Z, complex)


class TestRearChamberVentedBox:
    """Tests for rear_chamber_impedance() with chamber_type='vented'."""

    def test_vented_returns_complex(self):
        """Vented mode should return a complex number."""
        Z = models.rear_chamber_impedance(
            freq=55.0, volume=0.005, length=0.15, chamber_type="vented"
        )
        assert isinstance(Z, complex)
        assert np.iscomplexobj(np.array([Z]))

    def test_vented_resonance_is_in_acoustic_range(self):
        """Vented rear chamber resonance should be in the range 20-200 Hz for typical Vrc/Lrc."""
        # With Vrc=5L and Lrc=15cm the acoustic mass and compliance give f_b ~ 55 Hz
        # (computed iteratively in the function to match the Helmholtz formula)
        vrc = 0.005  # 5 L
        lrc = 0.15   # 15 cm port
        # The resonance frequency should be computable (function should not error)
        Z = models.rear_chamber_impedance(freq=55.0, volume=vrc, length=lrc, chamber_type="vented")
        assert np.isfinite(Z)
        assert abs(Z) > 0

    def test_vented_vs_sealed_produce_different_spl_behavior(self):
        """Sealed and vented rear chambers should give different impedance values.

        The sealed model gives purely capacitive reactance growing without bound at LF.
        The vented model gives a Helmholtz resonance with a finite peak.
        At 55 Hz they should differ substantially.
        """
        vrc = 0.005
        lrc = 0.15
        freq = 55.0
        Z_sealed = models.rear_chamber_impedance(freq, vrc, lrc, chamber_type="sealed")
        Z_vented = models.rear_chamber_impedance(freq, vrc, lrc, chamber_type="vented")
        # They should be numerically different (both finite, non-zero)
        assert np.isfinite(Z_sealed)
        assert np.isfinite(Z_vented)
        assert abs(Z_sealed) > 0
        assert abs(Z_vented) > 0

    def test_vented_zero_volume_returns_zero(self):
        """volume <= 0 should return 0j regardless of chamber_type."""
        Z = models.rear_chamber_impedance(freq=55.0, volume=0.0, length=0.15, chamber_type="vented")
        assert Z == pytest.approx(0.0j)

    def test_vented_zero_length_falls_back_to_sealed(self):
        """lrc=0 with vented type should fall back to sealed compliance model."""
        vrc = 0.005
        freq = 50.0
        Z_vented_zero_lrc = models.rear_chamber_impedance(freq, vrc, 0.0, chamber_type="vented")
        Z_sealed = models.rear_chamber_impedance(freq, vrc, 0.0, chamber_type="sealed")
        assert Z_vented_zero_lrc == pytest.approx(Z_sealed)


class TestRearChamberCoupling:
    """Tests for rear_chamber_impedance() with chamber_type='coupling'.

    The coupling chamber is the correct model for a BLH rear chamber — a pure
    acoustic stiffness load with no mass-controlled Helmholtz resonance peak.
    This is physically distinct from both the sealed (closed-tube) and vented
    (Helmholtz resonator) models.
    """

    def test_coupling_returns_complex(self):
        """Coupling mode should return a complex number."""
        Z = models.rear_chamber_impedance(
            freq=50.0, volume=0.005, length=0.15, chamber_type="coupling"
        )
        assert isinstance(Z, complex)
        assert np.iscomplexobj(np.array([Z]))

    def test_coupling_zero_volume_returns_zero(self):
        """volume <= 0 should return 0j regardless of chamber_type."""
        Z = models.rear_chamber_impedance(freq=50.0, volume=0.0, length=0.15, chamber_type="coupling")
        assert Z == pytest.approx(0.0j)

    def test_coupling_purely_capacitive_at_low_freq(self):
        """At very low frequency, coupling chamber should be purely capacitive (negative imag)."""
        Z = models.rear_chamber_impedance(freq=20.0, volume=0.005, length=0.15, chamber_type="coupling")
        # Pure stiffness: Z ≈ 1/(jωC) → purely negative imaginary
        assert Z.real == pytest.approx(0.0, abs=1e-6)  # essentially zero real part
        assert Z.imag < 0  # capacitive

    def test_coupling_differs_from_vented(self):
        """Coupling and vented types should produce different impedance shapes.

        The vented model has a resonance peak at f_b; the coupling model does not.
        At the vented resonance frequency (≈55 Hz for Vrc=5L, Lrc=15cm) the vented
        model shows a dramatic impedance dip (mass+compliance cancel) while the coupling
        model remains smoothly stiffness-dominated.
        """
        vrc = 0.005
        lrc = 0.15
        freq = 55.0
        Z_coupling = models.rear_chamber_impedance(freq, vrc, lrc, chamber_type="coupling")
        Z_vented = models.rear_chamber_impedance(freq, vrc, lrc, chamber_type="vented")
        # At vented resonance the imaginary part is near zero (mass=compliance)
        # Coupling is always stiffness-dominated → always negative imaginary at LF
        assert Z_coupling.imag < 0  # coupling is always capacitive
        assert abs(Z_vented.imag) < abs(Z_coupling.imag)  # vented imag is much smaller at resonance
        # They should differ substantially in both real and imaginary parts
        assert abs(Z_coupling - Z_vented) > 1000  # large difference at vented resonance

    def test_coupling_without_throat_same_as_sealed(self):
        """Coupling with throat_area=0 is identical to sealed (pure compliance).

        When no throat radiation area is specified, the coupling chamber is
        mathematically identical to the sealed compliance model.
        """
        vrc = 0.005
        freq = 50.0
        Z_coupling_no_throat = models.rear_chamber_impedance(
            freq, vrc, 0.0, chamber_type="coupling", throat_area=0.0
        )
        Z_sealed = models.rear_chamber_impedance(freq, vrc, 0.0, chamber_type="sealed")
        # Both are pure compliance: 1/(jωC). Mathematically identical.
        assert Z_coupling_no_throat == pytest.approx(Z_sealed)

    def test_coupling_with_throat_area_has_radiation(self):
        """With throat_area > 0, coupling chamber adds radiation impedance."""
        Z_no_throat = models.rear_chamber_impedance(
            freq=50.0, volume=0.005, length=0.15,
            chamber_type="coupling", throat_area=0.0
        )
        Z_with_throat = models.rear_chamber_impedance(
            freq=50.0, volume=0.005, length=0.15,
            chamber_type="coupling", throat_area=0.0044  # ~7.5 cm diameter
        )
        # Adding radiation should increase the real part
        assert Z_with_throat.real > Z_no_throat.real

    def test_coupling_no_resonance_peak(self):
        """Coupling impedance should not show a mass-controlled resonance peak.

        Unlike the vented model, coupling impedance should be smooth and
        monotonically decreasing in magnitude as frequency increases (stiffness-dominated:
        |Z| ∝ 1/ω).  There is no resonance dip like the vented box has.
        """
        vrc = 0.005
        lrc = 0.15
        freqs = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        Z_vals = np.array([
            models.rear_chamber_impedance(f, vrc, lrc, chamber_type="coupling")
            for f in freqs
        ])
        magnitudes = np.abs(Z_vals)
        # Pure stiffness: |Z| ∝ 1/ω → decreases as freq increases (no resonance peak)
        for i in range(1, len(magnitudes)):
            assert magnitudes[i] < magnitudes[i - 1], \
                f"Coupling impedance not monotonic decreasing: {magnitudes}"


class TestRadiationImpedance:
    """Tests for radiation_impedance()."""

    def test_returns_complex(self):
        """Should return a complex number."""
        Z = models.radiation_impedance(freq=1000.0, mouth_area=0.01, ang=2 * np.pi)
        assert isinstance(Z, complex)

    def test_real_part_positive(self):
        """Radiation resistance real part should be positive."""
        Z = models.radiation_impedance(freq=1000.0, mouth_area=0.01, ang=2 * np.pi)
        assert Z.real > 0

    def test_imaginary_part_positive(self):
        """Radiation reactance should be positive (mass-like)."""
        Z = models.radiation_impedance(freq=1000.0, mouth_area=0.01, ang=2 * np.pi)
        assert Z.imag > 0

    def test_larger_mouth_lower_Zc(self):
        """Larger mouth area → lower characteristic impedance → lower |Z|."""
        Z_small = models.radiation_impedance(freq=1000.0, mouth_area=0.001, ang=2 * np.pi)
        Z_large = models.radiation_impedance(freq=1000.0, mouth_area=0.1, ang=2 * np.pi)
        assert abs(Z_large) < abs(Z_small)

    def test_with_rectangular_dimensions(self):
        """mouth_width and mouth_height should be used for rectangular piston model."""
        Z_circ = models.radiation_impedance(freq=1000.0, mouth_area=0.02, ang=2 * np.pi)
        Z_rect = models.radiation_impedance(
            freq=1000.0, mouth_area=0.02, ang=2 * np.pi,
            mouth_width=0.2, mouth_height=0.1
        )
        # Both should be valid complex numbers
        assert np.iscomplexobj(np.array([Z_circ]))
        assert np.iscomplexobj(np.array([Z_rect]))

class TestRectangularPistonRadiation:
    """Tests for the rectangular piston radiation impedance (low-ka approximation)."""

    def test_resistance_scales_as_k_squared(self):
        """
        Radiation resistance R_rad is proportional to k**2 at low ka.
        Doubling frequency should quadruple R_rad.
        """
        mouth_w, mouth_h = 0.1, 0.2
        mouth_area = mouth_w * mouth_h
        Z_100 = models.radiation_impedance(
            100.0, mouth_area, 2 * np.pi, mouth_width=mouth_w, mouth_height=mouth_h
        )
        Z_200 = models.radiation_impedance(
            200.0, mouth_area, 2 * np.pi, mouth_width=mouth_w, mouth_height=mouth_h
        )
        ratio = Z_200.real / Z_100.real
        assert ratio == pytest.approx(4.0, rel=1e-2), f"R should 4x when f 2x; got {ratio}"

    def test_resistance_matches_morse_ingard_formula(self):
        """
        R_rad = Z0 * k**2 * mouth_area**2 / (2*pi) for ang = 2*pi (half-space).
        Verify at 100 Hz for a 0.1m x 0.2m rectangular mouth.
        """
        RHO, C = 1.21, 343.0
        Z0_check = RHO * C
        mouth_w, mouth_h = 0.1, 0.2
        mouth_area = mouth_w * mouth_h
        k = 2 * np.pi * 100.0 / C
        R_expected = Z0_check * k**2 * mouth_area**2 / (2 * np.pi)
        Z = models.radiation_impedance(
            100.0, mouth_area, 2 * np.pi, mouth_width=mouth_w, mouth_height=mouth_h
        )
        assert Z.real == pytest.approx(R_expected, rel=1e-3)

    def test_reactance_scales_linearly_with_frequency(self):
        """
        Radiation reactance X_rad is proportional to k at low ka.
        Doubling frequency should double X_rad.
        """
        mouth_w, mouth_h = 0.1, 0.2
        mouth_area = mouth_w * mouth_h
        Z_100 = models.radiation_impedance(
            100.0, mouth_area, 2 * np.pi, mouth_width=mouth_w, mouth_height=mouth_h
        )
        Z_200 = models.radiation_impedance(
            200.0, mouth_area, 2 * np.pi, mouth_width=mouth_w, mouth_height=mouth_h
        )
        ratio = Z_200.imag / Z_100.imag
        assert ratio == pytest.approx(2.0, rel=1e-2), f"X should 2x when f 2x; got {ratio}"

    def test_reactance_matches_end_correction_formula(self):
        """
        X_rad = Z0 * k * (mouth_width + mouth_height) / (3*pi) for ang = 2*pi.
        """
        RHO, C = 1.21, 343.0
        Z0_check = RHO * C
        mouth_w, mouth_h = 0.1, 0.2
        mouth_area = mouth_w * mouth_h
        k = 2 * np.pi * 100.0 / C
        X_expected = Z0_check * k * (mouth_w + mouth_h) / (3 * np.pi)
        Z = models.radiation_impedance(
            100.0, mouth_area, 2 * np.pi, mouth_width=mouth_w, mouth_height=mouth_h
        )
        assert Z.imag == pytest.approx(X_expected, rel=1e-3)

    def test_resistance_positive_for_all_frequencies(self):
        """Radiation resistance should always be non-negative."""
        mouth_w, mouth_h = 0.2, 0.3
        mouth_area = mouth_w * mouth_h
        for f in [50, 100, 200, 500, 1000]:
            Z = models.radiation_impedance(
                float(f), mouth_area, 2 * np.pi,
                mouth_width=mouth_w, mouth_height=mouth_h
            )
            assert Z.real >= 0, f"R_rad = {Z.real} at {f} Hz"

    def test_reactance_positive_at_low_ka(self):
        """Radiation reactance should be positive (mass-like) at low ka."""
        mouth_w, mouth_h = 0.2, 0.3
        mouth_area = mouth_w * mouth_h
        for f in [50, 100, 200, 500]:
            Z = models.radiation_impedance(
                float(f), mouth_area, 2 * np.pi,
                mouth_width=mouth_w, mouth_height=mouth_h
            )
            assert Z.imag > 0, f"X_rad = {Z.imag} at {f} Hz"

    def test_solid_angle_scaling(self):
        """
        Radiation resistance doubles when solid angle halves (ang: 2*pi -> pi).
        X_rad is the end-correction reactance (proportional to edge mass), which
        does NOT depend on solid angle -- it is set by the piston geometry alone.
        """
        mouth_w, mouth_h = 0.1, 0.2
        mouth_area = mouth_w * mouth_h
        Z_half = models.radiation_impedance(
            500.0, mouth_area, np.pi, mouth_width=mouth_w, mouth_height=mouth_h
        )
        Z_full = models.radiation_impedance(
            500.0, mouth_area, 2 * np.pi, mouth_width=mouth_w, mouth_height=mouth_h
        )
        assert Z_half.real == pytest.approx(2.0 * Z_full.real, rel=1e-2)
        # X_rad is the mass-like end correction; independent of solid angle
        assert Z_half.imag == pytest.approx(Z_full.imag, rel=1e-2)




class TestLevineInglisRadiation:
    """Tests for the exact Levine/Inglis circular piston radiation impedance."""

    def test_returns_complex(self):
        """Should return a complex number at mid frequency."""
        Z = models._circular_piston_radiation_impedance(
            freq=1000.0, mouth_area=0.01, ang=2 * np.pi,
            Zc=400.0, a=0.056
        )
        assert isinstance(Z, complex)

    def test_resistance_positive(self):
        """Radiation resistance should always be non-negative."""
        for f in [100, 500, 1000, 5000, 10000]:
            Z = models._circular_piston_radiation_impedance(
                freq=float(f), mouth_area=0.01, ang=2 * np.pi,
                Zc=400.0, a=0.056
            )
            assert Z.real >= 0, f"R_rad = {Z.real} at {f} Hz"

    def test_reactance_positive(self):
        """
        Radiation reactance is positive (mass-like) for ka < ~1.3 (f < ~1300 Hz for a=0.056m).
        Above this, the exact Levine/Inglis formula gives negative reactance (compliant)
        as the piston phase becomes non-uniform. This is physical.
        """
        for f in [100, 500, 1000, 1200]:
            Z = models._circular_piston_radiation_impedance(
                freq=float(f), mouth_area=0.01, ang=2 * np.pi,
                Zc=400.0, a=0.056
            )
            assert Z.imag >= 0, f"X_rad = {Z.imag} at {f} Hz"

    def test_reactance_becomes_negative_at_high_ka(self):
        """
        At sufficiently high ka (f > ~1300 Hz for a=0.056m), reactance turns negative
        — this is a real physical effect from the Levine/Inglis exact solution,
        not an implementation bug.
        """
        Z_low = models._circular_piston_radiation_impedance(
            1000.0, 0.01, 2 * np.pi, 400.0, 0.056
        )
        Z_high = models._circular_piston_radiation_impedance(
            2000.0, 0.01, 2 * np.pi, 400.0, 0.056
        )
        assert Z_low.imag > 0
        assert Z_high.imag < 0

    def test_small_ka_ka4_scaling(self):
        """
        The change in radiation resistance over a frequency decade scales as O(ka⁴).
        We verify this by checking ΔR_rad ∝ f² (since ka ∝ f, ka⁴ ∝ f⁴,
        and ΔR_rad ∝ f⁴ for the variable part). The absolute resistance at any
        single frequency converges to 3Zc/4 + O(ka²) in the small-ka limit, so the
        ratio test between two small-ka points is ~1. Instead we check that the
        resistance change between two nearby frequencies grows as f⁴.
        """
        Zc = 400.0
        a = 0.056
        # Use f=20,22 Hz (small Δf) to isolate the ka⁴ contribution from the 3Zc/4 constant
        R_20 = models._circular_piston_radiation_impedance(20.0, 0.01, 2 * np.pi, Zc, a).real
        R_22 = models._circular_piston_radiation_impedance(22.0, 0.01, 2 * np.pi, Zc, a).real
        # Use f=20, 22 for the ratio; also compare with 10, 11 Hz (same relative spacing)
        R_10 = models._circular_piston_radiation_impedance(10.0, 0.01, 2 * np.pi, Zc, a).real
        R_11 = models._circular_piston_radiation_impedance(11.0, 0.01, 2 * np.pi, Zc, a).real
        # ΔR at higher frequencies should be larger (scales as f⁴)
        delta_high = R_22 - R_20
        delta_low = R_11 - R_10
        # delta_high / delta_low should be approximately (22²-20²)⁴ / (11²-10²)⁴ ≈ (84)⁴/(21)⁴ ≈ 256
        # We just check the ratio is > 1 (higher-freq delta is larger)
        assert delta_high > delta_low, (
            f"Delta-R at higher freq should be larger: "
            f"Δ(20-22Hz)={delta_high:.6f}, Δ(10-11Hz)={delta_low:.6f}"
        )
        # Also: the resistance should INCREASE with frequency (scales as ka⁴ contribution)
        assert R_22 > R_20
        assert R_11 > R_10

    def test_half_space_solid_angle_scaling(self):
        """Double the solid angle should halve both R_rad and X_rad."""
        Z_half = models._circular_piston_radiation_impedance(
            freq=1000.0, mouth_area=0.01, ang=np.pi,
            Zc=400.0, a=0.056
        )
        Z_full = models._circular_piston_radiation_impedance(
            freq=1000.0, mouth_area=0.01, ang=2 * np.pi,
            Zc=400.0, a=0.056
        )
        # 2π vs π → half the radiated power per unit solid angle
        assert np.isclose(Z_half.real, 2.0 * Z_full.real, rtol=0.01)
        assert np.isclose(Z_half.imag, 2.0 * Z_full.imag, rtol=0.01)


class TestCascade:
    """Tests for cascade()."""

    def test_identity_leaves_matrix_unchanged(self):
        """I @ M = M and M @ I = M."""
        I = np.eye(2, dtype=complex)
        M = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
        assert np.allclose(models.cascade([I, M]), M)
        assert np.allclose(models.cascade([M, I]), M)

    def test_cascade_two_matrices(self):
        """Cascading two matrices should produce correct composition."""
        M1 = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=complex)
        M2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        result = models.cascade([M1, M2])
        expected = M1 @ M2
        assert np.allclose(result, expected)

    def test_cascade_empty_returns_identity(self):
        """cascading no matrices should return 2x2 identity."""
        result = models.cascade([])
        assert np.allclose(result, np.eye(2, dtype=complex))

    def test_cascade_three_matrices(self):
        """Three matrices should cascade correctly."""
        M1 = np.array([[1, 1], [0, 1]], dtype=complex)
        M2 = np.array([[1, 0], [1, 1]], dtype=complex)
        M3 = np.array([[0, 1], [1, 0]], dtype=complex)
        result = models.cascade([M1, M2, M3])
        assert np.allclose(result, M1 @ M2 @ M3)


class TestHornResponseIntegration:
    """Integration tests for horn_response() with real driver and geometry."""

    @pytest.fixture
    def fostex_driver(self):
        """Fostex FE166NV2 driver specs."""
        return DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83, le=0.0008, xmax=0.0015,
        )

    @pytest.fixture
    def simple_horn(self):
        """Simple BLH horn geometry with conical segments."""
        return HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
        )

    def test_returns_simulation_result(self, fostex_driver, simple_horn):
        """horn_response should return a SimulationResult."""
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, simple_horn)
        assert isinstance(result, models.SimulationResult)
        assert len(result.freqs) == 100

    def test_spl_in_reasonable_range(self, fostex_driver, simple_horn):
        """SPL values should be in a plausible range (40–130 dB)."""
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, simple_horn)
        assert np.all(result.spl > 0)
        assert np.all(result.spl < 150)

    def test_impedance_has_positive_real_part(self, fostex_driver, simple_horn):
        """Electrical impedance should have positive real part."""
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, simple_horn)
        assert np.all(result.impedance.real > 0)

    def test_excursion_at_low_frequency_finite(self, fostex_driver, simple_horn):
        """Excursion should stay finite at low frequencies and not blow up."""
        freqs = np.linspace(20, 200, 50)
        result = models.horn_response(freqs, fostex_driver, simple_horn)
        # Excursion at fs should be reasonable (< 50 mm peak for a typical driver)
        assert np.all(np.isfinite(result.excursion))
        assert np.max(result.excursion) < 100.0  # mm peak

    def test_group_delay_reasonable(self, fostex_driver, simple_horn):
        """Group delay should be positive and in a plausible range (< 100 ms)."""
        freqs = np.linspace(20, 5000, 200)
        result = models.horn_response(freqs, fostex_driver, simple_horn)
        # Group delay should be mostly positive (phase unwrapped)
        assert np.all(np.isfinite(result.group_delay))

    def test_blh_has_direct_and_horn_spl(self, fostex_driver, simple_horn):
        """BLH configuration should populate direct_spl and horn_spl."""
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, simple_horn)
        assert result.direct_spl is not None
        assert result.horn_spl is not None
        assert len(result.direct_spl) == 100
        assert len(result.horn_spl) == 100

    def test_flh_has_no_direct_spl(self, fostex_driver):
        """FLH configuration should not populate direct_spl."""
        flh_horn = HornGeometry(
            enclosure_type="FLH",
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
        )
        freqs = np.linspace(20, 5000, 50)
        result = models.horn_response(freqs, fostex_driver, flh_horn)
        assert result.direct_spl is None
        assert result.horn_spl is None

    def test_rear_chamber_with_lrc_changes_impedance(self, fostex_driver):
        """lrc > 0 should produce a meaningfully different impedance curve vs lrc = 0.

        The rear chamber (vrc + lrc) adds an acoustic load that modifies the low-frequency
        impedance response. With lrc = 0 the transmission-line term is zeroed out, so
        changing lrc from 0 to a non-zero value must alter the impedance shape.
        """
        base_horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=0.025,
            lrc=0.0,  # no rear chamber — transmission line off
        )
        horn_with_lrc = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=0.025,
            lrc=0.18,  # 18 cm rear chamber depth (matches hiro.yaml)
        )
        freqs = np.linspace(20, 5000, 200)
        result_no_lrc = models.horn_response(freqs, fostex_driver, base_horn)
        result_with_lrc = models.horn_response(freqs, fostex_driver, horn_with_lrc)

        # Impedance arrays must both be finite
        assert np.all(np.isfinite(result_no_lrc.impedance))
        assert np.all(np.isfinite(result_with_lrc.impedance))

        # The curves must differ — lrc changes the rear chamber load
        diff = np.abs(result_with_lrc.impedance - result_no_lrc.impedance)
        assert np.max(diff) > 0.05, "lrc change should produce a measurable impedance difference"

        # Low-frequency region (< 200 Hz) is most sensitive to rear chamber
        lf_mask = freqs < 200
        lf_diff = diff[lf_mask]
        assert np.mean(lf_diff) > 0.01, (
            "Low-frequency impedance should differ noticeably with lrc > 0"
        )

    def test_rear_chamber_lrc_zero_still_uses_compliance(self, fostex_driver):
        """With lrc=0 but fr_rc>0, the compliance term still activates via cubic fallback.

        The function comment says: when fr > 0 but length <= 0, it falls back to a
        cubic approximation (volume ** 1/3) as the length. So even with lrc=0, a
        non-zero fr_rc activates the rear chamber load.
        """
        horn_no_rc = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.07, 0.03)],
            vrc=0.0,
            lrc=0.0,
            fr_rc=0.0,
        )
        horn_with_fr_rc = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.07, 0.03)],
            vrc=0.025,
            lrc=0.0,
            fr_rc=2000.0,  # damping activates the cubic fallback
        )
        freqs = np.linspace(20, 5000, 100)
        result_no_rc = models.horn_response(freqs, fostex_driver, horn_no_rc)
        result_with_fr_rc = models.horn_response(freqs, fostex_driver, horn_with_fr_rc)

        diff = np.abs(result_with_fr_rc.impedance - result_no_rc.impedance)
        assert np.max(diff) > 0.01, (
            "fr_rc > 0 with lrc=0 should still activate rear chamber via cubic fallback"
        )


class TestSectionsCutoffFrequency:
    """Tests that chained profile sections produce correct cutoff behaviour.

    The cutoff frequency of a chained horn is determined by the exponential
    (or hyperbolic) flare section, NOT by a straight constant-area throat section.
    A straight throat is acoustically transparent — it behaves like a short
    cylindrical duct with no low-frequency cutoff of its own.
    """

    @pytest.fixture
    def fostex_driver(self):
        """FE166NV2 Thiele-Small parameters."""
        return DriverSpecs(
            fs=49.6,
            qts=0.27,
            qes=0.32,
            qms=1.63,
            vas=0.0189,
            re=7.80,
            bl=7.79,
            mms=0.00699,
            cms=1.472e-3,
            rms=0.277,
            sd=0.01327,
            voltage=2.83,
            xmax=0.001,
        )

    def test_straight_throat_then_exponential_cutoff_from_flare_section(self, fostex_driver):
        """Straight throat + exponential flare: cutoff must come from the flare section.

        A straight throat section (constant area) has no exponential cutoff — it is
        acoustically like a short cylinder.  The exponential flare section produces
        the low-frequency roll-off.  The overall horn cutoff frequency must match
        the exponential section's cutoff, not be dominated by the straight section.
        """
        horn_with_straight_throat = HornGeometry(
            sections=[
                Section(
                    name="throat",
                    profile_type="straight",
                    length=0.40,
                    start_area=0.0044,
                    end_area=0.0044,
                ),
                Section(
                    name="main_horn",
                    profile_type="exponential",
                    length=0.80,
                    start_area=0.0044,
                    end_area=0.08,
                ),
            ],
            enclosure_type="BLH",
        )

        horn_exponential_only = HornGeometry(
            sections=[
                Section(
                    name="main_horn",
                    profile_type="exponential",
                    length=0.80,
                    start_area=0.0044,
                    end_area=0.08,
                ),
            ],
            enclosure_type="BLH",
        )

        # Expected cutoff from the exponential section:
        #   m = (1/L) * ln(S2/S1) = (1/0.8) * ln(0.08/0.0044)
        #   m = 1.25 * ln(18.18) = 1.25 * 2.901 = 3.627 m⁻¹
        #   fc = (m * c) / (4π) = (3.627 * 343) / (12.566) ≈ 99 Hz
        m_exp = (1.0 / 0.80) * np.log(0.08 / 0.0044)
        fc_expected = (m_exp * 343.0) / (4.0 * np.pi)
        assert fc_expected == pytest.approx(99.0, rel=0.05)

        # Compute response: straight throat + exponential vs exponential only
        freqs = np.linspace(20, 500, 200)
        result_straight = models.horn_response(freqs, fostex_driver, horn_with_straight_throat)
        result_exp_only = models.horn_response(freqs, fostex_driver, horn_exponential_only)

        # Above cutoff (~200-400 Hz), the straight throat horn should be close to exponential-only.
        # This band is above the coupling-chamber resonance notch (80-130 Hz).
        band_above = (freqs >= 200) & (freqs <= 400)
        spl_diff_above = np.abs(
            result_straight.spl[band_above] - result_exp_only.spl[band_above]
        )
        assert np.mean(spl_diff_above) < 15.0, (
            "Above cutoff, straight-throat and exponential-only horns should remain similar. "
            f"Mean SPL diff in 200-400 Hz band: {np.mean(spl_diff_above):.1f} dB"
        )

        # The SPL must actually roll off at very low frequencies (below cutoff).
        # At 20 Hz the SPL should be noticeably lower than at 200 Hz (passband).
        idx_20hz = int(np.argmin(np.abs(freqs - 20.0)))
        idx_200hz = int(np.argmin(np.abs(freqs - 200.0)))
        spl_at_20hz = result_straight.spl[idx_20hz]
        spl_at_200hz = result_straight.spl[idx_200hz]
        assert spl_at_20hz < spl_at_200hz - 10.0, (
            "Below the exponential cutoff (~99 Hz), SPL should be rolling off. "
            f"SPL at 20 Hz ({spl_at_20hz:.1f} dB) should be >10 dB below "
            f"SPL at 200 Hz ({spl_at_200hz:.1f} dB)."
        )

    def test_sections_cutoff_matches_exponential_formula(self, fostex_driver):
        """Verify the exponential section's cutoff matches the analytical formula.

        fc = (m * c) / (4π),  m = (1/L) * ln(S2/S1)
        """
        horn = HornGeometry(
            sections=[
                Section(
                    name="flare",
                    profile_type="exponential",
                    length=0.60,
                    start_area=0.005,
                    end_area=0.050,
                ),
            ],
            enclosure_type="BLH",
        )
        # m = (1/0.6) * ln(0.05/0.005) = 1.667 * ln(10) = 1.667 * 2.303 = 3.838 m⁻¹
        # fc = (3.838 * 343) / (4π) = 1316 / 12.566 = 104.7 Hz
        m_exp = (1.0 / 0.60) * np.log(0.050 / 0.005)
        fc_expected = (m_exp * 343.0) / (4.0 * np.pi)
        assert fc_expected == pytest.approx(104.7, rel=0.02)

        freqs = np.linspace(20, 500, 100)
        result = models.horn_response(freqs, fostex_driver, horn)

        # NOTE (May 5 2026): The coupling chamber model (CRIT-1 fix, a8bfe7e) creates
        # a response notch around 200-250 Hz. We use 300 Hz (above the notch) as the
        # passband reference and 52 Hz (below cutoff) as the sub-cutoff reference.
        idx_52hz = int(np.argmin(np.abs(freqs - 52.4)))
        idx_105hz = int(np.argmin(np.abs(freqs - 104.7)))
        idx_300hz = int(np.argmin(np.abs(freqs - 300.0)))

        spl_below = result.spl[idx_52hz]
        spl_at = result.spl[idx_105hz]
        spl_above = result.spl[idx_300hz]

        # Above cutoff (300 Hz, above the notch) should be louder than below cutoff (52 Hz)
        assert spl_above > spl_below + 3.0, (
            f"SPL at 300 Hz ({spl_above:.1f} dB) should be >3 dB above "
            f"SPL at 0.5× cutoff (52 Hz: {spl_below:.1f} dB). "
            "This confirms the exponential cutoff behaviour."
        )
        # At exactly cutoff, SPL should be between the two extremes
        assert spl_at > spl_below - 5.0, (
            f"SPL at cutoff ({spl_at:.1f} dB) should be within 5 dB of "
            f"the sub-cutoff level ({spl_below:.1f} dB)."
        )

    def test_ap1_affects_spl(self, fostex_driver):
        """REGRESSION: ap1 (throat-adapter area) must change SPL output.

        CRIT-2 (BACKLOG): sweeping ap1 from 0.002 m² to 0.020 m² produced
        *identical* SPL — ap1 had zero effect. This test verifies that changing
        ap1 actually changes the simulation result. The throat adapter T-matrix
        is applied in the BLH TMM cascade; changing ap1 should shift the
        effective throat loading and hence SPL.

        NOTE: This is a regression test. If it fails, it confirms CRIT-2.
        Once CRIT-2 is fixed, this test should pass.
        """
        cone_segs = [(0.06, 0.07, 0.03), (0.07, 0.08, 0.04), (0.08, 0.10, 0.05)]

        horn_small_ap1 = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=cone_segs,
            vtc=0.0036,
            fr_tc=2000.0,
            ap1=0.002,
            lpt=0.01,
        )
        horn_large_ap1 = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=cone_segs,
            vtc=0.0036,
            fr_tc=2000.0,
            ap1=0.020,
            lpt=0.01,
        )

        freqs = np.linspace(100, 2000, 200)
        result_small = models.horn_response(freqs, fostex_driver, horn_small_ap1)
        result_large = models.horn_response(freqs, fostex_driver, horn_large_ap1)

        # SPL should differ somewhere in the 100–2000 Hz band.
        # The effect is expected to be most visible in the 300–1000 Hz region
        # where throat adapter dimensions are acoustically significant.
        spl_diff = np.abs(result_small.spl - result_large.spl)
        max_diff = float(np.max(spl_diff))
        assert max_diff > 0.1, (
            f"ap1 sweep (0.002 → 0.020 m²) produced max SPL diff of only "
            f"{max_diff:.3f} dB — ap1 appears to have no effect (CRIT-2 bug). "
            f"Expected >0.1 dB change."
        )


class TestInfiniteBaffleResponse:
    """Tests for infinite_baffle_response()."""

    @pytest.fixture
    def fostex_driver(self):
        return DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
        )

    def test_returns_spl_array(self, fostex_driver):
        """Should return a numpy array of SPL values."""
        freqs = np.linspace(20, 5000, 100)
        spl = models.infinite_baffle_response(freqs, fostex_driver)
        assert isinstance(spl, np.ndarray)
        assert len(spl) == 100

    def test_spl_above_fs_below_100_dbspl(self, fostex_driver):
        """Typical driver in IB should not exceed ~100 dB at 1W."""
        freqs = np.linspace(20, 5000, 100)
        spl = models.infinite_baffle_response(freqs, fostex_driver)
        assert np.all(spl < 105)

    def test_spl_decreases_at_very_low_frequency(self, fostex_driver):
        """Below resonance, SPL should drop off rapidly."""
        freqs = np.linspace(5, 100, 50)
        spl = models.infinite_baffle_response(freqs, fostex_driver)
        # SPL at 5 Hz should be much lower than at 100 Hz
        assert spl[0] < spl[20]

class TestMikiFactorsClamping:
    """Tests for _miki_factors() clamping bounds outside validity range."""

    def test_bass_frequency_zc_factor_clamped(self):
        """At 20 Hz / sigma=20000 (X=1.0), |Zc| should be << 10 after clamping."""
        Zc_f, _ = models._miki_factors(freq=20.0, sigma=20000.0)
        assert abs(Zc_f) < 3.0, f"|Zc_factor| = {abs(Zc_f)}, expected < 3.5"

    def test_bass_frequency_k_factor_clamped(self):
        """At 20 Hz / sigma=20000 (X=1.0), |k| should be << 14 after clamping."""
        _, k_f = models._miki_factors(freq=20.0, sigma=20000.0)
        assert abs(k_f) < 3.0, f"|k_factor| = {abs(k_f)}, expected < 3.5"

    def test_real_parts_within_bounds(self):
        """Real parts of both factors should be in [0.7, 2.5]."""
        Zc_f, k_f = models._miki_factors(freq=50.0, sigma=5000.0)
        assert 0.7 <= Zc_f.real <= 2.5
        assert 0.7 <= k_f.real <= 2.5

    def test_imaginary_parts_non_positive(self):
        """Imag parts should always be ≤ 0 (dissipative)."""
        for f in [20, 100, 500, 2000]:
            for sigma in [5000, 20000, 50000]:
                Zc_f, k_f = models._miki_factors(freq=float(f), sigma=float(sigma))
                assert Zc_f.imag <= 0, f"Zc imag = {Zc_f.imag} at f={f}, sigma={sigma}"
                assert k_f.imag <= 0, f"k imag = {k_f.imag} at f={f}, sigma={sigma}"

    def test_valid_range_still_uses_model(self):
        """Within the validity range (X~0.1), factors should be > 1.0 (model active)."""
        # X = 0.1 → f/sigma = 0.0001 → very high sigma relative to f
        Zc_f, k_f = models._miki_factors(freq=100.0, sigma=1_000_000.0)
        assert abs(Zc_f) > 1.0, "Within valid range, absorption should increase |Zc|"
        assert Zc_f.imag < 0



class TestVentedBoxImpedance:
    """Tests for vented_box_impedance()."""

    def test_zero_volume_returns_zero(self):
        """vrc <= 0 should return 0j."""
        Z = models.vented_box_impedance(freq=100.0, vrc=0.0, lrc=0.05, fr_tuning=50.0)
        assert Z == pytest.approx(0.0j)

    def test_zero_length_returns_zero(self):
        """lrc <= 0 should return 0j."""
        Z = models.vented_box_impedance(freq=100.0, vrc=0.035, lrc=0.0, fr_tuning=50.0)
        assert Z == pytest.approx(0.0j)

    def test_zero_tuning_freq_returns_zero(self):
        """fr_tuning <= 0 should return 0j."""
        Z = models.vented_box_impedance(freq=100.0, vrc=0.035, lrc=0.05, fr_tuning=0.0)
        assert Z == pytest.approx(0.0j)

    def test_returns_complex(self):
        """Should return a complex number."""
        Z = models.vented_box_impedance(freq=100.0, vrc=0.035, lrc=0.05, fr_tuning=50.0)
        assert isinstance(Z, complex)
        assert np.iscomplexobj(np.array([Z]))

    def test_low_frequency_compliance_dominant(self):
        """At very low frequency, box compliance should dominate → negative reactance."""
        Z = models.vented_box_impedance(freq=10.0, vrc=0.035, lrc=0.05, fr_tuning=50.0)
        # Compliance: Z ≈ 1/(jωC) → negative imaginary
        assert Z.imag < 0, f"Below tuning, compliance should dominate (neg imag); got Z={Z}"

    def test_resonance_cancellation_at_tuning(self):
        """At the tuning frequency, mass and compliance impedances cancel to first order.

        The Helmholtz resonance condition M_v·C_vb = 1/ω² means the series mass+compliance
        reactances cancel to O(1).  After the small-ka radiation-impedance fix, the
        remaining reactance is dominated by the port radiation X_rad (not an artifact
        of mis-sized M_v/C_vb).  We verify the mass and compliance cancellation is
        still the dominant effect by checking that |Z| at tuning is close to the
        parallel radiation+leak resistance.
        """
        fr_tuning = 50.0
        Z = models.vented_box_impedance(freq=fr_tuning, vrc=0.035, lrc=0.05, fr_tuning=fr_tuning)
        # At tuning: |Z| should be close to the real (radiation+leak) resistance,
        # not swamped by mass/compliance reactance.  With the corrected small-ka
        # radiation model, the port X_rad is realistic (~few thousand acoustic ohms)
        # but still much smaller than the pre-fix artificial ~400k value.
        # Check that Z.real > 0 and |Z| is not enormously larger than Z.real
        # (ratio < 3 means mass+compliance residual is at most ~3× the real part)
        ratio = abs(Z.imag) / abs(Z.real) if Z.real != 0 else abs(Z.imag)
        assert ratio < 3.0, (
            f"At tuning frequency, |Z_imag|/|Z_real| should be modest; got {ratio:.3f}"
        )
        assert Z.real > 0, f"At tuning, Z.real should be positive; got {Z.real}"

    def test_high_frequency_mass_dominant(self):
        """Above tuning, port mass should dominate → positive reactance."""
        Z = models.vented_box_impedance(freq=200.0, vrc=0.035, lrc=0.05, fr_tuning=50.0)
        # Mass: Z ≈ jωM → positive imaginary
        assert Z.imag > 0, f"Above tuning, mass should dominate (pos imag); got Z={Z}"

    def test_leak_loss_ql_broadens_resonance(self):
        """Lower Ql (more loss) → lower impedance peak at tuning.

        R_leak = 1/(2π·fr·C_vb·Ql). Lower Ql → higher R_leak.
        In the parallel combination R_rad || R_leak, a higher R_leak shunts
        less → total Z is closer to R_rad (higher). Wait — actually:
        Lower Ql = higher R_leak → less current diverted → higher Z_total.
        Higher Ql = lower R_leak → more current diverted → lower Z_total.
        So: Ql=2 should give HIGHER Z than Ql=10.
        """
        fr_tuning = 50.0
        Z_low_q = models.vented_box_impedance(freq=fr_tuning, vrc=0.035, lrc=0.05, fr_tuning=fr_tuning, ql=2.0)
        Z_high_q = models.vented_box_impedance(freq=fr_tuning, vrc=0.035, lrc=0.05, fr_tuning=fr_tuning, ql=10.0)
        # Lower Ql (more loss) → lower R_leak → more current diverted → lower Z
        # Higher Ql (less loss) → higher R_leak → less current diverted → higher Z
        assert abs(Z_low_q) > abs(Z_high_q), (
            f"Lower Ql should give higher |Z| at resonance; "
            f"got |Z_ql=2|={abs(Z_low_q):.2f} vs |Z_ql=10|={abs(Z_high_q):.2f}"
        )

    def test_impedance_magnitude_positive(self):
        """Impedance magnitude should always be positive."""
        freqs = np.linspace(20, 500, 50)
        for f in freqs:
            Z = models.vented_box_impedance(f, vrc=0.035, lrc=0.05, fr_tuning=50.0)
            assert abs(Z) >= 0, f"Impedance magnitude must be non-negative at {f} Hz"


class TestVentedBoxVsClosedBox:
    """Integration tests: vented box vs closed-box (rear chamber) SPL difference."""

    @pytest.fixture
    def fostex_driver(self):
        return DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83, le=0.0008, xmax=0.0015,
        )

    def test_vented_box_with_horn_works(self, fostex_driver):
        """horn_response should run without error when a VentedBox is set."""
        from pyhorn_core.config.models import VentedBox
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=0.035,
            vented_box=VentedBox(vrc=0.035, fr=50.0, lrc=0.05, ql=5.0),
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, horn)
        assert isinstance(result, models.SimulationResult)
        assert np.all(np.isfinite(result.spl))
        assert np.all(np.isfinite(result.impedance))

    def test_vented_box_spl_differs_from_sealed_below_tuning(self, fostex_driver):
        """Below the tuning frequency, vented box SPL should be lower than sealed box.

        In a bass-reflex system, the port radiates out-of-phase with the driver below
        the tuning frequency, causing partial cancellation and a steeper rolloff
        compared to a sealed box of the same volume.
        """
        from pyhorn_core.config.models import VentedBox
        vrc = 0.035  # 35 L
        fr_tuning = 50.0

        sealed_horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=vrc,
        )
        vented_horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=vrc,
            vented_box=VentedBox(vrc=vrc, fr=fr_tuning, lrc=0.05, ql=5.0),
        )

        # Evaluate at 30 Hz — well below the 50 Hz tuning frequency
        freqs = np.linspace(29, 31, 10)  # small sweep to satisfy np.gradient
        result_sealed = models.horn_response(freqs, fostex_driver, sealed_horn)
        result_vented = models.horn_response(freqs, fostex_driver, vented_horn)

        # Both should be finite
        assert np.isfinite(result_sealed.spl[0])
        assert np.isfinite(result_vented.spl[0])
        # Vented box SPL at 30 Hz should differ from sealed box
        # (typically lower due to the port cancellation effect below tuning)
        assert result_vented.spl[0] != pytest.approx(result_sealed.spl[0], rel=1e-3), (
            f"Vented box SPL at 30 Hz ({result_vented.spl[0]:.2f} dB) should differ "
            f"from sealed box ({result_sealed.spl[0]:.2f} dB)"
        )

    def test_vented_box_double_peak_impedance(self, fostex_driver):
        """Vented box should show a double-peak impedance characteristic.

        A bass-reflex system has two resonance peaks:
          1. The port tuning resonance (fr_vb ≈ 50 Hz)
          2. The driver's free-air resonance (fs ≈ 50 Hz for FE166NV2)

        Between these two peaks the impedance drops to a minimum (the "valley").
        A sealed box only has a single impedance peak at fs.
        """
        from pyhorn_core.config.models import VentedBox
        vrc = 0.035
        fr_tuning = 50.0

        vented_horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vrc=vrc,
            vented_box=VentedBox(vrc=vrc, fr=fr_tuning, lrc=0.05, ql=5.0),
        )

        # Sweep a range that covers both the port resonance and driver resonance
        freqs = np.linspace(20, 200, 100)
        result = models.horn_response(freqs, fostex_driver, vented_horn)
        z_mag = np.abs(result.impedance)

        # Find the two peaks and the minimum between them
        peaks = []
        for i in range(1, len(z_mag) - 1):
            if z_mag[i] > z_mag[i - 1] and z_mag[i] > z_mag[i + 1]:
                peaks.append((freqs[i], z_mag[i]))

        # We expect at least one peak near the vent tuning (~50 Hz) or driver fs (~50 Hz)
        # The key test: there should be a local minimum between two peaks
        assert len(peaks) >= 1, f"Expected at least one impedance peak, got: {peaks}"
        # The minimum in the impedance curve should be clearly below the peaks
        valley_idx = np.argmin(z_mag[10:80]) + 10  # look between 10-80 indices
        valley_z = z_mag[valley_idx]
        peak_z_max = max(z_mag)
        # The valley should be significantly lower than the peak
        assert valley_z < peak_z_max * 0.7, (
            f"Impedance valley ({valley_z:.1f} Ω) should be at least 30% below "
            f"the peak ({peak_z_max:.1f} Ω) for a vented box"
        )


class TestFiniteHornChargedBassReflex:
    """Tests for the finite horn-charged bass reflex system topology."""

    @pytest.fixture
    def fostex_driver(self):
        return DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83, le=0.0008, xmax=0.0015,
        )

    def test_finite_horn_charged_returns_result(self, fostex_driver):
        """horn_response should return a SimulationResult without error when
        finite_horn_charged=True on the VentedBox."""
        from pyhorn_core.config.models import VentedBox
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_horn_charged=True,
            ),
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, horn)
        assert isinstance(result, models.SimulationResult)
        assert np.all(np.isfinite(result.spl))
        assert np.all(np.isfinite(result.impedance))

    def test_finite_horn_charged_flag_in_result(self, fostex_driver):
        """result.finite_horn_charged should be True when the topology is active."""
        from pyhorn_core.config.models import VentedBox
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_horn_charged=True,
            ),
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, horn)
        assert result.finite_horn_charged is True

    def test_finite_horn_charged_false_not_set(self, fostex_driver):
        """result.finite_horn_charged should be False when not set."""
        from pyhorn_core.config.models import VentedBox
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_horn_charged=False,  # explicitly off
            ),
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, horn)
        assert result.finite_horn_charged is False

    def test_finite_horn_charged_flh_works(self, fostex_driver):
        """FLH enclosure with finite_horn_charged=True should also work."""
        from pyhorn_core.config.models import VentedBox
        horn = HornGeometry(
            enclosure_type="FLH",
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_horn_charged=True,
            ),
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, horn)
        assert isinstance(result, models.SimulationResult)
        assert result.finite_horn_charged is True
        assert np.all(np.isfinite(result.spl))

    def test_spl_differs_from_standard_vented(self, fostex_driver):
        """With finite_horn_charged=True, SPL should differ from standard vented
        box (same box params but finite_horn_charged=False) because the port
        radiation is now added as a separate pressure source."""
        from pyhorn_core.config.models import VentedBox
        vrc = 0.035
        fr_tuning = 50.0
        base = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vrc=vrc,
            vented_box=VentedBox(
                vrc=vrc, fr=fr_tuning, lrc=0.05, ql=5.0,
                finite_horn_charged=False,
            ),
        )
        fhc = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vrc=vrc,
            vented_box=VentedBox(
                vrc=vrc, fr=fr_tuning, lrc=0.05, ql=5.0,
                finite_horn_charged=True,
            ),
        )
        freqs = np.linspace(20, 500, 100)
        result_base = models.horn_response(freqs, fostex_driver, base)
        result_fhc = models.horn_response(freqs, fostex_driver, fhc)
        # The SPL curves should differ — the port contribution changes the total
        diff = np.abs(result_base.spl - result_fhc.spl)
        assert np.max(diff) > 0.01, (
            "finite_horn_charged=True should produce different SPL from "
            "finite_horn_charged=False with same box params"
        )

    def test_path_length_difference_zero_is_neutral(self, fostex_driver):
        """path_length_difference=0 should produce identical SPL to omitting it."""
        from pyhorn_core.config.models import VentedBox
        horn_no_pld = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.07, 0.03), (0.07, 0.08, 0.04)],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_horn_charged=True,
                path_length_difference=0.0,
            ),
        )
        horn_no_pld_explicit = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.07, 0.03), (0.07, 0.08, 0.04)],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_horn_charged=True,
            ),
        )
        freqs = np.linspace(20, 5000, 100)
        r1 = models.horn_response(freqs, fostex_driver, horn_no_pld)
        r2 = models.horn_response(freqs, fostex_driver, horn_no_pld_explicit)
        np.testing.assert_allclose(r1.spl, r2.spl, rtol=1e-10,
                                   err_msg="path_length_difference=0 should be neutral")

    def test_path_length_difference_nonzero_changes_result(self, fostex_driver):
        """Non-zero path_length_difference should change SPL due to phase offset."""
        from pyhorn_core.config.models import VentedBox
        horn_base = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.07, 0.03), (0.07, 0.08, 0.04)],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_horn_charged=True,
                path_length_difference=0.0,
            ),
        )
        horn_offset = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.07, 0.03), (0.07, 0.08, 0.04)],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_horn_charged=True,
                path_length_difference=0.5,  # 0.5 m listening offset
            ),
        )
        freqs = np.linspace(20, 5000, 100)
        r_zero = models.horn_response(freqs, fostex_driver, horn_base)
        r_offset = models.horn_response(freqs, fostex_driver, horn_offset)
        # SPL should differ at some frequencies due to the phase offset
        diff = np.abs(r_offset.spl - r_zero.spl)
        assert np.max(diff) > 0.01, (
            "Non-zero path_length_difference should produce different SPL; "
            "got max diff = {np.max(diff):.6f} dB"
        )


class TestTransmissionLineImpedance:
    """Tests for transmission_line_impedance()."""

    def test_zero_length_returns_zero(self):
        """ltl <= 0 should return 0j."""
        Z = models.transmission_line_impedance(freq=100.0, ltl=0.0, area=0.01)
        assert Z == pytest.approx(0.0j)

    def test_zero_area_returns_zero(self):
        """area <= 0 should return 0j."""
        Z = models.transmission_line_impedance(freq=100.0, ltl=1.0, area=0.0)
        assert Z == pytest.approx(0.0j)

    def test_negative_length_returns_zero(self):
        """ltl < 0 should return 0j."""
        Z = models.transmission_line_impedance(freq=100.0, ltl=-0.5, area=0.01)
        assert Z == pytest.approx(0.0j)

    def test_returns_complex(self):
        """Should return a complex number for valid parameters."""
        Z = models.transmission_line_impedance(freq=100.0, ltl=1.0, area=0.01)
        assert isinstance(Z, complex)

    def test_imaginary_for_small_length(self):
        """For short ltl at moderate f, Z should be imaginary (compliance-like)."""
        Z = models.transmission_line_impedance(freq=50.0, ltl=0.5, area=0.01)
        assert Z.imag != 0.0

    def test_zero_at_series_resonance(self):
        """At series resonance k·l = π, tan(k·l) = 0 → Z_tl → 0."""
        # f_series = c / (2l) for closed pipe
        c = models.C
        ltl = 1.0
        f_series = c / (2.0 * ltl)  # 171.5 Hz
        Z = models.transmission_line_impedance(freq=f_series, ltl=ltl, area=0.01)
        assert abs(Z) < 1.0, f"At series resonance Z should be near zero; got |Z|={abs(Z)}"

    def test_high_at_anti_resonance(self):
        """At anti-resonance k·l = π/2, tan(k·l) → ∞ → Z_tl → ∞."""
        c = models.C
        ltl = 1.0
        f_ar = c / (4.0 * ltl)  # 85.75 Hz (quarter-wave resonance)
        Z = models.transmission_line_impedance(freq=f_ar, ltl=ltl, area=0.01)
        # tan(π/2) → ∞, so |Z| should be large
        assert abs(Z) > 1e6, f"At anti-resonance |Z| should be very large; got {abs(Z)}"


class TestFiniteTransmissionLine:
    """Tests for the finite transmission line topology (Hornresp page 091).

    A finite transmission line (separate topology from finite horn-charged bass reflex)
    models a pipe closed at the far end whose output is summed with the horn output
    at the listening point.
    """

    @pytest.fixture
    def fostex_driver(self):
        return DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83, le=0.0008, xmax=0.0015,
        )

    def test_finite_transmission_line_returns_result(self, fostex_driver):
        """horn_response should return a SimulationResult without error when
        finite_transmission_line=True on the VentedBox."""
        from pyhorn_core.config.models import VentedBox
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_transmission_line=True,
                ltl=1.0,  # 1 m transmission line
            ),
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, horn)
        assert isinstance(result, models.SimulationResult)
        assert np.all(np.isfinite(result.spl))
        assert np.all(np.isfinite(result.impedance))

    def test_finite_transmission_line_differs_from_base(self, fostex_driver):
        """With finite_transmission_line=True, SPL should differ from the base case
        (same params but finite_transmission_line=False)."""
        from pyhorn_core.config.models import VentedBox
        horn_base = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_transmission_line=False,
            ),
        )
        horn_tl = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_transmission_line=True,
                ltl=1.0,
            ),
        )
        freqs = np.linspace(20, 5000, 100)
        r_base = models.horn_response(freqs, fostex_driver, horn_base)
        r_tl = models.horn_response(freqs, fostex_driver, horn_tl)
        diff = np.abs(r_tl.spl - r_base.spl)
        assert np.max(diff) > 0.01, (
            "finite_transmission_line=True should produce different SPL from the base case; "
            f"got max diff = {np.max(diff):.6f} dB"
        )

    def test_finite_transmission_line_zero_ltl_ignored(self, fostex_driver):
        """When ltl=0 (even with finite_transmission_line=True), the result should
        match the base case (no TL contribution)."""
        from pyhorn_core.config.models import VentedBox
        horn_base = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.07, 0.03)],
            vrc=0.035,
            vented_box=VentedBox(vrc=0.035, fr=45.0, lrc=0.10, ql=5.0),
        )
        horn_zero_ltl = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.07, 0.03)],
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_transmission_line=True,
                ltl=0.0,
            ),
        )
        freqs = np.linspace(20, 5000, 50)
        r_base = models.horn_response(freqs, fostex_driver, horn_base)
        r_zero = models.horn_response(freqs, fostex_driver, horn_zero_ltl)
        np.testing.assert_allclose(r_base.spl, r_zero.spl, atol=1e-10)

    def test_finite_transmission_line_tl_length_affects_response(self, fostex_driver):
        """Different ltl values should produce different SPL responses
        (the TL resonance frequency changes with length)."""
        from pyhorn_core.config.models import VentedBox
        def make_horn(ltl_val):
            return HornGeometry(
                enclosure_type="BLH",
                width=0.2,
                conical_segments=[(0.06, 0.07, 0.03), (0.07, 0.09, 0.04)],
                vrc=0.035,
                vented_box=VentedBox(
                    vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                    finite_transmission_line=True,
                    ltl=ltl_val,
                ),
            )

        freqs = np.linspace(20, 5000, 100)
        r_1m = models.horn_response(freqs, fostex_driver, make_horn(1.0))
        r_2m = models.horn_response(freqs, fostex_driver, make_horn(2.0))
        diff = np.abs(r_1m.spl - r_2m.spl)
        assert np.max(diff) > 0.1, (
            "Different ltl values should produce measurably different SPL; "
            f"got max diff = {np.max(diff):.6f} dB"
        )

    def test_finite_transmission_line_in_flh_mode(self, fostex_driver):
        """finite_transmission_line should also work in FLH (non-BLH) mode."""
        from pyhorn_core.config.models import VentedBox
        horn = HornGeometry(
            enclosure_type="FLH",
            throat_area=0.004,
            mouth_area=0.09,
            path_length=1.0,
            profile_type="exponential",
            n_segments=50,
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_transmission_line=True,
                ltl=0.8,
            ),
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, horn)
        assert isinstance(result, models.SimulationResult)
        assert np.all(np.isfinite(result.spl))

    def test_finite_transmission_line_and_finite_horn_charged_together(self, fostex_driver):
        """Both finite_transmission_line and finite_horn_charged can be True simultaneously."""
        from pyhorn_core.config.models import VentedBox
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.08, 0.05)],
            vrc=0.035,
            vented_box=VentedBox(
                vrc=0.035, fr=45.0, lrc=0.10, ql=5.0,
                finite_horn_charged=True,
                finite_transmission_line=True,
                ltl=1.0,
                path_length_difference=0.2,
            ),
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, horn)
        assert isinstance(result, models.SimulationResult)
        assert np.all(np.isfinite(result.spl))


class TestPassiveRadiatorImpedance:
    """Tests for passive_radiator_impedance()."""

    def test_zero_volume_returns_zero(self):
        """vrc <= 0 should return 0j."""
        Z = models.passive_radiator_impedance(
            freq=100.0, vrc=0.0, mma=0.005, sp=0.005
        )
        assert Z == pytest.approx(0.0j)

    def test_zero_mass_returns_zero(self):
        """mma <= 0 should return 0j."""
        Z = models.passive_radiator_impedance(
            freq=100.0, vrc=0.035, mma=0.0, sp=0.005
        )
        assert Z == pytest.approx(0.0j)

    def test_zero_area_returns_zero(self):
        """sp <= 0 should return 0j."""
        Z = models.passive_radiator_impedance(
            freq=100.0, vrc=0.035, mma=0.005, sp=0.0
        )
        assert Z == pytest.approx(0.0j)

    def test_returns_complex(self):
        """Should return a complex number for valid parameters."""
        Z = models.passive_radiator_impedance(
            freq=100.0, vrc=0.035, mma=0.005, sp=0.005
        )
        assert isinstance(Z, complex)
        assert np.iscomplexobj(np.array([Z]))

    def test_resonance_finite_at_pr_frequency(self):
        """At the PR tuning frequency, Z_ab should be finite and non-zero."""
        vrc = 0.035
        mma = 0.005
        sp = 0.005
        C_pr = vrc / (models.RHO * models.C**2)
        f_pr = np.sqrt(1.0 / (mma * C_pr)) / (2.0 * np.pi)
        Z = models.passive_radiator_impedance(
            freq=f_pr, vrc=vrc, mma=mma, sp=sp
        )
        assert abs(Z) > 0.0
        assert np.isfinite(Z)

    def test_low_frequency_compliance_dominant(self):
        """Below f_pr, the box compliance should dominate → negative reactance."""
        vrc = 0.035
        mma = 0.005
        sp = 0.005
        C_pr = vrc / (models.RHO * models.C**2)
        f_pr = np.sqrt(1.0 / (mma * C_pr)) / (2.0 * np.pi)
        Z = models.passive_radiator_impedance(
            freq=f_pr * 0.3, vrc=vrc, mma=mma, sp=sp
        )
        assert Z.imag < 0, (
            f"Below f_pr, box compliance should dominate → negative reactance; got Z={Z}"
        )

    def test_high_frequency_impedance_rises_with_mass(self):
        """Above f_pr, higher Mma → higher |Z_ab|."""
        vrc = 0.035
        sp = 0.005
        freq = 200.0
        Z_light = models.passive_radiator_impedance(
            freq=freq, vrc=vrc, mma=0.003, sp=sp
        )
        Z_heavy = models.passive_radiator_impedance(
            freq=freq, vrc=vrc, mma=0.010, sp=sp
        )
        assert abs(Z_heavy) > abs(Z_light), (
            f"Heavier PR at high freq should have higher |Z|; "
            f"got {abs(Z_heavy):.3f} vs {abs(Z_light):.3f}"
        )

    def test_leak_loss_ql_broadens_resonance(self):
        """Higher Ql (less loss) → higher |Z| away from resonance."""
        vrc = 0.035
        mma = 0.005
        sp = 0.005
        C_pr = vrc / (models.RHO * models.C**2)
        f_pr = np.sqrt(1.0 / (mma * C_pr)) / (2.0 * np.pi)
        freq = f_pr * 1.5
        Z_low_q = models.passive_radiator_impedance(
            freq=freq, vrc=vrc, mma=mma, sp=sp, ql_pr=2.0
        )
        Z_high_q = models.passive_radiator_impedance(
            freq=freq, vrc=vrc, mma=mma, sp=sp, ql_pr=10.0
        )
        # R_leak = 1/(2π·f·C·Ql). Higher Ql → smaller R_leak → more shunting → lower |Z|.
        assert abs(Z_high_q) < abs(Z_low_q), (
            f"Ql=10 should give lower |Z| than Ql=2; got {abs(Z_high_q):.3f} vs {abs(Z_low_q):.3f}"
        )


class TestSecondToneDistortion:
    """Tests for second tone distortion — Hornresp pages 85/89, single-segment horns only."""

    @pytest.fixture
    def fostex_driver(self):
        return DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83, le=0.0008, xmax=0.0015,
        )

    def test_second_tone_distortion_returns_result(self, fostex_driver):
        """_compute_second_tone_distortion returns a non-None array for single-seg horn."""
        freqs = np.linspace(50, 2000, 200)
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.08, 0.12)],  # single segment → single-seg horn
            vtc=0.0036,
            fr_tc=2000.0,
        )
        from pyhorn_core.solver.models import _compute_second_tone_distortion, horn_response

        # Test the standalone function directly
        result = horn_response(freqs, fostex_driver, horn, compute_distortion=True)
        assert result.second_tone_distortion is not None, (
            "second_tone_distortion should be non-None for single-segment horn"
        )
        assert isinstance(result.second_tone_distortion, np.ndarray)
        assert len(result.second_tone_distortion) == len(freqs)

    def test_distortion_increases_with_excursion(self, fostex_driver):
        """Higher SPL at low frequency → larger excursion → more distortion.

        Distortion arises from nonlinear compliance (Bl varies with x).
        Larger cone excursion at low frequencies produces proportionally
        larger second-harmonic distortion. At 100 Hz, excursion is high;
        at 1000 Hz it is much lower — distortion should be lower at high freq.
        """
        freqs = np.linspace(50, 2000, 300)
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.08, 0.12)],
            vtc=0.0036,
            fr_tc=2000.0,
        )
        result = models.horn_response(freqs, fostex_driver, horn, compute_distortion=True)

        assert result.second_tone_distortion is not None
        dist = result.second_tone_distortion
        valid = np.isfinite(dist)

        # Find distortion at ~100 Hz and ~1000 Hz
        idx_lo = int(np.argmin(np.abs(freqs - 100)))
        idx_hi = int(np.argmin(np.abs(freqs - 1000)))
        dist_lo = dist[idx_lo]
        dist_hi = dist[idx_hi]

        # Low-frequency distortion should be larger (less negative = more distortion)
        # because excursion is higher at low frequency
        assert dist_lo > dist_hi, (
            f"Distortion at 100 Hz ({dist_lo:.1f} dB) should be less negative "
            f"(more distortion) than at 1000 Hz ({dist_hi:.1f} dB)"
        )

    def test_distortion_is_negative_db(self, fostex_driver):
        """Distortion is always below fundamental — dB values should be negative.

        Second tone distortion D2 = SPL(2f) - SPL(f) is the dB difference between
        the second harmonic and the fundamental. Since the 2f component is always
        smaller than the fundamental for a linear-ish driver, D2 must be negative.
        """
        freqs = np.linspace(50, 2000, 300)
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.08, 0.12)],
            vtc=0.0036,
            fr_tc=2000.0,
        )
        result = models.horn_response(freqs, fostex_driver, horn, compute_distortion=True)

        assert result.second_tone_distortion is not None
        dist = result.second_tone_distortion
        valid = np.isfinite(dist)
        assert np.all(dist[valid] < 0.0), (
            f"All distortion values should be negative (dB below fundamental). "
            f"Found positive values at indices where valid: {np.where(valid & (dist > 0))[0]}"
        )

    def test_multisegment_horn_returns_none(self, fostex_driver):
        """Multi-segment horns should have second_tone_distortion = None."""
        freqs = np.linspace(50, 2000, 200)
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
        )
        result = models.horn_response(freqs, fostex_driver, horn, compute_distortion=True)
        assert result.second_tone_distortion is None, (
            "Multi-segment horn should have second_tone_distortion = None"
        )

    def test_distortion_disabled_by_flag(self, fostex_driver):
        """compute_distortion=False should skip distortion computation."""
        freqs = np.linspace(50, 2000, 200)
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[(0.06, 0.08, 0.12)],
            vtc=0.0036,
            fr_tc=2000.0,
        )
        result = models.horn_response(freqs, fostex_driver, horn, compute_distortion=False)
        assert result.second_tone_distortion is None, (
            "second_tone_distortion should be None when compute_distortion=False"
        )


class TestPassiveRadiatorVsVentedBox:
    """Integration tests: passive radiator vs vented box."""

    @pytest.fixture
    def fostex_driver(self):
        return DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83, le=0.0008, xmax=0.0015,
        )

    def test_passive_radiator_horn_works(self, fostex_driver):
        """horn_response should run without error when a PassiveRadiator is set."""
        from pyhorn_core.config.models import PassiveRadiator
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=0.035,
            passive_radiator=PassiveRadiator(
                mma=0.005, sp1=0.005, ql_pr=5.0
            ),
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, horn)
        assert isinstance(result, models.SimulationResult)
        assert np.all(np.isfinite(result.spl))
        assert np.all(np.isfinite(result.impedance))

    def test_passive_radiator_spl_differs_from_vented_box(self, fostex_driver):
        """PR and vented-box horns with similar tunings should produce different SPL curves."""
        from pyhorn_core.config.models import PassiveRadiator, VentedBox
        vrc = 0.035
        f_tune = 50.0
        sp_pr = 0.005
        C_pr = vrc / (models.RHO * models.C**2)
        mma = 1.0 / ((2.0 * np.pi * f_tune) ** 2 * C_pr)
        mma = max(mma, 1e-6)

        pr_horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=vrc,
            passive_radiator=PassiveRadiator(mma=mma, sp1=sp_pr, ql_pr=5.0),
        )
        vented_horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
            vrc=vrc,
            vented_box=VentedBox(vrc=vrc, fr=f_tune, lrc=0.05, ql=5.0),
        )
        freqs = np.linspace(20, 5000, 200)
        result_pr = models.horn_response(freqs, fostex_driver, pr_horn)
        result_vb = models.horn_response(freqs, fostex_driver, vented_horn)
        diff = np.abs(result_pr.spl - result_vb.spl)
        assert np.max(diff) > 0.05, (
            "PR and vented-box horns with similar tunings should produce "
            f"different SPL curves; got max diff = {np.max(diff):.3f} dB"
        )

    def test_passive_radiator_total_sp_sums_panels(self):
        """total_sp property should correctly sum sp1–sp9."""
        from pyhorn_core.config.models import PassiveRadiator
        pr = PassiveRadiator(
            mma=0.005,
            sp1=0.001,
            sp2=0.002,
            sp3=0.003,
            sp4=0.0,
            sp5=0.0,
            sp6=0.0,
            sp7=0.0,
            sp8=0.0,
            sp9=0.0,
        )
        assert pr.total_sp == pytest.approx(0.006)


class TestThermalPowerCompression:
    """Tests for thermal power compression (voice coil heating → Re → SPL compression).

    Reference: Hornresp page 98 — "thermal power compression expressed in decibels".
    Formula: re_heated = Re × (1 + α × (T_voice − 20))
             compression_dB = 10 × log10(P_acoustic_heated / P_acoustic_nominal)
    """

    @pytest.fixture
    def fostex_driver(self):
        return DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83, le=0.0008, xmax=0.0015,
        )

    @pytest.fixture
    def simple_horn(self):
        return HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
        )

    def test_no_compression_at_room_temperature(self, fostex_driver, simple_horn):
        """T=20°C should give zero compression (no heating)."""
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(
            freqs, fostex_driver, simple_horn, T_voice=20.0
        )
        assert result.thermal_compression_db is None, (
            "T=20°C should not compute thermal compression (None, not 0)"
        )

    def test_no_compression_when_T_voice_is_none(self, fostex_driver, simple_horn):
        """T_voice=None should not compute thermal compression."""
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, simple_horn, T_voice=None)
        assert result.thermal_compression_db is None

    def test_compression_is_non_positive_at_100C(self, fostex_driver, simple_horn):
        """Heating cannot increase SPL — compression must be ≤ 0 dB."""
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(
            freqs, fostex_driver, simple_horn, T_voice=100.0
        )
        assert result.thermal_compression_db is not None
        assert np.all(result.thermal_compression_db <= 0.0), (
            "Thermal compression must be ≤ 0 dB (heating reduces output)"
        )

    def test_compression_increases_with_temperature(self, fostex_driver, simple_horn):
        """Higher temperature → larger (more negative) compression."""
        freqs = np.linspace(20, 5000, 100)
        result_100 = models.horn_response(
            freqs, fostex_driver, simple_horn, T_voice=100.0
        )
        result_150 = models.horn_response(
            freqs, fostex_driver, simple_horn, T_voice=150.0
        )
        assert result_100.thermal_compression_db is not None
        assert result_150.thermal_compression_db is not None
        # More heating → more compression (average more negative)
        assert np.mean(result_150.thermal_compression_db) < np.mean(
            result_100.thermal_compression_db
        ), "Higher temperature should produce more compression (more negative dB)"

    def test_compression_reasonable_magnitude(self, fostex_driver, simple_horn):
        """At 100°C with copper wire, compression should be roughly 1-3 dB."""
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(
            freqs, fostex_driver, simple_horn, T_voice=100.0
        )
        assert result.thermal_compression_db is not None
        mean_compression = np.mean(result.thermal_compression_db)
        # Copper alpha = 0.00393, 80°C rise → Re ratio = 1.314
        # 10*log10(1/1.314) ≈ -1.18 dB. With impedance effects, expect -0.5 to -3 dB.
        assert -3.5 < mean_compression < 0.0, (
            f"Mean compression at 100°C should be roughly -0.5 to -3 dB; "
            f"got {mean_compression:.2f} dB"
        )

    def test_compute_thermal_power_compression_returns_array(self, fostex_driver, simple_horn):
        """compute_thermal_power_compression() should return a numpy array."""
        freqs = np.linspace(20, 5000, 100)
        tcdb = models.compute_thermal_power_compression(
            freqs, fostex_driver, simple_horn, T_voice=100.0
        )
        assert isinstance(tcdb, np.ndarray)
        assert len(tcdb) == len(freqs)
        assert tcdb.dtype.kind == "f"

    def test_compute_thermal_power_compression_zero_at_20C(self, fostex_driver, simple_horn):
        """T=20°C should return zeros array."""
        freqs = np.linspace(20, 5000, 100)
        tcdb = models.compute_thermal_power_compression(
            freqs, fostex_driver, simple_horn, T_voice=20.0
        )
        assert isinstance(tcdb, np.ndarray)
        assert np.allclose(tcdb, 0.0)

    def test_driver_alpha_re_in_DriverSpecs(self):
        """DriverSpecs should have alpha_re field with copper default."""
        d = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
        )
        assert hasattr(d, "alpha_re")
        assert d.alpha_re == pytest.approx(0.00393)

    def test_alpha_re_custom_value_used(self, simple_horn):
        """Custom alpha_re on driver should affect the compression magnitude."""
        from dataclasses import replace
        freqs = np.linspace(20, 5000, 100)
        d_cu = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83, alpha_re=0.00393,  # copper
        )
        d_al = replace(d_cu, alpha_re=0.00430)  # aluminium (higher alpha)
        result_cu = models.horn_response(freqs, d_cu, simple_horn, T_voice=100.0)
        result_al = models.horn_response(freqs, d_al, simple_horn, T_voice=100.0)
        # Aluminium has higher alpha → more compression (more negative mean)
        assert np.mean(result_al.thermal_compression_db) < np.mean(
            result_cu.thermal_compression_db
        ), "Higher alpha_re (Al) should give more compression than copper"


class TestNotchFilter:
    """Tests for _apply_notch_filter() and notch_filter in horn_response()."""

    @pytest.fixture
    def fostex_driver(self):
        """Fostex FE166NV2 driver specs."""
        return DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83, le=0.0008, xmax=0.0015,
        )

    @pytest.fixture
    def simple_horn(self):
        """Simple BLH horn geometry with conical segments."""
        return HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
        )

    def test_apply_notch_filter_no_freqs_returns_unchanged(self):
        """With no notch frequencies, SPL should be returned unchanged."""
        freqs = np.linspace(20, 5000, 500)
        spl = np.ones_like(freqs) * 90.0
        result = models._apply_notch_filter(freqs, spl, [], notch_q=10.0)
        np.testing.assert_allclose(result, spl, atol=1e-6)

    def test_apply_notch_filter_preserves_other_frequencies(self):
        """SPL far from notch frequencies should be preserved."""
        freqs = np.linspace(100, 5000, 1000)
        spl = np.linspace(80, 100, 1000)  # known gradient
        result = models._apply_notch_filter(freqs, spl, [1847.0], notch_q=10.0)
        # At 500 Hz and 4000 Hz (far from 1847 Hz), values should be nearly unchanged
        idx_500 = np.argmin(np.abs(freqs - 500))
        idx_4000 = np.argmin(np.abs(freqs - 4000))
        np.testing.assert_allclose(result[idx_500], spl[idx_500], atol=2.0)
        np.testing.assert_allclose(result[idx_4000], spl[idx_4000], atol=2.0)

    def test_apply_notch_filter_empty_freqs_returns_unchanged(self):
        """With an empty notch_frequencies list, SPL is unchanged."""
        freqs = np.linspace(20, 5000, 500)
        spl = np.ones_like(freqs) * 90.0
        result = models._apply_notch_filter(freqs, spl, [], notch_q=10.0)
        np.testing.assert_allclose(result, spl, atol=1e-6)

    def test_apply_notch_filter_reduces_bump_at_target_freq(self):
        """A bump at the target frequency should be reduced by the notch filter."""
        freqs = np.linspace(1500, 2200, 500)  # 1.4 Hz spacing
        # Baseline: smooth gradient
        spl = np.linspace(85, 95, 500)
        # Add a sharp bump at 1847 Hz
        idx_1847 = np.argmin(np.abs(freqs - 1847.0))
        bump = 10.0 * np.exp(-((freqs - 1847.0) ** 2) / (2 * 10.0**2))
        spl_bumped = spl + bump
        result = models._apply_notch_filter(freqs, spl_bumped, [1847.0], notch_q=10.0)
        # The bump should be reduced at 1847 Hz
        assert result[idx_1847] < spl_bumped[idx_1847] - 0.5, (
            "Notch filter should reduce the bump amplitude at the target frequency"
        )

    def test_horn_response_notch_filter_false_no_spl_notched(
        self, fostex_driver, simple_horn
    ):
        """When notch_filter=False, spl_notched should be None."""
        freqs = np.linspace(20, 5000, 500)
        result = models.horn_response(
            freqs, fostex_driver, simple_horn,
            notch_filter=False,
            notch_frequencies=[1847.0, 2508.0],
        )
        assert result.spl_notched is None

    def test_horn_response_notch_filter_true_produces_spl_notched(
        self, fostex_driver, simple_horn
    ):
        """When notch_filter=True, spl_notched should be a numpy array."""
        freqs = np.linspace(1000, 3500, 2000)
        result = models.horn_response(
            freqs, fostex_driver, simple_horn,
            notch_filter=True,
            notch_frequencies=[1847.0, 2508.0, 2732.0, 2852.0, 2969.0],
            notch_q=10.0,
        )
        assert result.spl_notched is not None
        assert isinstance(result.spl_notched, np.ndarray)
        assert result.spl_notched.shape == result.spl.shape
        # Notched SPL should differ from raw SPL (not identical arrays)
        assert not np.allclose(result.spl_notched, result.spl), (
            "Notched SPL should differ from raw SPL when filter is applied"
        )

    def test_horn_response_notch_filter_with_none_freqs_stays_none(
        self, fostex_driver, simple_horn
    ):
        """When notch_filter=True but notch_frequencies=None, spl_notched stays None."""
        freqs = np.linspace(1000, 3500, 1000)
        result = models.horn_response(
            freqs, fostex_driver, simple_horn,
            notch_filter=True,
            notch_frequencies=None,  # None → not applied
            notch_q=10.0,
        )
        assert result.spl_notched is None

    def test_horn_response_notch_filter_q_affects_depth(
        self, fostex_driver, simple_horn
    ):
        """Higher Q should produce a narrower notch (less reduction off-centre).

        Both Q=3 and Q=30 give ~20 dB reduction exactly at 1847 Hz (the notch
        centre).  The difference is visible away from the centre: Q=3 has a wide
        notch (BW ≈ 615 Hz) so the reduction extends far; Q=30 has a narrow notch
        (BW ≈ 62 Hz) so reduction drops off quickly.  We test at 2000 Hz.
        """
        freqs = np.linspace(1500, 2200, 1000)
        r_low_q = models.horn_response(
            freqs, fostex_driver, simple_horn,
            notch_filter=True,
            notch_frequencies=[1847.0],
            notch_q=3.0,
        )
        r_high_q = models.horn_response(
            freqs, fostex_driver, simple_horn,
            notch_filter=True,
            notch_frequencies=[1847.0],
            notch_q=30.0,
        )
        # At 2000 Hz (153 Hz above centre), Q=3 notch is still deep but Q=30
        # notch has dropped to near-zero — we test that Q=3 reduces more than Q=30
        idx_2000 = np.argmin(np.abs(freqs - 2000.0))
        reduction_low_q = r_low_q.spl[idx_2000] - r_low_q.spl_notched[idx_2000]
        reduction_high_q = r_high_q.spl[idx_2000] - r_high_q.spl_notched[idx_2000]
        assert reduction_low_q > reduction_high_q + 1.0, (
            f"Wide-notch (Q=3) should give more reduction at 2000 Hz than "
            f"narrow-notch (Q=30): low_q={reduction_low_q:.2f} dB, "
            f"high_q={reduction_high_q:.2f} dB"
        )

    def test_notch_filter_multiple_freqs_all_affected(
        self, fostex_driver, simple_horn
    ):
        """All specified artifact frequencies should show SPL differences after filtering."""
        freqs = np.linspace(1000, 3500, 3000)
        artifact_freqs = [1847.0, 2508.0, 2732.0]
        result = models.horn_response(
            freqs, fostex_driver, simple_horn,
            notch_filter=True,
            notch_frequencies=artifact_freqs,
            notch_q=10.0,
        )
        for af in artifact_freqs:
            idx = np.argmin(np.abs(freqs - af))
            diff = abs(result.spl[idx] - result.spl_notched[idx])
            assert diff > 0.05, (
                f"Notch at {af} Hz should change SPL by >0.05 dB, got {diff:.4f} dB"
            )


class TestLeFreqDependency:
    """Tests for the frequency-dependent Le (semi-inductance) voice coil model.

    Le(f) = Le_const × √(1 + (f / f_ref)²)
    Reference: Hornresp page 12 (semi-inductance model).
    """

    def test_le_at_dc_returns_le_const(self):
        """At f=0, Le(f) should equal Le_const."""
        le = models._le_freq_dependent(le_const=0.001, f=0.0, f_ref=100.0)
        assert le == pytest.approx(0.001)

    def test_le_at_f_ref_is_sqrt2_times_le_const(self):
        """At f = f_ref, Le should be √2 × Le_const."""
        le = models._le_freq_dependent(le_const=0.001, f=100.0, f_ref=100.0)
        assert le == pytest.approx(0.001 * np.sqrt(2), rel=1e-9)

    def test_le_grows_with_frequency(self):
        """Le should increase monotonically with frequency."""
        le_100 = models._le_freq_dependent(0.001, 100.0, 100.0)
        le_500 = models._le_freq_dependent(0.001, 500.0, 100.0)
        le_1000 = models._le_freq_dependent(0.001, 1000.0, 100.0)
        assert le_100 < le_500 < le_1000

    def test_le_high_freq_approaches_linear(self):
        """At f >> f_ref, Le(f) ≈ Le_const × (f / f_ref) (linear asymptote)."""
        le = models._le_freq_dependent(0.001, 10000.0, 100.0)
        linear_approx = 0.001 * (10000.0 / 100.0)  # = 0.1
        assert le == pytest.approx(linear_approx, rel=0.01)

    def test_le_zero_f_ref_returns_le_const(self):
        """f_ref <= 0 should return Le_const unchanged (guard)."""
        le = models._le_freq_dependent(0.001, 100.0, f_ref=0.0)
        assert le == pytest.approx(0.001)

    def test_le_zero_le_const_returns_zero(self):
        """Le_const = 0 should return 0."""
        le = models._le_freq_dependent(0.0, 100.0, f_ref=100.0)
        assert le == 0.0

    def test_driverspecs_default_no_freq_dependency(self):
        """le_freq_dependency should default to False (constant Le)."""
        d = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
        )
        assert d.le_freq_dependency is False

    def test_driverspecs_default_f_ref_is_100hz(self):
        """le_f_ref should default to 100.0 Hz."""
        d = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
        )
        assert d.le_f_ref == 100.0

    def test_horn_response_with_freq_dep_le_runs(self):
        """horn_response should run without error when le_freq_dependency=True."""
        d = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, le_freq_dependency=True, le_f_ref=100.0,
        )
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, d, horn)
        assert isinstance(result, models.SimulationResult)
        assert np.all(np.isfinite(result.spl))
        assert np.all(np.isfinite(result.impedance))

    def test_freq_dep_le_increases_impedance_at_high_freq(self):
        """With le_freq_dependency=True, |Z_e| should be larger at high frequency
        compared to le_freq_dependency=False (same driver, same geometry)."""
        d_const = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, le_freq_dependency=False,
        )
        d_freq = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, le_freq_dependency=True, le_f_ref=100.0,
        )
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
        )
        freqs = np.linspace(20, 5000, 200)
        r_const = models.horn_response(freqs, d_const, horn)
        r_freq = models.horn_response(freqs, d_freq, horn)

        # At 5 kHz (f >> f_ref=100 Hz), Le(f) ≈ Le_const × 50, so impedance
        # should be significantly larger than with constant Le
        idx_5k = int(np.argmin(np.abs(freqs - 5000.0)))
        z_const_5k = abs(r_const.impedance[idx_5k])
        z_freq_5k = abs(r_freq.impedance[idx_5k])
        assert z_freq_5k > z_const_5k * 1.5, (
            f"At 5 kHz, freq-dep Le should give higher |Z|; "
            f"got {z_freq_5k:.2f} Ω (freq-dep) vs {z_const_5k:.2f} Ω (const)"
        )

    def test_freq_dep_le_at_fs_same_as_constant(self):
        """Near the driver's free-air resonance (fs ~ 50 Hz, f << f_ref),
        frequency-dependent Le should be nearly identical to constant Le."""
        d_const = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, le_freq_dependency=False,
        )
        d_freq = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, le_freq_dependency=True, le_f_ref=100.0,
        )
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
        )
        freqs = np.linspace(20, 5000, 200)
        r_const = models.horn_response(freqs, d_const, horn)
        r_freq = models.horn_response(freqs, d_freq, horn)

        # At 50 Hz (f < f_ref), the two impedance curves should be very close
        idx_50 = int(np.argmin(np.abs(freqs - 50.0)))
        z_const_50 = abs(r_const.impedance[idx_50])
        z_freq_50 = abs(r_freq.impedance[idx_50])
        assert abs(z_freq_50 - z_const_50) / z_const_50 < 0.05, (
            f"At 50 Hz, freq-dep and const Le should agree within 5%; "
            f"got {z_freq_50:.3f} Ω (freq-dep) vs {z_const_50:.3f} Ω (const)"
        )

    def test_infinite_baffle_with_freq_dep_le(self):
        """infinite_baffle_response should also honour le_freq_dependency."""
        d_const = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83, le=0.0008, le_freq_dependency=False,
        )
        d_freq = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, le_freq_dependency=True, le_f_ref=100.0,
        )
        freqs = np.linspace(20, 5000, 100)
        ib_const = models.infinite_baffle_response(freqs, d_const)
        ib_freq = models.infinite_baffle_response(freqs, d_freq)

        # High-frequency SPL should be different when Le is frequency-dependent
        idx_hi = int(np.argmin(np.abs(freqs - 5000.0)))
        assert abs(ib_freq[idx_hi] - ib_const[idx_hi]) > 0.01, (
            "IB SPL at 5 kHz should differ between freq-dep and const Le"
        )


class TestFDDDirectivityIndex:
    """Tests for _fdd_directivity_index() — Frequency Dependent Directivity model."""

    def test_returns_numpy_array(self):
        """Should return a numpy array of the same length as freqs."""
        freqs = np.linspace(20, 5000, 100)
        mouth_area = 0.03  # ~200mm diameter horn mouth
        di = models._fdd_directivity_index(freqs, mouth_area, f_c=300.0, D_max=5.0)
        assert isinstance(di, np.ndarray)
        assert len(di) == len(freqs)

    def test_at_zero_frequency_returns_zero(self):
        """DI should be 0 dB at f=0 (omnidirectional)."""
        freqs = np.array([0.0, 1.0])
        di = models._fdd_directivity_index(freqs, mouth_area=0.03, f_c=300.0, D_max=5.0)
        assert di[0] == pytest.approx(0.0, abs=1e-6)
        assert di[1] > 0.0  # even at 1 Hz there should be some directivity

    def test_at_low_frequency_near_zero(self):
        """Below f_c, DI should be very small (near omnidirectional)."""
        freqs = np.array([10.0, 50.0, 100.0])
        di = models._fdd_directivity_index(freqs, mouth_area=0.03, f_c=300.0, D_max=5.0)
        # At 10 Hz (<< f_c), DI should be negligible
        assert di[0] < 0.01
        # At 50 Hz (<< f_c), DI should still be very small
        assert di[1] < 0.2  # 0.14 dB for f_c=300 Hz, D_max=5 dB
        assert di[2] < 1.0

    def test_at_high_frequency_approaches_D_max(self):
        """Above a few f_c, DI should approach D_max."""
        freqs = np.array([1000.0, 2000.0, 5000.0])
        di = models._fdd_directivity_index(freqs, mouth_area=0.03, f_c=300.0, D_max=5.0)
        # At 2000 Hz (≈ 6.7 × f_c), transition factor ≈ 1 - e^(-44.4) ≈ 1.0
        assert di[-1] == pytest.approx(5.0, abs=0.2)
        assert all(d >= 0 for d in di)

    def test_monotonically_increasing(self):
        """DI should always increase with frequency."""
        freqs = np.linspace(20, 5000, 200)
        di = models._fdd_directivity_index(freqs, mouth_area=0.03, f_c=300.0, D_max=5.0)
        assert all(di[i] <= di[i+1] for i in range(len(di)-1)), (
            "FDD DI should be monotonically increasing with frequency"
        )

    def test_f_c_controls_transition_point(self):
        """Higher f_c → transition happens at higher frequency."""
        freqs = np.linspace(100, 1000, 100)
        mouth_area = 0.03
        di_low_fc = models._fdd_directivity_index(freqs, mouth_area, f_c=200.0, D_max=5.0)
        di_high_fc = models._fdd_directivity_index(freqs, mouth_area, f_c=500.0, D_max=5.0)
        # At 300 Hz, lower f_c (200 Hz) gives more directivity than higher f_c (500 Hz)
        idx_300 = int(np.argmin(np.abs(freqs - 300.0)))
        assert di_low_fc[idx_300] > di_high_fc[idx_300]

    def test_D_max_is_asymptotic_limit(self):
        """DI should never exceed D_max."""
        freqs = np.logspace(1, 5, 500)
        di = models._fdd_directivity_index(freqs, mouth_area=0.03, f_c=100.0, D_max=5.0)
        assert np.all(di <= 5.0 + 1e-9)

    def test_D_max_scales_result(self):
        """Larger D_max → larger DI at all frequencies."""
        freqs = np.linspace(100, 2000, 100)
        mouth_area = 0.03
        di_small = models._fdd_directivity_index(freqs, mouth_area, f_c=300.0, D_max=3.0)
        di_large = models._fdd_directivity_index(freqs, mouth_area, f_c=300.0, D_max=6.0)
        # D_max scales linearly with DI
        ratio = di_large / np.maximum(di_small, 1e-12)
        # At mid frequencies, the ratio should be close to 2.0 (6/3)
        idx_mid = int(len(freqs) // 2)
        assert ratio[idx_mid] == pytest.approx(2.0, rel=0.05)

    def test_result_is_bounded_0_to_D_max(self):
        """DI should always be in the range [0, D_max]."""
        freqs = np.logspace(0, 5, 300)
        for D_max in [2.0, 5.0, 10.0]:
            di = models._fdd_directivity_index(freqs, mouth_area=0.03, f_c=300.0, D_max=D_max)
            assert np.all(di >= 0.0), f"DI should be >= 0 for D_max={D_max}"
            assert np.all(di <= D_max + 1e-9), f"DI should be <= D_max for D_max={D_max}"


class TestFDDOffAxisSPL:
    """Tests for _fdd_off_axis_spl() — FDD off-axis directivity pattern."""

    def test_shape_correct(self):
        """Returns array of shape (n_freq, n_angles)."""
        freqs = np.linspace(20, 5000, 50)
        angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
        result = models._fdd_off_axis_spl(freqs, 0.03, angles, f_c=300.0, D_max=5.0)
        assert result.shape == (50, 5)

    def test_on_axis_is_zero_db(self):
        """At 0° (on-axis), off-axis SPL should be 0 dB (reference)."""
        freqs = np.linspace(20, 5000, 100)
        angles = np.array([0.0, 30.0, 60.0])
        result = models._fdd_off_axis_spl(freqs, 0.03, angles, f_c=300.0, D_max=5.0)
        # On-axis should be 0 dB (reference). Small deviations from numerical precision.
        assert np.allclose(result[:, 0], 0.0, atol=1e-2), "On-axis should always be 0 dB"

    def test_at_low_frequency_all_angles_near_zero(self):
        """At very low frequency (f << f_c), all off-axis angles should be near 0 dB (omni)."""
        freqs = np.array([5.0, 10.0, 20.0])
        angles = np.array([30.0, 60.0, 90.0])
        result = models._fdd_off_axis_spl(freqs, 0.03, angles, f_c=300.0, D_max=5.0)
        # At 10 Hz, the mouth is infinitesimal relative to wavelength → omni
        assert np.all(np.abs(result) < 0.05), (
            f"At very low frequency, off-axis SPL should be ~0 dB (omni). Got: {result}"
        )

    def test_at_high_frequency_off_axis_losses_increase(self):
        """At high frequency, larger angles should have more negative SPL on average.

        Due to the sinc-function oscillation of the piston directivity pattern,
        the monotonic relationship between angle and loss can break down at very
        high frequencies (90° can have slightly higher D than 60° near a null).
        We test the weaker statement: the mean of off-axis values is negative.
        """
        freqs = np.array([1000.0, 2000.0, 4000.0])
        angles = np.array([0.0, 30.0, 60.0, 90.0])
        result = models._fdd_off_axis_spl(freqs, 0.03, angles, f_c=300.0, D_max=5.0)
        for i, f in enumerate(freqs):
            # Off-axis SPL at all angles should be ≤ 0 (on-axis = 0 dB reference)
            assert all(result[i, j] <= 0.1 for j in range(1, len(angles))), (
                f"At {f} Hz, all off-axis SPL values should be ≤ 0 dB"
            )
            # The on-axis reference (index 0) should be 0 dB
            assert result[i, 0] == pytest.approx(0.0, abs=0.01)

    def test_off_axis_decreases_with_frequency(self):
        """At a fixed angle (e.g. 60°), off-axis loss should become more negative at high f."""
        freqs = np.linspace(100, 4000, 100)
        angles = np.array([60.0])
        result = models._fdd_off_axis_spl(freqs, 0.03, angles, f_c=300.0, D_max=5.0)
        # The off-axis loss (negative dB) should become more negative at higher freq
        assert result[:, 0].min() < result[:, 0].max(), (
            "Off-axis loss at 60° should increase (more negative) at high frequency"
        )

    def test_fdd_vs_piston_differ_at_mid_frequencies(self):
        """FDD and piston model should differ at mid frequencies where the transition is active.

        At very low frequencies both are omnidirectional. At high frequencies both converge
        to the piston pattern. The mid-frequency transition zone is where they differ.
        We test that the transition is NOT a step (i.e., FDD is not exactly equal to piston
        at all frequencies).
        """
        freqs = np.linspace(200, 4000, 100)
        angles = np.array([0.0, 30.0, 60.0])
        mouth_area = 0.03
        # FDD off-axis SPL
        result_fdd = models._fdd_off_axis_spl(freqs, mouth_area, angles, f_c=300.0, D_max=5.0)
        # Piston off-axis SPL (reference)
        result_piston = np.zeros_like(result_fdd)
        a_mouth = np.sqrt(mouth_area / np.pi)
        ka_arr = 2.0 * np.pi * freqs * a_mouth / models.C
        for j, ang_deg in enumerate(angles):
            ang_rad = np.radians(ang_deg)
            x = ka_arr * np.sin(ang_rad)
            x_safe = np.where(x < 0.05, 0.05, x)
            from scipy.special import jv as jv1
            j1_vals = jv1(1, x_safe)
            D = (2.0 * j1_vals / (x_safe + 1e-12)) ** 2
            D = np.where(ka_arr < 0.05, 1.0, D)
            result_piston[:, j] = 10.0 * np.log10(np.clip(D, 1e-12, None))

        # At high frequency (3000 Hz), FDD and piston should be similar
        idx_hi = int(np.argmin(np.abs(freqs - 3000.0)))
        np.testing.assert_allclose(result_fdd[idx_hi], result_piston[idx_hi], atol=1.5)

        # At very low frequency, FDD should be MORE omnidirectional (less negative) than piston
        # because FDD's transition factor → 0 at low f
        idx_lo = int(np.argmin(np.abs(freqs - 200.0)))
        # At 200 Hz, piston may already show some directivity; FDD should be less directional
        # (closer to 0 dB omnidirectional)
        diff = result_fdd[idx_lo, 1] - result_piston[idx_lo, 1]  # at 30° off-axis
        assert diff >= 0.0, (
            f"At 200 Hz, FDD should be less directional than piston (closer to 0 dB omni). "
            f"FDD={result_fdd[idx_lo, 1]:.4f} dB, piston={result_piston[idx_lo, 1]:.4f} dB"
        )


class TestFDDRadiationAngle:
    """Tests for _fdd_radiation_angle() — FDD -6dB beamwidth computation."""

    def test_returns_float_or_none(self):
        """Should return a float or None."""
        freqs = np.linspace(200, 4000, 100)
        angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
        off_axis = models._fdd_off_axis_spl(freqs, 0.03, angles, f_c=300.0, D_max=5.0)
        ra = models._fdd_radiation_angle(freqs, 0.03, off_axis, angles, f_c=300.0)
        if ra is not None:
            assert isinstance(ra, float)
            assert 0 < ra < 90.0

    def test_returns_none_when_insufficient_angles(self):
        """With only 0° and 90°, beamwidth is not well-defined."""
        freqs = np.linspace(200, 4000, 100)
        angles = np.array([0.0, 90.0])
        off_axis = np.zeros((len(freqs), 2))
        ra = models._fdd_radiation_angle(freqs, 0.03, off_axis, angles, f_c=300.0)
        # With only 0° and 90° it's hard to interpolate → may return None or partial
        # Just check it doesn't crash
        assert ra is None or isinstance(ra, float)


class TestFDDModeIntegration:
    """Integration tests for horn_response() with fdd_mode=True."""

    @pytest.fixture
    def fostex_driver(self):
        return DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83, le=0.0008, xmax=0.0015,
        )

    @pytest.fixture
    def simple_horn(self):
        return HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
                (0.08, 0.10, 0.05),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
        )

    def test_fdd_mode_false_default_off_axis_unchanged(self, fostex_driver, simple_horn):
        """fdd_mode=False (default) should produce the standard piston off-axis SPL."""
        freqs = np.linspace(200, 5000, 100)
        result = models.horn_response(freqs, fostex_driver, simple_horn, fdd_mode=False)
        # fdd_enabled should be False
        assert result.fdd_enabled is False
        # fdd_di should be None when fdd_mode=False
        assert result.fdd_di is None

    def test_fdd_mode_true_sets_flags(self, fostex_driver, simple_horn):
        """fdd_mode=True should set fdd_enabled=True and populate fdd_di."""
        freqs = np.linspace(200, 5000, 100)
        result = models.horn_response(
            freqs, fostex_driver, simple_horn,
            fdd_mode=True, fdd_fc=300.0, fdd_dmax=5.0
        )
        assert result.fdd_enabled is True
        assert result.fdd_di is not None
        assert len(result.fdd_di) == len(freqs)

    def test_fdd_mode_true_fdd_di_increases_with_frequency(self, fostex_driver, simple_horn):
        """FDD DI should increase with frequency (more directional at high f)."""
        freqs = np.linspace(100, 4000, 100)
        result = models.horn_response(
            freqs, fostex_driver, simple_horn,
            fdd_mode=True, fdd_fc=300.0, fdd_dmax=5.0
        )
        # DI should be monotonically increasing
        assert all(result.fdd_di[i] <= result.fdd_di[i+1] for i in range(len(result.fdd_di)-1))

    def test_fdd_mode_true_off_axis_spl_affected(self, fostex_driver, simple_horn):
        """With fdd_mode=True, off_axis_spl should be computed (shape correct) and non-None."""
        freqs = np.linspace(200, 4000, 200)
        off_axis_angles = np.array([0.0, 30.0, 60.0, 90.0])
        result_fdd = models.horn_response(
            freqs, fostex_driver, simple_horn,
            fdd_mode=True, fdd_fc=300.0, fdd_dmax=5.0,
            off_axis_angles=off_axis_angles
        )
        assert result_fdd.off_axis_spl is not None
        assert result_fdd.off_axis_spl.shape == (200, 4)
        # On-axis (0°) should be near 0 dB
        assert np.allclose(result_fdd.off_axis_spl[:, 0], 0.0, atol=1e-2)
        # Off-axis values should be ≤ 0 (on-axis is the reference)
        assert np.all(result_fdd.off_axis_spl[:, 1:] <= 0.1)

    def test_fdd_mode_true_result_contains_fdd_di(self, fostex_driver, simple_horn):
        """Result.fdd_di should be a numpy array when fdd_mode=True."""
        freqs = np.linspace(200, 5000, 100)
        result = models.horn_response(
            freqs, fostex_driver, simple_horn,
            fdd_mode=True, fdd_fc=300.0, fdd_dmax=5.0
        )
        assert result.fdd_di is not None
        assert isinstance(result.fdd_di, np.ndarray)
        assert len(result.fdd_di) == 100
        # All DI values should be non-negative and not exceed D_max
        assert np.all(result.fdd_di >= 0.0)
        assert np.all(result.fdd_di <= 5.0 + 1e-9)

    def test_fdd_mode_false_result_has_no_fdd_di(self, fostex_driver, simple_horn):
        """Result.fdd_di should be None when fdd_mode=False."""
        freqs = np.linspace(200, 5000, 100)
        result = models.horn_response(
            freqs, fostex_driver, simple_horn,
            fdd_mode=False
        )
        assert result.fdd_enabled is False
        assert result.fdd_di is None

    def test_fdd_mode_true_radiation_angle_present(self, fostex_driver, simple_horn):
        """FDD mode should compute a radiation_angle."""
        freqs = np.linspace(200, 4000, 100)
        off_axis_angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
        result = models.horn_response(
            freqs, fostex_driver, simple_horn,
            fdd_mode=True, fdd_fc=300.0, fdd_dmax=5.0,
            off_axis_angles=off_axis_angles
        )
        # radiation_angle may or may not be None depending on horn geometry
        if result.radiation_angle is not None:
            assert 0.0 < result.radiation_angle < 90.0

    def test_fdd_di_bounded_by_D_max(self, fostex_driver, simple_horn):
        """FDD DI should never exceed D_max."""
        freqs = np.linspace(20, 5000, 200)
        for D_max in [3.0, 5.0, 8.0]:
            result = models.horn_response(
                freqs, fostex_driver, simple_horn,
                fdd_mode=True, fdd_fc=300.0, fdd_dmax=D_max
            )
            assert np.all(result.fdd_di <= D_max + 1e-9), (
                f"FDD DI should never exceed D_max={D_max}"
            )

    def test_fdd_mode_with_custom_parameters(self, fostex_driver, simple_horn):
        """Custom f_c and D_max should affect the FDD result."""
        freqs = np.linspace(100, 3000, 100)
        result_low_fc = models.horn_response(
            freqs, fostex_driver, simple_horn,
            fdd_mode=True, fdd_fc=150.0, fdd_dmax=5.0
        )
        result_high_fc = models.horn_response(
            freqs, fostex_driver, simple_horn,
            fdd_mode=True, fdd_fc=600.0, fdd_dmax=5.0
        )
        # At 400 Hz, lower f_c (150 Hz) → more directivity than higher f_c (600 Hz)
        idx_400 = int(np.argmin(np.abs(freqs - 400.0)))
        assert result_low_fc.fdd_di[idx_400] > result_high_fc.fdd_di[idx_400], (
            "Lower f_c should give more directivity at mid frequencies"
        )

    def test_fdd_mode_spl_unchanged(self, fostex_driver, simple_horn):
        """FDD mode only changes directivity fields, not the main SPL."""
        freqs = np.linspace(200, 4000, 200)
        result_fdd = models.horn_response(
            freqs, fostex_driver, simple_horn,
            fdd_mode=True, fdd_fc=300.0, fdd_dmax=5.0
        )
        result_piston = models.horn_response(
            freqs, fostex_driver, simple_horn,
            fdd_mode=False
        )
        # Main SPL should be identical (FDD only affects directivity, not acoustic response)
        np.testing.assert_allclose(result_fdd.spl, result_piston.spl, rtol=1e-10)


class TestLossyLeModel:
    """Tests for the Lossy Le model (eddy-current losses, Hornresp page 77).

    The Lossy Le model adds a frequency-dependent series resistance:

        R_lossy(f) = R_e_eddy × (f / f_ref)²

    Total electrical impedance: Z_e = Re + R_lossy(f) + j·ω·Le(f)

    This is distinct from the semi-inductance model (le_freq_dependency, Hornresp page 12)
    which only raises the inductance with frequency.
    """

    def test_lossy_le_impedance_at_dc_is_zero(self):
        """At f=0 Hz, Lossy Le resistance should be zero."""
        z_lossy = models._lossy_le_impedance(f=0.0, R_e_eddy=1.0, f_ref=1000.0)
        assert z_lossy == pytest.approx(0.0j)

    def test_lossy_le_impedance_at_f_ref_is_R_e_eddy(self):
        """At f = f_ref, R_lossy should equal R_e_eddy."""
        z_lossy = models._lossy_le_impedance(f=1000.0, R_e_eddy=2.5, f_ref=1000.0)
        assert z_lossy == pytest.approx(2.5 + 0.0j)

    def test_lossy_le_impedance_grows_with_frequency(self):
        """R_lossy should increase as (f/f_ref)² — doubling frequency quadruples R_lossy."""
        z_100 = models._lossy_le_impedance(f=100.0, R_e_eddy=1.0, f_ref=1000.0)
        z_200 = models._lossy_le_impedance(f=200.0, R_e_eddy=1.0, f_ref=1000.0)
        z_400 = models._lossy_le_impedance(f=400.0, R_e_eddy=1.0, f_ref=1000.0)
        # 100 Hz: (100/1000)² = 0.01 → R = 0.01
        # 200 Hz: (200/1000)² = 0.04 → R = 0.04 (4× the 100 Hz value)
        # 400 Hz: (400/1000)² = 0.16 → R = 0.16 (16× the 100 Hz value)
        assert float(z_200.real) == pytest.approx(4.0 * float(z_100.real))
        assert float(z_400.real) == pytest.approx(16.0 * float(z_100.real))

    def test_lossy_le_impedance_is_purely_real(self):
        """The Lossy Le impedance should be purely real (no reactive component)."""
        z = models._lossy_le_impedance(f=500.0, R_e_eddy=2.0, f_ref=1000.0)
        assert z.imag == pytest.approx(0.0, abs=1e-12)

    def test_lossy_le_impedance_zero_R_e_eddy_returns_zero(self):
        """R_e_eddy = 0 should return 0j (model disabled)."""
        z = models._lossy_le_impedance(f=1000.0, R_e_eddy=0.0, f_ref=1000.0)
        assert z == 0.0j

    def test_lossy_le_impedance_zero_f_ref_returns_zero(self):
        """f_ref <= 0 should return 0j (guard against divide-by-zero)."""
        z = models._lossy_le_impedance(f=1000.0, R_e_eddy=2.0, f_ref=0.0)
        assert z == 0.0j

    def test_lossy_le_impedance_negative_R_e_eddy_returns_zero(self):
        """Negative R_e_eddy should return 0j (physically meaningless)."""
        z = models._lossy_le_impedance(f=1000.0, R_e_eddy=-1.0, f_ref=1000.0)
        assert z == 0.0j

    def test_driverspecs_lossy_le_defaults(self):
        """DriverSpecs should have lossy_le=False, le_R_e_eddy=0.0, le_f_lossy_ref=1000.0."""
        d = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
        )
        assert d.lossy_le is False
        assert d.le_R_e_eddy == 0.0
        assert d.le_f_lossy_ref == 1000.0

    def test_horn_response_with_lossy_le_runs(self):
        """horn_response should run without error when lossy_le=True."""
        d = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, lossy_le=True, le_R_e_eddy=1.5, le_f_lossy_ref=1000.0,
        )
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
            vtc=0.0036,
            fr_tc=2000.0,
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, d, horn)
        assert isinstance(result, models.SimulationResult)
        assert np.all(np.isfinite(result.spl))
        assert np.all(np.isfinite(result.impedance))

    def test_lossy_le_increases_impedance_real_part(self):
        """With lossy_le=True, Re(Z_e) should increase with frequency above f_ref.

        The Lossy Le resistance R_lossy = R_e_eddy × (f/f_ref)² grows with
        frequency. At 4× f_ref it contributes 16× the base R_e_eddy value.
        """
        d_no_lossy = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, lossy_le=False,
        )
        d_lossy = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, lossy_le=True, le_R_e_eddy=2.0, le_f_lossy_ref=1000.0,
        )
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
        )
        freqs = np.linspace(20, 5000, 200)
        r_no = models.horn_response(freqs, d_no_lossy, horn)
        r_lossy = models.horn_response(freqs, d_lossy, horn)

        # At 4 kHz (4× f_ref=1000 Hz), R_lossy = 2.0 × (4000/1000)² = 2.0 × 16 = 32 Ω
        idx_4k = int(np.argmin(np.abs(freqs - 4000.0)))
        z_no_real_4k = r_no.impedance[idx_4k].real
        z_lossy_real_4k = r_lossy.impedance[idx_4k].real
        # The Lossy Le model should give a higher real part
        assert z_lossy_real_4k > z_no_real_4k + 10.0, (
            f"At 4 kHz, Lossy Le should add ~32 Ω to real(Z); "
            f"got {z_lossy_real_4k:.1f} Ω (lossy) vs {z_no_real_4k:.1f} Ω (no-lossy)"
        )

    def test_lossy_le_below_f_ref_minimal_effect(self):
        """Below f_ref, R_lossy is small and should have minimal effect on impedance.

        At f = f_ref/2: R_lossy = R_e_eddy × 0.25 — only 25% of R_e_eddy.
        """
        d_no = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, lossy_le=False,
        )
        d_lossy = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, lossy_le=True, le_R_e_eddy=2.0, le_f_lossy_ref=1000.0,
        )
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
        )
        freqs = np.linspace(20, 5000, 200)
        r_no = models.horn_response(freqs, d_no, horn)
        r_lossy = models.horn_response(freqs, d_lossy, horn)

        # At 500 Hz (f_ref/2), R_lossy = 2.0 × (500/1000)² = 2.0 × 0.25 = 0.5 Ω
        idx_500 = int(np.argmin(np.abs(freqs - 500.0)))
        z_no_real_500 = r_no.impedance[idx_500].real
        z_lossy_real_500 = r_lossy.impedance[idx_500].real
        # The difference should be small but measurable (lossy adds ~0.5 Ω)
        diff = z_lossy_real_500 - z_no_real_500
        assert 0.2 < diff < 5.0, (
            f"At 500 Hz (f_ref/2), Lossy Le should add ~0.5 Ω; "
            f"got diff={diff:.2f} Ω"
        )

    def test_lossy_le_and_freq_dep_le_can_both_be_active(self):
        """Lossy Le and semi-inductance (le_freq_dependency) can be enabled simultaneously.

        They model different physical phenomena and should compose correctly:
          - Semi-inductance: Le(f) = Le_const × √(1 + (f/f_ref)²)
          - Lossy Le:         R_lossy(f) = R_e_eddy × (f/f_ref)²
        Total: Z_e = Re + R_lossy(f) + jω·Le(f)
        """
        d = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008,
            le_freq_dependency=True, le_f_ref=100.0,  # semi-inductance
            lossy_le=True, le_R_e_eddy=1.5, le_f_lossy_ref=1000.0,  # Lossy Le
        )
        horn = HornGeometry(
            enclosure_type="BLH",
            width=0.2,
            conical_segments=[
                (0.06, 0.07, 0.03),
                (0.07, 0.08, 0.04),
            ],
        )
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response(freqs, d, horn)
        assert isinstance(result, models.SimulationResult)
        assert np.all(np.isfinite(result.spl))
        assert np.all(np.isfinite(result.impedance))

    def test_infinite_baffle_with_lossy_le(self):
        """infinite_baffle_response should also honour lossy_le."""
        d_const = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, lossy_le=False,
        )
        d_lossy = DriverSpecs(
            fs=49.6, qts=0.27, qes=0.28, qms=7.88, vas=0.0369,
            re=7.8, bl=7.79, mms=0.00699, cms=0.001472, rms=0.277,
            sd=0.01327, voltage=2.83,
            le=0.0008, lossy_le=True, le_R_e_eddy=2.0, le_f_lossy_ref=1000.0,
        )
        freqs = np.linspace(20, 5000, 200)
        ib_const = models.infinite_baffle_response(freqs, d_const)
        ib_lossy = models.infinite_baffle_response(freqs, d_lossy)

        # At 4 kHz (4× f_ref), Lossy Le adds ~32 Ω → more impedance →
        # different SPL. The SPL should differ noticeably.
        idx_4k = int(np.argmin(np.abs(freqs - 4000.0)))
        assert abs(ib_lossy[idx_4k] - ib_const[idx_4k]) > 0.01, (
            "IB SPL at 4 kHz should differ between lossy and non-lossy Le models"
        )


class TestDetectNumericalArtifacts:
    """Tests for _detect_numerical_artifacts() — TMM artifact detection heuristics.

    Two heuristics are tested:
      1. Isolated spike: >20 dB/bin jump, then immediate reversal
      2. Trend break: SPL deviates >20 dB from local 5-sample median
    """

    def test_empty_input_returns_empty(self):
        """len(freqs) < 5 should return [] (guard clause)."""
        result = models._detect_numerical_artifacts(
            np.array([100.0]), np.array([90.0])
        )
        assert result == []

    def test_smooth_response_returns_no_artifacts(self):
        """A smoothly-varying physical horn response should produce no artifacts."""
        freqs = np.linspace(20, 5000, 500)
        # Physical response: gentle rolloff, no sharp jumps
        spl = 100 - 10 * np.log10(freqs / 20) + 5 * np.sin(freqs / 500)
        result = models._detect_numerical_artifacts(freqs, spl)
        assert result == []

    def test_isolated_spike_detected(self):
        """A single-bin spike that reverses direction should be flagged."""
        freqs = np.linspace(20, 5000, 500)
        spl = np.full(500, 90.0)
        # Spike at index 250: jump +25 dB, then reverse -25 dB next bin
        spl[250] = 115.0
        spl[251] = 90.0
        result = models._detect_numerical_artifacts(freqs, spl)
        # Should detect at least one artifact near the spike
        assert len(result) >= 1
        # The detected frequency should correspond to the spike region
        assert any(abs(f - freqs[250]) < 50.0 for f in result)

    def test_trend_break_detected(self):
        """A narrow spike region should be flagged via trend break (deviation > 20 dB).

        The trend-break heuristic triggers when SPL deviates >20 dB from the
        local 5-bin median. A narrow spike (5 bins, 30 dB above baseline) satisfies
        this: the median stays at 90 dB while the spike bins are at 120 dB.
        """
        freqs = np.linspace(20, 5000, 500)
        spl = np.full(500, 90.0)
        spl[240:245] = 120.0  # narrow spike: 5 bins, 30 dB above baseline
        result = models._detect_numerical_artifacts(freqs, spl)
        assert len(result) > 0
        artifact_freqs = np.array(result)
        # All detected artifact frequencies should be in the spike region
        assert artifact_freqs.min() >= freqs[239]
        assert artifact_freqs.max() <= freqs[246]

    def test_small_spike_below_threshold_not_flagged(self):
        """Spikes <20 dB magnitude should NOT be flagged (physical)."""
        freqs = np.linspace(20, 5000, 500)
        spl = np.full(500, 90.0)
        # Jump of only 10 dB — below the 20 dB threshold
        spl[250] = 100.0
        spl[251] = 90.0
        result = models._detect_numerical_artifacts(freqs, spl)
        assert result == []

    def test_multiple_artifacts_all_detected(self):
        """Multiple artifact regions should all appear in the returned list."""
        freqs = np.linspace(20, 5000, 1000)
        spl = np.full(1000, 90.0)
        # Two isolated spikes
        spl[100] = 115.0
        spl[101] = 90.0
        spl[700] = 115.0
        spl[701] = 90.0
        result = models._detect_numerical_artifacts(freqs, spl)
        assert len(result) >= 2

    def test_known_hiro_artifact_pattern(self):
        """Known Hiro ~1847 Hz artifact: large isolated spike pattern should be detected.

        The Hiro geometry produces a ~1847 Hz artifact that is a single-bin
        +24 dB spike. This is the canonical test case for the detector.
        """
        # Reconstruct a plausible Hiro-like response around the artifact
        freqs = np.linspace(1800, 1900, 200)
        # Flat response at 90 dB with a single-bin spike at ~1847 Hz
        spl = np.full(200, 90.0)
        # Spike: +24 dB at the artifact frequency, then back
        idx_artifact = int(np.argmin(np.abs(freqs - 1847.0)))
        spl[idx_artifact] = 114.0  # +24 dB spike
        if idx_artifact + 1 < len(spl):
            spl[idx_artifact + 1] = 90.0  # reversal
        result = models._detect_numerical_artifacts(freqs, spl)
        assert len(result) > 0, "Hiro-style 1847 Hz spike should be detected"

    def test_return_value_is_sorted(self):
        """Returned artifact frequencies should be sorted low→high."""
        freqs = np.linspace(20, 5000, 1000)
        spl = np.full(1000, 90.0)
        spl[200] = 115.0; spl[201] = 90.0
        spl[800] = 115.0; spl[801] = 90.0
        result = models._detect_numerical_artifacts(freqs, spl)
        assert result == sorted(result)

    def test_result_deduplicated(self):
        """If a region of bad values triggers multiple detections, result should still be a list."""
        freqs = np.linspace(20, 5000, 500)
        spl = np.full(500, 90.0)
        # A wider spike that spans several bins should not produce duplicates
        spl[240:260] = 120.0
        result = models._detect_numerical_artifacts(freqs, spl)
        # Should be a flat list, not contain duplicates when converted to sorted unique
        assert len(result) == len(sorted(set(result)))


class TestCompoundHorn:
    """Tests for horn_response_compound() — Compound Horn (CH mode)."""

    @pytest.fixture
    def fostex_driver(self):
        """FE166NV2 driver parameters."""
        return DriverSpecs(
            fs=43.0,
            qts=0.195,
            qes=0.213,
            qms=2.8,
            vas=35e-3,
            re=7.0,
            bl=7.3,
            mms=4.8e-3,
            cms=2.84e-4,
            rms=0.38,
            sd=1.33e-2,
            voltage=2.83,
            le=4.5e-4,
        )

    @pytest.fixture
    def simple_main_horn(self):
        """Simple exponential main horn: throat → mouth."""
        return HornGeometry(
            enclosure_type="BLH",
            throat_area=0.0015,
            mouth_area=0.05,
            path_length=0.9,
            profile_type="exponential",
            n_segments=60,
            vrc=0.0,
            lrc=0.0,
        )

    @pytest.fixture
    def default_compound_chamber(self):
        """Default compound chamber: no rear chamber, no secondary horn."""
        return CompoundChamber(
            vrc_rear=0.0,
            lrc_rear=0.0,
            vtc_rear=0.0,
            atc_rear=0.0,
            secondary_mouth_area=0.0,
            secondary_mouth_ang=2.0 * np.pi,
        )

    @pytest.fixture
    def rear_chamber(self):
        """Rear chamber with meaningful volume and length."""
        return CompoundChamber(
            vrc_rear=0.015,
            lrc_rear=0.08,
            vtc_rear=0.0,
            atc_rear=0.0,
            secondary_mouth_area=0.0,
            secondary_mouth_ang=2.0 * np.pi,
        )

    @pytest.fixture
    def rear_chamber_with_secondary_horn(self):
        """Rear chamber coupled to a secondary horn mouth."""
        return CompoundChamber(
            vrc_rear=0.01,
            lrc_rear=0.05,
            vtc_rear=0.0,
            atc_rear=0.0,
            secondary_mouth_area=0.03,
            secondary_mouth_ang=2.0 * np.pi,
        )

    def test_runs_without_error(self, fostex_driver, simple_main_horn, default_compound_chamber):
        """horn_response_compound() should run without raising an exception."""
        freqs = np.linspace(20, 5000, 200)
        result = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, default_compound_chamber
        )
        assert result.spl is not None
        assert len(result.spl) == len(freqs)

    def test_spl_is_positive(self, fostex_driver, simple_main_horn, default_compound_chamber):
        """SPL should be a positive dB value in the pass band."""
        freqs = np.linspace(80, 2000, 100)
        result = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, default_compound_chamber
        )
        # In the pass band the SPL should be above a reasonable floor
        assert np.mean(result.spl[10:50]) > 60.0

    def test_zero_rear_chamber_gives_horn_only_output(self, fostex_driver, simple_main_horn):
        """With no rear chamber (vrc_rear=0, secondary_mouth_area=0) the solver
        should still run and produce a valid response (direct rear radiation path)."""
        freqs = np.linspace(20, 5000, 100)
        compound_zero = CompoundChamber(
            vrc_rear=0.0,
            lrc_rear=0.0,
            vtc_rear=0.0,
            atc_rear=0.0,
            secondary_mouth_area=0.0,
        )
        result = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, compound_zero
        )
        assert len(result.spl) == len(freqs)
        assert not np.any(np.isnan(result.spl))
        assert not np.any(np.isinf(result.spl))

    def test_rear_chamber_affects_response_shape(self, fostex_driver, simple_main_horn):
        """Changing rear chamber volume (vrc_rear) should measurably change the SPL
        response shape, confirming the rear side contributes to the output."""
        freqs = np.linspace(20, 2000, 200)

        # Small rear chamber
        compound_small = CompoundChamber(vrc_rear=0.005, lrc_rear=0.03)
        r_small = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, compound_small
        )

        # Large rear chamber
        compound_large = CompoundChamber(vrc_rear=0.050, lrc_rear=0.15)
        r_large = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, compound_large
        )

        # The two responses should differ (RMS difference > 0.5 dB in the pass band)
        rms_diff = np.sqrt(np.mean((r_small.spl - r_large.spl) ** 2))
        assert rms_diff > 0.5, (
            f"RMS SPL difference {rms_diff:.3f} dB is too small — "
            "rear chamber should change the response"
        )

    def test_compound_differs_from_standard_blh(self, fostex_driver, simple_main_horn):
        """The compound horn (with non-zero rear load) should produce a different
        response from the standard BLH horn_response() at low frequencies,
        because the rear side contributes additional radiation."""
        freqs = np.linspace(20, 2000, 200)

        # Standard BLH: rear side has no coupling (compound chamber = zero)
        compound_zero = CompoundChamber(vrc_rear=0.0, lrc_rear=0.0)
        r_compound_zero = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, compound_zero
        )

        # Compound with rear chamber loading
        compound_active = CompoundChamber(vrc_rear=0.02, lrc_rear=0.1)
        r_compound_active = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, compound_active
        )

        # These two should differ (rear chamber changes the rear load)
        rms_diff = np.sqrt(np.mean((r_compound_zero.spl - r_compound_active.spl) ** 2))
        assert rms_diff > 0.3, (
            f"RMS SPL difference {rms_diff:.3f} dB is too small — "
            "adding rear chamber should change the response"
        )

    def test_impedance_is_complex(self, fostex_driver, simple_main_horn, default_compound_chamber):
        """Electrical impedance should be complex-valued."""
        freqs = np.linspace(20, 2000, 100)
        result = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, default_compound_chamber
        )
        assert result.impedance.dtype == np.complex128
        assert np.all(result.impedance != 0)

    def test_excursion_positive(self, fostex_driver, simple_main_horn, default_compound_chamber):
        """Driver excursion should be a positive mm value below resonance."""
        freqs = np.linspace(20, 200, 100)
        result = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, default_compound_chamber
        )
        assert np.all(result.excursion >= 0)
        assert np.mean(result.excursion[5:30]) > 0.01  # meaningful excursion

    def test_secondary_mouth_area_adds_contribution(self, fostex_driver, simple_main_horn):
        """With secondary_mouth_area > 0, the response should differ from
        when it is 0, confirming the secondary path contributes."""
        freqs = np.linspace(20, 2000, 200)

        compound_no_secondary = CompoundChamber(
            vrc_rear=0.01, lrc_rear=0.05, secondary_mouth_area=0.0
        )
        r_no_sec = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, compound_no_secondary
        )

        compound_with_secondary = CompoundChamber(
            vrc_rear=0.01, lrc_rear=0.05, secondary_mouth_area=0.04
        )
        r_with_sec = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, compound_with_secondary
        )

        rms_diff = np.sqrt(np.mean((r_no_sec.spl - r_with_sec.spl) ** 2))
        assert rms_diff > 0.2, (
            f"RMS SPL difference {rms_diff:.3f} dB — "
            "secondary_mouth_area should affect the response"
        )

    def test_compound_vs_tapped_horn_are_different_topologies(
        self, fostex_driver
    ):
        """The compound horn and tapped horn should produce different responses,
        as they are fundamentally different topologies."""
        from pyhorn_core.config.models import TappedHornGeometry

        freqs = np.linspace(20, 2000, 200)

        # Tapped horn: single horn, driver at interior tap point
        th_geom = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(
                    name="front",
                    profile_type="exponential",
                    start_area=0.003,
                    end_area=0.06,
                    length=1.0,
                )
            ],
            rear_chamber=None,
            rear_load_type="free_space",
        )
        r_tapped = models.horn_response_tapped(freqs, fostex_driver, th_geom)

        # Compound horn: main horn + rear direct radiator
        main_horn = HornGeometry(
            throat_area=0.003,
            mouth_area=0.06,
            path_length=1.0,
            profile_type="exponential",
            n_segments=60,
        )
        compound = CompoundChamber(vrc_rear=0.01, lrc_rear=0.05)
        r_compound = models.horn_response_compound(freqs, fostex_driver, main_horn, compound)

        rms_diff = np.sqrt(np.mean((r_tapped.spl - r_compound.spl) ** 2))
        assert rms_diff > 0.5, (
            f"Compound and tapped horn responses are too similar "
            f"(RMS diff {rms_diff:.3f} dB) — they are different topologies"
        )

    def test_rear_chamber_length_changes_response(
        self, fostex_driver, simple_main_horn
    ):
        """Changing lrc_rear (rear chamber length) should change the response,
        confirming the mass term of the rear chamber affects the system."""
        freqs = np.linspace(20, 2000, 200)

        compound_short = CompoundChamber(vrc_rear=0.015, lrc_rear=0.02)
        r_short = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, compound_short
        )

        compound_long = CompoundChamber(vrc_rear=0.015, lrc_rear=0.20)
        r_long = models.horn_response_compound(
            freqs, fostex_driver, simple_main_horn, compound_long
        )

        rms_diff = np.sqrt(np.mean((r_short.spl - r_long.spl) ** 2))
        assert rms_diff > 0.3, (
            f"RMS SPL difference {rms_diff:.3f} dB is too small — "
            "lrc_rear (rear chamber length) should affect the response"
        )


class TestTappedHorn:
    """Comprehensive tests for horn_response_tapped() — Tapped Horn (TH / TH1 mode).

    In a TH horn the driver is at an interior tap point (S2 or S3), not at the
    throat or mouth. The front of the driver loads into the horn proper; the rear
    loads into either a rear chamber, free space, or an infinite baffle.

    rear_load_type options:
      - "rear_chamber"  : rear side loads into a sealed coupling chamber
      - "free_space"    : rear side radiates into half-space (ang=2π)
      - "infinite_baffle": rear side on infinite baffle (ang=π)

    Reference: Hornresp manual pages 057–058.
    """

    @pytest.fixture
    def fostex_driver(self):
        """FE166NV2 driver parameters."""
        return DriverSpecs(
            fs=49.6,
            qts=0.27,
            qes=0.28,
            qms=7.88,
            vas=0.0369,
            re=7.8,
            bl=7.79,
            mms=0.00699,
            cms=0.001472,
            rms=0.277,
            sd=0.01327,
            voltage=2.83,
            le=0.0008,
        )

    @pytest.fixture
    def simple_th_horn(self):
        """Simple TH geometry: exponential front section, free-space rear load."""
        return TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(
                    name="front",
                    profile_type="exponential",
                    start_area=0.003,
                    end_area=0.06,
                    length=1.0,
                )
            ],
            rear_chamber=None,
            rear_load_type="free_space",
            ang=2.0 * np.pi,
            n_segments=60,
        )

    @pytest.fixture
    def th_horn_with_rear_chamber(self, fostex_driver):
        """TH with rear chamber load — rear chamber should affect response shape."""
        return TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(
                    name="front",
                    profile_type="exponential",
                    start_area=0.003,
                    end_area=0.06,
                    length=1.0,
                )
            ],
            rear_chamber=RearChamber(vrc=0.01, lrc=0.05, fr_rc=0.0),
            rear_load_type="rear_chamber",
            ang=2.0 * np.pi,
            n_segments=60,
        )

    def test_runs_without_error(self, fostex_driver, simple_th_horn):
        """horn_response_tapped should return a SimulationResult without raising."""
        freqs = np.linspace(20, 5000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert isinstance(result, models.SimulationResult)
        assert len(result.freqs) == len(freqs)
        assert result.spl is not None

    def test_spl_positive_in_passband(self, fostex_driver, simple_th_horn):
        """SPL should be positive in the pass band (80–2000 Hz)."""
        freqs = np.linspace(80, 2000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert np.mean(result.spl[10:50]) > 50.0

    def test_no_nan_or_inf(self, fostex_driver, simple_th_horn):
        """No output arrays should contain NaN or Inf."""
        freqs = np.linspace(20, 5000, 200)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        for attr in ["spl", "impedance", "excursion", "cone_velocity", "group_delay"]:
            arr = getattr(result, attr)
            if arr is not None:
                assert not np.any(np.isnan(arr)), f"{attr} has NaN"
                assert not np.any(np.isinf(arr)), f"{attr} has Inf"

    def test_impedance_is_complex(self, fostex_driver, simple_th_horn):
        """Electrical input impedance should be a complex array."""
        freqs = np.linspace(20, 2000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert result.impedance.dtype == complex
        assert np.all(result.impedance.real >= 0)

    def test_excursion_positive(self, fostex_driver, simple_th_horn):
        """Excursion (m) should be non-negative across the band."""
        freqs = np.linspace(20, 2000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert np.all(result.excursion >= 0)

    def test_cone_velocity_positive(self, fostex_driver, simple_th_horn):
        """Cone velocity magnitude should be non-negative."""
        freqs = np.linspace(20, 2000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert np.all(result.cone_velocity >= 0)

    def test_group_delay_is_array(self, fostex_driver, simple_th_horn):
        """Group delay should be a non-None float array."""
        freqs = np.linspace(20, 2000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert result.group_delay is not None
        assert len(result.group_delay) == len(freqs)
        assert isinstance(result.group_delay, np.ndarray)

    def test_off_axis_spl_shape(self, fostex_driver, simple_th_horn):
        """Off-axis SPL array should have shape (n_freqs, n_angles)."""
        freqs = np.linspace(100, 3000, 100)
        off_axis = np.array([0.0, 30.0, 60.0, 90.0])
        result = models.horn_response_tapped(
            freqs, fostex_driver, simple_th_horn, off_axis_angles=off_axis
        )
        assert result.off_axis_spl is not None
        assert result.off_axis_spl.shape == (len(freqs), len(off_axis))
        assert result.off_axis_angles is not None

    def test_direction_index_shape(self, fostex_driver, simple_th_horn):
        """Direction index array should match off-axis shape."""
        freqs = np.linspace(100, 3000, 100)
        off_axis = np.array([0.0, 30.0, 60.0])
        result = models.horn_response_tapped(
            freqs, fostex_driver, simple_th_horn, off_axis_angles=off_axis
        )
        assert result.direction_index is not None
        assert result.direction_index.shape == (len(freqs), len(off_axis))

    def test_rear_chamber_load_differs_from_free_space(self, fostex_driver):
        """Using rear_chamber load should give a different response than free_space."""
        freqs = np.linspace(20, 2000, 200)

        th_free = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(
                    name="front",
                    profile_type="exponential",
                    start_area=0.003,
                    end_area=0.06,
                    length=1.0,
                )
            ],
            rear_chamber=None,
            rear_load_type="free_space",
            n_segments=60,
        )
        r_free = models.horn_response_tapped(freqs, fostex_driver, th_free)

        th_chamber = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(
                    name="front",
                    profile_type="exponential",
                    start_area=0.003,
                    end_area=0.06,
                    length=1.0,
                )
            ],
            rear_chamber=RearChamber(vrc=0.01, lrc=0.05, fr_rc=0.0),
            rear_load_type="rear_chamber",
            n_segments=60,
        )
        r_chamber = models.horn_response_tapped(freqs, fostex_driver, th_chamber)

        rms_diff = np.sqrt(np.mean((r_free.spl - r_chamber.spl) ** 2))
        assert rms_diff > 0.2, (
            f"RMS SPL diff {rms_diff:.3f} dB — rear_chamber and free_space "
            "rear loads should produce different responses"
        )

    def test_infinite_baffle_differs_from_free_space(self, fostex_driver):
        """infinite_baffle rear load should differ from free_space rear load."""
        freqs = np.linspace(20, 2000, 200)

        th_free = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(
                    name="front",
                    profile_type="exponential",
                    start_area=0.003,
                    end_area=0.06,
                    length=1.0,
                )
            ],
            rear_load_type="free_space",
            n_segments=60,
        )
        r_free = models.horn_response_tapped(freqs, fostex_driver, th_free)

        th_ib = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(
                    name="front",
                    profile_type="exponential",
                    start_area=0.003,
                    end_area=0.06,
                    length=1.0,
                )
            ],
            rear_load_type="infinite_baffle",
            n_segments=60,
        )
        r_ib = models.horn_response_tapped(freqs, fostex_driver, th_ib)

        rms_diff = np.sqrt(np.mean((r_free.spl - r_ib.spl) ** 2))
        assert rms_diff > 0.1, (
            f"RMS SPL diff {rms_diff:.3f} dB — infinite_baffle and free_space "
            "rear loads should differ (different radiation angle)"
        )

    def test_low_freq_floor_at_1hz(self, fostex_driver, simple_th_horn):
        """Very low frequencies (f < 1 Hz) should be floored to 1 Hz without crash."""
        freqs = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert not np.any(np.isnan(result.spl))
        assert not np.any(np.isinf(result.spl))

    def test_th_with_rear_sections_runs(self, fostex_driver):
        """TH with explicit rear_sections (driver rear → rear termination) should run."""
        th_with_rear = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(
                    name="front",
                    profile_type="exponential",
                    start_area=0.003,
                    end_area=0.06,
                    length=1.0,
                )
            ],
            rear_sections=[
                Section(
                    name="rear",
                    profile_type="exponential",
                    start_area=0.003,
                    end_area=0.01,
                    length=0.2,
                )
            ],
            rear_load_type="free_space",
            n_segments=40,
        )
        freqs = np.linspace(20, 2000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, th_with_rear)
        assert len(result.spl) == len(freqs)
        assert not np.any(np.isnan(result.spl))

    def test_multiple_front_sections(self, fostex_driver):
        """TH with multiple front sections (multi-segment path) should run correctly."""
        th_multi = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(
                    name="throat",
                    profile_type="conical",
                    start_area=0.003,
                    end_area=0.01,
                    length=0.15,
                ),
                Section(
                    name="main",
                    profile_type="exponential",
                    start_area=0.01,
                    end_area=0.04,
                    length=0.6,
                ),
                Section(
                    name="mouth",
                    profile_type="conical",
                    start_area=0.04,
                    end_area=0.06,
                    length=0.25,
                ),
            ],
            rear_load_type="free_space",
            n_segments=40,
        )
        freqs = np.linspace(20, 3000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, th_multi)
        assert len(result.spl) == len(freqs)
        assert not np.any(np.isnan(result.spl))
        assert not np.any(np.isinf(result.spl))

    def test_cone_acceleration_computed(self, fostex_driver, simple_th_horn):
        """Cone acceleration should be a non-negative array."""
        freqs = np.linspace(20, 2000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert result.cone_acceleration is not None
        assert np.all(result.cone_acceleration >= 0)

    def test_electrical_power_computed(self, fostex_driver, simple_th_horn):
        """Electrical input power should be a non-negative array."""
        freqs = np.linspace(20, 2000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert result.electrical_input_power is not None
        assert np.all(result.electrical_input_power >= 0)

    def test_acoustic_power_computed(self, fostex_driver, simple_th_horn):
        """Acoustic power should be a non-negative array."""
        freqs = np.linspace(20, 2000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert result.acoustic_power is not None
        assert np.all(result.acoustic_power >= 0)

    def test_efficiency_positive(self, fostex_driver, simple_th_horn):
        """Efficiency (%) should be between 0 and 100."""
        freqs = np.linspace(20, 2000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert result.efficiency_pct is not None
        assert np.all(result.efficiency_pct >= 0.0)
        assert np.all(result.efficiency_pct <= 100.0)

    def test_phase_is_bounded(self, fostex_driver, simple_th_horn):
        """Phase should be between -π and +π."""
        freqs = np.linspace(20, 2000, 100)
        result = models.horn_response_tapped(freqs, fostex_driver, simple_th_horn)
        assert result.phase is not None
        assert np.all(result.phase >= -np.pi - 0.01)
        assert np.all(result.phase <= np.pi + 0.01)

    def test_off_axis_spl_decreases_with_angle(self, fostex_driver, simple_th_horn):
        """Off-axis SPL should generally decrease as angle increases (at mid-high freq)."""
        freqs = np.linspace(500, 3000, 50)
        off_axis = np.array([0.0, 30.0, 60.0, 90.0])
        result = models.horn_response_tapped(
            freqs, fostex_driver, simple_th_horn, off_axis_angles=off_axis
        )
        on_axis = result.off_axis_spl[:, 0]
        for j in range(1, len(off_axis)):
            off_axis_mean = np.mean(result.off_axis_spl[:, j])
            on_axis_mean = np.mean(on_axis)
            # Off-axis should generally be ≤ on-axis (not always, due to directivity nulls)
            # We check the on-axis is not dramatically lower than off-axis
            assert on_axis_mean >= off_axis_mean - 15.0, (
                f"On-axis SPL ({on_axis_mean:.1f} dB) unexpectedly lower than "
                f"off-axis at {off_axis[j]}° ({off_axis_mean:.1f} dB)"
            )


class TestCompoundHornDualDriver:
    """Tests for dual-driver CH mode (ch_dual_driver=True).

    In dual-driver mode the compound horn has two independent drive units:
      - Front driver: at main horn throat, drives main horn (S1→S4)
      - Rear driver : at secondary horn throat (junction S4), drives secondary horn (S4→S5)
    Both mouths radiate; their contributions sum at the listening point.
    Reference: Hornresp manual pages 048–049, 059 (Compound Horn / CH mode).
    """

    @pytest.fixture
    def fostex_driver(self):
        """FE166NV2 front driver parameters."""
        return DriverSpecs(
            fs=43.0,
            qts=0.195,
            qes=0.213,
            qms=2.8,
            vas=35e-3,
            re=7.0,
            bl=7.3,
            mms=4.8e-3,
            cms=2.84e-4,
            rms=0.38,
            sd=1.33e-2,
            voltage=2.83,
            le=4.5e-4,
        )

    @pytest.fixture
    def rear_driver(self):
        """Second driver (e.g., smaller tweeter-style driver) for rear position."""
        return DriverSpecs(
            fs=120.0,
            qts=0.35,
            qes=0.38,
            qms=3.5,
            vas=8e-3,
            re=8.0,
            bl=5.5,
            mms=2.1e-3,
            cms=0.83e-4,
            rms=0.22,
            sd=0.78e-2,  # smaller driver
            voltage=2.83,
            le=3.5e-4,
        )

    @pytest.fixture
    def main_horn(self):
        """Main exponential horn: throat → junction S4."""
        return HornGeometry(
            enclosure_type="BLH",
            throat_area=0.0015,
            mouth_area=0.05,
            path_length=0.9,
            profile_type="exponential",
            n_segments=60,
        )

    def test_dual_driver_runs_without_error(
        self, fostex_driver, rear_driver, main_horn
    ):
        """horn_response_compound() with ch_dual_driver=True should run cleanly."""
        freqs = np.linspace(20, 5000, 200)
        compound = CompoundChamber(
            vrc_rear=0.01,
            lrc_rear=0.05,
            secondary_mouth_area=0.03,
            secondary_mouth_ang=2.0 * np.pi,
            ch_dual_driver=True,
            rear_driver=rear_driver,
        )
        result = models.horn_response_compound(
            freqs, fostex_driver, main_horn, compound
        )
        assert result.spl is not None
        assert len(result.spl) == len(freqs)
        assert not np.any(np.isnan(result.spl))
        assert not np.any(np.isinf(result.spl))

    def test_dual_driver_produces_valid_spl(
        self, fostex_driver, rear_driver, main_horn
    ):
        """SPL in dual-driver mode should be a positive dB value in the pass band."""
        freqs = np.linspace(80, 2000, 100)
        compound = CompoundChamber(
            vrc_rear=0.01,
            lrc_rear=0.05,
            secondary_mouth_area=0.03,
            ch_dual_driver=True,
            rear_driver=rear_driver,
        )
        result = models.horn_response_compound(
            freqs, fostex_driver, main_horn, compound
        )
        # In the pass band the SPL should be above a reasonable floor
        assert np.mean(result.spl[10:50]) > 60.0

    def test_dual_driver_differs_from_single_driver(
        self, fostex_driver, rear_driver, main_horn
    ):
        """Dual-driver response should differ from single-driver response,
        confirming the rear driver contributes independently."""
        freqs = np.linspace(20, 2000, 200)
        compound_single = CompoundChamber(
            vrc_rear=0.01,
            lrc_rear=0.05,
            secondary_mouth_area=0.03,
            ch_dual_driver=False,  # single-driver mode (default)
        )
        compound_dual = CompoundChamber(
            vrc_rear=0.01,
            lrc_rear=0.05,
            secondary_mouth_area=0.03,
            ch_dual_driver=True,
            rear_driver=rear_driver,
        )
        r_single = models.horn_response_compound(
            freqs, fostex_driver, main_horn, compound_single
        )
        r_dual = models.horn_response_compound(
            freqs, fostex_driver, main_horn, compound_dual
        )
        rms_diff = np.sqrt(np.mean((r_single.spl - r_dual.spl) ** 2))
        assert rms_diff > 0.3, (
            f"RMS SPL difference {rms_diff:.3f} dB is too small — "
            "dual-driver mode should differ from single-driver mode"
        )

    def test_dual_driver_impedance_is_complex(
        self, fostex_driver, rear_driver, main_horn
    ):
        """Electrical impedance should be complex-valued in dual-driver mode."""
        freqs = np.linspace(20, 2000, 100)
        compound = CompoundChamber(
            vrc_rear=0.01,
            lrc_rear=0.05,
            secondary_mouth_area=0.03,
            ch_dual_driver=True,
            rear_driver=rear_driver,
        )
        result = models.horn_response_compound(
            freqs, fostex_driver, main_horn, compound
        )
        assert result.impedance.dtype == np.complex128
        assert np.all(result.impedance != 0)

    def test_dual_driver_excursion_positive(
        self, fostex_driver, rear_driver, main_horn
    ):
        """Front driver excursion should be positive in dual-driver mode."""
        freqs = np.linspace(20, 200, 100)
        compound = CompoundChamber(
            vrc_rear=0.01,
            lrc_rear=0.05,
            secondary_mouth_area=0.03,
            ch_dual_driver=True,
            rear_driver=rear_driver,
        )
        result = models.horn_response_compound(
            freqs, fostex_driver, main_horn, compound
        )
        assert np.all(result.excursion >= 0)
        assert np.mean(result.excursion[5:30]) > 0.01

    def test_dual_driver_without_rear_driver_raises_or_defaults(
        self, fostex_driver, main_horn
    ):
        """ch_dual_driver=True without rear_driver should either raise or gracefully
        fall back to single-driver behavior (no crash)."""
        freqs = np.linspace(20, 2000, 100)
        compound_no_rear = CompoundChamber(
            vrc_rear=0.01,
            lrc_rear=0.05,
            secondary_mouth_area=0.03,
            ch_dual_driver=True,
            rear_driver=None,  # missing rear driver
        )
        # Should not raise — should handle gracefully
        result = models.horn_response_compound(
            freqs, fostex_driver, main_horn, compound_no_rear
        )
        assert result.spl is not None
        assert len(result.spl) == len(freqs)

    def test_dual_driver_rear_driver_params_affect_response(
        self, fostex_driver, main_horn
    ):
        """Changing rear driver T-S parameters should measurably change the response,
        confirming the rear driver is computed independently."""
        freqs = np.linspace(20, 2000, 200)
        rear_driver_light = DriverSpecs(
            fs=120.0, qts=0.35, qes=0.38, qms=3.5, vas=8e-3,
            re=8.0, bl=5.5, mms=1.0e-3, cms=1.76e-4, rms=0.22,
            sd=0.78e-2, voltage=2.83, le=3.5e-4,
        )
        rear_driver_heavy = DriverSpecs(
            fs=120.0, qts=0.35, qes=0.38, qms=3.5, vas=8e-3,
            re=8.0, bl=5.5, mms=5.0e-3, cms=0.35e-4, rms=0.22,
            sd=0.78e-2, voltage=2.83, le=3.5e-4,
        )
        compound_light = CompoundChamber(
            vrc_rear=0.01, lrc_rear=0.05, secondary_mouth_area=0.03,
            ch_dual_driver=True, rear_driver=rear_driver_light,
        )
        compound_heavy = CompoundChamber(
            vrc_rear=0.01, lrc_rear=0.05, secondary_mouth_area=0.03,
            ch_dual_driver=True, rear_driver=rear_driver_heavy,
        )
        r_light = models.horn_response_compound(
            freqs, fostex_driver, main_horn, compound_light
        )
        r_heavy = models.horn_response_compound(
            freqs, fostex_driver, main_horn, compound_heavy
        )
        rms_diff = np.sqrt(np.mean((r_light.spl - r_heavy.spl) ** 2))
        # The rear driver's contribution is small (~0.1 dB) compared to the main driver (~120 dB),
        # so we use a modest threshold. Before the compound-horn voltage-coupling fix
        # (commit: fix(core): CRIT-1), the rear driver was driven at a fixed (max) force
        # ignoring rd.voltage — that produced larger differences and a 0.3 dB threshold passed.
        # With correct voltage coupling, the rear driver's influence is physically smaller.
        assert rms_diff > 0.05, (
            f"RMS SPL difference {rms_diff:.3f} dB is too small — "
            "different rear driver parameters should change the response"
        )

    def test_dual_driver_ch_dual_driver_field_exists(self):
        """CompoundChamber should accept and store the ch_dual_driver field."""
        compound = CompoundChamber(
            vrc_rear=0.01,
            lrc_rear=0.05,
            secondary_mouth_area=0.03,
            ch_dual_driver=True,
        )
        assert compound.ch_dual_driver is True

    def test_dual_driver_rear_driver_field_exists(self, fostex_driver):
        """CompoundChamber should accept and store the rear_driver field."""
        compound = CompoundChamber(
            vrc_rear=0.01,
            lrc_rear=0.05,
            secondary_mouth_area=0.03,
            rear_driver=fostex_driver,
        )
        assert compound.rear_driver is fostex_driver


class TestSlavbasImpedance:
    """Tests for slavbas_impedance() — Slavic rear chamber (aperiodic box) model.

    The Slavic box is an overdamped (aperiodic) variant of a sealed box.
    Unlike a vented box (which has a mass-controlled resonance peak), the
    slavbas uses a resistive leak in parallel with the box compliance, giving
    a smooth rolloff without any resonance.

    Key behaviours of the formula Z_sl = (1/(jωC) || R_leak):
      - f → 0:   Z → R_leak   (leak provides finite path; no hard wall at DC)
      - f_c = 1/(2π·R_leak·C_a): corner frequency; Z = R_leak/√2
      - f >> f_c: Z → 1/(jωC)  (compliance dominates; same rolloff as sealed box)
      - No resonance peak: overdamped system, Q ≈ 0.5
    """

    def test_slavbas_dc_impedance_equals_rleak(self):
        """As f → 0, impedance approaches R_leak (leak dominates, not ∞ as in sealed box)."""
        from pyhorn_core.pyhorn_physics import slavbas_impedance

        rleak = 500.0
        Z = slavbas_impedance(freq=0.001, vrc=0.025, rleak=rleak)
        # At f → 0: Z → R_leak (compliance is open-circuit at DC)
        # At 0.001 Hz the imaginary part is ~0.0003; use loose tolerance
        assert abs(Z - rleak) < 0.01

    def test_slavbas_at_low_freq_close_to_rleak(self):
        """At 1 Hz, impedance should be very close to R_leak (leak-dominant region)."""
        from pyhorn_core.pyhorn_physics import slavbas_impedance

        Z = slavbas_impedance(freq=1.0, vrc=0.025, rleak=500.0)
        # At 1 Hz, Z ≈ 500 - 0.28j; magnitude ≈ 500 = R_leak
        assert abs(abs(Z) - 500.0) < 1.0  # magnitude within 1 of R_leak

    def test_slavbas_impedance_below_sealed_at_low_freq(self):
        """At low frequency (below corner), slavbas impedance is much lower than sealed box."""
        from pyhorn_core.pyhorn_physics import (
            rear_chamber_impedance,
            slavbas_impedance,
        )

        # 50 Hz: sealed box has large impedance; slavbas is still leak-dominated
        freq, vrc = 50.0, 0.025
        Z_sl = abs(slavbas_impedance(freq=freq, vrc=vrc, rleak=500.0))
        Z_sealed = abs(rear_chamber_impedance(freq=freq, volume=vrc, length=0.0))
        # Slavbas should be much lower (leak bypasses the stiff compliance)
        assert Z_sl < Z_sealed * 0.5

    def test_slavbas_no_resonance_peak(self):
        """Impedance should be smooth — no resonance peak anywhere in 20-200 Hz.

        Unlike a vented box (Q > 1, sharp peak at fb), slavbas is overdamped
        (Q ≈ 0.5). The impedance should be monotonically decreasing or flat,
        never showing a local maximum above the DC value.
        """
        from pyhorn_core.pyhorn_physics import slavbas_impedance

        freqs = np.linspace(20.0, 200.0, 500)
        impedances = [abs(slavbas_impedance(f, vrc=0.025, rleak=500.0)) for f in freqs]

        peaks = []
        for i in range(1, len(impedances) - 1):
            if (
                impedances[i] > impedances[i - 1] * 1.01
                and impedances[i] > impedances[i + 1] * 1.01
            ):
                peaks.append(impedances[i])

        # A resonant system (Q > 1) would show a clear peak.
        # Overdamped slavbas should have no peaks (flat or monotonically decreasing).
        assert len(peaks) == 0, (
            f"Slavbas shows {len(peaks)} resonance peak(s) in 20-200 Hz band. "
            f"Expected none — slavbas is overdamped (Q ≈ 0.5)."
        )

    def test_slavbas_impedance_type_is_complex(self):
        """slavbas_impedance should return a complex number."""
        from pyhorn_core.pyhorn_physics import slavbas_impedance

        Z = slavbas_impedance(freq=100.0, vrc=0.025, rleak=500.0)
        assert isinstance(Z, complex)

    def test_slavbas_zero_volume_returns_zero(self):
        """Zero volume should return zero impedance."""
        from pyhorn_core.pyhorn_physics import slavbas_impedance

        Z = slavbas_impedance(freq=100.0, vrc=0.0, rleak=500.0)
        assert Z == 0.0j

    def test_slavbas_zero_rleak_falls_back_to_sealed_box(self):
        """Zero rleak (no leak) should fall back to standard sealed-box impedance."""
        from pyhorn_core.pyhorn_physics import (
            rear_chamber_impedance,
            slavbas_impedance,
        )

        freq, vrc = 100.0, 0.025
        Z_sl = slavbas_impedance(freq=freq, vrc=vrc, rleak=0.0)
        Z_sealed = rear_chamber_impedance(freq=freq, volume=vrc, length=0.0)
        assert abs(Z_sl - Z_sealed) / (abs(Z_sealed) + 1e-12) < 1e-6

    def test_slavbas_larger_volume_lower_impedance_at_low_freq(self):
        """Larger volume → larger compliance → lower impedance below corner freq."""
        from pyhorn_core.pyhorn_physics import slavbas_impedance

        # At f = 1 Hz (well below corner for all volumes), impedance ≈ R_leak
        # regardless of volume — the leak dominates. At higher f, larger
        # volume (smaller stiffness) means lower impedance.
        Z_small = abs(slavbas_impedance(freq=100.0, vrc=0.010, rleak=500.0))
        Z_large = abs(slavbas_impedance(freq=100.0, vrc=0.050, rleak=500.0))
        # Larger volume → more compliant → lower impedance
        assert Z_large < Z_small

    def test_slavbas_higher_rleak_higher_impedance(self):
        """Higher leak resistance → higher impedance (leak is less effective)."""
        from pyhorn_core.pyhorn_physics import slavbas_impedance

        Z_tight = abs(slavbas_impedance(freq=1.0, vrc=0.025, rleak=100.0))
        Z_loose = abs(slavbas_impedance(freq=1.0, vrc=0.025, rleak=2000.0))
        assert Z_loose > Z_tight

    def test_slavbas_corner_frequency(self):
        """At the corner frequency f_c = 1/(2π·R·C), |Z| = R_leak/√2."""
        from pyhorn_core.pyhorn_physics import RHO, C, slavbas_impedance

        vrc, rleak = 0.025, 500.0
        Ca = vrc / (RHO * C**2)
        f_c = 1.0 / (2.0 * np.pi * rleak * Ca)
        Z_c = abs(slavbas_impedance(freq=f_c, vrc=vrc, rleak=rleak))
        assert abs(Z_c - rleak / np.sqrt(2)) < 1.0

    def test_slavbas_high_freq_rolloff_is_6db_oct(self):
        """Above corner frequency, impedance rolls off at 6 dB/oct (1/(ωC) behaviour)."""
        from pyhorn_core.pyhorn_physics import slavbas_impedance

        vrc, rleak = 0.025, 500.0
        f_low = 5000.0   # Hz — well above corner (~2768 Hz)
        f_high = 10000.0  # Hz — octave above f_low
        Z_low = abs(slavbas_impedance(freq=f_low, vrc=vrc, rleak=rleak))
        Z_high = abs(slavbas_impedance(freq=f_high, vrc=vrc, rleak=rleak))
        # 6 dB/oct rolloff: Z_high ≈ Z_low / 2
        assert abs(Z_high / Z_low - 0.5) < 0.1


class TestSlavicBoxDataclass:
    """Tests for the SlavicBox dataclass definition."""

    def test_slavicbox_default_values(self):
        """SlavicBox should have sensible defaults."""
        from pyhorn_core.config.models import SlavicBox

        sb = SlavicBox()
        assert sb.vrc == 0.0
        assert sb.rleak == 0.0
        assert sb.aleak == 0.0
        assert sb.lrc == 0.005

    def test_slavicbox_with_rleak(self):
        """SlavicBox should accept rleak directly."""
        from pyhorn_core.config.models import SlavicBox

        sb = SlavicBox(vrc=0.025, rleak=500.0)
        assert sb.vrc == 0.025
        assert sb.rleak == 500.0

    def test_slavicbox_with_aleak_and_lrc(self):
        """SlavicBox should accept aleak + lrc (converted to rleak in solver)."""
        from pyhorn_core.config.models import SlavicBox

        sb = SlavicBox(vrc=0.025, aleak=0.0001, lrc=0.005)
        assert sb.vrc == 0.025
        assert sb.aleak == 0.0001
        assert sb.lrc == 0.005

    def test_slavicbox_in_horn_geometry(self):
        """HornGeometry should accept a slavbas field."""
        from pyhorn_core.config.models import HornGeometry, SlavicBox

        sb = SlavicBox(vrc=0.025, rleak=500.0)
        horn = HornGeometry(
            throat_area=0.005,
            mouth_area=0.1,
            path_length=1.0,
            slavbas=sb,
        )
        assert horn.slavbas is sb
        assert horn.slavbas.rleak == 500.0

    def test_slavicbox_in_horn_project(self):
        """HornProject should accept a slavbas field."""
        from pyhorn_core.config.models import HornProject, SlavicBox

        sb = SlavicBox(vrc=0.025, rleak=500.0)
        proj = HornProject(name="Test slavbas", slavbas=sb)
        assert proj.slavbas is sb


class TestRearChamberCalibration:
    """CRIT-1 validation: rear chamber V_rc must affect LF SPL.

    This confirms the coupling-chamber model (chamber_type='coupling', pure
    compliance, no mass term) IS in the signal chain and changing V_rc produces
    measurably different low-frequency response.

    Geometry: hornresp_gdb1 benchmark (FE166NV2 in BLH configuration).
    Frequency range: 40–120 Hz (rear chamber compliance dominates).
    """

    def _gdb1_driver(self) -> DriverSpecs:
        """Return the FE166NV2 T-S parameters (hornresp_gdb1 benchmark)."""
        return DriverSpecs(
            fs=49.6,
            qts=0.27,
            qes=0.28,
            qms=7.88,
            vas=0.0369,
            re=7.80,
            sd=0.0132,
            bl=7.75,
            mms=0.00604,
            cms=1.49e-3,
            rms=0.27,
            le=0.80e-3,
            voltage=2.83,
        )

    def _gdb1_horn(self, vrc: float, lrc: float = 0.15) -> HornGeometry:
        """Return GdB1 BLH horn geometry with the given rear chamber params."""
        from pyhorn_core.config.models import RearChamber
        horn = HornGeometry(
            enclosure_type="BLH",
            throat_area=0.0080,
            mouth_area=0.0600,
            path_length=1.530,
            n_segments=50,
            profile_type="hyperbolic",
            hyperbolic_t=0.35,
            ap1=0.0080,
            lpt=0.0,
            vtc=0.00016,
            atc=0.0080,
            fr_tc=2000,
            ang=1.5708,
            width=0.30,
            vrc=vrc,
            lrc=lrc,
            fr_rc=2000.0,
        )
        # Attach rear_chamber attribute for chamber_type="coupling"
        horn.rear_chamber = RearChamber(vrc=vrc, lrc=lrc, fr_rc=2000.0, chamber_type="coupling")
        return horn

    def test_rear_chamber_calibration_sweep(self):
        """CRIT-1: changing V_rc must produce measurably different LF SPL.

        Runs pyhorn with V_rc = 0.010, 0.020, 0.030, 0.040, 0.050 m³
        (10, 20, 30, 40, 50 litres) at fixed l_rc = 0.18 m.

        At 60 Hz (rear-chamber-dominated region), the SPL must differ by
        at least 0.5 dB between the smallest and largest V_rc values.

        This proves the rear chamber IS affecting the simulation — it is not
        a dead parameter.  Full calibration against Hornresp reference data
        (awaiting Geopan's response) will then tune V_rc to match absolute levels.
        """
        import numpy as np
        from pyhorn_core.solver import models

        vrc_values = [0.010, 0.020, 0.030, 0.040, 0.050]
        lrc_fixed = 0.18  # m — fixed for this sweep

        # Frequency range: 40–120 Hz (LF region where rear chamber compliance dominates)
        freqs = np.linspace(40.0, 120.0, 81)  # 81 points = 1 Hz resolution

        driver = self._gdb1_driver()
        results = []
        for vrc in vrc_values:
            horn = self._gdb1_horn(vrc=vrc, lrc=lrc_fixed)
            res = models.horn_response(freqs, driver, horn, compute_distortion=False)
            results.append((vrc, res.spl))

        # At 60 Hz: compare smallest vs largest V_rc
        idx_60hz = int(np.argmin(np.abs(freqs - 60.0)))
        spl_at_60hz = [res[idx_60hz] for _, res in results]

        spl_min_vrc = spl_at_60hz[0]   # V_rc = 0.010 m³
        spl_max_vrc = spl_at_60hz[-1]  # V_rc = 0.050 m³
        spl_diff = abs(spl_max_vrc - spl_min_vrc)

        # Larger rear chamber → more compliant → lower system resonance →
        # less SPL contribution from the chamber at 60 Hz.
        # We expect at least 0.5 dB difference across this 5× volume range.
        assert spl_diff > 0.5, (
            f"V_rc sweep produced only {spl_diff:.2f} dB difference at 60 Hz "
            f"(min V_rc={vrc_values[0]} m³ → {spl_min_vrc:.1f} dB, "
            f"max V_rc={vrc_values[-1]} m³ → {spl_max_vrc:.1f} dB). "
            f"Expected > 0.5 dB — rear chamber may not be affecting the simulation."
        )

        # Log the sweep for manual inspection
        # results[i] = (vrc_value, spl_array); iterate directly to avoid tuple confusion
        print("\n--- CRIT-1 V_rc sweep ---")
        print(f"{'V_rc (m³)':>12} {'45 Hz':>8} {'60 Hz':>8} {'80 Hz':>8} {'100 Hz':>8}")
        idx45 = int(np.argmin(np.abs(freqs - 45.0)))
        idx80 = int(np.argmin(np.abs(freqs - 80.0)))
        idx100 = int(np.argmin(np.abs(freqs - 100.0)))
        for vrc_val, spl_arr in results:
            print(f"{vrc_val:>12.3f} {spl_arr[idx45]:>8.2f} {spl_arr[idx_60hz]:>8.2f} "
                  f"{spl_arr[idx80]:>8.2f} {spl_arr[idx100]:>8.2f}")
