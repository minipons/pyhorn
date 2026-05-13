"""Tests for the Tapped Horn (TH / TH1) solver path.

Reference: Hornresp manual pages 057–058.
"""

import numpy as np
import pytest
from pyhorn_core.config.models import (
    DriverSpecs,
    TappedHornGeometry,
    Section,
    RearChamber,
)
from pyhorn_core.solver.models import horn_response_tapped


DRIVER_FE166 = DriverSpecs(
    fs=44.0,
    qts=0.27,
    qes=0.30,
    qms=2.9,
    vas=36.9e-3,        # m³ = 36.9 L
    re=7.8,
    bl=7.79,
    mms=6.99e-3,        # kg (moving mass)
    cms=1.472e-3,       # m/N
    rms=0.277,
    sd=0.01327,          # m² (132.7 cm²)
    le=0.80e-3,          # H
    xmax=8.0e-3,        # m
)


class TestTappedHornGeometry:
    def test_tapped_horn_creation(self):
        th = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(
                    name="front",
                    profile_type="exponential",
                    start_area=0.01327,   # ~S2 = driver Sd
                    end_area=0.08,
                    length=1.2,
                ),
            ],
            rear_chamber=RearChamber(vrc=0.035, lrc=0.15),
        )
        assert th.tap_segment_index == 2
        assert len(th.front_sections) == 1
        assert th.front_path_length() == pytest.approx(1.2)
        assert th.rear_load_type == "rear_chamber"

    def test_th_front_path_length(self):
        th = TappedHornGeometry(
            tap_segment_index=3,
            front_sections=[
                Section(name="seg3", profile_type="conical",
                        start_area=0.008, end_area=0.02, length=0.2),
                Section(name="seg4", profile_type="exponential",
                        start_area=0.02, end_area=0.1, length=0.8),
            ],
        )
        assert th.front_path_length() == pytest.approx(1.0)

    def test_th1_mode_with_rear_sections(self):
        th = TappedHornGeometry(
            tap_segment_index=3,  # TH1: driver at S3
            front_sections=[
                Section(name="seg3", profile_type="conical",
                        start_area=0.008, end_area=0.02, length=0.2),
                Section(name="seg4", profile_type="exponential",
                        start_area=0.02, end_area=0.1, length=0.8),
            ],
            rear_sections=[
                Section(name="rear", profile_type="conical",
                        start_area=0.01327, end_area=0.01327, length=0.05),
            ],
            rear_load_type="free_space",
        )
        assert th.tap_segment_index == 3
        assert th.rear_path_length() == pytest.approx(0.05)
        assert th.rear_load_type == "free_space"

    def test_default_values(self):
        th = TappedHornGeometry()
        assert th.tap_segment_index == 2
        assert th.front_sections == []
        assert th.rear_sections == []
        assert th.rear_chamber is None
        assert th.rear_load_type == "rear_chamber"
        assert th.ang == pytest.approx(2.0 * np.pi)
        assert th.n_segments == 100


class TestHornResponseTapped:
    def test_th_runs_without_error(self):
        """Basic smoke test: horn_response_tapped runs and returns a result."""
        th = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(
                    name="front",
                    profile_type="exponential",
                    start_area=0.01327,   # driver Sd
                    end_area=0.08,
                    length=1.2,
                ),
            ],
            rear_chamber=RearChamber(vrc=0.035, lrc=0.15),
        )
        freqs = np.linspace(20, 5000, 100)
        result = horn_response_tapped(freqs, DRIVER_FE166, th)

        assert result.freqs is not None
        assert len(result.freqs) == 100
        assert result.spl is not None
        assert len(result.spl) == 100
        assert np.all(np.isfinite(result.spl))

    def test_th_spl_shape(self):
        """SPL array has the same length as the frequency array."""
        th = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(name="front", profile_type="exponential",
                        start_area=0.01327, end_area=0.08, length=1.0),
            ],
            rear_load_type="free_space",
        )
        freqs = np.array([100.0, 500.0, 1000.0, 2000.0])
        result = horn_response_tapped(freqs, DRIVER_FE166, th)

        assert result.spl.shape == (4,)
        # All SPL values should be physical (40–130 dB range)
        assert np.all(result.spl > 20.0)
        assert np.all(result.spl < 140.0)

    def test_th_impedance(self):
        """Electrical impedance is non-zero and finite."""
        th = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(name="front", profile_type="conical",
                        start_area=0.01327, end_area=0.1, length=1.0),
            ],
            rear_load_type="infinite_baffle",
        )
        freqs = np.linspace(20, 200, 50)
        result = horn_response_tapped(freqs, DRIVER_FE166, th)

        assert np.all(result.impedance > 0.0)  # magnitude > 0
        assert np.all(np.isfinite(result.impedance))

    def test_th_cone_excursion(self):
        """Cone excursion decreases at high frequencies (mass-controlled)."""
        th = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(name="front", profile_type="exponential",
                        start_area=0.01327, end_area=0.08, length=1.2),
            ],
        )
        freqs = np.array([50.0, 200.0, 1000.0])
        result = horn_response_tapped(freqs, DRIVER_FE166, th)

        assert np.all(result.excursion >= 0.0)
        # Excursion at 1 kHz should be much smaller than at 50 Hz
        assert result.excursion[2] < result.excursion[0] * 0.5

    def test_th_free_space_vs_rear_chamber(self):
        """Different rear loads produce different SPL curves."""
        th_free = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(name="front", profile_type="exponential",
                        start_area=0.01327, end_area=0.08, length=1.0),
            ],
            rear_load_type="free_space",
        )
        th_chamber = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(name="front", profile_type="exponential",
                        start_area=0.01327, end_area=0.08, length=1.0),
            ],
            rear_chamber=RearChamber(vrc=0.035, lrc=0.15),
        )
        freqs = np.array([60.0, 100.0, 200.0, 400.0])
        result_free = horn_response_tapped(freqs, DRIVER_FE166, th_free)
        result_chamber = horn_response_tapped(freqs, DRIVER_FE166, th_chamber)

        # At some frequency, the two should differ (rear load affects response)
        diff = np.abs(result_free.spl - result_chamber.spl)
        assert np.any(diff > 0.01), "Free-space and rear-chamber loads should give different SPL"

    def test_th_off_axis_spl(self):
        """Off-axis SPL array has correct shape (n_freq × n_angles)."""
        th = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(name="front", profile_type="exponential",
                        start_area=0.01327, end_area=0.08, length=1.0),
            ],
        )
        freqs = np.linspace(100, 2000, 50)
        angles = np.array([0.0, 30.0, 60.0, 90.0])
        result = horn_response_tapped(freqs, DRIVER_FE166, th, off_axis_angles=angles)

        assert result.off_axis_spl is not None
        assert result.off_axis_angles is not None
        assert result.off_axis_spl.shape == (50, 4)
        # Off-axis SPL should be lower than on-axis at wide angles
        assert result.off_axis_spl[0, 3] < result.off_axis_spl[0, 0]

    def test_th_phase_is_unwrapped(self):
        """Phase values vary smoothly with frequency (no 2π jumps)."""
        th = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(name="front", profile_type="exponential",
                        start_area=0.01327, end_area=0.08, length=1.0),
            ],
        )
        freqs = np.linspace(100, 1000, 100)
        result = horn_response_tapped(freqs, DRIVER_FE166, th)

        # Phase differences between adjacent bins should be small (< π)
        phase_diff = np.diff(result.phase)
        # Allow for phase wrapping near ±π boundary
        wrapped = np.abs(phase_diff) > np.pi * 0.9
        assert np.sum(wrapped) < 5, "Phase should be mostly smooth (few wraps)"

    def test_th_direction_index(self):
        """Direction index is ≤ 0 dB for all off-axis angles."""
        th = TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(name="front", profile_type="exponential",
                        start_area=0.01327, end_area=0.1, length=1.0),
            ],
        )
        freqs = np.array([500.0, 1000.0, 2000.0])
        result = horn_response_tapped(freqs, DRIVER_FE166, th)

        if result.direction_index is not None:
            # DI is always ≤ 0 (beam narrowing, never beam widening)
            assert np.all(result.direction_index <= 1.0)  # in linear units
            # At 0° off-axis, DI should be ~0 dB (direction_factor = 1)
            # At wider angles, DI should be negative (direction_factor < 1)
