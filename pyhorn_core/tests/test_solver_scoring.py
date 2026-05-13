"""Tests for pyhorn_core/solver/scoring.py — acoustic scoring primitives."""

import numpy as np
import pytest

from pyhorn_core.solver.scoring import (
    _sigmoid,
    compute_response_metrics,
    cutoff_penalty,
    estimate_cutoff_frequency,
    mean_spl_in_band,
    spl_in_band,
)


def test_compute_response_metrics_returns_expected_band_values():
    freq = np.array([50.0, 80.0, 120.0, 160.0, 200.0, 260.0])
    spl = np.array([80.0, 90.0, 96.0, 94.0, 92.0, 91.0])

    metrics = compute_response_metrics(spl, freq, f_min=80.0, f_max=200.0)

    assert metrics.mean_spl == np.mean([90.0, 96.0, 94.0, 92.0])
    assert metrics.flatness_db == np.std([90.0, 96.0, 94.0, 92.0])
    assert metrics.bass_mean_spl == np.mean([90.0, 96.0, 94.0])
    assert metrics.bass_deficit_db == 0.0


def test_estimate_cutoff_frequency_finds_first_minus_3db_crossing():
    freq = np.array([80.0, 100.0, 120.0, 160.0, 200.0, 300.0, 400.0])
    spl = np.array([97.0, 98.0, 97.5, 95.0, 94.0, 87.0, 88.0])

    cutoff = estimate_cutoff_frequency(spl, freq, f_min=80.0)

    assert cutoff == 300.0


def test_mean_spl_in_band_returns_none_when_empty():
    freq = np.array([100.0, 200.0])
    spl = np.array([90.0, 91.0])

    assert mean_spl_in_band(spl, freq, 300.0, 400.0) is None


def test_cutoff_penalty_sigmoid_decay():
    # f_c <= f_min → full credit (1.0)
    assert cutoff_penalty(70.0, 80.0) == 1.0
    # f_c >= f_min * 2.5 → zero penalty (0.0)
    assert cutoff_penalty(250.0, 80.0) == 0.0
    # f_c = exactly f_min → returns 1.0 (first branch)
    assert cutoff_penalty(80.0, 80.0) == 1.0
    # f_c in transition zone → smooth sigmoid decay, penalty strictly between 0 and 1
    p_120 = cutoff_penalty(120.0, 80.0)
    assert 0.0 < p_120 < 1.0
    p_160 = cutoff_penalty(160.0, 80.0)
    assert p_160 < p_120  # closer to f_min*2.5 → smaller penalty


# ─── spl_in_band ──────────────────────────────────────────────────────────────

class TestSplInBand:
    """Tests for spl_in_band helper."""

    def test_returns_matching_samples(self):
        freq = np.array([50.0, 100.0, 200.0, 500.0, 1000.0])
        spl = np.array([80.0, 90.0, 95.0, 92.0, 88.0])
        result = spl_in_band(spl, freq, 100.0, 500.0)
        np.testing.assert_array_equal(result, [90.0, 95.0, 92.0])

    def test_boundary_inclusive(self):
        freq = np.array([99.9, 100.0, 200.0, 500.0, 500.1])
        spl = np.array([79.0, 90.0, 95.0, 92.0, 78.0])
        result = spl_in_band(spl, freq, 100.0, 500.0)
        np.testing.assert_array_equal(result, [90.0, 95.0, 92.0])

    def test_empty_when_no_samples_in_band(self):
        freq = np.array([10.0, 20.0, 30.0])
        spl = np.array([80.0, 85.0, 90.0])
        result = spl_in_band(spl, freq, 100.0, 200.0)
        assert len(result) == 0

    def test_full_range(self):
        freq = np.array([50.0, 100.0, 200.0])
        spl = np.array([80.0, 90.0, 95.0])
        result = spl_in_band(spl, freq, 50.0, 200.0)
        np.testing.assert_array_equal(result, [80.0, 90.0, 95.0])


# ─── _sigmoid ────────────────────────────────────────────────────────────────

class TestSigmoid:
    """Tests for _sigmoid helper."""

    def test_t_0_returns_less_than_half(self):
        # sigmoid(t) = 1/(1+exp(-4*(t-0.5))); at t=0: exp(2) ≈ 7.39 → ≈0.119
        assert _sigmoid(0.0) < 0.5

    def test_t_1_returns_greater_than_half(self):
        # at t=1: exp(-2) ≈ 0.135 → ≈0.881
        assert _sigmoid(1.0) > 0.5

    def test_t_0_5_returns_0_5(self):
        assert _sigmoid(0.5) == pytest.approx(0.5)

    def test_monotonically_increasing(self):
        values = [_sigmoid(t) for t in np.linspace(0, 1, 20)]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1]

    def test_steepness_affects_slope_near_center(self):
        # Near t=0.5, lower steepness → value closer to 0.5 (flatter)
        # At t=0.4, steep=10.0 gives steep falloff → further from 0.5
        flat = _sigmoid(0.4, steepness=1.0)
        steep = _sigmoid(0.4, steepness=10.0)
        # flat (steepness=1) is closer to 0.5 than steep (steepness=10)
        assert abs(flat - 0.5) < abs(steep - 0.5)


# ─── estimate_cutoff_frequency edge cases ─────────────────────────────────────

class TestEstimateCutoffFrequency:
    """Edge case tests for estimate_cutoff_frequency."""

    def test_no_crossing_found_returns_last_freq(self):
        """When SPL never drops below threshold, return last freq above f_min."""
        freq = np.array([80.0, 100.0, 120.0, 160.0, 200.0])
        spl = np.array([97.0, 98.0, 97.5, 97.0, 97.2])  # never dips below ~95
        cutoff = estimate_cutoff_frequency(spl, freq, f_min=80.0)
        assert cutoff == 200.0

    def test_empty_bands_returns_f_min(self):
        """When all search bands are empty, return f_min."""
        freq = np.array([50.0, 60.0, 70.0])  # all below f_min*1.5
        spl = np.array([90.0, 91.0, 92.0])
        cutoff = estimate_cutoff_frequency(spl, freq, f_min=80.0)
        assert cutoff == 80.0

    def test_freq_array_all_below_f_min(self):
        """When no freq >= f_min, return f_min."""
        freq = np.array([30.0, 40.0, 50.0])
        spl = np.array([80.0, 85.0, 90.0])
        cutoff = estimate_cutoff_frequency(spl, freq, f_min=80.0)
        assert cutoff == 80.0

    def test_crossing_found_returns_first_below_threshold(self):
        """Crossing below threshold returns first such frequency."""
        freq = np.array([80.0, 100.0, 150.0, 200.0, 250.0])
        # avg in [160,400] = mean([98,97,95,93]) = 95.75; threshold = 92.75
        # 200 Hz: 93 < 92.75? No. 250 Hz: 90 < 92.75? Yes → returns 250
        spl = np.array([97.0, 98.0, 97.0, 93.0, 90.0])
        cutoff = estimate_cutoff_frequency(spl, freq, f_min=80.0)
        assert cutoff == 250.0


# ─── cutoff_penalty boundaries ────────────────────────────────────────────────

class TestCutoffPenalty:
    """Boundary and edge case tests for cutoff_penalty."""

    def test_f_c_exactly_f_min_returns_one(self):
        assert cutoff_penalty(80.0, 80.0) == 1.0

    def test_f_c_below_f_min_returns_one(self):
        assert cutoff_penalty(50.0, 80.0) == 1.0

    def test_f_c_at_f_min_times_2_5_returns_zero(self):
        assert cutoff_penalty(80.0 * 2.5, 80.0) == 0.0

    def test_f_c_well_above_limit_returns_zero(self):
        assert cutoff_penalty(500.0, 80.0) == 0.0

    def test_f_c_in_transition_zone_is_between_zero_and_one(self):
        # f_c = 120 Hz, f_min = 80 Hz → t = (120-80)/(80*1.5) = 40/120 = 0.333
        # sigmoid(0.333, 4.0) ≈ 0.27 → penalty ≈ 0.73
        p = cutoff_penalty(120.0, 80.0)
        assert 0.0 < p < 1.0

    def test_penalty_decreases_as_f_c_increases_in_transition(self):
        p_100 = cutoff_penalty(100.0, 80.0)
        p_140 = cutoff_penalty(140.0, 80.0)
        p_190 = cutoff_penalty(190.0, 80.0)
        assert p_100 > p_140 > p_190
