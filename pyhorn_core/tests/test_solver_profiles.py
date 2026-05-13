"""Unit tests for pyhorn.solver.profiles — horn profile discretisation."""

import math
import numpy as np
import pytest
from pyhorn_core.solver.profiles import (
    discretise_profile,
    discretise_conical_segments,
    discretise_rectangular_segments,
    horn_profile_metrics,
    profile_area_at_distance,
)


# ─── discretise_profile ────────────────────────────────────────────────────────


class TestDiscretiseProfile:
    """Tests for discretise_profile()."""

    def test_conical_profile_returns_list_of_tuples(self):
        """Should return a list of (length, area, fr) tuples."""
        result = discretise_profile(
            "conical", throat_area=0.01, mouth_area=0.05, length=0.5, n_segments=10
        )
        assert isinstance(result, list)
        assert all(isinstance(seg, tuple) and len(seg) == 3 for seg in result)

    def test_conical_n_segments(self):
        """n_segments controls the number of returned segments."""
        result_10 = discretise_profile("conical", 0.01, 0.05, 0.5, n_segments=10)
        result_50 = discretise_profile("conical", 0.01, 0.05, 0.5, n_segments=50)
        assert len(result_10) == 10
        assert len(result_50) == 50

    def test_conical_segment_lengths_all_equal(self):
        """All segments should have equal length = total_length / n_segments."""
        n = 20
        length = 0.4
        result = discretise_profile("conical", 0.01, 0.05, length, n_segments=n)
        expected_dx = length / n
        for seg in result:
            assert seg[0] == pytest.approx(expected_dx)

    def test_conical_area_monotonically_increases(self):
        """Area should monotonically increase from throat to mouth."""
        result = discretise_profile("conical", 0.01, 0.05, 0.5, n_segments=50)
        areas = [seg[1] for seg in result]
        for i in range(1, len(areas)):
            assert areas[i] >= areas[i - 1] - 1e-12

    def test_conical_throat_area_at_start(self):
        """First segment area ≈ throat_area."""
        result = discretise_profile("conical", 0.01, 0.05, 0.5, n_segments=100)
        # First segment area is average of first two cross-sections near throat
        assert result[0][1] == pytest.approx(0.01, rel=0.1)

    def test_conical_mouth_area_at_end(self):
        """Last segment area ≈ mouth_area."""
        result = discretise_profile("conical", 0.01, 0.05, 0.5, n_segments=100)
        # Last segment area is average of last two cross-sections near mouth
        assert result[-1][1] == pytest.approx(0.05, rel=0.1)

    def test_exponential_mouth_area_approached(self):
        """Exponential last segment average should approach mouth_area.

        Because discretisation uses (A[i] + A[i+1])/2 averaging with the final
        point at A_mouth, the last segment average slightly undershoots A_mouth
        for exponential (while conical, being linear in radius, is more exact).
        We check it's within 5% — clearly approaching the target.
        """
        throat, mouth, length = 0.01, 0.1, 0.5
        result = discretise_profile(
            "exponential", throat, mouth, length, n_segments=100
        )
        # Last segment ends at x=L, area should be close to mouth_area
        assert result[-1][1] == pytest.approx(mouth, rel=0.05)
        # And notably higher than the throat area
        assert result[-1][1] > throat

    def test_parabolic_profile(self):
        """Parabolic (linear area) profile should give linear area interpolation."""
        result = discretise_profile("parabolic", 0.02, 0.08, 0.5, n_segments=100)
        areas = [seg[1] for seg in result]
        # Linear area growth: area at x=L/2 should be midpoint of throat and mouth
        expected_mid = (0.02 + 0.08) / 2
        assert areas[49] == pytest.approx(expected_mid, rel=0.05)

    def test_hyperbolic_profile(self):
        """Hyperbolic profile: A(L) = mouth_area exactly."""
        throat = 0.01
        mouth = 0.09
        result = discretise_profile("hyperbolic", throat, mouth, 0.5, n_segments=100)
        # The last segment's average area should be close to mouth
        assert result[-1][1] == pytest.approx(mouth, rel=0.05)

    def test_hyperbolic_slower_start_than_exponential(self):
        """Hyperbolic flare should start slower than exponential."""
        result_hyp = discretise_profile("hyperbolic", 0.01, 0.1, 0.5, n_segments=50, hyperbolic_t=0.5)
        result_exp = discretise_profile("exponential", 0.01, 0.1, 0.5, n_segments=50)
        # Early segments: hyperbolic with t=0.5 < exponential (t<1 gives sub-exponential start)
        assert result_hyp[3][1] < result_exp[3][1]

    def test_aliases_con_and_conical(self):
        """'con' and 'conical' should produce identical results."""
        r1 = discretise_profile("con", 0.01, 0.05, 0.5, n_segments=20)
        r2 = discretise_profile("conical", 0.01, 0.05, 0.5, n_segments=20)
        for seg1, seg2 in zip(r1, r2):
            assert seg1[0] == pytest.approx(seg2[0])
            assert seg1[1] == pytest.approx(seg2[1])

    def test_aliases_exp_and_exponential(self):
        """'exp' and 'exponential' should produce identical results."""
        r1 = discretise_profile("exp", 0.01, 0.05, 0.5, n_segments=20)
        r2 = discretise_profile("exponential", 0.01, 0.05, 0.5, n_segments=20)
        for seg1, seg2 in zip(r1, r2):
            assert seg1[0] == pytest.approx(seg2[0])
            assert seg1[1] == pytest.approx(seg2[1])

    def test_hyperbolic_t_one_matches_exponential(self):
        """Hyperbolic T=1 should reduce to the exponential family."""
        throat = 0.01
        mouth = 0.08
        length = 0.6
        r_hyp = discretise_profile(
            "hyperbolic",
            throat,
            mouth,
            length,
            n_segments=40,
            hyperbolic_t=1.0,
        )
        r_exp = discretise_profile(
            "exponential",
            throat,
            mouth,
            length,
            n_segments=40,
        )
        for seg_hyp, seg_exp in zip(r_hyp, r_exp):
            assert seg_hyp[1] == pytest.approx(seg_exp[1], rel=1e-6)

    def test_profile_area_at_distance_honours_hyperbolic_t(self):
        """Generalized hyperbolic profile should still hit the requested mouth area."""
        throat = 0.01
        mouth = 0.05
        length = 0.8
        area = profile_area_at_distance(
            "hyperbolic",
            throat,
            mouth,
            length,
            length,
            hyperbolic_t=0.6,
        )
        assert area == pytest.approx(mouth, rel=1e-6)

    def test_raises_on_negative_length(self):
        """length <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="Length must be > 0"):
            discretise_profile("conical", 0.01, 0.05, 0.0)

    def test_raises_on_zero_throat_area(self):
        """throat_area <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="Areas must be > 0"):
            discretise_profile("conical", 0.0, 0.05, 0.5)

    def test_raises_on_zero_mouth_area(self):
        """mouth_area <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="Areas must be > 0"):
            discretise_profile("conical", 0.01, 0.0, 0.5)

    def test_raises_on_unknown_profile(self):
        """Unknown profile type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown profile type"):
            discretise_profile("unknown_profile", 0.01, 0.05, 0.5)

    def test_fr_field_is_zero(self):
        """discretise_profile should return fr=0.0 for all segments (no absorption)."""
        result = discretise_profile("conical", 0.01, 0.05, 0.5, n_segments=20)
        for seg in result:
            assert seg[2] == pytest.approx(0.0)


# ─── discretise_conical_segments ───────────────────────────────────────────────


class TestDiscretiseConicalSegments:
    """Tests for discretise_conical_segments()."""

    def test_returns_segments_and_bends(self):
        """Should return (segments, bends) tuple."""
        result = discretise_conical_segments(
            [(0.06, 0.07, 0.03)],
            width=0.2,
            n_per_segment=10,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        segments, bends = result
        assert isinstance(segments, list)
        assert isinstance(bends, list)

    def test_n_per_segment_controls_subdivisions(self):
        """n_per_segment=10 on one segment should give 10 sub-segments."""
        segs, bends = discretise_conical_segments(
            [(0.06, 0.07, 0.03)],
            width=0.2,
            n_per_segment=10,
        )
        assert len(segs) == 10

    def test_n_per_segment_5_on_two_segments_gives_10_subsegments(self):
        """Two segments with n_per_segment=5 should give 10 total sub-segments."""
        segs, bends = discretise_conical_segments(
            [(0.06, 0.07, 0.03), (0.07, 0.09, 0.04)],
            width=0.2,
            n_per_segment=5,
        )
        assert len(segs) == 10

    def test_without_width_areas_are_dim_values(self):
        """Without width, areas are the dim values directly."""
        segs, bends = discretise_conical_segments(
            [(0.01, 0.02, 0.05)],
            width=None,
            n_per_segment=5,
        )
        # With width=None: area_start=0.01, area_end=0.02, dx=0.01
        # A_x = [0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
        # First segment avg area = (0.01 + 0.012) / 2 = 0.011
        assert segs[0][1] == pytest.approx(0.011, rel=1e-6)

    def test_with_width_areas_are_height_times_width(self):
        """With width set, areas = height * width."""
        segs, bends = discretise_conical_segments(
            [(0.06, 0.07, 0.03)],
            width=0.2,
            n_per_segment=5,
        )
        # area_start = 0.06 * 0.2 = 0.012
        expected_avg = (0.06 * 0.2 + 0.062 * 0.2) / 2  # midpoint area
        assert segs[0][1] == pytest.approx(expected_avg, rel=1e-6)

    def test_bends_detected_at_area_mismatch(self):
        """A bend should be recorded when consecutive segments have mismatched areas."""
        segs, bends = discretise_conical_segments(
            [
                (0.06, 0.07, 0.03),
                (0.08, 0.09, 0.04),
            ],  # end of first = 0.07*w, start of second = 0.08*w
            width=0.2,
            n_per_segment=5,
        )
        # 0.07*0.2 = 0.014, 0.08*0.2 = 0.016 → different → one bend expected
        assert len(bends) == 1

    def test_no_bends_when_areas_match(self):
        """No bend recorded when end of one segment equals start of next."""
        segs, bends = discretise_conical_segments(
            [
                (0.06, 0.07, 0.03),
                (0.07, 0.09, 0.04),
            ],  # seg1 end = 0.07, seg2 start = 0.07
            width=None,  # areas directly in dim values
            n_per_segment=5,
        )
        assert len(bends) == 0

    def test_flow_resistivity_fr_preserved(self):
        """4th element of segment tuple should become fr in output."""
        segs, bends = discretise_conical_segments(
            [(0.06, 0.07, 0.03, 5000.0)],
            width=0.2,
            n_per_segment=5,
        )
        for seg in segs:
            assert seg[2] == pytest.approx(5000.0)

    def test_default_fr_is_zero(self):
        """Without 4th element, fr should default to 0.0."""
        segs, bends = discretise_conical_segments(
            [(0.06, 0.07, 0.03)],
            width=0.2,
            n_per_segment=5,
        )
        for seg in segs:
            assert seg[2] == pytest.approx(0.0)

    def test_raises_on_segment_with_fewer_than_3_elements(self):
        """Segment with fewer than 3 elements should raise ValueError."""
        with pytest.raises(ValueError, match="at least 3 elements"):
            discretise_conical_segments([(0.06, 0.07)], width=0.2)

    def test_segments_have_length_area_fr(self):
        """Each segment should be a 3-tuple (length, area, fr)."""
        segs, bends = discretise_conical_segments(
            [(0.06, 0.07, 0.03)],
            width=0.2,
            n_per_segment=5,
        )
        for seg in segs:
            assert len(seg) == 3
            assert seg[0] > 0  # length positive
            assert seg[1] > 0  # area positive


# ─── discretise_rectangular_segments ──────────────────────────────────────────


class TestDiscretiseRectangularSegments:
    """Tests for discretise_rectangular_segments()."""

    def test_returns_segments_bends_and_widths(self):
        """Should return (segments, bends, node_widths) tuple."""
        result = discretise_rectangular_segments(
            [(0.1, 0.05, 0.1, 0.08, 0.03)],
            n_per_segment=5,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        segs, bends, widths = result
        assert isinstance(segs, list)
        assert isinstance(bends, list)
        assert isinstance(widths, list)

    def test_n_per_segment_controls_subdivisions(self):
        """n_per_segment=10 on one segment should give 10 sub-segments."""
        segs, _, _ = discretise_rectangular_segments(
            [(0.1, 0.05, 0.1, 0.08, 0.03)],
            n_per_segment=10,
        )
        assert len(segs) == 10

    def test_segment_count_equals_n_per_segment(self):
        """One segment with n_per_segment=7 should return exactly 7 segments."""
        segs, _, _ = discretise_rectangular_segments(
            [(0.1, 0.05, 0.1, 0.08, 0.03)],
            n_per_segment=7,
        )
        assert len(segs) == 7

    def test_area_is_width_times_height(self):
        """Segment area should be width * height."""
        segs, _, _ = discretise_rectangular_segments(
            [(0.1, 0.05, 0.1, 0.08, 0.03)],
            n_per_segment=5,
        )
        # First segment avg area should be close to start area = 0.1*0.05 = 0.005
        assert segs[0][1] == pytest.approx(0.1 * 0.05, rel=0.2)

    def test_bends_detected_at_area_mismatch(self):
        """Bend recorded when consecutive segments have different end/start areas."""
        segs, bends, _ = discretise_rectangular_segments(
            [
                (0.1, 0.05, 0.1, 0.08, 0.03),  # end area = 0.1*0.08 = 0.008
                (
                    0.1,
                    0.10,
                    0.1,
                    0.12,
                    0.04,
                ),  # start area = 0.1*0.10 = 0.010  → mismatch
            ],
            n_per_segment=5,
        )
        assert len(bends) == 1

    def test_no_bends_when_contiguous_areas_match(self):
        """No bend when one segment's end area equals next segment's start area."""
        segs, bends, _ = discretise_rectangular_segments(
            [
                (0.1, 0.05, 0.1, 0.08, 0.03),  # end = 0.008
                (0.1, 0.08, 0.1, 0.10, 0.04),  # start = 0.008  → match, no bend
            ],
            n_per_segment=5,
        )
        assert len(bends) == 0

    def test_node_widths_length_equals_segments_plus_one(self):
        """node_widths should have n_per_segment + 1 entries (start + after each sub-segment)."""
        segs, _, widths = discretise_rectangular_segments(
            [(0.1, 0.05, 0.1, 0.08, 0.03)],
            n_per_segment=7,
        )
        assert len(widths) == len(segs) + 1 == 8

    def test_node_widths_tracks_width_changes(self):
        """node_widths should reflect changing width along segment."""
        segs, _, widths = discretise_rectangular_segments(
            [(0.05, 0.05, 0.15, 0.05, 0.5)],  # width expands from 0.05 to 0.15
            n_per_segment=10,
        )
        assert widths[0] == pytest.approx(0.05, rel=1e-6)
        assert widths[-1] == pytest.approx(0.15, rel=1e-6)

    def test_fr_preserved_from_6th_element(self):
        """6th element in segment tuple should become fr in output."""
        segs, _, _ = discretise_rectangular_segments(
            [(0.1, 0.05, 0.1, 0.08, 0.03, 8000.0)],
            n_per_segment=5,
        )
        for seg in segs:
            assert seg[2] == pytest.approx(8000.0)

    def test_default_fr_is_zero(self):
        """Without 6th element, fr should be 0.0."""
        segs, _, _ = discretise_rectangular_segments(
            [(0.1, 0.05, 0.1, 0.08, 0.03)],
            n_per_segment=5,
        )
        for seg in segs:
            assert seg[2] == pytest.approx(0.0)

    def test_raises_on_segment_with_fewer_than_5_elements(self):
        """Rectangular segment with fewer than 5 elements should raise."""
        with pytest.raises(ValueError, match="at least 5 elements"):
            discretise_rectangular_segments([(0.1, 0.05, 0.1, 0.08)], n_per_segment=5)

    def test_segments_are_length_area_fr_tuples(self):
        """Each segment should be a 3-tuple (length, area, fr)."""
        segs, _, _ = discretise_rectangular_segments(
            [(0.1, 0.05, 0.1, 0.08, 0.03)],
            n_per_segment=5,
        )
        for seg in segs:
            assert len(seg) == 3
            assert seg[0] > 0  # length positive
            assert seg[1] > 0  # area positive


# ─── horn_profile_metrics ─────────────────────────────────────────────────────


class TestHornProfileMetrics:
    """Tests for horn_profile_metrics()."""

    def test_exponential_cutoff_formula(self):
        """fc = (m·c)/(4π) where m = (1/L)·ln(Am/At)."""
        # throat=0.01m², mouth=0.1m², L=0.5m
        # m = ln(10)/0.5 = 2.3026 per metre
        # fc = 2.3026 * 343 / (4π) = 62.9 Hz
        result = horn_profile_metrics("exponential", 0.01, 0.1, 0.5)
        m_expected = (1.0 / 0.5) * np.log(0.1 / 0.01)  # ln(10)/0.5
        fc_expected = (m_expected * 343.0) / (4.0 * np.pi)
        assert result["flare_constant_m"] == pytest.approx(m_expected, rel=1e-6)
        assert result["cutoff_hz"] == pytest.approx(fc_expected, rel=1e-6)

    def test_exponential_krm_at_cutoff(self):
        """krm_fc = rm·m/2 (from k·rm = 2πfc·rm/c = m·rm/2)."""
        result = horn_profile_metrics("exponential", 0.01, 0.1, 0.5)
        rm = np.sqrt(0.1 / np.pi)
        m = (1.0 / 0.5) * np.log(10.0)
        krm_expected = (rm * m) / 2.0
        assert result["krm"] == pytest.approx(krm_expected, rel=1e-6)

    def test_undersized_mouth_krm_below_07(self):
        """Small mouth area should give krm < 0.7 → 'undersized'."""
        # Very small mouth for given length: throat=0.01, mouth=0.02, L=2m
        result = horn_profile_metrics("exponential", 0.01, 0.02, 2.0)
        assert result["krm"] < 0.7
        assert result["mouth_rating"] == "undersized"

    def test_large_mouth_krm_above_1(self):
        """Large mouth area should give krm ≥ 1.0 → 'midrange_ok'."""
        # throat=0.01, mouth=0.5, L=0.3m (aggressive expansion)
        result = horn_profile_metrics("exponential", 0.01, 0.5, 0.3)
        assert result["krm"] >= 1.0
        assert result["mouth_rating"] == "midrange_ok"

    def test_bass_ok_rating(self):
        """krm between 0.7 and 1.0 → 'bass_ok'."""
        # At=0.005, Am=0.05, L=0.2 → ratio=10, m=11.51, rm=0.126, krm=0.726
        result = horn_profile_metrics("exponential", 0.005, 0.05, 0.2)
        assert 0.7 <= result["krm"] < 1.0, f"krm={result['krm']:.4f} not in [0.7, 1.0)"
        assert result["mouth_rating"] == "bass_ok"

    def test_conical_has_zero_cutoff(self):
        """Conical horns have no exponential cutoff; cutoff_hz should be 0."""
        result = horn_profile_metrics("conical", 0.01, 0.1, 0.5)
        assert result["cutoff_hz"] == 0.0
        assert result["flare_constant_m"] == 0.0

    def test_hyperbolic_profile(self):
        """Hyperbolic (hypex) profile should return a valid cutoff."""
        result = horn_profile_metrics("hyperbolic", 0.01, 0.1, 0.5)
        assert result["cutoff_hz"] > 0
        assert result["krm"] > 0
        assert result["mouth_rating"] in ("undersized", "bass_ok", "midrange_ok")

    def test_parabolic_profile(self):
        """Parabolic (linear area) profile should compute cutoff via ln ratio."""
        result = horn_profile_metrics("parabolic", 0.01, 0.1, 0.5)
        # Uses same ln-based m approximation as exponential
        assert result["cutoff_hz"] > 0

    def test_tractrix_profile(self):
        """Tractrix profile should return valid metrics."""
        result = horn_profile_metrics("tractrix", 0.01, 0.1, 0.5)
        assert result["cutoff_hz"] > 0
        assert result["krm"] > 0
        assert result["expansion_ratio"] == 10.0
        assert result["mouth_radius_m"] == pytest.approx(np.sqrt(0.1 / np.pi), rel=1e-6)

    def test_mouth_krm_min_hz_formula(self):
        """m mouth_krm_min_hz = 0.7·c / (2π·rm)."""
        result = horn_profile_metrics("exponential", 0.01, 0.1, 0.5)
        rm = np.sqrt(0.1 / np.pi)
        expected_min_hz = (0.7 * 343.0) / (2.0 * np.pi * rm)
        assert result["mouth_krm_min_hz"] == pytest.approx(expected_min_hz, rel=1e-6)

    def test_zero_invalid_input(self):
        """Invalid inputs (zero/negative) should return zeros without crashing."""
        result = horn_profile_metrics("exponential", 0.0, 0.1, 0.5)
        assert result["cutoff_hz"] == 0.0
        assert result["krm"] == 0.0

    def test_all_keys_present(self):
        """Should return all expected keys."""
        result = horn_profile_metrics("exponential", 0.01, 0.1, 0.5)
        expected_keys = {
            "flare_constant_m",
            "cutoff_hz",
            "krm",
            "mouth_radius_m",
            "mouth_diameter_cm",
            "expansion_ratio",
            "mouth_rating",
            "mouth_krm_min_hz",
            "tl_tuning_hz",
            "mouth_ko",
        }
        assert set(result.keys()) == expected_keys

    def test_tl_tuning_formula(self):
        """TL tuning: f_tl = c / (4L)."""
        result = horn_profile_metrics("exponential", 0.01, 0.1, 0.5)
        assert result["tl_tuning_hz"] == pytest.approx(343.0 / (4 * 0.5), rel=1e-6)

    def test_tl_tuning_short_path_lower(self):
        """Shorter path → higher TL tuning."""
        r1 = horn_profile_metrics("exponential", 0.01, 0.1, 0.3)
        r2 = horn_profile_metrics("exponential", 0.01, 0.1, 0.6)
        assert r1["tl_tuning_hz"] > r2["tl_tuning_hz"]

    def test_tl_tuning_zero_for_zero_length(self):
        """Zero length should give tl_tuning_hz = 0."""
        result = horn_profile_metrics("exponential", 0.01, 0.1, 0.0)
        assert result["tl_tuning_hz"] == 0.0

    def test_mouth_ko_equals_twice_radius(self):
        """mouth_ko = 2*rm = mouth diameter (circle or square side)."""
        result = horn_profile_metrics("exponential", 0.01, 0.1, 0.5)
        rm = np.sqrt(0.1 / np.pi)
        assert result["mouth_ko"] == pytest.approx(2.0 * rm, rel=1e-6)

    def test_tl_tuning_still_computes_for_conical(self):
        """Conical has no exponential cutoff but tl_tuning_hz still computes."""
        result = horn_profile_metrics("conical", 0.01, 0.1, 0.5)
        assert result["tl_tuning_hz"] == pytest.approx(343.0 / (4 * 0.5), rel=1e-6)
        assert result["cutoff_hz"] == 0.0


# ─── tractrix discretisation ───────────────────────────────────────────────────


class TestTractrixProfile:
    """Tests for tractrix profile discretisation."""

    def test_tractrix_aliases(self):
        """'tractrix' and 'trx' should produce identical discretisation."""
        r1 = discretise_profile("tractrix", 0.01, 0.05, 0.5, n_segments=20)
        r2 = discretise_profile("trx", 0.01, 0.05, 0.5, n_segments=20)
        for seg1, seg2 in zip(r1, r2):
            assert seg1[0] == pytest.approx(seg2[0])
            assert seg1[1] == pytest.approx(seg2[1])

    def test_tractrix_area_increases_monotonically(self):
        """Area should monotonically increase from throat to mouth."""
        result = discretise_profile("tractrix", 0.01, 0.1, 0.5, n_segments=50)
        areas = [seg[1] for seg in result]
        for i in range(1, len(areas)):
            assert (
                areas[i] > areas[i - 1]
            ), f"Area decreased at index {i}: {areas[i-1]:.6f} → {areas[i]:.6f}"

    def test_tractrix_different_curvature_than_exponential(self):
        """Tractrix curvature should differ from pure exponential."""
        result_trx = discretise_profile("tractrix", 0.01, 0.1, 0.5, n_segments=50)
        result_exp = discretise_profile("exponential", 0.01, 0.1, 0.5, n_segments=50)
        # Mid-segments must clearly diverge
        assert result_trx[10][1] != pytest.approx(result_exp[10][1])
        # Both first segments should be near the throat area (within 10%:
        # seg[0] = avg of throat cross-section and first interior point)
        rt = np.sqrt(0.01 / np.pi)
        throat_cs = np.pi * rt**2
        assert result_trx[0][1] == pytest.approx(throat_cs, rel=0.10)
        assert result_exp[0][1] == pytest.approx(throat_cs, rel=0.10)

    def test_tractrix_unknown_alias_raises(self):
        """Unknown alias for tractrix should raise."""
        with pytest.raises(ValueError, match="Unknown profile type"):
            discretise_profile("not_tractrix", 0.01, 0.05, 0.5)
