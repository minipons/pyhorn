"""Tests for pyhorn.solver.hornresp — Hornresp-style horn parameter solver."""

import math
import pytest

from pyhorn_core.solver.hornresp import (
    hornresp_hyperbolic_ratio,
    solve_hornresp_hyperbolic,
    solve_hornresp_profile,
    _solve_u_from_ratio_and_t,
    _validate_positive,
)


# ─── _validate_positive ────────────────────────────────────────────────────────

class TestValidatePositive:
    def test_positive_value_does_not_raise(self):
        _validate_positive("S1", 100.0)
        _validate_positive("F12", 80.0)
        _validate_positive("Hyp", 50.0)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            _validate_positive("S1", 0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            _validate_positive("S1", -10.0)


# ─── _solve_u_from_ratio_and_t ────────────────────────────────────────────────

class TestSolveUFromRatioAndT:
    def test_t_minus_one_with_s2_less_than_s1(self):
        # T = -1, S2 < S1 → should return -log(sqrt(S2/S1))
        result = _solve_u_from_ratio_and_t(area_ratio_sqrt=0.5, t=-1.0)
        assert math.isclose(result, -math.log(0.5), rel_tol=1e-9)

    def test_t_minus_one_with_s2_greater_than_s1_raises(self):
        with pytest.raises(ValueError, match="T = -1 is only valid when S2 < S1"):
            _solve_u_from_ratio_and_t(area_ratio_sqrt=2.0, t=-1.0)

    def test_discriminant_negative_raises(self):
        # area_ratio_sqrt=0.3, t=0.2 → discriminant = 0.09+0.04-1 = -0.87
        with pytest.raises(ValueError, match="inconsistent for a real hyperbolic"):
            _solve_u_from_ratio_and_t(area_ratio_sqrt=0.3, t=0.2)

    def test_no_positive_candidates_raises(self):
        # S2=S1 (ratio=1), t=0 → y = 1/(1+0) = 1, but we need y>1 for log
        with pytest.raises(ValueError, match="non-positive horn rate"):
            _solve_u_from_ratio_and_t(area_ratio_sqrt=1.0, t=0.0)

    def test_best_not_greater_than_one_raises(self):
        # When both candidates <= 1, the function raises
        with pytest.raises(ValueError, match="non-positive horn rate"):
            _solve_u_from_ratio_and_t(area_ratio_sqrt=0.5, t=2.0)

    def test_valid_solutions(self):
        # Known case: exponential horn (t=1)
        result = _solve_u_from_ratio_and_t(area_ratio_sqrt=math.e, t=1.0)
        assert result == pytest.approx(1.0, abs=1e-9)


# ─── hornresp_hyperbolic_ratio ─────────────────────────────────────────────────

class TestHornrespHyperbolicRatio:
    def test_exponential_t_1_returns_cosh_plus_sinh(self):
        f12, hyp, t = 80.0, 100.0, 1.0
        length_m = hyp / 100.0
        u = (2.0 * math.pi * f12 * length_m) / 343.0
        expected = math.cosh(u) + t * math.sinh(u)
        result = hornresp_hyperbolic_ratio(f12, hyp, t)
        assert result == pytest.approx(expected)

    def test_hyperbolic_t_0_returns_cosh(self):
        f12, hyp, t = 60.0, 80.0, 0.0
        length_m = hyp / 100.0
        u = (2.0 * math.pi * f12 * length_m) / 343.0
        result = hornresp_hyperbolic_ratio(f12, hyp, t)
        assert result == pytest.approx(math.cosh(u))

    def test_negative_f12_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            hornresp_hyperbolic_ratio(-80.0, 100.0, 1.0)

    def test_zero_hyp_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            hornresp_hyperbolic_ratio(80.0, 0.0, 1.0)

    def test_zero_f12_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            hornresp_hyperbolic_ratio(0.0, 100.0, 1.0)


# ─── solve_hornresp_hyperbolic ─────────────────────────────────────────────────

class TestSolveHornrespHyperbolic:
    def test_provides_all_five_validates_consistency(self):
        # All five provided — validates against predicted ratio.
        # Use values that satisfy the hyperbolic equation:
        # s1=50, t=1, hyp=100 → ratio=cosh(2π*80*1/343) ≈ 4.33
        # → s2 = s1 * ratio² ≈ 937.25
        s1, t, hyp, f12 = 50.0, 1.0, 100.0, 80.0
        ratio = hornresp_hyperbolic_ratio(f12, hyp, t)
        s2 = s1 * (ratio**2)
        result = solve_hornresp_hyperbolic(s1, s2, f12, t, hyp)
        assert result["s1_cm2"] == pytest.approx(s1)
        assert result["s2_cm2"] == pytest.approx(s2)
        assert result["f12_hz"] == pytest.approx(f12)
        assert result["t"] == pytest.approx(t)
        assert result["hyp_cm"] == pytest.approx(hyp)
        assert result["path_length_m"] == pytest.approx(hyp / 100.0)
        assert "throat_area_m2" in result
        assert "mouth_area_m2" in result
        assert "u" in result
        assert "flare_constant_m" in result

    def test_provides_all_five_inconsistent_raises(self):
        # S2=400 but ratio computed from S1=50 and f12/hyp/t gives different ratio
        with pytest.raises(ValueError, match="inconsistent"):
            solve_hornresp_hyperbolic(s1_cm2=50.0, s2_cm2=400.0, f12_hz=80.0, t=1.0, hyp_cm=50.0)

    def test_missing_s1_solves_s1(self):
        s2, f12, t, hyp = 400.0, 80.0, 1.0, 100.0
        result = solve_hornresp_hyperbolic(s1_cm2=None, s2_cm2=s2, f12_hz=f12, t=t, hyp_cm=hyp)
        assert result["s1_cm2"] is not None
        assert result["s1_cm2"] < s2  # S1 < S2 for exponential

    def test_missing_s2_solves_s2(self):
        s1, f12, t, hyp = 50.0, 80.0, 1.0, 100.0
        result = solve_hornresp_hyperbolic(s1_cm2=s1, s2_cm2=None, f12_hz=f12, t=t, hyp_cm=hyp)
        assert result["s2_cm2"] is not None
        assert result["s2_cm2"] > s1

    def test_missing_t_solves_t(self):
        s1, s2, f12, hyp = 50.0, 400.0, 80.0, 100.0
        result = solve_hornresp_hyperbolic(s1_cm2=s1, s2_cm2=s2, f12_hz=f12, t=None, hyp_cm=hyp)
        assert result["t"] is not None

    def test_missing_t_zero_sinh_raises(self):
        # F12 and Hyp that give u≈0 (horn rate ≈ 0) — cannot solve T
        with pytest.raises(ValueError, match="zero horn rate"):
            solve_hornresp_hyperbolic(s1_cm2=50.0, s2_cm2=50.0, f12_hz=1e-20, t=None, hyp_cm=1e-10)

    def test_missing_f12_solves_f12(self):
        s1, s2, t, hyp = 50.0, 400.0, 1.0, 100.0
        result = solve_hornresp_hyperbolic(s1_cm2=s1, s2_cm2=s2, f12_hz=None, t=t, hyp_cm=hyp)
        assert result["f12_hz"] is not None
        assert result["f12_hz"] > 0

    def test_missing_hyp_solves_hyp(self):
        s1, s2, f12, t = 50.0, 400.0, 80.0, 1.0
        result = solve_hornresp_hyperbolic(s1_cm2=s1, s2_cm2=s2, f12_hz=f12, t=t, hyp_cm=None)
        assert result["hyp_cm"] is not None
        assert result["hyp_cm"] > 0

    def test_two_missing_raises(self):
        with pytest.raises(ValueError, match="underdetermined"):
            solve_hornresp_hyperbolic(s1_cm2=None, s2_cm2=None, f12_hz=80.0, t=1.0, hyp_cm=100.0)

    def test_negative_input_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            solve_hornresp_hyperbolic(s1_cm2=-50.0, s2_cm2=400.0, f12_hz=80.0, t=1.0, hyp_cm=100.0)


# ─── solve_hornresp_profile — Con (conical) ────────────────────────────────────

class TestSolveConical:
    def test_conical_s1_s2_l12(self):
        result = solve_hornresp_profile("con", s1_cm2=40.0, s2_cm2=300.0, l12_cm=150.0)
        assert result["profile_type"] == "conical"
        assert result["s1_cm2"] == 40.0
        assert result["s2_cm2"] == 300.0
        assert result["l12_cm"] == 150.0
        assert result["path_length_m"] == 1.5
        assert result["throat_area_m2"] == 0.004
        assert result["mouth_area_m2"] == 0.03
        assert result["f12_hz"] == 0.0  # N/A for conical

    def test_conical_requires_all_three(self):
        with pytest.raises(ValueError, match="requires S1, S2, and L12"):
            solve_hornresp_profile("con", s1_cm2=40.0, s2_cm2=300.0, l12_cm=None)

    def test_conical_negative_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            solve_hornresp_profile("con", s1_cm2=-40.0, s2_cm2=300.0, l12_cm=150.0)


# ─── solve_hornresp_profile — Exp (exponential) ─────────────────────────────────

class TestSolveExponential:
    def test_exp_s1_s2_f12_solves_l12(self):
        result = solve_hornresp_profile("exp", s1_cm2=40.0, s2_cm2=300.0, f12_hz=50.0)
        assert result["profile_type"] == "exponential"
        assert result["s1_cm2"] == 40.0
        assert result["s2_cm2"] == 300.0
        assert result["f12_hz"] == 50.0
        assert result["l12_cm"] > 0
        assert result["throat_area_m2"] == 0.004
        assert result["mouth_area_m2"] == 0.03

    def test_exp_s1_s2_l12_solves_f12(self):
        result = solve_hornresp_profile("exp", s1_cm2=40.0, s2_cm2=300.0, l12_cm=150.0)
        assert result["profile_type"] == "exponential"
        assert result["s1_cm2"] == 40.0
        assert result["s2_cm2"] == 300.0
        assert result["l12_cm"] == 150.0
        assert result["f12_hz"] > 0  # derived from ln(S2/S1)/L12

    def test_exp_requires_two_of_four(self):
        with pytest.raises(ValueError, match="requires at least two"):
            solve_hornresp_profile("exp", s1_cm2=40.0, s2_cm2=None, f12_hz=None, l12_cm=None)

    def test_exp_alias_l12_cm_same_as_hyp_cm(self):
        r1 = solve_hornresp_profile("exp", s1_cm2=40.0, s2_cm2=300.0, l12_cm=150.0)
        r2 = solve_hornresp_profile("exp", s1_cm2=40.0, s2_cm2=300.0, hyp_cm=150.0)
        assert r1["l12_cm"] == r2["l12_cm"]
        assert r1["f12_hz"] == r2["f12_hz"]


# ─── solve_hornresp_profile — Par (parabolic) ──────────────────────────────────

class TestSolveParabolic:
    def test_parabolic_s1_s2_l12(self):
        result = solve_hornresp_profile("par", s1_cm2=40.0, s2_cm2=300.0, l12_cm=150.0)
        assert result["profile_type"] == "parabolic"
        assert result["s1_cm2"] == 40.0
        assert result["s2_cm2"] == 300.0
        assert result["l12_cm"] == 150.0
        assert result["path_length_m"] == 1.5
        assert result["throat_area_m2"] == 0.004
        assert result["mouth_area_m2"] == 0.03
        assert result["f12_hz"] == 0.0  # N/A for parabolic

    def test_parabolic_requires_all_three(self):
        with pytest.raises(ValueError, match="requires S1, S2, and L12"):
            solve_hornresp_profile("par", s1_cm2=40.0, s2_cm2=None, l12_cm=150.0)


# ─── solve_hornresp_profile — Cat (catenoidal) ────────────────────────────────

class TestSolveCatenoidal:
    def test_catenoidal_s1_s2_l12(self):
        result = solve_hornresp_profile("cat", s1_cm2=40.0, s2_cm2=300.0, l12_cm=150.0)
        assert result["profile_type"] == "catenoidal"
        assert result["s1_cm2"] == 40.0
        assert result["s2_cm2"] == 300.0
        assert result["l12_cm"] == 150.0
        assert result["path_length_m"] == 1.5
        assert result["t"] == 1.0  # catenoidal = hyperbolic with T=1
        assert result["f12_hz"] > 0  # derived from catenoidal m
        assert result["throat_area_m2"] == 0.004
        assert result["mouth_area_m2"] == 0.03

    def test_catenoidal_requires_s1_s2_l12(self):
        with pytest.raises(ValueError, match="requires S1, S2, and L12"):
            solve_hornresp_profile("cat", s1_cm2=40.0, s2_cm2=300.0, l12_cm=None)

    def test_catenoidal_s2_must_exceed_s1(self):
        with pytest.raises(ValueError, match="S2 must be > S1"):
            solve_hornresp_profile("cat", s1_cm2=300.0, s2_cm2=40.0, l12_cm=150.0)


# ─── solve_hornresp_profile — unified dispatcher ──────────────────────────────

class TestSolveHornrespProfile:
    def test_all_profile_type_aliases(self):
        # Full names map to canonical short forms; each profile needs valid inputs
        cases = [
            # hyp needs 4 of 5, here T is missing
            (("hyperbolic", "hyperbolic"), {"s1_cm2": 40.0, "s2_cm2": 300.0, "f12_hz": 50.43, "hyp_cm": 152.7}),
            (("hyp", "hyperbolic"), {"s1_cm2": 40.0, "s2_cm2": 300.0, "f12_hz": 50.43, "hyp_cm": 152.7}),
            # exp needs 2 of 4
            (("exponential", "exponential"), {"s1_cm2": 40.0, "s2_cm2": 300.0, "f12_hz": 50.0}),
            (("exp", "exponential"), {"s1_cm2": 40.0, "s2_cm2": 300.0, "f12_hz": 50.0}),
            # con needs S1, S2, L12
            (("conical", "conical"), {"s1_cm2": 40.0, "s2_cm2": 300.0, "l12_cm": 150.0}),
            (("con", "conical"), {"s1_cm2": 40.0, "s2_cm2": 300.0, "l12_cm": 150.0}),
            # par needs S1, S2, L12
            (("parabolic", "parabolic"), {"s1_cm2": 40.0, "s2_cm2": 300.0, "l12_cm": 150.0}),
            (("par", "parabolic"), {"s1_cm2": 40.0, "s2_cm2": 300.0, "l12_cm": 150.0}),
            # cat needs S1, S2, L12
            (("catenoidal", "catenoidal"), {"s1_cm2": 40.0, "s2_cm2": 300.0, "l12_cm": 150.0}),
            (("cat", "catenoidal"), {"s1_cm2": 40.0, "s2_cm2": 300.0, "l12_cm": 150.0}),
        ]
        for (pt_in, pt_out), kwargs in cases:
            r = solve_hornresp_profile(pt_in, **kwargs)
            assert r["profile_type"] == pt_out, f"Failed for {pt_in}"

    def test_unknown_profile_type_raises(self):
        with pytest.raises(ValueError, match="Unknown profile type"):
            solve_hornresp_profile("invalid", s1_cm2=40.0, s2_cm2=300.0, l12_cm=150.0)

    def test_hyp_delegates_to_solve_hornresp_hyperbolic(self):
        # When T is missing, hyp should solve it (like solve_hornresp_hyperbolic)
        r = solve_hornresp_profile("hyp", s1_cm2=40.0, s2_cm2=300.0, f12_hz=50.43, hyp_cm=152.7)
        assert r["profile_type"] == "hyperbolic"
        assert r["f12_hz"] == 50.43
        assert r["l12_cm"] == 152.7
        assert "t" in r
