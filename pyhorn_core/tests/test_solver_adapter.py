"""Tests for pyhorn.solver.adapter — throat adapter geometry designer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyhorn_core.config.models import ThroatAdapter
from pyhorn_core.solver.adapter import (
    ThroatAdapterInput,
    compute_throat_adapter,
    throat_adapter_profile,
    _diameter_to_area,
    _area_to_diameter,
    _minimum_length_conical,
)


class TestDiameterAreaConversions:
    def test_diameter_to_area(self):
        # D = 0.1 m  →  A = π*(0.05)² = 0.007854 m²
        area = _diameter_to_area(0.1)
        assert math.isclose(area, math.pi * (0.05) ** 2, rel_tol=1e-12)

    def test_area_to_diameter(self):
        area = math.pi * (0.05) ** 2
        diam = _area_to_diameter(area)
        assert math.isclose(diam, 0.1, rel_tol=1e-12)

    def test_round_trip(self):
        d = 0.078
        assert math.isclose(_area_to_diameter(_diameter_to_area(d)), d, rel_tol=1e-12)


class TestThroatAdapterInput:
    def test_valid_inputs(self):
        inp = ThroatAdapterInput(D1=0.05, D2=0.10, A1_deg=30.0, A2_deg=30.0, profile_type="conical")
        assert inp.D1 == 0.05
        assert inp.D2 == 0.10
        assert inp.profile_type == "conical"

    def test_invalid_D1_zero(self):
        with pytest.raises(ValueError, match="D1 must be a positive"):
            ThroatAdapterInput(D1=0.0, D2=0.10, A1_deg=30.0, A2_deg=30.0)

    def test_invalid_D2_negative(self):
        with pytest.raises(ValueError, match="D2 must be a positive"):
            ThroatAdapterInput(D1=0.05, D2=-0.1, A1_deg=30.0, A2_deg=30.0)

    def test_invalid_profile_type(self):
        with pytest.raises(ValueError, match="profile_type must be one of"):
            ThroatAdapterInput(D1=0.05, D2=0.10, A1_deg=30.0, A2_deg=30.0, profile_type="spirallic")


class TestMinimumLengthConical:
    def test_expansion_conical(self):
        # D1=50mm, D2=100mm, 30° flare → L = (100-50)/(2*tan30°) = 50/1.155 = 43.3mm
        L = _minimum_length_conical(0.050, 0.100, 30.0, 30.0)
        assert math.isclose(L, 0.0433, rel_tol=0.01)

    def test_constriction_conical(self):
        # D1=100mm, D2=50mm, 30° → same minimum length
        L = _minimum_length_conical(0.100, 0.050, 30.0, 30.0)
        assert math.isclose(L, 0.0433, rel_tol=0.01)

    def test_equal_diameters(self):
        # D1==D2 → L_min = 0
        L = _minimum_length_conical(0.100, 0.100, 30.0, 30.0)
        assert L == 0.0

    def test_asymmetric_angles_uses_smaller(self):
        # D1=50mm, D2=100mm, A1=30°, A2=15° → use 15° (more restrictive)
        L_30 = _minimum_length_conical(0.050, 0.100, 30.0, 30.0)
        L_asymm = _minimum_length_conical(0.050, 0.100, 30.0, 15.0)
        assert L_asymm > L_30  # smaller angle → longer minimum length

    def test_flare_angle_too_small(self):
        # 0.0° → tan(0) = 0 < 1e-12 → raises ValueError
        with pytest.raises(ValueError, match="[Ff]lare angle.*too small"):
            _minimum_length_conical(0.050, 0.100, 0.0, 0.0)


class TestComputeThroatAdapter:
    def test_cylindrical(self):
        adapter = compute_throat_adapter(0.050, 0.050, 0.0, 0.0, "cylindrical")
        assert adapter.type == "cylindrical"
        assert adapter.ap1 == _diameter_to_area(0.050)
        assert adapter.lpt == 0.0

    def test_conical_expansion(self):
        adapter = compute_throat_adapter(0.050, 0.100, 30.0, 30.0, "conical")
        assert adapter.type == "conical"
        assert math.isclose(adapter.ap1, _diameter_to_area(0.100), rel_tol=1e-12)
        # Minimum length: ~43.3 mm
        assert math.isclose(adapter.lpt, 0.0433, rel_tol=0.01)

    def test_conical_explicit_length(self):
        # Override minimum length with a longer explicit value
        adapter = compute_throat_adapter(0.050, 0.100, 30.0, 30.0, "conical", length=0.10)
        assert math.isclose(adapter.lpt, 0.10, rel_tol=1e-12)

    def test_conical_length_too_short(self):
        with pytest.raises(ValueError, match="shorter than the geometric minimum"):
            compute_throat_adapter(0.050, 0.100, 30.0, 30.0, "conical", length=0.01)

    def test_exponential(self):
        adapter = compute_throat_adapter(0.050, 0.100, 30.0, 30.0, "exponential")
        assert adapter.type == "exponential"
        assert adapter.lpt > 0

    def test_parabolic(self):
        adapter = compute_throat_adapter(0.050, 0.100, 30.0, 30.0, "parabolic")
        assert adapter.type == "parabolic"
        assert adapter.lpt > 0

    def test_output_area_equals_D2_area(self):
        D2 = 0.080
        adapter = compute_throat_adapter(0.050, D2, 30.0, 30.0, "conical")
        assert math.isclose(adapter.ap1, _diameter_to_area(D2), rel_tol=1e-12)


class TestThroatAdapterProfile:
    def _standard_adapter(self):
        return compute_throat_adapter(0.050, 0.100, 30.0, 30.0, "conical")

    def test_profile_keys(self):
        adapter = self._standard_adapter()
        A0 = math.pi * (0.050 / 2) ** 2
        result = throat_adapter_profile(adapter, A0=A0, n_points=51)
        assert set(result.keys()) == {"x", "area", "diam", "A0", "Ap1"}

    def test_x_range(self):
        adapter = self._standard_adapter()
        A0 = math.pi * (0.050 / 2) ** 2
        result = throat_adapter_profile(adapter, A0=A0, n_points=51)
        assert result["x"][0] == 0.0
        assert math.isclose(result["x"][-1], adapter.lpt, rel_tol=1e-12)

    def test_conical_endpoints(self):
        adapter = self._standard_adapter()
        A0 = math.pi * (0.050 / 2) ** 2
        result = throat_adapter_profile(adapter, A0=A0, n_points=51)
        assert math.isclose(result["area"][0], A0, rel_tol=1e-12)
        assert math.isclose(result["area"][-1], adapter.ap1, rel_tol=1e-12)

    def test_conical_linear_taper(self):
        adapter = self._standard_adapter()
        A0 = math.pi * (0.050 / 2) ** 2
        result = throat_adapter_profile(adapter, A0=A0, n_points=101)
        # At mid-length, area should be the arithmetic mean
        mid_idx = 50
        expected_mid = (A0 + adapter.ap1) / 2.0
        assert math.isclose(result["area"][mid_idx], expected_mid, rel_tol=1e-6)

    def test_cylindrical_profile_constant(self):
        adapter = compute_throat_adapter(0.050, 0.050, 0.0, 0.0, "cylindrical")
        A0 = math.pi * (0.050 / 2) ** 2
        result = throat_adapter_profile(adapter, A0=A0, n_points=11)
        areas = result["area"]
        assert all(math.isclose(a, areas[0], rel_tol=1e-12) for a in areas)

    def test_exponential_profile_at_ends(self):
        adapter = compute_throat_adapter(0.050, 0.100, 30.0, 30.0, "exponential")
        A0 = math.pi * (0.050 / 2) ** 2
        result = throat_adapter_profile(adapter, A0=A0, n_points=101)
        assert math.isclose(result["area"][0], A0, rel_tol=1e-12)
        assert math.isclose(result["area"][-1], adapter.ap1, rel_tol=1e-12)

    def test_parabolic_profile_at_ends(self):
        adapter = compute_throat_adapter(0.050, 0.100, 30.0, 30.0, "parabolic")
        A0 = math.pi * (0.050 / 2) ** 2
        result = throat_adapter_profile(adapter, A0=A0, n_points=101)
        assert math.isclose(result["area"][0], A0, rel_tol=1e-12)
        assert math.isclose(result["area"][-1], adapter.ap1, rel_tol=1e-12)

    def test_diameter_roundtrip(self):
        adapter = self._standard_adapter()
        A0 = math.pi * (0.050 / 2) ** 2
        result = throat_adapter_profile(adapter, A0=A0, n_points=51)
        # Diameters should be consistent with areas
        for area, diam in zip(result["area"], result["diam"]):
            assert math.isclose(area, _diameter_to_area(diam), rel_tol=1e-12)

    def test_n_points_validation(self):
        adapter = self._standard_adapter()
        A0 = math.pi * (0.050 / 2) ** 2
        with pytest.raises(ValueError, match="n_points must be >= 2"):
            throat_adapter_profile(adapter, A0=A0, n_points=1)

    def test_A0_none_uses_ap1(self):
        # When A0 is None, the adapter is treated as cylindrical (A0 = ap1)
        adapter = self._standard_adapter()
        result = throat_adapter_profile(adapter, A0=None, n_points=11)
        assert math.isclose(result["area"][0], adapter.ap1, rel_tol=1e-12)

    def test_n_points_101(self):
        adapter = self._standard_adapter()
        A0 = math.pi * (0.050 / 2) ** 2
        result = throat_adapter_profile(adapter, A0=A0, n_points=101)
        assert len(result["x"]) == 101
        assert len(result["area"]) == 101
        assert len(result["diam"]) == 101
