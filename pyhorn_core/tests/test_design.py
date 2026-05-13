"""Unit tests for pyhorn_core.solver.design — build_horn_from_params."""

import pytest
from pyhorn_core.solver.design import build_horn_from_params
from pyhorn_core.config.models import HornGeometry


class TestBuildHornFromParams:
    """Tests for build_horn_from_params builder function."""

    def _make_params(self, **overrides):
        """Minimal valid params mapping with safe defaults."""
        base = {
            "throat_area": 0.0044,
            "mouth_area": 0.09,
            "path_length": 1.8,
            "profile_type": "exponential",
        }
        base.update(overrides)
        return base

    # ─── 1. Required fields ─────────────────────────────────────────────────

    def test_basic_construction_returns_horn_geometry(self):
        """build_horn_from_params should return a HornGeometry instance."""
        params = self._make_params()
        result = build_horn_from_params(params)
        assert isinstance(result, HornGeometry)

    def test_required_fields_throat_mouth_path(self):
        """throat_area, mouth_area, path_length must be set correctly."""
        params = self._make_params(
            throat_area=0.005,
            mouth_area=0.1,
            path_length=2.0,
        )
        result = build_horn_from_params(params)
        assert result.throat_area == 0.005
        assert result.mouth_area == 0.1
        assert result.path_length == 2.0

    def test_profile_type_from_params(self):
        """profile_type from the params dict should be used when no override."""
        params = self._make_params(profile_type="conical")
        result = build_horn_from_params(params)
        assert result.profile_type == "conical"

    # ─── 2. Default values ──────────────────────────────────────────────────

    def test_default_enclosure_type_is_BLH(self):
        """When enclosure_type is not specified, defaults to 'BLH'."""
        params = self._make_params()
        result = build_horn_from_params(params)
        assert result.enclosure_type == "BLH"

    def test_default_n_segments_is_100(self):
        """When n_segments is not specified, defaults to 100."""
        params = self._make_params()
        result = build_horn_from_params(params)
        assert result.n_segments == 100

    def test_default_hyperbolic_t_is_1_0(self):
        """When hyperbolic_t is not specified, defaults to 1.0."""
        params = self._make_params()
        result = build_horn_from_params(params)
        assert result.hyperbolic_t == 1.0

    def test_default_lrc_is_zero(self):
        """When lrc is absent from params, defaults to 0.0."""
        params = self._make_params()
        result = build_horn_from_params(params)
        assert result.lrc == 0.0

    def test_default_vrc_matches_lrc_times_throat_area(self):
        """vrc defaults to lrc * throat_area when not explicitly set."""
        params = self._make_params(lrc=0.15)
        result = build_horn_from_params(params)
        # vrc = lrc * throat_area = 0.15 * 0.0044 = 0.00066
        assert result.vrc == pytest.approx(0.15 * 0.0044)

    def test_default_vtc_is_zero(self):
        """When vtc is absent, defaults to 0.0."""
        params = self._make_params()
        result = build_horn_from_params(params)
        assert result.vtc == 0.0

    # ─── 3. Explicit overrides via params ────────────────────────────────────

    def test_explicit_hyperbolic_t_in_params(self):
        """hyperbolic_t from params should be reflected in output."""
        params = self._make_params(hyperbolic_t=0.8)
        result = build_horn_from_params(params)
        assert result.hyperbolic_t == 0.8

    def test_explicit_vrc_in_params(self):
        """vrc explicitly set in params should override the lrc*throat_area default."""
        params = self._make_params(vrc=0.0005)
        result = build_horn_from_params(params)
        assert result.vrc == 0.0005

    def test_explicit_vtc_in_params(self):
        """vtc explicitly set in params should be reflected in output."""
        params = self._make_params(vtc=0.0001)
        result = build_horn_from_params(params)
        assert result.vtc == 0.0001

    def test_explicit_lrc_in_params(self):
        """lrc explicitly set in params should be reflected in output."""
        params = self._make_params(lrc=0.2)
        result = build_horn_from_params(params)
        assert result.lrc == 0.2

    def test_explicit_n_segments_in_params(self):
        """n_segments from params dict should be used when no override arg."""
        params = self._make_params(n_segments=200)
        result = build_horn_from_params(params)
        assert result.n_segments == 200

    # ─── 4. Function-argument overrides take priority ───────────────────────

    def test_profile_type_override_arg_wins_over_params(self):
        """profile_type passed as function arg should override params dict."""
        params = self._make_params(profile_type="conical")
        result = build_horn_from_params(params, profile_type="hyperbolic")
        assert result.profile_type == "hyperbolic"

    def test_enclosure_type_override_arg_wins(self):
        """enclosure_type passed as function arg should override params default."""
        params = self._make_params()
        result = build_horn_from_params(params, enclosure_type="FLH")
        assert result.enclosure_type == "FLH"

    def test_n_segments_override_arg_wins_over_params(self):
        """n_segments passed as function arg should override params dict."""
        params = self._make_params(n_segments=50)
        result = build_horn_from_params(params, n_segments=150)
        assert result.n_segments == 150

    def test_n_segments_override_arg_wins_over_default(self):
        """n_segments override should also win when params omits it."""
        params = self._make_params()
        result = build_horn_from_params(params, n_segments=300)
        assert result.n_segments == 300

    # ─── 5. Error handling ──────────────────────────────────────────────────

    def test_missing_throat_area_raises(self):
        """Missing throat_area should raise KeyError."""
        params = {"mouth_area": 0.1, "path_length": 1.0, "profile_type": "exponential"}
        with pytest.raises(KeyError):
            build_horn_from_params(params)

    def test_missing_mouth_area_raises(self):
        """Missing mouth_area should raise KeyError."""
        params = {"throat_area": 0.004, "path_length": 1.0, "profile_type": "exponential"}
        with pytest.raises(KeyError):
            build_horn_from_params(params)

    def test_missing_path_length_raises(self):
        """Missing path_length should raise KeyError."""
        params = {"throat_area": 0.004, "mouth_area": 0.1, "profile_type": "exponential"}
        with pytest.raises(KeyError):
            build_horn_from_params(params)

    def test_missing_profile_type_raises_valueerror(self):
        """Missing profile_type with no override should raise ValueError."""
        params = {"throat_area": 0.004, "mouth_area": 0.1, "path_length": 1.0}
        with pytest.raises(ValueError, match="profile_type is required"):
            build_horn_from_params(params)

    # ─── 6. Type coercion ──────────────────────────────────────────────────

    def test_string_numeric_params_coerced_to_float(self):
        """Numeric values passed as strings in params should be coerced to float."""
        params = {
            "throat_area": "0.0044",
            "mouth_area": "0.09",
            "path_length": "1.8",
            "profile_type": "exponential",
            "hyperbolic_t": "0.7",
        }
        result = build_horn_from_params(params)
        assert result.throat_area == 0.0044
        assert result.mouth_area == 0.09
        assert result.path_length == 1.8
        assert result.hyperbolic_t == 0.7

    def test_n_segments_coerced_to_int(self):
        """n_segments is coerced to int even when passed as float."""
        params = self._make_params(n_segments=150.0)
        result = build_horn_from_params(params)
        assert result.n_segments == 150
        assert isinstance(result.n_segments, int)
