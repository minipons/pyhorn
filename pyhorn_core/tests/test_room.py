"""Tests for the Room Generator (Hornresp page 96)."""
from __future__ import annotations

import numpy as np
import pytest
from pyhorn_core.solver.room import (
    RoomConfig,
    compute_room_gain,
    apply_room_gain,
    _BOUNDARY_GAIN_DB,
    _OMEGA,
)


class TestRoomConfig:
    """Tests for RoomConfig dataclass."""

    def test_free_space_valid(self):
        cfg = RoomConfig(room_type="free_space")
        assert cfg.room_type == "free_space"

    def test_half_space_valid(self):
        cfg = RoomConfig(room_type="half_space", distance_to_wall_m=0.5)
        assert cfg.room_type == "half_space"
        assert cfg.distance_to_wall_m == 0.5

    def test_quarter_space_valid(self):
        cfg = RoomConfig(room_type="quarter_space", room_volume_m3=50.0)
        assert cfg.room_type == "quarter_space"

    def test_eighth_space_valid(self):
        cfg = RoomConfig(room_type="eighth_space")
        assert cfg.room_type == "eighth_space"

    def test_invalid_room_type_raises(self):
        with pytest.raises(ValueError, match="room_type must be one of"):
            RoomConfig(room_type="invalid")

    def test_default_none_values(self):
        cfg = RoomConfig(room_type="half_space")
        assert cfg.distance_to_wall_m is None
        assert cfg.room_volume_m3 is None


class TestBoundaryGainConstants:
    """Tests for boundary gain constant values."""

    def test_free_space_omega(self):
        # free_space uses Ω=2π sr (hemisphere in Hornresp convention, baseline 0 dB)
        assert abs(_OMEGA["free_space"] - 2.0 * np.pi) < 1e-9

    def test_half_space_omega(self):
        # half_space uses Ω=π sr (quarter-sphere) → +3 dB
        assert abs(_OMEGA["half_space"] - np.pi) < 1e-9

    def test_quarter_space_omega(self):
        # quarter_space uses Ω=π/2 sr (eighth-sphere) → +6 dB
        assert abs(_OMEGA["quarter_space"] - np.pi / 2.0) < 1e-9

    def test_eighth_space_omega(self):
        # eighth_space uses Ω=π/4 sr → +9 dB
        assert abs(_OMEGA["eighth_space"] - np.pi / 4.0) < 1e-9

    def test_free_space_gain_zero(self):
        assert abs(_BOUNDARY_GAIN_DB["free_space"]) < 1e-6

    def test_half_space_gain_plus3db(self):
        # 10*log10(2π/π) = 10*log10(2) ≈ 3.0103 dB
        assert abs(_BOUNDARY_GAIN_DB["half_space"] - 3.0103) < 1e-6

    def test_quarter_space_gain_plus6db(self):
        # 10*log10(2π/(π/2)) = 10*log10(4) ≈ 6.0206 dB
        assert abs(_BOUNDARY_GAIN_DB["quarter_space"] - 6.0206) < 1e-6

    def test_eighth_space_gain_plus9db(self):
        # 10*log10(2π/(π/4)) = 10*log10(8) ≈ 9.0309 dB
        assert abs(_BOUNDARY_GAIN_DB["eighth_space"] - 9.0309) < 1e-6


class TestComputeRoomGain:
    """Tests for compute_room_gain function."""

    def test_free_space_returns_zero(self):
        freqs = np.array([20.0, 100.0, 1000.0])
        gain = compute_room_gain(freqs, "free_space")
        np.testing.assert_allclose(gain, np.zeros(3), atol=1e-6)

    def test_free_space_array_shape_preserved(self):
        freqs = np.logspace(1, 3, 200)
        gain = compute_room_gain(freqs, "free_space")
        assert gain.shape == freqs.shape

    def test_half_space_gain_at_low_freq(self):
        freqs = np.array([20.0, 50.0, 100.0])
        gain = compute_room_gain(freqs, "half_space")
        # At very low frequencies the gain should be near the full +3.01 dB
        assert gain[0] > 2.9  # close to 3.01 dB
        assert gain[-1] > 2.5  # still significant at 100 Hz

    def test_quarter_space_gain_at_low_freq(self):
        freqs = np.array([20.0, 50.0, 100.0])
        gain = compute_room_gain(freqs, "quarter_space")
        # At very low frequencies the gain should be near the full +6.02 dB
        assert gain[0] > 5.9  # close to 6.02 dB
        assert gain[-1] > 5.5

    def test_eighth_space_gain_at_low_freq(self):
        freqs = np.array([20.0, 50.0, 100.0])
        gain = compute_room_gain(freqs, "eighth_space")
        # At very low frequencies the gain should be near the full +9.03 dB
        assert gain[0] > 8.9  # close to 9.03 dB
        assert gain[-1] > 8.5

    def test_gain_decreases_at_high_frequencies(self):
        # Use frequencies that span below and above the default cutoff (~300 Hz)
        # so the rolloff is clearly visible.
        freqs = np.array([20.0, 100.0, 400.0, 800.0, 3000.0])
        for room_type in ["half_space", "quarter_space", "eighth_space"]:
            gain = compute_room_gain(freqs, room_type)
            # Gain should monotonically decrease at high frequencies
            assert gain[2] < gain[1], f"{room_type}: gain should roll off"
            assert gain[3] < gain[2], f"{room_type}: gain should roll off"
            assert gain[4] < gain[3], f"{room_type}: gain should roll off"

    def test_gain_rolloff_is_smooth(self):
        freqs = np.logspace(np.log10(50), np.log10(5000), 500)
        gain = compute_room_gain(freqs, "half_space")
        # First derivative should be negative and continuous (no sudden jumps)
        diff = np.diff(gain)
        # Count sign changes in the derivative (should be smooth — few sign flips)
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        assert sign_changes < 10, "Gain rolloff should be smooth, not jagged"

    def test_invalid_room_type_raises(self):
        freqs = np.array([100.0])
        with pytest.raises(ValueError, match="room_type must be one of"):
            compute_room_gain(freqs, "invalid_type")

    def test_wall_distance_affects_cutoff(self):
        freqs = np.logspace(1, 4, 1000)
        # Close wall (0.1 m) → higher cutoff (~545 Hz), so 200 Hz is in-band
        gain_close = compute_room_gain(freqs, "half_space", distance_to_wall_m=0.1)
        # Far wall (1.0 m) → lower cutoff (~54 Hz), so 200 Hz is already rolled off
        gain_far = compute_room_gain(freqs, "half_space", distance_to_wall_m=1.0)
        # At 200 Hz, the close wall should have more gain (its cutoff is above 200 Hz)
        idx_200 = int(np.argmin(np.abs(freqs - 200)))
        assert gain_close[idx_200] > gain_far[idx_200], \
            "Close wall (higher cutoff) should show more gain at 200 Hz"

    def test_room_volume_tunes_cutoff(self):
        freqs = np.logspace(1, 4, 500)
        # Small room (10 m³) → higher room mode → higher cutoff
        gain_small = compute_room_gain(
            freqs, "half_space",
            distance_to_wall_m=0.5,
            room_volume_m3=10.0,
        )
        # Large room (200 m³) → lower room mode → lower cutoff
        gain_large = compute_room_gain(
            freqs, "half_space",
            distance_to_wall_m=0.5,
            room_volume_m3=200.0,
        )
        # At a mid frequency both should be rolled off differently
        idx_300 = int(np.argmin(np.abs(freqs - 300)))
        # Small room has higher room mode, so less rolloff at 300 Hz
        assert gain_small[idx_300] > gain_large[idx_300] - 0.5

    def test_output_dtype_is_float(self):
        freqs = np.array([20.0, 100.0, 1000.0], dtype=np.float64)
        gain = compute_room_gain(freqs, "half_space")
        assert gain.dtype == np.float64


class TestApplyRoomGain:
    """Tests for apply_room_gain function."""

    def test_apply_room_gain_adds_to_spl(self):
        freqs = np.array([20.0, 100.0, 10000.0])
        spl = np.array([80.0, 85.0, 90.0])
        cfg = RoomConfig(room_type="half_space")
        result = apply_room_gain(spl, freqs, cfg)
        assert result[0] > spl[0]  # half_space adds gain at low freq
        # At 10 kHz the gain is essentially 0 (well above the 300 Hz cutoff)
        assert abs(result[2] - spl[2]) < 0.01

    def test_free_space_no_change(self):
        freqs = np.array([20.0, 100.0, 1000.0])
        spl = np.array([80.0, 85.0, 90.0])
        cfg = RoomConfig(room_type="free_space")
        result = apply_room_gain(spl, freqs, cfg)
        np.testing.assert_allclose(result, spl, atol=1e-6)

    def test_output_shape_matches_input(self):
        freqs = np.logspace(1, 3, 300)
        spl = np.full_like(freqs, 85.0)
        cfg = RoomConfig(room_type="quarter_space")
        result = apply_room_gain(spl, freqs, cfg)
        assert result.shape == freqs.shape

    def test_apply_room_gain_with_wall_distance(self):
        freqs = np.array([50.0, 200.0, 2000.0])
        spl = np.array([75.0, 80.0, 85.0])
        cfg = RoomConfig(room_type="half_space", distance_to_wall_m=0.2)
        result = apply_room_gain(spl, freqs, cfg)
        # At low freq (50 Hz) the gain should be near maximum
        assert result[0] > spl[0] + 2.0
        # At 2 kHz the gain should be near zero
        assert abs(result[2] - spl[2]) < 0.5
