"""
Room Generator — Hornresp page 96
=================================
Models the effect of room boundaries (walls, floor, ceiling, corners) on the
acoustic power response of a loudspeaker.

The gain follows the boundary gain factor:
    gain_dB = 10 × log10(2π / Ω)
where Ω is the radiation solid angle in steradians:
    free_space  (full sphere 4π):  gain = 0 dB
    half_space  (hemisphere 2π):  gain = +3 dB
    quarter_space (quarter-sphere π): gain = +6 dB
    eighth_space (octant π/2):    gain = +9 dB

The gain is frequency-dependent: at very low frequencies (f << c/wall_dist) the
gain is maximum; it rolls off as 1/f² above the first room mode.

Room mode frequency (Sabine approximation):
    f_room = c / (2×π) × √(A / V)
where A = wall area, V = room volume.

Reference: Hornresp page 96 — Loudspeaker Wizard acoustical power diagram.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─── Physical constants ────────────────────────────────────────────────────────
_C = 343.0  # Speed of sound m/s


# ─── Radiation solid angles & boundary gain constants ─────────────────────────
# Boundary gain = 10×log10(2π/Ω) relative to free-space radiation (Ω=2π sr → 0 dB).
# As the radiation solid angle Ω shrinks (more walls blocking radiation),
# the gain increases:
#   free_space  (Ω=2π sr, hemisphere):  0 dB  — Hornresp reference baseline
#   half_space  (Ω=π sr, quarter-sphere): +3 dB
#   quarter_space (Ω=π/2 sr, eighth-sphere): +6 dB
#   eighth_space (Ω=π/4 sr): +9 dB
_OMEGA = {
    "free_space":  2.0 * np.pi,     # hemisphere (half-space) in Hornresp convention → 0 dB
    "half_space":  np.pi,            # quarter-sphere → +3 dB
    "quarter_space": np.pi / 2.0,    # eighth-sphere → +6 dB
    "eighth_space": np.pi / 4.0,     # → +9 dB
}
# Verify gain values: 10*log10(2π/Ω)
assert abs(10.0 * np.log10(2.0 * np.pi / (2.0 * np.pi))) < 1e-9, "free_space should be 0 dB"
assert abs(10.0 * np.log10(2.0 * np.pi / np.pi) - 3.0103) < 1e-6, "half_space should be ~+3 dB"
assert abs(10.0 * np.log10(2.0 * np.pi / (np.pi / 2.0)) - 6.0206) < 1e-6, "quarter_space should be ~+6 dB"
assert abs(10.0 * np.log10(2.0 * np.pi / (np.pi / 4.0)) - 9.0309) < 1e-6, "eighth_space should be ~+9 dB"
_BOUNDARY_GAIN_DB = {
    key: 10.0 * np.log10(2.0 * np.pi / omega)
    for key, omega in _OMEGA.items()
}


@dataclass
class RoomConfig:
    """Configuration for room boundary gain modeling.

    Parameters
    ----------
    room_type : str
        One of 'free_space', 'half_space', 'quarter_space', 'eighth_space'.
        Describes how many room boundaries the speaker is adjacent to:
          - free_space  : speaker away from all walls (no boundary gain)
          - half_space  : speaker near one wall (e.g. on a stand)
          - quarter_space: speaker in a corner against two walls
          - eighth_space : speaker in a bookshelf/ recesses against three surfaces
    distance_to_wall_m : float, optional
        Distance from the speaker to the nearest wall in metres.
        Used to compute the first room-mode cutoff frequency.
        f_cutoff ≈ c / (2π × distance_to_wall_m)
        If None, a default cutoff of ~300 Hz is used.
    room_volume_m3 : float, optional
        Room volume in cubic metres (for Sabine room-mode estimation).
        Only used when distance_to_wall_m is also provided.
    """
    room_type: str
    distance_to_wall_m: Optional[float] = None
    room_volume_m3: Optional[float] = None

    def __post_init__(self):
        valid = list(_OMEGA.keys())
        if self.room_type not in valid:
            raise ValueError(
                f"room_type must be one of {valid}, got '{self.room_type}'"
            )


def compute_room_gain(
    frequencies: np.ndarray,
    room_type: str,
    distance_to_wall_m: Optional[float] = None,
    room_volume_m3: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the room boundary gain (dB) for each frequency.

    The gain model has two regions:
      1. Low-frequency plateau: the full boundary gain is applied.
         The low-frequency limit is set by the distance to the nearest wall
         (or a default cutoff of ~300 Hz if distance is not specified).
      2. High-frequency rolloff: above the cutoff, gain rolls off as 1/f²
         (i.e. −6 dB/octave), reflecting the fact that wavelengths become
         small compared to the wall distance and the boundary effect diminishes.

    The cutoff frequency is:
        f_cutoff = c / (2π × d)   (when distance_to_wall_m is provided)
    If d is not provided, f_cutoff defaults to ~300 Hz (typical room dimension).

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array in Hz. Must be sorted ascending.
    room_type : str
        One of 'free_space', 'half_space', 'quarter_space', 'eighth_space'.
    distance_to_wall_m : float, optional
        Distance from the speaker to the nearest wall (metres).
    room_volume_m3 : float, optional
        Room volume in cubic metres (used for Sabine room-mode estimation).

    Returns
    -------
    np.ndarray
        Room gain in dB at each frequency. Shape matches ``frequencies``.
        Returns zeros for free_space.
    """
    if room_type == "free_space":
        return np.zeros_like(frequencies, dtype=float)

    if room_type not in _OMEGA:
        raise ValueError(
            f"room_type must be one of {list(_OMEGA.keys())}, got '{room_type}'"
        )

    gain_db = _BOUNDARY_GAIN_DB[room_type]

    # ── Cutoff frequency from wall distance ───────────────────────────────────
    if distance_to_wall_m is not None and distance_to_wall_m > 0:
        # f_cutoff ≈ c / (2π × d)
        f_cutoff = _C / (2.0 * np.pi * distance_to_wall_m)
    else:
        # Default cutoff ≈ 300 Hz (typical for a room with ~0.18 m wall distance)
        f_cutoff = 300.0

    # ── Room mode correction (Sabine) ──────────────────────────────────────────
    # If both room volume and a wall area are available, use Sabine to refine
    # the cutoff. Sabine: f_room = c/(2π) × √(A/V)
    # We approximate A as the relevant boundary area based on room_type.
    if (
        room_volume_m3 is not None
        and room_volume_m3 > 0
        and distance_to_wall_m is not None
        and distance_to_wall_m > 0
    ):
        # Estimate wall area as a square of side √(room_volume/ceiling_height).
        # Assume a typical ceiling height of 2.5 m.
        ceiling_height = 2.5  # m
        wall_area_estimate = room_volume_m3 / ceiling_height  # m²

        f_room = _C / (2.0 * np.pi) * np.sqrt(wall_area_estimate / room_volume_m3)
        # Use the lower of the two cutoff estimates (more conservative rolloff)
        f_cutoff = min(f_cutoff, f_room * 2.0)

    # ── Frequency-dependent gain ──────────────────────────────────────────────
    # Below f_cutoff: full boundary gain
    # Above f_cutoff: rolls off as 1/f² (slope = −6 dB/octave)
    # The transition is smooth (2nd-order Butterworth characteristic).
    gain = np.zeros_like(frequencies, dtype=float)

    # Second-order Butterworth low-pass squared magnitude: H² = 1/(1 + (f/f_c)⁴)
    # At f = f_cutoff: H² = 0.5 → H = −3 dB (standard Butterworth)
    f_ratio = frequencies / f_cutoff
    low_pass_sq = 1.0 / (1.0 + f_ratio**4)
    gain = gain_db * np.sqrt(low_pass_sq)

    return gain


def apply_room_gain(
    spl: np.ndarray,
    frequencies: np.ndarray,
    room_config: RoomConfig,
) -> np.ndarray:
    """
    Apply room boundary gain to an SPL or power response.

    Parameters
    ----------
    spl : np.ndarray
        Sound pressure level or power level in dB (same shape as frequencies).
    frequencies : np.ndarray
        Frequency array in Hz (sorted ascending).
    room_config : RoomConfig
        Room boundary configuration.

    Returns
    -------
    np.ndarray
        SPL/power with room boundary gain added (dB).
    """
    room_gain = compute_room_gain(
        frequencies,
        room_config.room_type,
        room_config.distance_to_wall_m,
        room_config.room_volume_m3,
    )
    return spl + room_gain
