from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Section:
    """A single section of a chained horn profile."""

    name: str
    profile_type: str
    length: float
    start_area: float
    end_area: float
    hyperbolic_t: Optional[float] = None
    fr1: float = 0.0
    tal1: float = 0.0


@dataclass
class HornGeometry:
    """Geometric parameters for the horn enclosure."""

    throat_area: float = 0.0
    mouth_area: float = 0.0
    path_length: float = 0.0
    enclosure_type: str = "FLH"
    path_diff: float = 0.0
    ang: float = 6.283185307
    mouth_radiation: str = "levine"
    vrc: float = 0.0
    lrc: float = 0.0
    fr_rc: float = 0.0
    vented_box: Optional["VentedBox"] = None
    passive_radiator: Optional["PassiveRadiator"] = None
    slavbas: Optional["SlavicBox"] = None
    vtc: float = 0.0
    atc: float = 0.0
    fr_tc: float = 0.0
    ap1: float = 0.0
    lpt: float = 0.0
    throat_adapter_type: str = "cylindrical"
    profile_type: Optional[str] = None
    hyperbolic_t: float = 1.0
    n_segments: int = 100
    width: Optional[float] = None
    sections: Optional[List[Section]] = None
    conical_segments: Optional[List[Tuple[float, ...]]] = None
    rectangular_segments: Optional[List[Tuple[float, ...]]] = None
    coordinates: Optional[List[Tuple[float, float]]] = None
    enclosure_dims: Optional[Tuple[float, float]] = None
    driver_coord: Optional[Tuple[float, float]] = None
    discretisation: Optional[str] = None
    bend_angles: Optional[List[float]] = None
    lem_step_model: Optional[str] = None
    lem_step_strength: float = 1.0
    lem_step_resistance: float = 0.0
    segments: List[Tuple[float, ...]] = field(default_factory=list)
    sensitivity_db: Optional[np.ndarray] = None
    bends: Optional[List[Tuple[float, float]]] = None
    _folded_plot_override: Optional[List[Tuple[float, ...]]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.sections is None or self.conical_segments is not None:
            return
        expanded: List[Tuple[float, ...]] = []
        for sec in self.sections:
            h_start = 2.0 * math.sqrt(max(sec.start_area, 1e-12) / math.pi)
            h_end = 2.0 * math.sqrt(max(sec.end_area, 1e-12) / math.pi)
            expanded.append((h_start, h_end, sec.length))
            if self.profile_type is None and sec.profile_type is not None:
                self.profile_type = sec.profile_type
            if sec.hyperbolic_t is not None and self.hyperbolic_t == 1.0:
                self.hyperbolic_t = sec.hyperbolic_t
        if expanded:
            self.conical_segments = expanded
            self.profile_type = None
            if self.throat_area == 0.0:
                self.throat_area = self.sections[0].start_area
            if self.mouth_area == 0.0:
                self.mouth_area = self.sections[-1].end_area

    def geometry_diagnostics(self) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {}

        if self.rectangular_segments:
            areas: List[float] = []
            lengths = [seg[4] for seg in self.rectangular_segments if len(seg) >= 5]
            widths: List[float] = []
            heights: List[float] = []

            for seg in self.rectangular_segments:
                w_start, h_start, w_end, h_end = seg[0], seg[1], seg[2], seg[3]
                widths.extend([w_start, w_end])
                heights.extend([h_start, h_end])
                areas.extend([w_start * h_start, w_end * h_end])

            diagnostics["segment_count"] = float(len(self.rectangular_segments))
            diagnostics["min_segment_length_m"] = min(lengths)
            diagnostics["max_segment_length_m"] = max(lengths)
            diagnostics["min_area_m2"] = min(areas)
            diagnostics["max_area_m2"] = max(areas)
            diagnostics["min_width_m"] = min(widths)
            diagnostics["max_width_m"] = max(widths)
            diagnostics["min_height_m"] = min(heights)
            diagnostics["max_height_m"] = max(heights)

            max_area_ratio = 1.0
            for i in range(1, len(self.rectangular_segments)):
                prev = self.rectangular_segments[i - 1]
                curr = self.rectangular_segments[i]
                prev_end = prev[2] * prev[3]
                curr_start = curr[0] * curr[1]
                if prev_end > 0 and curr_start > 0:
                    ratio = max(prev_end, curr_start) / min(prev_end, curr_start)
                    max_area_ratio = max(max_area_ratio, ratio)
            diagnostics["max_area_step_ratio"] = max_area_ratio

        elif self.conical_segments:
            areas: List[float] = []
            lengths = [seg[2] for seg in self.conical_segments if len(seg) >= 3]

            for seg in self.conical_segments:
                dim_start, dim_end = seg[0], seg[1]
                if self.width is not None:
                    area_start = dim_start * self.width
                    area_end = dim_end * self.width
                else:
                    area_start = dim_start
                    area_end = dim_end
                areas.extend([area_start, area_end])

            diagnostics["segment_count"] = float(len(self.conical_segments))
            diagnostics["min_segment_length_m"] = min(lengths)
            diagnostics["max_segment_length_m"] = max(lengths)
            diagnostics["min_area_m2"] = min(areas)
            diagnostics["max_area_m2"] = max(areas)

            max_area_ratio = 1.0
            for i in range(1, len(self.conical_segments)):
                prev = self.conical_segments[i - 1]
                curr = self.conical_segments[i]
                prev_end = prev[1] * self.width if self.width is not None else prev[1]
                curr_start = curr[0] * self.width if self.width is not None else curr[0]
                if prev_end > 0 and curr_start > 0:
                    ratio = max(prev_end, curr_start) / min(prev_end, curr_start)
                    max_area_ratio = max(max_area_ratio, ratio)
            diagnostics["max_area_step_ratio"] = max_area_ratio

        if self.bend_angles:
            max_bend_rad = max(self.bend_angles)
            diagnostics["max_bend_angle_deg"] = math.degrees(max_bend_rad)
            diagnostics["mean_bend_angle_deg"] = math.degrees(
                sum(self.bend_angles) / len(self.bend_angles)
            )

        if self.lem_step_model and self.lem_step_model.lower() != "ideal":
            diagnostics["lem_enabled"] = 1.0

        if self.profile_type and self.path_length > 0 and self.throat_area > 0:
            try:
                from pyhorn_core.solver.profiles import horn_profile_metrics

                m = horn_profile_metrics(
                    self.profile_type,
                    self.throat_area,
                    self.mouth_area,
                    self.path_length,
                    hyperbolic_t=self.hyperbolic_t,
                )
                diagnostics["krm"] = float(m["krm"])
                diagnostics["cutoff_hz"] = float(m["cutoff_hz"])
                diagnostics["tl_tuning_hz"] = float(m["tl_tuning_hz"])
                diagnostics["mouth_rating"] = (
                    1.0
                    if m["mouth_rating"] == "midrange_ok"
                    else (0.7 if m["mouth_rating"] == "bass_ok" else 0.0)
                )
                diagnostics["mouth_krm_min_hz"] = float(m["mouth_krm_min_hz"])
                diagnostics["mouth_ko_cm"] = float(m["mouth_ko"]) * 100
            except Exception:
                pass

        return diagnostics

    def folded_plot_segments(self) -> Optional[List[Tuple[float, ...]]]:
        override = getattr(self, "_folded_plot_override", None)
        if override is not None:
            return override
        if self.rectangular_segments:
            return [(seg[1], seg[3], seg[4]) for seg in self.rectangular_segments]
        if self.conical_segments:
            return self.conical_segments
        if self.sections:
            result = []
            for seg in self.sections:
                h_start = 2.0 * math.sqrt(max(seg.start_area, 1e-12) / math.pi)
                h_end = 2.0 * math.sqrt(max(seg.end_area, 1e-12) / math.pi)
                result.append((h_start, h_end, seg.length))
            return result
        return None

    def override_driver_coord(self, coord: Tuple[float, float]) -> None:
        self.driver_coord = coord

    def override_folded_plot_segments(self, segments: List[Tuple[float, ...]]) -> None:
        self._folded_plot_override = segments


@dataclass
class TappedHornGeometry:
    """Geometric parameters for a Tapped Horn."""

    tap_segment_index: int = 2
    front_sections: List[Section] = field(default_factory=list)
    rear_sections: List[Section] = field(default_factory=list)
    rear_chamber: Optional["RearChamber"] = None
    rear_load_type: str = "rear_chamber"
    ang: float = 6.283185307
    n_segments: int = 100

    def front_path_length(self) -> float:
        return sum(sec.length for sec in self.front_sections)

    def rear_path_length(self) -> float:
        return sum(sec.length for sec in self.rear_sections)


@dataclass
class CompoundChamber:
    """Rear-facing chamber parameters for Compound Horn mode."""

    vrc_rear: float = 0.0
    lrc_rear: float = 0.0
    vtc_rear: float = 0.0
    atc_rear: float = 0.0
    secondary_mouth_area: float = 0.0
    secondary_mouth_ang: float = 6.283185307
    ch_dual_driver: bool = False
    rear_driver: Optional["DriverSpecs"] = None
