from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple, Optional, Any, Literal
import numpy as np


@dataclass
class Section:
    """A single section of a chained horn profile.

    Each section has its own profile type (straight, exponential, hyperbolic, etc.)
    and defines the transition from start_area to end_area over the given length.

    Profile type strings map to existing profile functions in pyhorn_core/solver/profiles.py:
      - "straight"     — constant-area cylinder
      - "exponential"  — exponential flare
      - "hyperbolic"   — hyperbolic (Miki) flare; requires hyperbolic_t parameter
      - "catenoidal"   — catenoidal (surface of revolution)
      - "parabolic"    — parabolic flare
      - "conical"      — linear conical flare

    Damping material (Hornresp pages 73-74):
      - fr1  : flow resistivity of fill material (Rayls/m = Pa·s/m²)
      - tal1 : fill fraction — what fraction of the segment is filled (0.0–1.0)
    """

    name: str
    profile_type: str  # "straight", "exponential", "hyperbolic", "catenoidal", "parabolic", "conical"
    length: float  # metres
    start_area: float  # m²
    end_area: float  # m²
    hyperbolic_t: Optional[float] = None  # only for "hyperbolic" profile_type
    fr1: float = 0.0  # Flow resistivity of damping material (Rayls/m)
    tal1: float = 0.0  # Fill fraction of segment that is damped (0.0–1.0)


@dataclass(frozen=True)
class DriverSpecs:
    """Thiele-Small parameters for the loudspeaker driver. All values in SI metric units."""

    fs: float  # Resonant frequency (Hz)
    qts: float  # Total Q
    qes: float  # Electrical Q
    qms: float  # Mechanical Q
    vas: float  # Equivalent compliance volume (m³)
    re: float  # Voice coil resistance (Ohms)
    bl: float  # Force factor (N/A)
    mms: float  # Moving mass (kg)
    cms: float  # Compliance (m/N)
    rms: float  # Mechanical resistance (kg/s)
    sd: float  # Piston area (m²)
    voltage: float = 2.83  # Input voltage (V)
    le: float = 0.0  # Voice coil inductance (H)
    xmax: float = 0.0  # One-way linear excursion limit (m)
    alpha_re: float = 0.00393  # Temperature coefficient of resistance for copper (1/°C)
    le_freq_dependency: bool = False  # Use frequency-dependent Le model: Le(f) = le * sqrt(1 + (f/f_ref)²)
    le_f_ref: float = 100.0  # Reference frequency (Hz) for semi-inductance Le model (Hornresp page 12)
    # Lossy Le model (Hornresp page 77): accounts for eddy current losses in the motor
    # system — adds a frequency-dependent series resistance R_lossy = R_e_eddy × (f/f_ref)²
    lossy_le: bool = False  # Enable Lossy Le model for large motor systems
    le_R_e_eddy: float = 0.0  # Eddy-current resistance coefficient (ohms) — empirically derived
    le_f_lossy_ref: float = 1000.0  # Reference frequency (Hz) for Lossy Le resistance (Hornresp page 77)
    # Sensitivity calibration for CRIT-3: HF SPL dB/W/m reference offset.
    # Hornresp normalizes HF SPL to dB/W/m (1 W electrical input at 1 m).  pyhorn's
    # raw pressure-based SPL (_pressure_to_spl) can be ~15 dB hot at HF at V=2.83.
    # Set sensitivity_db to the negative of the HF excess (e.g., −15.0 dB) to
    # calibrate pyhorn's absolute HF level to match Hornresp's dB/W/m reference.
    # Can also be set per-driver in the YAML: sensitivity_db: -15.0
    sensitivity_db: float = 0.0  # dB offset applied to acoustic-power-based SPL (scalar or array)
    # Measured free-air SPL response of the bare driver (1 W / 1 m, baffled).
    # When set, the orchestrator overrides the TS-model-predicted direct-cone
    # radiation with this curve — captures cone breakup and motor non-linearities
    # that the lumped TS model cannot predict (esp. above ~2 kHz).
    # Two formats accepted:
    #   - path to CSV with columns "Frequency_Hz, SPL_dB" (resolved relative to driver YAML)
    #   - inline (N, 2) numpy array of [freq_hz, spl_db] pairs
    spl_response: Optional[Any] = None

    def get_sensitivity_db(self, freqs: np.ndarray) -> np.ndarray:
        """
        Return sensitivity_db evaluated at the given frequency array.

        If sensitivity_db is a scalar, returns that scalar broadcast to all freqs.
        If sensitivity_db is an ndarray with the same length as freqs, returns it directly.
        If sensitivity_db is an ndarray of shape (N, 2) with [freq_hz, delta_db] pairs,
        linearly interpolates to the given frequency grid.
        """
        sd = self.sensitivity_db
        if isinstance(sd, np.ndarray):
            # Check if it's a 2-column interpolation table [freq, value]
            if sd.ndim == 2 and sd.shape[1] == 2:
                # Piecewise-linear interpolation table
                table_freqs = sd[:, 0]
                table_vals = sd[:, 1]
                return np.interp(freqs, table_freqs, table_vals, left=table_vals[0], right=table_vals[-1])
            # Already a frequency-matched array
            return np.asarray(sd)
        # Scalar: broadcast
        return np.full_like(freqs, sd, dtype=float)

    def get_spl_response(self, freqs: np.ndarray) -> Optional[np.ndarray]:
        """Return measured driver SPL response at the given frequencies, or None.

        Linearly interpolates the (N, 2) ``spl_response`` table in log-frequency
        space.  Extrapolates by holding the endpoint value.
        """
        sr = self.spl_response
        if sr is None:
            return None
        arr = np.asarray(sr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        table_f = arr[:, 0]
        table_db = arr[:, 1]
        log_f = np.log10(np.maximum(freqs, 1e-6))
        log_tf = np.log10(np.maximum(table_f, 1e-6))
        return np.interp(log_f, log_tf, table_db, left=table_db[0], right=table_db[-1])

    @property
    def reference_spl(self) -> float:
        """Calculates the theoretical half-space (2pi) reference SPL of the bare driver at the given voltage."""
        rho = 1.21
        c = 343.0
        # Reference efficiency
        eta_0 = (rho / (2 * 3.14159265 * c)) * (
            (self.bl**2 * self.sd**2) / (self.re * self.mms**2)
        )
        # SPL at 1W/1m
        spl_1w = 112.2 + 10 * __import__("math").log10(max(eta_0, 1e-12))
        # Adjust for actual voltage
        power = (self.voltage**2) / self.re
        return spl_1w + 10 * __import__("math").log10(max(power, 1e-12))


@dataclass
class HornGeometry:
    """Geometric parameters for the horn enclosure. All values in SI metric units."""

    throat_area: float = 0.0  # Initial throat area (m²)
    mouth_area: float = 0.0  # Final mouth area (m²)
    path_length: float = 0.0  # Total path length of the horn (m)

    # Enclosure Type
    enclosure_type: str = "FLH"  # 'FLH', 'BLH', or 'TL' (finite transmission line)
    path_diff: float = 0.0  # Path difference between driver and horn mouth (m)

    ang: float = 6.283185307  # Solid radiation angle (steradians), default 2*pi

    # Rear chamber
    vrc: float = 0.0  # Rear chamber volume (m³)
    lrc: float = 0.0  # Rear chamber average length (m)
    fr_rc: float = 0.0  # Rear chamber flow resistivity (Rayls/m)

    # Bass-reflex / vented box
    vented_box: Optional["VentedBox"] = None  # Set for vented-box enclosure

    # Passive radiator
    passive_radiator: Optional["PassiveRadiator"] = None  # Set for passive-radiator enclosure

    # Slavic rear chamber (aperiodic / slave bass)
    slavbas: Optional["SlavicBox"] = None  # Set for Slavic rear chamber

    # Throat chamber
    vtc: float = 0.0  # Throat chamber volume (m³)
    atc: float = 0.0  # Throat chamber cross-sectional area (m²)
    fr_tc: float = 0.0  # Throat chamber flow resistivity (Rayls/m)

    # Throat adapter (Ap1 + Lpt) — transition duct between throat chamber and horn throat
    ap1: float = 0.0  # Throat-adapter cross-sectional area at the horn (throat) end (m²)
    lpt: float = 0.0  # Throat-adapter axial length (m)
    throat_adapter_type: str = (
        "cylindrical"  # throat adapter profile type: cylindrical | conical | exponential | parabolic
    )

    # Expansion Profile
    profile_type: Optional[str] = None  # 'Conical' or 'Exponential'
    hyperbolic_t: float = 1.0  # Hornresp/Salmon hyperbolic family parameter T
    n_segments: int = 100  # Discretisation segments

    # Complex Folded Horn support
    width: Optional[float] = None  # Internal width (m) for constant-width folded horns
    # Chained profile sections (new format — preferred over conical/rectangular_segments)
    sections: Optional[List["Section"]] = None

    # conical_segments: List of (dim_start, dim_end, length_m, [optional_fr])
    # If width is set, dim_start/dim_end are heights (m). Otherwise, they are areas (m²).
    # The 4th element is optional flow resistivity (Rayls/m).
    conical_segments: Optional[List[Tuple[float, ...]]] = None
    # rectangular_segments: List of
    # (width_start_m, height_start_m, width_end_m, height_end_m, length_m, [optional_fr])
    # Use this when both width and height vary and area alone is not descriptive enough.
    rectangular_segments: Optional[List[Tuple[float, ...]]] = None

    # Folded 2D Visualization
    coordinates: Optional[List[Tuple[float, float]]] = (
        None  # List of (x, y) coordinates for the start of each segment + mouth
    )
    enclosure_dims: Optional[Tuple[float, float]] = (
        None  # (depth_m, height_m) for the bounding box
    )
    driver_coord: Optional[Tuple[float, float]] = (
        None  # (x, y) coordinate of the driver
    )

    # Geometry-aware discretisation
    discretisation: Optional[str] = (
        None  # None = legacy, "geometry" = angle-aware bends + true cross-sections
    )
    bend_angles: Optional[List[float]] = (
        None  # radians, one per interior coordinate point
    )

    # Local lumped-element corrections for discontinuities.
    # None or "ideal" keeps the legacy pure area-ratio step.
    # "basic" adds a local inertive term and an expansion compliance.
    lem_step_model: Optional[str] = None
    lem_step_strength: float = 1.0
    lem_step_resistance: float = 0.0

    # Segments represented as a list of (length_m, area_m2, [optional_fr])
    segments: List[Tuple[float, ...]] = field(default_factory=list)
    # Bends represented as area steps: list of (area_before_m2, area_after_m2)
    bends: Optional[List[Tuple[float, float]]] = None
    _folded_plot_override: Optional[List[Tuple[float, ...]]] = field(
        default=None, init=False, repr=False
    )

    # ─── Expand sections into flat format ─────────────────────────────────────

    def __post_init__(self) -> None:
        """Expand ``sections`` high-level DSL into ``conical_segments`` for the TMM solver.

        The ``sections`` format lets callers describe a chained horn as a list of
        named profile sections (each with its own profile_type, length, start/end areas).
        The TMM solver only knows about ``conical_segments`` / ``rectangular_segments``,
        so this hook auto-expands ``sections`` on ``HornGeometry`` construction.
        """
        if self.sections is None or self.conical_segments is not None:
            return
        expanded: List[Tuple[float, ...]] = []
        for sec in self.sections:
            # Conical segment: (height_start_m, height_end_m, length_m)
            # heights are diameters derived from circular cross-section areas
            h_start = 2.0 * math.sqrt(max(sec.start_area, 1e-12) / math.pi)
            h_end = 2.0 * math.sqrt(max(sec.end_area, 1e-12) / math.pi)
            expanded.append((h_start, h_end, sec.length))
            # Carry profile type from section into HornGeometry so the solver sees it
            if self.profile_type is None and sec.profile_type is not None:
                self.profile_type = sec.profile_type
            # Carry hyperbolic_t from section (only relevant for "hyperbolic" profile)
            if sec.hyperbolic_t is not None and self.hyperbolic_t == 1.0:
                self.hyperbolic_t = sec.hyperbolic_t
        if expanded:
            self.conical_segments = expanded
            # Sections are already discretised — clear profile_type so the solver
            # uses conical_segments directly instead of re-discretising via
            # discretise_profile() (which would use path_length=0 and fail).
            self.profile_type = None
            # Also populate flat fields from sections for code that reads them directly
            if self.throat_area == 0.0:
                self.throat_area = self.sections[0].start_area
            if self.mouth_area == 0.0:
                self.mouth_area = self.sections[-1].end_area

    def geometry_diagnostics(self) -> Dict[str, float]:
        """Return basic geometry diagnostics for folded-horn inspection."""
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

        # Mouth size check: krm = k·rm = 2π·rm·f/c.
        # For horns with a profile_type and path_length, compute krm at the
        # exponential cutoff frequency. Report mouth rating per Keele (1973):
        #   krm ≥ 1.0  → midrange_ok  (smooth directivity control)
        #   krm ≥ 0.7  → bass_ok       (adequate for bass horns)
        #   krm < 0.7  → undersized     (avoid for bass; mouth reflections significant)
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
                pass  # Profile not recognised; skip krm diagnostics

        return diagnostics

    def folded_plot_segments(self) -> Optional[List[Tuple[float, ...]]]:
        """Return height-based folded segments for 2D plotting.

        Returns the project-level override if one has been set via
        override_folded_plot_segments().  Otherwise derives segments from
        rectangular_segments (if set) or conical_segments or sections.
        """
        override = getattr(self, "_folded_plot_override", None)
        if override is not None:
            return override
        if self.rectangular_segments:
            return [(seg[1], seg[3], seg[4]) for seg in self.rectangular_segments]
        if self.conical_segments:
            return self.conical_segments
        if self.sections:
            # sections format: list of Section(name, profile_type, length, start_area, end_area)
            # For 2D plot, convert circular cross-section areas to diameters (heights)
            result = []
            for seg in self.sections:
                h_start = 2.0 * math.sqrt(max(seg.start_area, 1e-12) / math.pi)
                h_end = 2.0 * math.sqrt(max(seg.end_area, 1e-12) / math.pi)
                result.append((h_start, h_end, seg.length))
            return result
        return None

    def override_driver_coord(self, coord: Tuple[float, float]) -> None:
        """Set driver coordinate for 2D schematic."""
        self.driver_coord = coord

    def override_folded_plot_segments(self, segments: List[Tuple[float, ...]]) -> None:
        """Set project-level override for folded 2D schematic segments."""
        self._folded_plot_override = segments


@dataclass
class ThroatAdapter:
    """Throat adapter — transition duct between the throat chamber and the horn throat.

    The throat adapter is a short profiled section that connects the throat chamber
    (modelled as a lumped compliance via ``ThroatChamber``) to the first horn segment.
    It smooths the area change and can reduce high-frequency reflections.

    Hornresp manual page 013 defines two key parameters:
      - **Ap1** — cross-sectional area at the *throat* (horn) end of the adapter
      - **Lpt** — axial length of the adapter

    The *driver* end of the adapter presents area **Atc** (ThroatChamber.atc).
    For a cylindrical adapter, Atc = Ap1 and the adapter is just a short tube.
    For a conical adapter, the area transitions linearly from Atc → Ap1 over length Lpt.

    Profile types (Hornresp page 087):
      - ``conical``       — straight conical flare, linear area taper
      - ``cylindrical``  — constant cross-section (Atc = Ap1), just a short tube
      - ``exponential``  — exponential area taper
      - ``parabolic``    — parabolic area taper
    """

    type: str = "cylindrical"  # conical | cylindrical | exponential | parabolic
    ap1: float = 0.0  # Cross-sectional area at the throat (horn) end (m²)
    lpt: float = 0.0  # Axial length of the adapter (m)


@dataclass
class RearChamber:
    """Rear chamber (sealed box) parameters.

    The rear chamber is the large sealed enclosure behind the driver.  Its
    compliance (Vrc / rho·c²) adds to the driver's suspension compliance,
    lowering the effective system tuning frequency.

    For a typical FE166NV2 in a 35 L cabinet the rear chamber is the main
    box volume minus the internal horn volume.

    The ``chamber_type`` field selects the acoustic model:

    - ``"sealed"`` (default): rear chamber modeled as a pure acoustic compliance
      (sealed box).  Low-frequency acoustic stiffness, no resonance peak.
    - ``"vented"``: rear chamber modeled as a vented-box / Helmholtz resonator.
      The ``lrc`` parameter is used as the port length, and the port area is
      derived from the Helmholtz resonance using the known Vrc and Lrc.
      This produces an impedance peak at the rear-chamber resonance frequency
      (e.g. ~55 Hz for Vrc=5 L, Lrc=15 cm), matching Hornresp's BLH rear
      chamber behaviour.
    """

    vrc: float = 0.0  # Rear chamber volume (m³)
    lrc: float = 0.0  # Rear chamber average length (m) — port length when chamber_type="vented"
    fr_rc: float = 0.0  # Flow resistivity of damping material inside (Rayls/m)
    fr_tuning: float = 0.0  # Helmholtz tuning frequency (Hz); when > 0, used as f_b in vented model; otherwise falls back to driver.fs in the orchestrator
    width: float = 0.0  # Width (m) — for 2D schematic
    height: float = 0.0  # Height (m) — for 2D schematic
    depth: float = 0.0  # Depth (m) — for 2D schematic
    chamber_type: Literal["vented", "coupling", "sealed"] = "sealed"  # "sealed" | "vented" | "coupling"


@dataclass
class PassiveRadiator:
    """Passive radiator (PR) parameters.

    A passive radiator is like a vented box but instead of a port tube, there is a
    free-moving mass (Mma) on a spider that radiates sound.  The PR has an effective
    moving-system mass ``mma`` and a radiating area ``Sp`` (one or more panels).

    The PR impedance is a series resonant circuit — the mass Mma in series with the
    box compliance C_pr:

        Z_pr = j·ω·Mma + 1/(j·ω·C_pr)

    where C_pr = Vrc / (ρ·c²) — the same compliance as the box air spring in a
    vented box.

    The Helmholtz tuning frequency of the PR is:

        f_pr = (1 / 2π) · √(Sp / (Mma · Vrc))

    Reference: Hornresp manual page 18 (bass reflex / passive radiator model).
    """

    mma: float = 0.0  # Effective mass of PR moving system (kg)
    sp1: float = 0.0  # Radiating area of panel 1 (m²)
    sp2: float = 0.0  # Radiating area of panel 2 (m²)
    sp3: float = 0.0  # Radiating area of panel 3 (m²)
    sp4: float = 0.0  # Radiating area of panel 4 (m²)
    sp5: float = 0.0  # Radiating area of panel 5 (m²)
    sp6: float = 0.0  # Radiating area of panel 6 (m²)
    sp7: float = 0.0  # Radiating area of panel 7 (m²)
    sp8: float = 0.0  # Radiating area of panel 8 (m²)
    sp9: float = 0.0  # Radiating area of panel 9 (m²)
    ql_pr: float = 5.0  # Leak-loss Q factor (dimensionless, default 5.0)

    @property
    def total_sp(self) -> float:
        """Total radiating area of all panels (m²)."""
        return (
            self.sp1
            + self.sp2
            + self.sp3
            + self.sp4
            + self.sp5
            + self.sp6
            + self.sp7
            + self.sp8
            + self.sp9
        )


@dataclass
class VentedBox:
    """Bass-reflex (vented) box parameters.

    The vented box is a Helmholtz resonator: the box volume acts as a compliance
    (C_vb = Vrc / ρ·c²) and the port tube acts as an acoustic mass
    (M_v = ρ·lrc / A_pipe).  The resonance frequency is:

        f_r = (1 / 2π) · √(A_pipe / (Vrc · L_eff))

    where L_eff ≈ lrc + 0.6·√(A_pipe/π) is the end-correction length.

    For a typical FE166NV2 in a 35 L cabinet tuned to ~50 Hz, the port tube
    is ~5–10 cm long with a cross-sectional area of a few cm².

    The vent impedance Z_vb = 1/(jωC_vb) + jωM_v + Z_rad_port is placed in
    parallel with the rear chamber compliance, giving the characteristic
    double-peak impedance response of a bass-reflex system.

    When ``finite_horn_charged=True`` this becomes a **finite horn-charged
    bass reflex** system (Hornresp page 89): the driver fires into both the
    horn and the box simultaneously.  The port radiation and horn radiation
    are summed as acoustic pressures at the listening position.

    When ``finite_transmission_line=True`` the driver additionally loads into
    a **finite transmission line** (pipe closed at the far end, Hornresp page 091).
    The TL output is summed with the horn mouth radiation at the listening point.
    The ``ltl`` parameter specifies the acoustic length of the transmission line;
    the TL mouth radiation is phase-shifted by exp(-j·k·l) relative to the horn
    mouth to account for the additional path length.
    """

    vrc: float = 0.0  # Net box volume (m³)
    fr: float = 0.0  # Vent tuning frequency (Hz)
    lrc: float = 0.0  # Vent tube length (m)
    ql: float = 5.0  # Leak losses Q factor (dimensionless, default 5.0)
    finite_horn_charged: bool = False  # Hybrid BLH + bass reflex topology
    path_length_difference: float = 0.0  # Listening distance offset (m); positive increases the effective path from port to listener, adding phase lag Δφ = 2π·pld/c·f to port radiation (Hornresp page 91)
    finite_transmission_line: bool = False  # Finite transmission line topology (Hornresp page 091)
    ltl: float = 0.0  # Transmission line acoustic length (m) — closed at far end


@dataclass
class SlavicBox:
    """Slavic rear chamber (aperiodic "slave bass") parameters.

    The Slavic rear chamber is a sealed box with a deliberate resistive leak —
    an aperiodic box variant. Unlike a standard vented box (Helmholtz resonator
    with a mass-controlled resonance peak), the Slavic box uses a resistive vent
    (flow-controlled) that bleeds off energy gradually, giving a smooth rolloff
    without a sharp resonance peak.

    The resistive vent is placed in *parallel* with the box compliance, forming
    an overdamped (Q ≈ 0.5) acoustic circuit. The leak effectively shorts DC
    pressure, so Z → 0 as f → 0, while at high frequencies the box behaves as
    a standard sealed enclosure.

    The corner frequency is ω_c = 1 / (R_leak · C_a) where C_a = Vrc / (ρ·c²).

    Reference: Hornresp manual page 65 — "Slavbas" (slave bass) rear chamber type.
    """

    vrc: float = 0.0  # Sealed rear chamber volume (m³)
    rleak: float = 0.0  # Acoustic leak resistance (N·s/m⁵ = Pa·s/m³)
    # Alternatively specify leak as hole area + effective length (converted to rleak):
    aleak: float = 0.0  # Leak hole area (m²)
    lrc: float = 0.005  # Effective length of leak path (m) — default 5 mm hole depth


@dataclass
class ThroatChamber:
    """Throat chamber (sealed rear volume at the horn throat).

    The throat chamber is the small sealed volume directly behind the driver
    cone, between the dust cap and the horn throat.  It is acoustically in
    series with the horn: the driver loads into this chamber, which in turn
    loads into the horn path.

    Typical dimensions for a small LF driver: 2–5 cm deep, area equal to
    the throat opening or the driver piston area (Sd).

    Miki (1990) flow resistivity values for common materials:
      - Wool fibre:       500 – 2 000 Rayls/m
      - Mineral wool:    5 000 – 20 000 Rayls/m
      - Felt (thin):     1 000 – 5 000 Rayls/m
      - Open-cell foam:  2 000 – 10 000 Rayls/m
    """

    vtc: float = 0.0  # Chamber volume (m³). If 0 the chamber is skipped.
    atc: float = 0.0  # Chamber cross-sectional area (m²) presented to the driver.
    fr_tc: float = 0.0  # Flow resistivity of damping material (Rayls/m = Pa·s/m²).


@dataclass
class TappedHornGeometry:
    """Geometric parameters for a Tapped Horn (TH / TH1 mode).

    In a tapped horn the loudspeaker driver is positioned at an *interior* point
    of the horn (at S2 or S3) rather than at the throat or mouth terminus.
    The driver's rear side loads into a rear chamber or the environment; the
    driver's front side loads into the horn proper, which runs from the tap
    point to the mouth.

    Reference: Hornresp manual pages 057–058.

    Topology (TH mode, driver at S2):
        [Rear of driver] ←→ [Rear chamber / free-space]
             ↓ (driver coupling)
        [S2 = tap point] ←→ [Horn sections S2→S3→S4→S5 = mouth]
             ↓ (horn mouth radiation)

    TMM reformulation:
        The horn proper (S2→mouth) is solved as a standard TMM chain with
        radiation impedance at the mouth.  The input impedance at S2 (Z_tap_horn)
        is found by transforming Z_radiation backward through the front sections.

        The driver's rear load (rear chamber, or Z_rear radiation) gives Z_tap_rear.
        The two loads are summed in *series* at the driver's mechanical node:

            1/Z_total = 1/Z_tap_horn + 1/Z_tap_rear   (acoustic admittance sum)

        The driver cone velocity U_d is found from the electrical input voltage,
        then the mouth volume velocity is:

            U_mouth = U_d × (Z_tap_horn / Z_total) × T_tap_to_mouth

        where T_tap_to_mouth is the forward-pressure transfer function of the
        front horn sections from S2 to mouth.

        SPL is then computed from U_mouth and the mouth radiation impedance.

    Parameters
    ----------
    tap_segment_index : int
        1-based index of the segment at which the driver is tapped.
        TH mode: tap_segment_index = 2  (driver at S2)
        TH1 mode: tap_segment_index = 3  (driver at S3)
        The front sections run from segment `tap_segment_index` to the mouth.
        The rear sections run from the driver back to the rear termination.

    front_sections : List[Section]
        Horn sections from the tap point (S_tap) to the mouth.
        Each section has its own profile type, length, start/end areas.
        The first front section starts at S_tap (driver position).

    rear_sections : List[Section]
        Optional sections behind the driver (driver rear → rear termination).
        If empty, the rear termination is free-space radiation (Z_rear = ρc/Sd).

    rear_chamber : Optional[RearChamber]
        If set, the rear of the driver loads into this sealed rear chamber
        (same model as the rear chamber in FLH/BLH).
        If None, rear radiation is directly into free space.

    rear_load_type : str
        "rear_chamber" (default) | "free_space" | "infinite_baffle"
        - "rear_chamber": use rear_chamber volume + compliance
        - "free_space": rear of driver radiates into half-space (ang = 2π)
        - "infinite_baffle": rear of driver on infinite baffle (ang = π)

    ang : float
        Solid radiation angle at the horn MOUTH (steradians).
        Default: 2π (half-space).

    n_segments : int
        Discretisation segments per section (passed to discretise_profile).

    Examples
    --------
    TH mode — 3-segment horn with driver at S2, exponential front section:

        TappedHornGeometry(
            tap_segment_index=2,
            front_sections=[
                Section(name="front", profile_type="exponential",
                        start_area=0.0044, end_area=0.08, length=1.2),
            ],
            rear_chamber=RearChamber(vrc=0.035, lrc=0.15),
        )

    TH1 mode — 4-segment horn with driver at S3, two front sections:

        TappedHornGeometry(
            tap_segment_index=3,
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
    """

    tap_segment_index: int = 2  # TH mode: 2 (S2), TH1 mode: 3 (S3)

    # Horn sections from tap point to mouth
    front_sections: List["Section"] = field(default_factory=list)

    # Sections from driver rear to rear termination (can be empty for free-space)
    rear_sections: List["Section"] = field(default_factory=list)

    # Rear load configuration
    rear_chamber: Optional["RearChamber"] = None
    rear_load_type: str = "rear_chamber"  # "rear_chamber" | "free_space" | "infinite_baffle"

    # Mouth radiation angle (steradians)
    ang: float = 6.283185307  # 2π = half-space

    n_segments: int = 100  # Discretisation per section

    def front_path_length(self) -> float:
        """Total acoustic path length from tap point to mouth."""
        return sum(sec.length for sec in self.front_sections)

    def rear_path_length(self) -> float:
        """Total acoustic path length from driver rear to rear termination."""
        return sum(sec.length for sec in self.rear_sections)


@dataclass
class CompoundChamber:
    """Rear-facing chamber parameters for Compound Horn (CH) mode.

    In Hornresp CH mode the driver is sandwiched between two horns:
      - Front (S1→S4): the main horn, driven at the throat (same as BLH)
      - Rear (driver rear → S4→S5): a secondary horn path sharing the S4 junction

    This dataclass models the rear side of the driver in CH mode — the coupling
    chamber between the driver rear surface and the secondary horn throat (S4).
    In Hornresp these are the Vrc/Lrc parameters (rear chamber) plus optionally
    Vtc/Atc (neck chamber at S4).

    The rear side of the driver radiates into this chamber, which in turn
    couples to the secondary horn path.  The secondary horn mouth (S5) adds to
    the total acoustic output, summing with the main horn mouth radiation.

    Reference: Hornresp manual pages 048–049, 059 (Compound Horn).

    Parameters
    ----------
    vrc_rear : float
        Volume of the rear coupling chamber (m³). Acts as a compliance
        (C_rc = Vrc / ρc²) in series with the secondary horn path.
        In Hornresp: the "rear chamber" volume parameter.
    lrc_rear : float
        Average length of the rear coupling chamber (m). Determines the
        acoustic mass term (L_rc = ρ·lrc / S_d) seen by the driver rear.
        In Hornresp: the "rear chamber length" parameter.
    vtc_rear : float
        Volume of the neck chamber at the secondary horn throat S4 (m³).
        Optional — set to 0 to disable.
        In Hornresp: Vtc for the secondary horn.
    atc_rear : float
        Cross-sectional area of the neck chamber at S4 (m²).
        Optional — set to 0 to disable.
        In Hornresp: Atc for the secondary horn.
    secondary_mouth_area : float
        Mouth area of the secondary horn (S5) in m².
        If 0, the secondary horn is omitted and only direct rear radiation
        from the rear chamber is modeled (chamber-loaded driver, no S4→S5 path).
    secondary_mouth_ang : float
        Solid radiation angle at the secondary mouth (steradians).
        Default: 2π (half-space, like BLH direct radiator).
    """

    vrc_rear: float = 0.0  # Rear coupling chamber volume (m³)
    lrc_rear: float = 0.0  # Rear coupling chamber average length (m)
    vtc_rear: float = 0.0  # Secondary horn neck chamber volume (m³)
    atc_rear: float = 0.0  # Secondary horn neck chamber area (m²)
    secondary_mouth_area: float = 0.0  # Secondary (rear) horn mouth area (m²)
    secondary_mouth_ang: float = 6.283185307  # 2π = half-space
    # ── Dual-driver mode ─────────────────────────────────────────────────────
    # When ch_dual_driver=True, the rear of the horn (secondary mouth side)
    # has its own independent driver unit. Each driver drives its own horn path:
    #   - Front driver: at main throat S1, drives main horn (S1→S4)
    #   - Rear driver : at secondary throat (junction S4), drives secondary horn (S4→S5)
    # Both mouths radiate; their contributions sum acoustically at the listening point.
    # This is a distinct topology from the single-driver CH mode where one driver
    # sits at the junction and both sides of the cone load different horns.
    ch_dual_driver: bool = False  # Enable dual-driver CH mode
    rear_driver: Optional["DriverSpecs"] = None  # Rear driver T-S parameters


@dataclass
class HornProject:
    """Project metadata for a horn simulation.

    Carries non-geometric information that is not produced by auto-segment
    and should not be overwritten by it. Stored in a separate YAML file
    alongside the geometry YAML.

    Example project YAML:
        name: "BK16 1m"
        geometry_path: source/bk16_imported.yaml   # path relative to project file
        driver_coord: [0.02, 0.0]           # (x, y) in metres
        width: 0.20                 # internal horn width (m) — overrides geometry
        notes: " Fostex FE166NV2 in BK16 1m cabinet"
        fold_plot_segments:  # optional override for 2D schematic rendering
          - [0.06, 0.06, 0.05]
          - [0.08, 0.10, 0.04]
        rear_chamber:    # large sealed box behind the driver
          vrc: 0.025     # volume (m³) — 25 L cabinet
          fr_rc: 2000    # flow resistivity of damping (Rayls/m)
        throat_chamber:  # small sealed volume at the horn throat
          vtc: 0.00008   # volume (m³) — ~8 cl
          atc: 0.0044    # cross-sectional area (m²) — throat opening area
          fr_tc: 5000    # flow resistivity of damping (Rayls/m)
    """

    name: Optional[str] = None
    geometry_path: Optional[str] = (
        None  # path to geometry YAML (relative to project file)
    )
    driver_coord: Optional[Tuple[float, float]] = None
    # Override internal width (m) — useful to tweak the horn width without
    # regenerating the geometry from Onshape.
    width: Optional[float] = None
    # Outer enclosure dimensions (m) — for 2D schematic
    enclosure: Optional[Tuple[float, float]] = None
    # Wall thickness (m) — for 2D schematic wall rendering
    thickness: float = 0.0
    material: Optional[str] = None
    notes: Optional[str] = None
    # Override for HornGeometry.folded_plot_segments() — list of (h_start, h_end, length)
    fold_plot_segments: Optional[List[Tuple[float, ...]]] = None
    # Large sealed rear chamber (behind the driver)
    rear_chamber: Optional["RearChamber"] = None
    # Throat chamber (sealed volume at the horn throat)
    throat_chamber: Optional["ThroatChamber"] = None
    # Bass-reflex / vented box
    vented_box: Optional["VentedBox"] = None
    # Passive radiator
    passive_radiator: Optional["PassiveRadiator"] = None
    # Slavic rear chamber (aperiodic / slave bass)
    slavbas: Optional["SlavicBox"] = None
