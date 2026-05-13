"""
pyhorn_core/solver/filter_schematic.py
=======================================
Generate ASCII art schematics for audio filter topologies used in the
Filter Wizard.  Shared by the FastAPI server and (in future) the CLI.

Presets
-------
The module ships three families of schematics:

  Le Cléac'h HP    — 2nd-order series-L / shunt-C high-pass
                    (André Le Cléac'h, 12 dB/oct acoustic crossover)
  LR2 Crossover    — Linkwitz-Riley 2nd-order LR crossover
                      (2-way and 3-way variants)
  Parametric/EQ     — generic peaking EQ, high-shelf, low-shelf symbols

Usage
-----
  from pyhorn_core.solver.filter_schematic import generate_schematic

  # Named preset
  asc = generate_schematic(preset="le_cleach")

  # Custom parameters
  asc = generate_schematic(type="le_cleach", fc=120.0, q=0.71, r_load=8.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

# ─── Filter type union ────────────────────────────────────────────────────────

FilterType = Literal[
    "le_cleach",
    "lr2_crossover_2way",
    "lr2_crossover_3way",
    "peaking_eq",
    "highshelf",
    "lowshelf",
    "lowpass",
    "highpass",
    "bandpass",
]


@dataclass
class FilterBand:
    """A single filter band, compatible with pyhorn_ui/server.py FilterBand."""
    type: str
    frequency: float       # Hz
    q: float = 1.0
    gain_db: float = 0.0
    order: int = 2
    enabled: bool = True


# ─── Passive R/L/C Filter ─────────────────────────────────────────────────────

@dataclass
class PassiveComponent:
    """A single passive electrical component (R, L, or C).

    Parameters
    ----------
    component_type : Literal["R", "L", "C"]
        Resistor, inductor, or capacitor.
    value : float
        R in ohms, L in henries, C in farads.
    position : Literal["series", "shunt"], default "series"
        Whether the component is in series with the signal path
        or in shunt (parallel) to ground.
    """
    component_type: Literal["R", "L", "C"]
    value: float
    position: Literal["series", "shunt"] = "series"


class PassiveFilter:
    """Passive R/L/C electrical network filter.

    Models series and parallel R, L, C networks as electrical filters.
    Used for passive crossover networks (Le Cléac'h filter, LR2 crossover, etc.)

    The filter represents a network of passive components placed between the
    source and the load.  Series components are placed in the signal path;
    shunt components are connected from the signal node to ground.

    Electrical model
    ---------------
    Impedance of each component at frequency f (Hz):
        R: Z(f) = R
        L: Z(f) = j · 2π · f · L
        C: Z(f) = 1 / (j · 2π · f · C)

    Series topology (components in series with the load):
        Z_total = Σ Z_i
        H(jω)   = R_load / (Z_source + Z_total)

    Parallel topology (components in parallel across input/output):
        1/Z_total = Σ (1/Z_i)   for shunt components only
        H(jω)     = Z_total / (Z_total + R_load)

    Application to horn response
    ----------------------------
    The voltage transfer function magnitude |H(jω)| (linear) scales the
    SPL of the horn driver connected at the output:
        SPL_filtered = SPL_baseline + 20·log10(|H(jω)|)

    Phase contribution is added to the baseline phase.
    Impedance magnitude is scaled by |H(jω)|.

    Parameters
    ----------
    topology : Literal["series", "parallel"]
        Whether the components form a series chain or a shunt/parallel network.
    components : list[PassiveComponent]
        Ordered list of components.  For series topology this is the signal
        path; for parallel topology shunt components are identified by
        ``position="shunt"``.
    """

    def __init__(
        self,
        topology: Literal["series", "parallel", "le_cleach"],
        components: list[PassiveComponent],
    ):
        if topology not in ("series", "parallel", "le_cleach"):
            raise ValueError("topology must be 'series', 'parallel', or 'le_cleach'")
        if not components:
            raise ValueError("components list must not be empty")
        self.topology = topology
        self.components = components

    # ── Impedance helpers ───────────────────────────────────────────────────

    def _z_single(self, comp: PassiveComponent, f: float) -> complex:
        """Impedance of one component at frequency f (Hz)."""
        if comp.component_type == "R":
            return complex(comp.value)
        omega = 2.0 * math.pi * f
        if comp.component_type == "L":
            return complex(0, omega * comp.value)
        # C
        if f <= 0.0:
            return complex(math.inf, 0)
        return complex(0, -1.0 / (omega * comp.value))

    def _z_series(self, f: float) -> complex:
        """Total impedance of all series components at f."""
        return sum(self._z_single(c, f) for c in self.components if c.position == "series")

    def _z_parallel(self, f: float) -> complex:
        """Total impedance of shunt components in parallel at f.

        For le_cleach topology all components are at the shunt/junction node
        (capacitor and inductor in parallel, representing the shunt branch).
        For other topologies only components with position='shunt' are included.
        """
        if self.topology == "le_cleach":
            shunt = self.components  # all components at junction
        else:
            shunt = [c for c in self.components if c.position == "shunt"]
        if not shunt:
            return complex(math.inf)
        z_inv = sum(1.0 / self._z_single(c, f) for c in shunt)
        if z_inv == 0:
            return complex(math.inf)
        return complex(1.0 / z_inv)

    def z_at(self, f: float) -> complex:
        """Total impedance of this filter network at frequency f (Hz).

        For series topology: sum of series-component impedances (looking into
        the network from source toward load).
        For le_cleach topology: impedance below the junction node (shunt
        branch = R_load || C; used for the transfer function).
        For parallel topology: equivalent impedance of shunt components
        in parallel (bypassing the signal path).
        """
        if self.topology == "series":
            return self._z_series(f)
        if self.topology == "le_cleach":
            # Return the shunt-branch impedance (R_load || C) for transfer function
            return self._z_parallel(f)
        return self._z_parallel(f)

    # ── Transfer function ───────────────────────────────────────────────────

    def transfer_function(self, f: float, r_load: float = 4.0) -> complex:
        """Voltage transfer function H(jω) = V_out / V_in at frequency f.

        Parameters
        ----------
        f : float
            Frequency in Hz.
        r_load : float
            Load resistance in ohms (default 4.0 Ω — typical midbass driver).

        Returns
        -------
        complex
            Complex transfer function H(jω).  Magnitude is the voltage
            attenuation (linear); phase is the phase shift in radians.

        Series topology (output across load resistor):
            H = R_load / (Z_series + R_load)

        Le Cléac'h topology (series C → junction → shunt (R_load || L) to ground):
            Series branch: Z_C = 1/(jωC)  (blocks LF, passes HF)
            Shunt branch: Z_shunt = R_load || jωL  (at junction node)
            Output is taken at the junction node (between series C and shunt L).
            H = Z_shunt / (Z_C + Z_shunt)

        Parallel topology (shunt network across input/output):
            H = Z_parallel / (Z_parallel + R_load)
        """
        omega = 2.0 * math.pi * f
        if self.topology == "series":
            z = self._z_series(f)
            H = r_load / (z + r_load)
        elif self.topology == "le_cleach":
            # Le Cléac'h HP: series C → junction node → shunt (R_load || L)
            # Series: Z_C = 1/(jωC)  — blocks low frequencies
            # Shunt: Z_L = jωL, in parallel with R_load
            C = self._get_capacitance()
            L = self._get_inductance()
            if f <= 0.0 or C >= 1e10:
                # DC: capacitor is open → no current → output = 0
                return complex(0.0)
            z_c = complex(0, -1.0 / (omega * C))  # series C
            if L <= 0.0:
                z_shunt = complex(r_load)  # no L → just R_load
            else:
                z_l = complex(0, omega * L)  # shunt L
                # Z_shunt = R_load || jωL = (R_load * jωL) / (R_load + jωL)
                z_shunt = (complex(r_load) * z_l) / (complex(r_load) + z_l)
            H = z_shunt / (z_c + z_shunt)
        else:
            # Parallel: shunt network
            z = self._z_parallel(f)
            H = z / (z + r_load)
        return H

    def _get_inductance(self) -> float:
        """Return the inductance (in henries) of the first L component."""
        for c in self.components:
            if c.component_type == "L":
                return c.value
        return 0.0

    def _get_capacitance(self) -> float:
        """Return the capacitance (in farads) of the first C component."""
        for c in self.components:
            if c.component_type == "C":
                return c.value
        return math.inf

    def magnitude_db(self, f: float, r_load: float = 4.0) -> float:
        """Magnitude of the transfer function in dB at frequency f."""
        H = self.transfer_function(f, r_load)
        mag = abs(H)
        if mag < 1e-30:
            return -600.0
        return 20.0 * math.log10(mag)

    def phase_deg(self, f: float, r_load: float = 4.0) -> float:
        """Phase of the transfer function in degrees at frequency f."""
        H = self.transfer_function(f, r_load)
        return math.degrees(math.atan2(H.imag, H.real))

    # ── Application to horn response ────────────────────────────────────────

    def apply_to_response(
        self,
        spl: np.ndarray,
        frequencies: np.ndarray,
        impedance: np.ndarray,
        phase: np.ndarray,
        r_load: float = 4.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply this passive filter to horn response data.

        Each frequency point is weighted by the filter's transfer function.
        SPL is modified in the dB domain:
            filtered_spl = baseline_spl + filter_magnitude_db
        Phase is accumulated:
            filtered_phase = baseline_phase + filter_phase_deg
        Impedance magnitude is scaled by the filter's linear magnitude.

        Parameters
        ----------
        spl : np.ndarray
            Baseline SPL in dB (one value per frequency).
        frequencies : np.ndarray
            Frequency points in Hz (same length as spl).
        impedance : np.ndarray
            Baseline impedance magnitude in ohms.
        phase : np.ndarray
            Baseline phase in degrees.
        r_load : float
            Load resistance in ohms (default 4.0 Ω).

        Returns
        -------
        (filtered_spl, filtered_impedance, filtered_phase)
            Tuple of modified arrays, all the same shape as the inputs.
        """
        spl = np.asarray(spl, dtype=float)
        frequencies = np.asarray(frequencies, dtype=float)
        impedance = np.asarray(impedance, dtype=float)
        phase = np.asarray(phase, dtype=float)

        # Compute filter magnitude (linear) and phase (deg) at each frequency
        mag_linear = np.empty_like(frequencies, dtype=float)
        phase_shift = np.empty_like(frequencies, dtype=float)

        for i, f in enumerate(frequencies):
            H = self.transfer_function(f, r_load)
            mag_linear[i] = abs(H)
            phase_shift[i] = math.degrees(math.atan2(H.imag, H.real))

        # Avoid log of zero
        mag_linear = np.clip(mag_linear, 1e-30, None)
        mag_db = 20.0 * np.log10(mag_linear)

        filtered_spl = spl + mag_db
        filtered_impedance = impedance * mag_linear
        filtered_phase = phase + phase_shift

        return filtered_spl, filtered_impedance, filtered_phase

    # ── Schematic ────────────────────────────────────────────────────────────

    def schematic(self, r_load: float = 4.0) -> str:
        """Return an ASCII art schematic of this passive filter network."""
        sep = "─" * 58
        lines = [f"Passive R/L/C Filter — {self.topology.capitalize()} Network", sep]

        # Component table
        lines.append(" Component List")
        lines.append(f"  {'#':<3} {'Type':<6} {'Value':<18} {'Position':<8}")
        lines.append("  " + "─" * 38)
        for i, comp in enumerate(self.components, 1):
            if comp.component_type == "R":
                val_str = f"{comp.value:.2f} Ω"
            elif comp.component_type == "L":
                val_str = _format_L(comp.value)
            else:
                val_str = _format_C(comp.value)
            lines.append(f"  {i:<3} {comp.component_type:<6} {val_str:<18} {comp.position:<8}")

        lines.append(sep)
        lines.append(f" Topology: {'Series chain (signal path)' if self.topology == 'series' else 'Shunt / parallel network (to ground)'}")
        lines.append(f" Load: {r_load:.1f} Ω")
        lines.append(sep)

        # ASCII diagram
        if self.topology == "series":
            comp_symbols = []
            for comp in self.components:
                if comp.component_type == "R":
                    comp_symbols.append("──┤R├──")
                elif comp.component_type == "L":
                    comp_symbols.append("──┤L├──")
                else:
                    comp_symbols.append("──┤C├──")
            net = "──".join(comp_symbols)
            lines.append(f"  input  ──{net}── output  (V_out across load)")
            lines.append(f"                    R_load = {r_load:.1f} Ω to ground")
        elif self.topology == "le_cleach":
            # Series C → junction → shunt (R_load || L) to ground
            C_val = next((c for c in self.components if c.component_type == "C"), None)
            L_val = next((c for c in self.components if c.component_type == "L"), None)
            C_sym = f"──┤{C_val.component_type}──" if C_val else "──┤C──"
            L_sym = f"┤{L_val.component_type}┤" if L_val else "┤L┤"
            lines.append(f"  input  ──{C_sym}── output (V_junction)")
            lines.append(f"                    │")
            lines.append(f"                    ┌──{L_sym}──┐")
            lines.append(f"                    │  jωL      │")
            lines.append(f"                    └───────────┘")
            lines.append(f"                     │ GND  (parallel with R_load)")
        else:
            # Parallel: shunt components drawn to ground
            shunt_cmps = [c for c in self.components if c.position == "shunt"]
            if shunt_cmps:
                shunt_str = "  ||  ".join(
                    f"┤{c.component_type}┤" for c in shunt_cmps
                )
                lines.append(f"  input  ──{shunt_str}── output")
                lines.append(f"                 │ GND")
            else:
                lines.append(f"  input  ──────────────────── output  (no shunt components)")

        lines.append(sep)
        lines.append(" Notes")
        lines.append("  • Series: Z_total = Σ Z_i;  H = R_load / (Z_total + R_load)")
        lines.append("  • Parallel: Z_total = 1/Σ(1/Z_i);  H = Z_total / (Z_total + R_load)")
        lines.append("  • Component impedances: Z_R = R, Z_L = jωL, Z_C = 1/(jωC)")
        lines.append(sep)
        return "\n".join(lines)


# ─── Component value formulas ─────────────────────────────────────────────────

def _le_cleach_L(fc: float, q: float, r_load: float) -> float:
    """Le Cléac'h HP: L = R / (2π·fc·Q)."""
    return r_load / (2.0 * math.pi * fc * q)


def _le_cleach_C(fc: float, q: float, r_load: float) -> float:
    """Le Cléac'h HP: C = Q / (2π·fc·R)."""
    return q / (2.0 * math.pi * fc * r_load)


def _lr2_L(fc: float, r_load: float) -> float:
    """LR2 LP/HP: L = R / (√2·π·fc)  (Q = 1/√2 for Butterworth LR2)."""
    return r_load / (math.sqrt(2) * math.pi * fc)


def _lr2_C(fc: float, r_load: float) -> float:
    """LR2 LP/HP: C = 1 / (√2·π·fc·R)."""
    return 1.0 / (math.sqrt(2) * math.pi * fc * r_load)


def _format_L(L_henry: float) -> str:
    if L_henry >= 1.0:
        return f"{L_henry:.2f} H"
    elif L_henry >= 1e-3:
        return f"{L_henry * 1e3:.2f} mH"
    else:
        return f"{L_henry * 1e6:.2f} µH"


def _format_C(farad: float) -> str:
    if farad >= 1.0:
        return f"{farad:.2f} F"
    elif farad >= 1e-3:
        return f"{farad * 1e3:.2f} mF"
    elif farad >= 1e-6:
        return f"{farad * 1e6:.2f} µF"
    elif farad >= 1e-9:
        return f"{farad * 1e9:.2f} nF"
    else:
        return f"{farad * 1e12:.2f} pF"


def _fmt_fc(fc: float) -> str:
    if fc >= 1000:
        return f"{fc / 1000:.2f} kHz"
    return f"{fc:.1f} Hz"


# ─── Individual schematic generators ─────────────────────────────────────────

def _schematic_le_cleach(fc: float, q: float, r_load: float) -> str:
    L = _le_cleach_L(fc, q, r_load)
    C = _le_cleach_C(fc, q, r_load)
    sep = "─" * 58
    return f"""\
 Le Cléac'h High-Pass Filter (2nd-order / 12 dB/oct)
{sep}
 Topology:  series L → shunt C to ground
{sep}
               ┌─────┐
   input  ─────┤  L  ├───┬──────── HP output ──── (to amplifier / driver)
               └─────┘   │
                         ┌┴┐
                         │C│  C ─┬── GND
                         └┬┘     │
                          │     ─┘
                          └────────── GND

 Component Formulas (2nd-order Butterworth alignment, Q = {q:.2f})
   L = R / (2π · fc · Q)          C = Q / (2π · fc · R)

 Parameters
   R  = {r_load:.1f} Ω     (load / source impedance)
   fc = {fc:.1f} Hz   ({_fmt_fc(fc)})
   Q  = {q:.2f}

 Calculated Values
   L ≈ {_format_L(L)}   ({L:.4e} H)
   C ≈ {_format_C(C)}  ({C:.4e} F)

 Notes
   • At f >> fc: signal passes (~0 dB insertion loss)
   • At f << fc: series-L blocks LF, shunt-C shortens return path
   • Le Cléac'h topology minimizes group-delay variation vs. pure LR2
   • André Le Cléac'h, " Haut-Parleur et Charge 3D", 2006
"""


def _schematic_lr2_xover_2way(fc: float, r_load: float) -> str:
    L = _lr2_L(fc, r_load)
    C = _lr2_C(fc, r_load)
    sep = "─" * 58
    return f"""\
 LR2 Crossover — 2-Way (Linkwitz-Riley 2nd-order / 12 dB/oct)
{sep}

 ── LOW-PASS (Woofer) ──────────────────────────────────────
               ┌─────┐
   input  ─────┤  L  ├───┬──────── LP output (to woofer)
               └─────┘   │
                         ┌┴┐
                         │C│  C ─┬── GND
                         └┴┘     │
                          │     ─┘
                          └──────── GND

 ── HIGH-PASS (Tweeter) ────────────────────────────────────
               ┌─────┐
   input  ─────┤  C  ├───┬──────── HP output (to tweeter)
               └─────┘   │
                         ┌┴┐
                         │L│  L ─┬── GND
                         └┬┘     │
                          │     ─┘
                          └──────── GND

 Component Formulas (LR2 Butterworth, Q = 1/√2 ≈ 0.707)
   L = R / (√2 · π · fc)          C = 1 / (√2 · π · fc · R)

 Parameters
   R  = {r_load:.1f} Ω
   fc = {fc:.1f} Hz   ({_fmt_fc(fc)})

 Calculated Values (per section)
   L ≈ {_format_L(L)}   ({L:.4e} H)
   C ≈ {_format_C(C)}  ({C:.4e} F)

 Notes
   • LR2: cascaded 1st-order IIR sections → Linkwitz-Riley alignment
   • Both outputs are in-phase at crossover frequency
   • Phase difference: 180° (acoustic), compensated by polarity flip
"""


def _schematic_lr2_xover_3way(fc1: float, fc2: float, r_load: float) -> str:
    # fc1 = low/mid crossover, fc2 = mid/high crossover
    L1 = _lr2_L(fc1, r_load)
    C1 = _lr2_C(fc1, r_load)
    L2 = _lr2_L(fc2, r_load)
    C2 = _lr2_C(fc2, r_load)
    sep = "─" * 58
    return f"""\
 LR2 Crossover — 3-Way (Linkwitz-Riley 2nd-order / 12 dB/oct)
{sep}

 ── LOW-PASS (Woofer)  fc = {fc1:.1f} Hz ─────────────────────
               ┌─────┐
   input  ─────┤  L  ├───┬──────── LP output (to woofer)
               └─────┘   │
                         ┌┴┐
                         │C│  C ─┬── GND
                         └┴┘     │
                          │     ─┘
                          └──────── GND

 ── BAND-PASS (Midrange)  fc1 = {fc1:.1f} Hz, fc2 = {fc2:.1f} Hz ─
 (HP section: C → L to input, LP section: L → C to output)
               ┌─────┐
   input  ─────┤  C  ├───┐   ┌─────┐
               └─────┘   ├───┤  L  ├───┬──────── BP output
                         │   └─────┘   │
                         │             ┌┴┐
                         │             │L│  L ─┬── GND
                         │             └┬┘     │
                         │              │     ─┘
                         │              └──────── GND
                         │
                         │  (HP section: C in series, L to GND)
                         └────────────────────────── GND

 ── HIGH-PASS (Tweeter)  fc = {fc2:.1f} Hz ──────────────────
               ┌─────┐
   input  ─────┤  C  ├───┬──────── HP output (to tweeter)
               └─────┘   │
                         ┌┴┐
                         │L│  L ─┬── GND
                         └┬┘     │
                          │     ─┘
                          └──────── GND

 Parameters
   R  = {r_load:.1f} Ω
   fc1 = {fc1:.1f} Hz   ({_fmt_fc(fc1)})
   fc2 = {fc2:.1f} Hz   ({_fmt_fc(fc2)})

 Calculated Values
   Section 1 (LP):  L ≈ {_format_L(L1)},  C ≈ {_format_C(C1)}
   Section 2 (HP):  L ≈ {_format_L(L2)},  C ≈ {_format_C(C2)}
"""


def _schematic_peaking_eq(fc: float, q: float, gain_db: float) -> str:
    sep = "─" * 58
    return f"""\
 Peaking / Parametric EQ
{sep}
 Topology:  frequency-dependent feedback network (RBJ / Bristow-Johnson)
{sep}

            ┌─────────────┐
   input ──┤  +          │
            │   Σ    H(z) ├───┬──── output
   feedback┤  -          │   │
   (from   └──────┬──────┘   │
   output)        │           │  ┌─────┐
                  └───────────┴──┤  z⁻¹ │  (delay)
                                    └─────┘

 Filter Parameters
   fc   = {fc:.1f} Hz   ({_fmt_fc(fc)})
   Q    = {q:.2f}
   Gain = {gain_db:+.1f} dB   ({'boost' if gain_db >= 0 else 'cut'})

 Formulas (RBJ Biquad)
   w0  = 2π · fc / fs
   A   = 10^(gain/40)
   α   = sin(w0) / (2·Q)

   b = [ 1 + α·A,  -2·cos(w0),  1 - α·A ]  (boost)
   b = [ 1 + α/A,  -2·cos(w0),  1 - α/A ]  (cut)
   a = [ 1 + α/A,  -2·cos(w0),  1 - α/A ]  (boost)
   a = [ 1 + α·A,  -2·cos(w0),  1 - α·A ]  (cut)

 Notes
   • Neutral at 0 dB gain (identity filter)
   • Phase shift accumulates from multiple cascaded sections
   • For narrow notches use Q > 2.0; for gentle shelves use Q ≈ 0.7
"""


def _schematic_highshelf(fc: float, q: float, gain_db: float) -> str:
    sep = "─" * 58
    return f"""\
 High-Shelf EQ (1st/2nd-order)
{sep}
 Topology:  RBJ high-shelf — frequency-independent gain above fc
{sep}

               ┌──────────────────┐
   input  ─────┤   High-Shelf      ├───┬──── output
               │   (boost/cut)     │   │
               └──────────────────┘   │
                                       │
   gain above fc = {gain_db:+.1f} dB        GND

 Filter Parameters
   fc   = {fc:.1f} Hz   ({_fmt_fc(fc)})
   Q    = {q:.2f}
   Gain = {gain_db:+.1f} dB   ({'boost' if gain_db >= 0 else 'cut'} above fc)

 Notes
   • Below fc: flat response (0 dB)
   • Above fc: gain = gain_db
   • 1st-order = 6 dB/oct slope, 2nd-order = 12 dB/oct slope
   • Higher Q = steeper initial roll-off near fc
"""


def _schematic_lowshelf(fc: float, q: float, gain_db: float) -> str:
    sep = "─" * 58
    return f"""\
 Low-Shelf EQ (1st/2nd-order)
{sep}
 Topology:  RBJ low-shelf — frequency-independent gain below fc
{sep}

               ┌──────────────────┐
   input  ─────┤   Low-Shelf       ├───┬──── output
               │   (boost/cut)     │   │
               └──────────────────┘   │
                                       │
   gain below fc = {gain_db:+.1f} dB        GND

 Filter Parameters
   fc   = {fc:.1f} Hz   ({_fmt_fc(fc)})
   Q    = {q:.2f}
   Gain = {gain_db:+.1f} dB   ({'boost' if gain_db >= 0 else 'cut'} below fc)

 Notes
   • Above fc: flat response (0 dB)
   • Below fc: gain = gain_db
   • 1st-order = 6 dB/oct slope, 2nd-order = 12 dB/oct slope
   • For sub-bass boost: fc = 80–200 Hz, Q = 0.7
"""


def _schematic_lowpass(fc: float, q: float, order: int, r_load: float) -> str:
    L = _lr2_L(fc, r_load) if order == 2 else 0
    C = _lr2_C(fc, r_load) if order == 2 else 0
    sep = "─" * 58
    order_str = {1: "1st-order (6 dB/oct)", 2: "2nd-order LR2 (12 dB/oct)",
                 3: "3rd-order (18 dB/oct)", 4: "4th-order LR4 (24 dB/oct)"}.get(order, f"{order}th-order")
    return f"""\
 Low-Pass Filter ({order_str})
{sep}

               ┌─────┐
   input  ─────┤  L  ├───┬──────── LP output (to woofer)
               └─────┘   │
                         ┌┴┐
                         │C│  C ─┬── GND
                         └┬┘     │
                          │     ─┘
                          └──────── GND

 Parameters
   fc = {fc:.1f} Hz   ({_fmt_fc(fc)})
   Q  = {q:.2f}
   R  = {r_load:.1f} Ω

{' Calculated Values (2nd-order LR2)' if order == 2 else ''}
{'   L ≈ ' + _format_L(L) + '   C ≈ ' + _format_C(C) if order == 2 else ''}
"""


def _schematic_highpass(fc: float, q: float, order: int, r_load: float) -> str:
    L = _lr2_L(fc, r_load) if order == 2 else 0
    C = _lr2_C(fc, r_load) if order == 2 else 0
    sep = "─" * 58
    order_str = {1: "1st-order (6 dB/oct)", 2: "2nd-order LR2 (12 dB/oct)",
                 3: "3rd-order (18 dB/oct)", 4: "4th-order LR4 (24 dB/oct)"}.get(order, f"{order}th-order")
    return f"""\
 High-Pass Filter ({order_str})
{sep}

               ┌─────┐
   input  ─────┤  C  ├───┬──────── HP output
               └─────┘   │
                         ┌┴┐
                         │L│  L ─┬── GND
                         └┬┘     │
                          │     ─┘
                          └──────── GND

 Parameters
   fc = {fc:.1f} Hz   ({_fmt_fc(fc)})
   Q  = {q:.2f}
   R  = {r_load:.1f} Ω

{' Calculated Values (2nd-order LR2)' if order == 2 else ''}
{'   L ≈ ' + _format_L(L) + '   C ≈ ' + _format_C(C) if order == 2 else ''}
"""


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_filter_schematic(
    bands: Union[list[FilterBand], "PassiveFilter"],
    r_load: float = 4.0,
) -> str:
    """
    Generate an ASCII schematic for a list of filter bands or a PassiveFilter.

    Handles single-band cases (Le Cléac'h, EQ, shelves, LP, HP, BP) and
    multi-band cases (2-way crossover = LP + HP at same fc,
    3-way crossover = LP + BP + HP).

    Parameters
    ----------
    bands : list[FilterBand] or PassiveFilter
        Filter bands (typically from a preset or user YAML config),
        or a PassiveFilter instance for electrical network simulation.
    r_load : float
        Load resistance in ohms (default 4.0 Ω).  Only used when
        ``bands`` is a PassiveFilter instance.

    Returns
    -------
    str
        Formatted ASCII schematic.
    """
    if isinstance(bands, PassiveFilter):
        return bands.schematic(r_load=r_load)

    enabled = [b for b in bands if b.enabled]
    if not enabled:
        return "No filter bands enabled."

    # ── Single band ───────────────────────────────────────────────────────────
    if len(enabled) == 1:
        b = enabled[0]
        return _schematic_for_band(b)

    # ── 2-way crossover: lowpass + highpass at same frequency ─────────────────
    types = {b.type for b in enabled}
    freqs = {b.frequency for b in enabled}
    if types == {"lowpass", "highpass"} and len(freqs) == 1:
        fc = enabled[0].frequency
        r_load = 8.0
        L_lp = _lr2_L(fc, r_load)
        C_lp = _lr2_C(fc, r_load)
        L_hp = L_lp
        C_hp = C_lp
        sep = "─" * 58
        return f"""\
 LR2 2-Way Crossover — {fc:.0f} Hz  (Linkwitz-Riley 2nd-order / 12 dB/oct)
{sep}

 ── LOW-PASS (Woofer) ──────────────────────────────────────
               ┌─────┐
   input  ─────┤  L  ├───┬──────── LP output (to woofer)
               └─────┘   │
                         ┌┴┐
                         │C│  C ─┬── GND
                         └┬┘     │
                          │     ─┘
                          └──────── GND

 ── HIGH-PASS (Tweeter) ────────────────────────────────────
               ┌─────┐
   input  ─────┤  C  ├───┬──────── HP output (to tweeter)
               └─────┘   │
                         ┌┴┐
                         │L│  L ─┬── GND
                         └┬┘     │
                          │     ─┘
                          └──────── GND

 Component Formulas (LR2 Butterworth, Q = 1/√2 ≈ 0.707)
   L = R / (√2 · π · fc)          C = 1 / (√2 · π · fc · R)

 Parameters
   R  = {r_load:.1f} Ω
   fc = {fc:.1f} Hz   ({_fmt_fc(fc)})

 Calculated Values (per section)
   L ≈ {_format_L(L_lp)}   ({L_lp:.4e} H)
   C ≈ {_format_C(C_lp)}  ({C_lp:.4e} F)

 Notes
   • Both outputs are in-phase at fc (acoustic alignment)
   • Phase difference = 180° — compensated by tweeter polarity flip
"""

    # ── 3-way crossover ────────────────────────────────────────────────────────
    if types == {"lowpass", "highpass", "bandpass"} and len(freqs) >= 2:
        lp_band = next(b for b in enabled if b.type == "lowpass")
        hp_band = next(b for b in enabled if b.type == "highpass")
        bp_band = next(b for b in enabled if b.type == "bandpass")
        fc1 = lp_band.frequency   # woofer/mid crossover
        fc2 = hp_band.frequency   # mid/tweeter crossover
        r_load = 8.0
        sep = "─" * 58
        return f"""\
LR2 3-Way Crossover  (Linkwitz-Riley 2nd-order / 12 dB/oct)
{sep}

 ── LOW-PASS (Woofer)  fc = {fc1:.0f} Hz ─────────────────────────────────
   LP:  L = R/(√2·π·fc)  C = 1/(√2·π·fc·R)

 ── BAND-PASS (Midrange)  fc1 = {fc1:.0f} Hz, fc2 = {fc2:.0f} Hz ─────────
   BP = HP(fc1) → LP(fc2) cascaded

 ── HIGH-PASS (Tweeter)  fc = {fc2:.0f} Hz ────────────────────────────────

 Parameters
   R  = {r_load:.1f} Ω
   fc1 = {fc1:.0f} Hz   ({_fmt_fc(fc1)})
   fc2 = {fc2:.0f} Hz   ({_fmt_fc(fc2)})

 See LR2 3-way schematic (3way_xover preset) for component-level diagram.
"""

    # ── Generic multi-band: show each band ────────────────────────────────────
    lines = ["Filter Network — Multiple Bands", "─" * 58]
    for i, b in enumerate(enabled, 1):
        lines.append(f"Band {i}: {b.type}  fc={b.frequency:.1f}Hz  Q={b.q:.2f}  gain={b.gain_db:+.1f}dB")
        lines.append(_schematic_for_band(b))
        lines.append("")
    return "\n".join(lines)


def _schematic_for_band(b: FilterBand) -> str:
    """Return the schematic for a single FilterBand."""
    t = b.type
    fc = b.frequency
    q = b.q
    gain_db = b.gain_db
    order = b.order
    r_load = 8.0

    if t == "le_cleach":
        return _schematic_le_cleach(fc=fc, q=q, r_load=r_load)
    elif t == "lowpass":
        return _schematic_lowpass(fc=fc, q=q, order=order, r_load=r_load)
    elif t == "highpass":
        return _schematic_highpass(fc=fc, q=q, order=order, r_load=r_load)
    elif t == "bandpass":
        sep = "─" * 58
        return f"""\
Band-Pass Filter
{sep}
   input ──┤ HP section ├───┤ LP section ├───┬── BP output
            (HP fc={fc:.0f}Hz)         (LP fc={fc:.0f}Hz)
  Q = {q:.2f}, BW = {fc / q:.1f} Hz
"""
    elif t == "peaking_eq":
        return _schematic_peaking_eq(fc=fc, q=q, gain_db=gain_db)
    elif t == "highshelf":
        return _schematic_highshelf(fc=fc, q=q, gain_db=gain_db)
    elif t == "lowshelf":
        return _schematic_lowshelf(fc=fc, q=q, gain_db=gain_db)
    else:
        return f"Unknown filter type: {t!r}"


# Named presets available via ?preset=<name>
PRESETS: dict[str, dict] = {
    "le_cleach": {
        "filter_type": "le_cleach",
        "description": "Le Cléac'h 2nd-order high-pass (André Le Cléac'h)",
        "default_fc": 80.0,
        "default_q": 0.7,
        "default_r_load": 8.0,
    },
    "2way_xover": {
        "filter_type": "lr2_crossover_2way",
        "description": "Linkwitz-Riley 2nd-order 2-way crossover",
        "default_fc": 3000.0,
        "default_q": 0.707,
        "default_r_load": 8.0,
    },
    "3way_xover": {
        "filter_type": "lr2_crossover_3way",
        "description": "Linkwitz-Riley 2nd-order 3-way crossover",
        "default_fc1": 400.0,
        "default_fc2": 4000.0,
        "default_q": 0.707,
        "default_r_load": 8.0,
    },
    "peaking_eq": {
        "filter_type": "peaking_eq",
        "description": "Peaking / parametric EQ (boost or cut)",
        "default_fc": 2500.0,
        "default_q": 1.4,
        "default_gain_db": 3.0,
    },
    "highshelf": {
        "filter_type": "highshelf",
        "description": "High-shelf EQ (boost/cut above fc)",
        "default_fc": 4000.0,
        "default_q": 0.707,
        "default_gain_db": -3.0,
    },
    "lowshelf": {
        "filter_type": "lowshelf",
        "description": "Low-shelf EQ (boost/cut below fc)",
        "default_fc": 200.0,
        "default_q": 0.707,
        "default_gain_db": 3.0,
    },
}


def generate_schematic(
    preset: Optional[str] = None,
    *,
    type: Optional[str] = None,
    fc: Optional[float] = None,
    fc1: Optional[float] = None,
    fc2: Optional[float] = None,
    q: Optional[float] = None,
    gain_db: Optional[float] = None,
    order: int = 2,
    r_load: float = 8.0,
) -> str:
    """
    Generate an ASCII schematic for the specified filter.

    Parameters
    ----------
    preset : str, optional
        Named preset. One of: le_cleach, 2way_xover, 3way_xover,
        peaking_eq, highshelf, lowshelf.
        When a preset is given the remaining parameters are used as
        overrides; defaults are taken from the preset definition.
    type : str, optional
        Filter topology (le_cleach, lr2_crossover_2way, peaking_eq,
        highshelf, lowshelf, lowpass, highpass, bandpass).
        Inferred from preset if not given.
    fc : float, optional
        Cutoff / centre frequency in Hz.
    fc1, fc2 : float, optional
        Band edges for 3-way crossover.
    q : float, optional
        Quality factor.
    gain_db : float, optional
        Boost/cut in dB (for shelving and peaking EQ).
    order : int, default 2
        Filter order (1st=6dB/oct … 4th=24dB/oct).
    r_load : float, default 8.0
        Load/source impedance in ohms.

    Returns
    -------
    str
        Formatted ASCII schematic.
    """
    # ── Resolve preset ────────────────────────────────────────────────────────
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset!r}. Available: {list(PRESETS)}")
        p = PRESETS[preset]
        type = p["filter_type"]
        fc = fc if fc is not None else p.get("default_fc", p.get("default_fc1", 1000.0))
        q = q if q is not None else p.get("default_q", 0.707)
        gain_db = gain_db if gain_db is not None else p.get("default_gain_db", 0.0)
        r_load = r_load
        fc1 = fc1 if fc1 is not None else p.get("default_fc1")
        fc2 = fc2 if fc2 is not None else p.get("default_fc2")
    else:
        if type is None:
            raise ValueError("Either preset= or type= must be provided")

    # ── Dispatch ───────────────────────────────────────────────────────────────
    if type == "le_cleach":
        return _schematic_le_cleach(fc=fc or 80.0, q=q or 0.7, r_load=r_load)

    elif type == "lr2_crossover_2way":
        return _schematic_lr2_xover_2way(fc=fc or 3000.0, r_load=r_load)

    elif type == "lr2_crossover_3way":
        return _schematic_lr2_xover_3way(
            fc1=fc1 or fc or 400.0,
            fc2=fc2 or (fc1 if fc1 else 4000.0),
            r_load=r_load,
        )

    elif type == "peaking_eq":
        return _schematic_peaking_eq(fc=fc or 1000.0, q=q or 1.0, gain_db=gain_db or 0.0)

    elif type == "highshelf":
        return _schematic_highshelf(fc=fc or 4000.0, q=q or 0.707, gain_db=gain_db or 0.0)

    elif type == "lowshelf":
        return _schematic_lowshelf(fc=fc or 200.0, q=q or 0.707, gain_db=gain_db or 0.0)

    elif type == "lowpass":
        return _schematic_lowpass(fc=fc or 1000.0, q=q or 0.707, order=order, r_load=r_load)

    elif type == "highpass":
        return _schematic_highpass(fc=fc or 1000.0, q=q or 0.707, order=order, r_load=r_load)

    elif type == "bandpass":
        sep = "─" * 58
        return f"""\
 Band-Pass Filter
{sep}
 Topology:  cascaded HP section → LP section
{sep}

   input ──┤ HP section ├───┤ LP section ├───┬── BP output
            (HP fc={fc or 1000:.0f}Hz)         (LP fc={fc or 1000:.0f}Hz)

 Parameters
   fc  = {fc or 1000.0:.1f} Hz   ({_fmt_fc(fc or 1000.0)})
   Q   = {q or 0.707:.2f}
   BW  = fc / Q = {(fc or 1000.0) / (q or 0.707):.1f} Hz

 Notes
   • Bandwidth = fc / Q
   • For narrow bandpass use Q > 5
   • Phase shift = sum of HP + LP phase contributions
"""

    else:
        raise ValueError(f"Unknown filter type: {type!r}")
