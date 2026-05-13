import matplotlib
import math

matplotlib.use("Agg")  # Headless mode: save files without spawning display windows
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING
from numpy.typing import NDArray
from pyhorn_core.solver.models import SimulationResult
from pyhorn_core.solver.spectrogram import compute_spectrogram, plot_spectrogram

if TYPE_CHECKING:
    from pyhorn_core.solver.wavefront import WavefrontGrid
    import matplotlib.pyplot as plt

# ─── Shared style ────────────────────────────────────────────────────────────
_COLORS = {
    "spl": "#2563eb",  # blue-600
    "impedance": "#dc2626",  # red-600
    "excursion": "#16a34a",  # green-600
    "reference": "#9ca3af",  # gray-400
    "target": "#f59e0b",  # amber-500
    "direct": "#0f766e",  # teal-700
    "horn": "#7c3aed",  # violet-600
    "group_delay": "#4b5563",  # gray-600
    "phase": "#0891b2",  # cyan-600
    "impedance_phase": "#a855f7",  # purple-500
    "efficiency": "#f97316",  # orange-500
    "driver_power": "#f97316",  # orange-500 (same as efficiency — same data family)
    "cone_acceleration": "#f97316",  # orange-500
}

_FREQ_TICKS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]


def _fmt_hz(val, pos):
    v = int(val)
    if v >= 1000:
        return f"{v // 1000}K"
    return str(v)


def _apply_freq_ticks(ax):
    ax.xaxis.set_major_locator(FixedLocator(_FREQ_TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_hz))
    ax.xaxis.set_minor_locator(FixedLocator([]))  # hide minor ticks for cleaner look


def _apply_style(
    ax,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    freq_axis: bool = False,
):
    ax.tick_params(labelsize=8, width=0.4, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.3)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.2)
    if freq_axis:
        _apply_freq_ticks(ax)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def plot_simulation_results(
    result: SimulationResult,
    output_path: Path,
    title: str = "Horn Acoustic Response",
    target_spl: Optional[float] = None,
    target_impedance: Optional[float] = None,
    target_excursion: Optional[float] = None,
    target_excursion_label: Optional[str] = None,
    output_mode: str = "combined",
    plot_phase: bool = True,
    plot_distortion: bool = True,
    polar_freq: Optional[float] = None,
    wavefront_grid: Optional["WavefrontGrid"] = None,
    wf_mesh_x: Optional[NDArray[np.float64]] = None,
    wf_mesh_y: Optional[NDArray[np.float64]] = None,
    wf_boundary_mask: Optional[NDArray[np.bool_]] = None,
    show_spectrogram: bool = False,
    spectrogram_window_ms: float = 50.0,
    spectrogram_overlap: float = 0.5,
    plot_spl_only: bool = False,
) -> None:
    """
    Generate and save a multi-panel figure with SPL, Impedance, Excursion, Phase,
    Acoustic Impedance, Efficiency, Second Tone Distortion, Direction Index,
    Spectrogram, and Polar Directivity.

    output_mode controls which SPL is shown as primary:
      - 'combined' (default): show total SPL as primary, horn and direct as dashed overlays
      - 'horn': show horn SPL as primary, total as dashed reference
      - 'element': show direct radiator SPL as primary, total as dashed reference

    plot_phase: when True (default), includes phase (degrees) + group delay (ms) panel
                and throat acoustic impedance panel.

    plot_distortion: when True (default), includes a second tone distortion panel
                (dB below fundamental: SPL(2f) - SPL(f)) for single-segment horns.

    polar_freq: when provided (Hz), adds a polar piston directivity panel at that frequency.

    wavefront_grid, wf_mesh_x, wf_mesh_y, wf_boundary_mask: when all four are provided,
                adds a second polar panel (last slot) derived from the 2-D wavefront
                pressure field at the given polar_freq.

    show_spectrogram: when True, adds a spectrogram (STFT) panel between the
                distortion panel and the polar panel, showing spectral intensity vs.
                frequency and time.

    spectrogram_window_ms: STFT window duration in milliseconds (default 50 ms).
                Smaller → better time resolution, worse frequency resolution.

    spectrogram_overlap: fraction of overlap between STFT windows (default 0.5 = 50%).

    plot_spl_only: when True, generates a single-panel SPL plot (no impedance,
                excursion, phase, or other panels). Useful for quick inspection.
    """
    # ── SPL-only single-panel shortcut ────────────────────────────────────────
    if plot_spl_only:
        fig, ax = plt.subplots(figsize=(10, 5))
        freqs = result.freqs
        if output_mode == "horn" and result.horn_spl is not None:
            ax.semilogx(freqs, result.horn_spl, color=_COLORS["horn"], linewidth=0.9, label="Horn")
            if result.spl is not None:
                ax.semilogx(freqs, result.spl, color=_COLORS["spl"], linewidth=0.5, linestyle="--", label="Total (ref)", alpha=0.7)
        elif output_mode == "element" and result.direct_spl is not None:
            ax.semilogx(freqs, result.direct_spl, color=_COLORS["direct"], linewidth=0.9, label="Direct radiator")
            if result.spl is not None:
                ax.semilogx(freqs, result.spl, color=_COLORS["spl"], linewidth=0.5, linestyle="--", label="Total (ref)", alpha=0.7)
        else:
            ax.semilogx(freqs, result.spl, color=_COLORS["spl"], linewidth=0.9, label="Total")
            if result.direct_spl is not None:
                ax.semilogx(freqs, result.direct_spl, color=_COLORS["direct"], linestyle="--", linewidth=0.7, label="Direct (cone)", alpha=0.7)
            if result.horn_spl is not None:
                ax.semilogx(freqs, result.horn_spl, color=_COLORS["horn"], linestyle="--", linewidth=0.7, label="Horn", alpha=0.3)
            if result.spl_power_based is not None:
                ax.semilogx(freqs, result.spl_power_based, color="#16a34a", linewidth=0.9, label="dB/W/m (calibrated)", alpha=0.85)
        if hasattr(result, "ib_spl") and result.ib_spl is not None:
            ax.semilogx(freqs, result.ib_spl, color=_COLORS["reference"], linestyle=":", linewidth=0.7, label="Infinite Baffle")
        if target_spl is not None:
            ax.axhline(target_spl, color=_COLORS["target"], linestyle="--", linewidth=0.7, label=f"Target ({target_spl:.1f} dB)")
        ax.legend(fontsize=8, framealpha=0.6, edgecolor="none")
        ymax = np.max(result.spl)
        ax.set_ylim(bottom=max(40, ymax - 50), top=ymax + 10)
        _apply_style(ax, xlabel="Frequency (Hz)", ylabel="SPL (dB @ 1W/1m)", freq_axis=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    _has_wf_polar = (
        wavefront_grid is not None
        and wf_mesh_x is not None
        and wf_mesh_y is not None
        and wf_boundary_mask is not None
        and polar_freq is not None
    )
    n_rows = 5 if plot_phase else 4
    _has_cone_velocity = result.cone_velocity is not None
    _has_cone_acceleration = result.cone_acceleration is not None
    _has_pv = (
        result.particle_velocity_throat is not None
        or result.particle_velocity_mouth is not None
        or result.particle_velocity_port is not None
    )
    if _has_cone_velocity:
        n_rows += 1
    if _has_cone_acceleration:
        n_rows += 1
    if _has_pv:
        n_rows += 1
    if result.efficiency_pct is not None:
        n_rows += 1
    if result.electrical_input_power is not None:
        n_rows += 1
    if result.second_tone_distortion is not None and plot_distortion:
        n_rows += 1
    if polar_freq is not None and result.off_axis_spl is not None:
        n_rows += 1
    if _has_wf_polar:
        n_rows += 1
    if show_spectrogram:
        n_rows += 1
    _has_di = result.direction_index is not None and result.off_axis_angles is not None
    if _has_di:
        n_rows += 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 10 + n_rows))
    fig.suptitle(title, fontsize=11, fontweight="medium", y=0.98)

    # Unpack all possible axes slots; unused ones stay None
    if n_rows == 4:
        ax1, ax2, ax3, ax4 = axes
        ax5 = ax6 = ax7 = ax8 = ax9 = ax10 = None
    elif n_rows == 5:
        ax1, ax2, ax3, ax4, ax5 = axes
        ax6 = ax7 = ax8 = ax9 = ax10 = None
    elif n_rows == 6:
        ax1, ax2, ax3, ax4, ax5, ax6 = axes
        ax7 = ax8 = ax9 = ax10 = None
    elif n_rows == 7:
        ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes
        ax8 = ax9 = ax10 = None
    elif n_rows == 8:
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes
        ax9 = ax10 = None
    elif n_rows == 9:
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes
        ax10 = None
    elif n_rows >= 10:
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10 = axes[:10]
    if n_rows >= 11:
        ax11 = axes[10]
    else:
        ax11 = None

    # ── Resolve which axes carry which panel ─────────────────────────────────
    # Layout (rows, top to bottom):
    #   0: SPL | 1: Impedance | 2: Excursion | 3: Phase+GD | 4: Throat Z
    #   5: Cone Velocity (if result.cone_velocity is not None)
    #   6: Cone Acceleration (if result.cone_acceleration is not None)
    #   Above optional panels (stacked from top): polar, wf_polar, spectrogram, DI, distortion, efficiency
    #
    # Cone Velocity slot: axes[5] if plot_phase else axes[4]
    # Cone Acceleration slot: axes[6] if plot_phase else axes[5]
    # Efficiency: always at top of optional stack = n_rows-1 minus panels above it
    # Throat Z: always axes[4] (if plot_phase)
    cone_vel_ax = axes[5] if (_has_cone_velocity and plot_phase) else (axes[4] if _has_cone_velocity else None)
    cone_acc_ax = axes[6] if (_has_cone_acceleration and plot_phase) else (axes[5] if _has_cone_acceleration else None)
    pv_ax = None
    if _has_pv:
        # Particle velocity panel slot: after cone_acc, before efficiency stack
        if plot_phase:
            pv_ax = axes[7] if n_rows >= 8 else (axes[6] if n_rows >= 7 else None)
        else:
            pv_ax = axes[6] if n_rows >= 7 else (axes[5] if n_rows >= 6 else None)

    _has_efficiency = result.efficiency_pct is not None
    _has_polar = polar_freq is not None and result.off_axis_spl is not None
    _has_distortion = (
        result.second_tone_distortion is not None
        and plot_distortion
        and n_rows >= (8 if _has_cone_velocity else 7)
    )
    # eff_base: base index for the efficiency panel, accounting for all stacked optional panels above it
    eff_base = n_rows - (4 + (_has_wf_polar or show_spectrogram or _has_di or _has_distortion or _has_efficiency))
    _eff_ax_pos = eff_base
    eff_ax = axes[eff_base] if _has_efficiency else None
    driver_power_ax = axes[eff_base + 1] if result.electrical_input_power is not None else None
    dist_ax = axes[eff_base + 2] if _has_distortion else None
    spec_ax = axes[eff_base + 3] if show_spectrogram else None
    di_ax = axes[eff_base + 4] if _has_di else None
    polar_ax = axes[n_rows - 1] if _has_polar else None

    if _has_wf_polar:
        # Two polar panels: piston at 2nd-to-last, wavefront at last
        # Replace the 2nd-to-last and last regular axes with polar projection axes
        axes = list(axes)
        axes[n_rows - 2] = fig.add_subplot(n_rows, 1, n_rows - 1, projection="polar")
        axes[n_rows - 1] = fig.add_subplot(n_rows, 1, n_rows, projection="polar")
        polar_ax = axes[n_rows - 2]
        wf_polar_ax: Optional["plt.Axes"] = axes[n_rows - 1]
    elif _has_polar:
        # Replace the last regular axis with polar projection
        axes = list(axes)
        axes[n_rows - 1] = fig.add_subplot(n_rows, 1, n_rows, projection="polar")
        polar_ax = axes[n_rows - 1]
        wf_polar_ax = None
    else:
        polar_ax = None
        wf_polar_ax = None

    freqs = result.freqs

    # 1. SPL Plot — select primary SPL based on output_mode
    if output_mode == "horn" and result.horn_spl is not None:
        primary_spl = result.horn_spl
        primary_label = "Horn"
        ax1.semilogx(freqs, primary_spl, color=_COLORS["horn"], linewidth=0.8, label=primary_label, alpha=0.3)
        if result.spl is not None:
            ax1.semilogx(freqs, result.spl, color=_COLORS["spl"], linewidth=0.5, linestyle="--", label="Total (ref)", alpha=0.7)
        if result.direct_spl is not None:
            ax1.semilogx(freqs, result.direct_spl, color=_COLORS["direct"], linewidth=0.5, linestyle=":", label="Element (ref)", alpha=0.6)
    elif output_mode == "element" and result.direct_spl is not None:
        primary_spl = result.direct_spl
        primary_label = "Direct radiator"
        ax1.semilogx(freqs, primary_spl, color=_COLORS["direct"], linewidth=0.8, label=primary_label)
        if result.spl is not None:
            ax1.semilogx(freqs, result.spl, color=_COLORS["spl"], linewidth=0.5, linestyle="--", label="Total (ref)", alpha=0.7)
        if result.horn_spl is not None:
            ax1.semilogx(freqs, result.horn_spl, color=_COLORS["horn"], linewidth=0.5, linestyle=":", label="Horn (ref)", alpha=0.3)
    else:
        # combined (default) — show total as primary, components as dashed overlays
        primary_spl = result.spl
        ax1.semilogx(freqs, result.spl, color=_COLORS["spl"], linewidth=0.8, label="Total")

        if result.direct_spl is not None:
            ax1.semilogx(
                freqs,
                result.direct_spl,
                color=_COLORS["direct"],
                linestyle="--",
                linewidth=0.7,
                label="Direct (cone)",
            )
        if result.horn_spl is not None:
            ax1.semilogx(
                freqs,
                result.horn_spl,
                color=_COLORS["horn"],
                linestyle="--",
                linewidth=0.7,
                label="Horn",
                alpha=0.3,
            )
        if result.spl_power_based is not None:
            ax1.semilogx(
                freqs,
                result.spl_power_based,
                color="#16a34a",
                linewidth=0.8,
                label="dB/W/m (calibrated)",
                alpha=0.85,
            )

    if hasattr(result, "ib_spl") and result.ib_spl is not None:
        ax1.semilogx(
            freqs,
            result.ib_spl,
            color=_COLORS["reference"],
            linestyle="--",
            linewidth=0.6,
            label="Infinite Baffle",
        )

    if target_spl is not None:
        ax1.axhline(
            target_spl,
            color=_COLORS["target"],
            linestyle="--",
            linewidth=0.5,
            label=f"Target ({target_spl:.1f} dB)",
        )

    if ax1.lines:
        ax1.legend(fontsize=7, framealpha=0.6, edgecolor="none")

    # Autoscale y-lim but keep sensible bounds
    ymax = max(np.max(primary_spl), target_spl + 5 if target_spl else 0)
    ax1.set_ylim(bottom=max(40, ymax - 50), top=ymax + 10)
    _apply_style(
        ax1, xlabel="Frequency (Hz)", ylabel="SPL (dB @ 1W/1m)", freq_axis=True
    )

    # 2. Electrical Impedance Plot (magnitude + phase as twin axis)
    ax2.semilogx(
        freqs, np.abs(result.impedance), color=_COLORS["impedance"], linewidth=0.8
    )
    if target_impedance is not None:
        ax2.axhline(
            target_impedance,
            color=_COLORS["target"],
            linestyle="--",
            linewidth=0.5,
            label=f"Target ({target_impedance:.2f} ohm)",
        )
        ax2.legend(fontsize=7, framealpha=0.6, edgecolor="none")
    ax2.set_xlim(min(freqs), max(freqs))
    _apply_style(
        ax2, xlabel="Frequency (Hz)", ylabel="Impedance (\u03a9)", freq_axis=True
    )
    # Electrical impedance phase on secondary y-axis (Hornresp page 116)
    if result.impedance_phase_deg is not None:
        ax2_z = ax2.twinx()
        imp_phase = np.clip(result.impedance_phase_deg, -120.0, 120.0)
        ax2_z.semilogx(
            freqs,
            imp_phase,
            color=_COLORS["impedance_phase"],
            linewidth=0.8,
            linestyle=":",
            label="Z phase",
        )
        ax2_z.set_ylabel("Impedance Phase (°)", fontsize=8, color=_COLORS["impedance_phase"])
        ax2_z.tick_params(axis="y", labelcolor=_COLORS["impedance_phase"])
        ax2_z.set_ylim(-120, 120)
        ax2_z.spines["right"].set_visible(True)
        # Legend for both lines on ax2
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines2_z, labels2_z = ax2_z.get_legend_handles_labels()
        if lines2 or lines2_z:
            ax2.legend(lines2 + lines2_z, labels2 + labels2_z, fontsize=7, framealpha=0.6, edgecolor="none")
    elif target_impedance is not None:
        ax2.legend(fontsize=7, framealpha=0.6, edgecolor="none")

    # 3. Excursion Plot
    ax3.semilogx(freqs, result.excursion, color=_COLORS["excursion"], linewidth=0.8)
    if target_excursion is not None:
        ax3.axhline(
            target_excursion,
            color=_COLORS["target"],
            linestyle="--",
            linewidth=0.5,
            label=target_excursion_label or f"Target ({target_excursion:.2f} mm)",
        )
        ax3.legend(fontsize=7, framealpha=0.6, edgecolor="none")
    ax3.set_xlim(min(freqs), max(freqs))
    _apply_style(
        ax3, xlabel="Frequency (Hz)", ylabel="Excursion (mm peak)", freq_axis=True
    )

    # 4. Phase (degrees) + Group Delay (ms) — dual-y-axis panel
    phase_deg = np.degrees(result.phase) if result.phase is not None else None
    if plot_phase and phase_deg is not None:
        color_phase = _COLORS["phase"]
        ax4.semilogx(freqs, phase_deg, color=color_phase, linewidth=0.8, label="Phase")
        ax4.set_xlim(min(freqs), max(freqs))
        phase_finite = phase_deg[np.isfinite(phase_deg)]
        if phase_finite.size:
            ax4.set_ylim(
                max(-185.0, np.min(phase_finite) - 15.0),
                min(185.0, np.max(phase_finite) + 15.0),
            )
        ax4.set_ylabel("Phase (°)", fontsize=8, color=color_phase)
        ax4.tick_params(axis="y", labelcolor=color_phase)

        # Group delay on secondary y-axis
        if result.group_delay is not None:
            ax4_gd = ax4.twinx()
            gd_display = np.clip(result.group_delay, -50.0, 50.0)
            ax4_gd.semilogx(
                freqs,
                gd_display,
                color=_COLORS["group_delay"],
                linewidth=0.8,
                linestyle="--",
                label="Group Delay",
            )
            finite_gd = gd_display[np.isfinite(gd_display)]
            if finite_gd.size:
                ax4_gd.set_ylim(
                    max(-50.0, np.min(finite_gd) - 2.0),
                    min(50.0, np.max(finite_gd) + 2.0),
                )
            ax4_gd.set_ylabel("Group Delay (ms)", fontsize=8, color=_COLORS["group_delay"])
            ax4_gd.tick_params(axis="y", labelcolor=_COLORS["group_delay"])
            ax4_gd.spines["right"].set_visible(True)

        _apply_style(ax4, xlabel="Frequency (Hz)", freq_axis=True)
        # Legend for phase line
        lines4, labels4 = ax4.get_legend_handles_labels()
        if lines4:
            ax4.legend(lines4, labels4, fontsize=7, framealpha=0.6, edgecolor="none")
    else:
        # Legacy: Group Delay only (plot_phase=False or no phase data)
        if result.group_delay is not None:
            gd_display = np.clip(result.group_delay, -50.0, 50.0)
            ax4.semilogx(
                freqs,
                gd_display,
                color=_COLORS["group_delay"],
                linewidth=0.8,
            )
            finite_gd = gd_display[np.isfinite(gd_display)]
            if finite_gd.size:
                ax4.set_ylim(
                    max(-50.0, np.min(finite_gd) - 2.0),
                    min(50.0, np.max(finite_gd) + 2.0),
                )
        ax4.set_xlim(min(freqs), max(freqs))
        _apply_style(
            ax4, xlabel="Frequency (Hz)", ylabel="Group Delay (ms)", freq_axis=True
        )

    # 5. Throat Acoustic Impedance (real + imaginary)
    if plot_phase and result.throat_impedance is not None:
        z_throat = result.throat_impedance
        ax5.semilogx(
            freqs,
            z_throat.real,
            color=_COLORS["impedance"],
            linewidth=0.8,
            label="Real (R)",
        )
        ax5.semilogx(
            freqs,
            z_throat.imag,
            color="#7c3aed",  # violet-600
            linewidth=0.8,
            linestyle="--",
            label="Imag (X)",
        )
        ax5.set_xlim(min(freqs), max(freqs))
        _apply_style(
            ax5, xlabel="Frequency (Hz)", ylabel="Acoustic Impedance (acoustic \u03a9)", freq_axis=True
        )
        if ax5.lines:
            ax5.legend(fontsize=7, framealpha=0.6, edgecolor="none")

    # 5b. Cone Velocity panel — peak cone speed in m/s (Hornresp page 126)
    if cone_vel_ax is not None and result.cone_velocity is not None:
        cone_vel_ax.semilogx(
            freqs,
            result.cone_velocity,
            color="#0891b2",  # teal-600
            linewidth=0.8,
        )
        cone_vel_ax.set_xlim(min(freqs), max(freqs))
        cv_max = float(np.max(result.cone_velocity[np.isfinite(result.cone_velocity)])) if result.cone_velocity.size else 1.0
        cone_vel_ax.set_ylim(bottom=0.0, top=cv_max * 1.2 + 0.1)
        _apply_style(
            cone_vel_ax,
            xlabel="Frequency (Hz)",
            ylabel="Cone Velocity (m/s)",
            freq_axis=True,
        )

    # 5c. Cone Acceleration panel — peak cone acceleration in m/s²
    # a_driver = |j·ω·v_driver| = ω × |v_driver| = ω² × |x_driver|
    if cone_acc_ax is not None and result.cone_acceleration is not None:
        cone_acc_ax.semilogx(
            freqs,
            result.cone_acceleration,
            color="#f97316",  # orange-500
            linewidth=0.8,
        )
        cone_acc_ax.set_xlim(min(freqs), max(freqs))
        ca_max = float(np.max(result.cone_acceleration[np.isfinite(result.cone_acceleration)])) if result.cone_acceleration.size else 1.0
        cone_acc_ax.set_ylim(bottom=0.0, top=ca_max * 1.2 + 0.1)
        cone_acc_ax.set_yscale("log")
        _apply_style(
            cone_acc_ax,
            xlabel="Frequency (Hz)",
            ylabel="Cone Acceleration (m/s²)",
            freq_axis=True,
        )

    # 5d. Particle Velocity panel — peak particle velocity in m/s (Hornresp page 106)
    # Shows throat, mouth, and port overlaid
    if pv_ax is not None and _has_pv:
        pv_configs = [
            (result.particle_velocity_throat, "Throat", "#dc2626"),  # red-600
            (result.particle_velocity_mouth, "Mouth", "#2563eb"),  # blue-600
            (result.particle_velocity_port, "Port", "#16a34a"),    # green-600
        ]
        any_positive = False
        for pv_arr, label, color in pv_configs:
            if pv_arr is not None:
                pv_ax.semilogx(freqs, pv_arr, color=color, linewidth=0.8, label=label)
                if np.any(pv_arr > 0):
                    any_positive = True
        pv_ax.set_xlim(min(freqs), max(freqs))
        if any_positive:
            pv_ax.set_yscale("log")
        _apply_style(
            pv_ax,
            xlabel="Frequency (Hz)",
            ylabel="Particle Velocity (m/s)",
            freq_axis=True,
        )
        pv_ax.legend(fontsize=7, loc="upper right")

    # 6. System Efficiency Panel (electrical → acoustic power conversion)
    if eff_ax is not None:
        eff_display = np.clip(result.efficiency_pct, 0.0, None)
        eff_ax.semilogx(
            freqs,
            eff_display,
            color=_COLORS["efficiency"],
            linewidth=0.8,
        )
        eff_ax.set_xlim(min(freqs), max(freqs))
        # Autoscale y: start at 0, top = max + 20% headroom, capped at 100
        eff_max = float(np.max(eff_display[np.isfinite(eff_display)])) if eff_display.size else 20.0
        eff_ax.set_ylim(bottom=0.0, top=min(100.0, eff_max * 1.2 + 0.5))
        _apply_style(
            eff_ax, xlabel="Frequency (Hz)", ylabel="Efficiency (%)", freq_axis=True
        )

    # 6b. Driver Power Panel (electrical input power in watts)
    # P_elec = |V|² · Re(Z_in) / |Z_e|² — real electrical power delivered to the voice coil
    # Reference: Hornresp page 105 (Driver power)
    if driver_power_ax is not None:
        p_elec = result.electrical_input_power
        valid = np.isfinite(p_elec) & (p_elec >= 0)
        driver_power_ax.semilogx(
            freqs[valid],
            p_elec[valid],
            color=_COLORS["driver_power"],
            linewidth=0.8,
        )
        driver_power_ax.set_xlim(min(freqs), max(freqs))
        # Autoscale y with a small floor
        p_finite = p_elec[valid]
        if p_finite.size:
            p_min = max(float(np.min(p_finite)), 1e-6)
            p_max = float(np.max(p_finite))
            driver_power_ax.set_ylim(p_min * 0.5, p_max * 1.5)
            driver_power_ax.set_yscale("log")
        _apply_style(
            driver_power_ax,
            xlabel="Frequency (Hz)",
            ylabel="Driver Power (W)",
            freq_axis=True,
        )

    # 7. Second Tone Distortion Panel (SPL(2f) - SPL(f), dB below fundamental)
    # Only shown for single-segment horns (result.second_tone_distortion is not None)
    if dist_ax is not None:
        dist_data = result.second_tone_distortion
        valid = np.isfinite(dist_data)
        dist_ax.semilogx(
            freqs[valid],
            dist_data[valid],
            color="#dc2626",  # red-600
            linewidth=0.8,
        )
        dist_ax.set_xlim(min(freqs), max(freqs))
        # Autoscale: most distortion is -20 to -80 dB; use symmetric around 0
        dist_finite = dist_data[valid]
        if dist_finite.size:
            d_min = np.min(dist_finite)
            d_max = np.max(dist_finite)
            d_range = max(abs(d_min), abs(d_max))
            dist_ax.set_ylim(-max(10.0, d_range * 1.2), 5.0)
        dist_ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
        dist_ax.set_ylabel("2nd Tone (dB rel)", fontsize=8)
        _apply_style(
            dist_ax, xlabel="Frequency (Hz)", ylabel="2nd Tone (dB rel)", freq_axis=True
        )

    # 7b. Spectrogram Panel (STFT of the impulse response)
    # Only shown when show_spectrogram=True and pressure data is available
    if show_spectrogram and result.pressure is not None and spec_ax is not None:
        try:
            time_ms, freq_bins, spec_db = compute_spectrogram(
                result.freqs,
                result.pressure,
                window_ms=spectrogram_window_ms,
                overlap=spectrogram_overlap,
            )
            plot_spectrogram(
                time_ms,
                freq_bins,
                spec_db,
                ax=spec_ax,
                f_min=float(min(result.freqs)),
                f_max=float(max(result.freqs)),
            )
        except Exception:
            # Spectrogram is non-critical; skip gracefully
            spec_ax.text(
                0.5, 0.5,
                "Spectrogram unavailable",
                ha="center", va="center",
                transform=spec_ax.transAxes,
                fontsize=8,
                color="gray",
            )
            spec_ax.set_axis_off()
    elif show_spectrogram and spec_ax is not None:
        spec_ax.text(
            0.5, 0.5,
            "Pressure data required for spectrogram",
            ha="center", va="center",
            transform=spec_ax.transAxes,
            fontsize=8,
            color="gray",
        )
        spec_ax.set_axis_off()

    # 8. Direction Index Panel — DI(f, θ) = 10×log10(direction_factor) at off-axis angles
    if di_ax is not None:
        di = result.direction_index
        angles = result.off_axis_angles
        # Plot 3 representative angles: 30°, 45°, 60° (or whatever angles are available)
        # Choose angles closest to 30, 45, 60
        target_angles = [30.0, 45.0, 60.0]
        di_colors = ["#f97316", "#2563eb", "#dc2626"]  # orange, blue, red

        plotted = []
        for target, color in zip(target_angles, di_colors):
            idx = int(np.argmin(np.abs(np.asarray(angles) - target)))
            angle_val = angles[idx]
            di_curve = di[:, idx]
            valid = np.isfinite(di_curve)
            di_ax.semilogx(
                freqs[valid],
                di_curve[valid],
                color=color,
                linewidth=0.8,
                label=f"{int(angle_val)}°",
            )
            plotted.append((angle_val, color))

        # 0 dB reference line (on-axis)
        di_ax.axhline(0, color="#e3b341", linewidth=0.5, linestyle="--", alpha=0.6)
        # Grid at -5, -10, -15 dB
        for db_ref in [-5.0, -10.0, -15.0]:
            di_ax.axhline(db_ref, color="#30363d", linewidth=0.3, linestyle="-")

        di_ax.set_xlim(min(freqs), max(freqs))
        di_ax.set_ylim(-20.0, 2.0)
        di_ax.set_ylabel("DI (dB)", fontsize=8)
        di_ax.set_xlabel("Frequency (Hz)", fontsize=8)
        if plotted:
            di_ax.legend(
                [f"{int(a)}°" for a, _ in plotted],
                fontsize=7,
                framealpha=0.6,
                edgecolor="none",
                loc="lower left",
                ncol=3,
            )
        _apply_style(
            di_ax, xlabel="Frequency (Hz)", ylabel="DI (dB)", freq_axis=True
        )

    # 9. Polar Directivity Panel
    if polar_ax is not None and result.off_axis_spl is not None and result.off_axis_angles is not None:
        freq_idx = int(np.argmin(np.abs(result.freqs - polar_freq)))
        chosen_freq = result.freqs[freq_idx]
        plot_polar_response(
            result.off_axis_spl[freq_idx],
            result.off_axis_angles,
            chosen_freq,
            polar_ax,
        )

    # 10. Wavefront-derived Polar Directivity Panel
    if wf_polar_ax is not None:
        from pyhorn_core.solver.wavefront import plot_wavefront_polar
        plot_wavefront_polar(
            pressure_field=wavefront_grid.pressure_field,
            mesh_x=wf_mesh_x,
            mesh_y=wf_mesh_y,
            boundary_mask=wf_boundary_mask,
            frequency=polar_freq,
            ax=wf_polar_ax,
        )

    plt.tight_layout(h_pad=1.5)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_polar_response(
    off_axis_spl: np.ndarray,
    off_axis_angles: np.ndarray,
    freq: float,
    ax: plt.Axes,
    mouth_area: Optional[float] = None,
) -> None:
    """
    Render a polar directivity plot (piston model) at a single frequency.

    Shows Levine/Inglis piston-in-baffle directivity: SPL dB relative to on-axis
    at 0° (top), 90° (right), full 360° symmetry.

    Parameters
    ----------
    off_axis_spl   : np.ndarray of shape (n_angles,) — dB relative to on-axis at one freq
    off_axis_angles: np.ndarray of shape (n_angles,) — angles in degrees [0,15,...,90]
    freq           : frequency in Hz to plot
    ax             : matplotlib Axes with projection='polar'
    mouth_area     : optional, used for title annotation
    """
    _ = mouth_area  # reserved for future annotation

    # Build full 360° symmetric piston pattern from half-plane data
    angles_deg = off_axis_angles  # e.g. [0, 15, 30, 45, 60, 75, 90]
    # Mirror: 0..90 → 90..0 → 180..0 → 180..90
    all_angles = np.concatenate([
        angles_deg,
        180.0 - angles_deg[1:][::-1],
        180.0 + angles_deg[1:],
        360.0 - angles_deg[1:][::-1],
    ])
    all_db = np.concatenate([
        off_axis_spl,
        off_axis_spl[1:][::-1],    # mirror 90→0
        off_axis_spl[1:],          # 180+0..90
        off_axis_spl[1:][::-1],    # mirror 270→360
    ])

    # Clip to physically plausible range
    all_db = np.clip(all_db, -40.0, 0.0)

    # Convert to radians for polar plot
    theta = np.deg2rad(all_angles)

    # r = -dB_rel so that 0 dB (on-axis) is at centre, -30 dB at edge
    r = -all_db  # 0→30 range

    ax.fill(theta, r, color="#2563eb", alpha=0.15, zorder=1)
    ax.plot(theta, r, color="#2563eb", linewidth=1.0, zorder=2)

    # Reference circles at -10, -20, -30 dB
    for db_ref in [-10.0, -20.0, -30.0]:
        ax.plot(theta, [-db_ref] * len(theta), color="#9ca3af",
                linewidth=0.3, linestyle="--", alpha=0.6, zorder=0)
        ax.text(np.deg2rad(90), -db_ref + 0.3, f"{db_ref:.0f}",
                fontsize=5, color="#9ca3af", ha="left", va="center", zorder=3)

    # On-axis marker
    ax.plot(0, 0, "o", color="#dc2626", markersize=3, zorder=4)

    ax.set_theta_zero_location("N")    # 0° at top
    ax.set_theta_direction(-1)        # clockwise (standard acoustic polar)
    ax.set_thetamin(0)
    ax.set_thetamax(270)
    ax.set_rlim(0, 30)
    ax.set_rgrids([0, 5, 10, 15, 20, 25, 30],
                  ["0", "−5", "−10", "−15", "−20", "−25", "−30"],
                  fontsize=5, color="#6b7280")
    ax.tick_params(labelsize=5, pad=1)
    ax.set_title(f"Polar Response @ {freq:.0f} Hz\n(SPL dB rel to on-axis)",
                 fontsize=6, pad=8, fontweight="medium")


def plot_horn_3d(
    segments: List[Tuple[float, ...]],
    output_path: Path,
    width: Optional[float] = None,
    width_profile: Optional[List[float]] = None,
) -> None:
    """
    Generate an unwrapped 3D wireframe plot of the horn geometry.
    If width is provided, assumes a rectangular horn with constant width.
    Otherwise, assumes a square horn.
    """
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, projection="3d")

    if not segments:
        return

    z_coords = [0.0]
    areas = [segments[0][1]]
    for seg in segments:
        L = seg[0]
        S = seg[1]
        z_coords.append(z_coords[-1] + L)
        areas.append(S)

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for i in range(len(z_coords) - 1):
        z1, z2 = z_coords[i], z_coords[i + 1]
        a1, a2 = areas[i], areas[i + 1]

        if width_profile and len(width_profile) == len(areas):
            w1 = width_profile[i]
            w2 = width_profile[i + 1]
            h1 = a1 / w1 if w1 > 0 else 0.0
            h2 = a2 / w2 if w2 > 0 else 0.0
        elif width:
            w1 = w2 = width
            h1 = a1 / width
            h2 = a2 / width
        else:
            w1 = h1 = np.sqrt(a1)
            w2 = h2 = np.sqrt(a2)

        x1 = [-w1 / 2, w1 / 2, w1 / 2, -w1 / 2, -w1 / 2]
        y1 = [-h1 / 2, -h1 / 2, h1 / 2, h1 / 2, -h1 / 2]
        x2 = [-w2 / 2, w2 / 2, w2 / 2, -w2 / 2, -w2 / 2]
        y2 = [-h2 / 2, -h2 / 2, h2 / 2, h2 / 2, -h2 / 2]

        verts = [
            [
                (z1, x1[0], y1[0]),
                (z1, x1[1], y1[1]),
                (z2, x2[1], y2[1]),
                (z2, x2[0], y2[0]),
            ],
            [
                (z1, x1[1], y1[1]),
                (z1, x1[2], y1[2]),
                (z2, x2[2], y2[2]),
                (z2, x2[1], y2[1]),
            ],
            [
                (z1, x1[2], y1[2]),
                (z1, x1[3], y1[3]),
                (z2, x2[3], y2[3]),
                (z2, x2[2], y2[2]),
            ],
            [
                (z1, x1[3], y1[3]),
                (z1, x1[0], y1[0]),
                (z2, x2[0], y2[0]),
                (z2, x2[3], y2[3]),
            ],
        ]

        t = i / max(len(z_coords) - 2, 1)
        face_color = colormaps["Blues"](0.2 + 0.4 * t)
        collection = Poly3DCollection(
            verts, alpha=0.3, facecolor=face_color, edgecolors="#374151", linewidths=0.3
        )
        ax.add_collection3d(collection)

        ax.plot([z1] * 5, x1, y1, color="#374151", alpha=0.5, linewidth=0.3)
        if i == len(z_coords) - 2:
            ax.plot([z2] * 5, x2, y2, color="#374151", alpha=0.5, linewidth=0.3)

    ax.set_box_aspect((3, 1, 1))
    ax.set_xlabel("Length (m)", fontsize=6)
    ax.set_ylabel("Width (m)", fontsize=6)
    ax.set_zlabel("Height (m)", fontsize=6)
    ax.tick_params(labelsize=6, width=0.25, length=1.5)
    ax.set_title("Horn Profile (Unwrapped)", fontsize=7, fontweight="medium")
    ax.view_init(elev=18, azim=40)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_horn_2d_folded(
    conical_segments: List[Tuple[float, ...]],
    coordinates: List[Tuple[float, float]],
    enclosure_dims: Optional[Tuple[float, float]],
    output_path: Path,
    driver_coord: Optional[Tuple[float, float]] = None,
    throat_chamber_side: Optional[float] = None,
    title: str = "Folded Horn (2D)",
    wall_t: float = 0.0,
    rear_chamber: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Draw a 2D side-profile of the folded horn using center coordinates and perpendicular cross-sections.

    Args:
        conical_segments: List of (h_start, h_end) per segment
        coordinates: Center-line coordinates [x, y]
        enclosure_dims: (depth, height) of outer enclosure
        output_path: Where to save the PNG
        driver_coord: (x, y) driver reference point — right edge will be flush with rear wall
        throat_chamber_side: Throat chamber side length (if any)
        title: Plot title
        wall_t: Wall thickness in metres (draws enclosure wall fill)
        rear_chamber: (w, h, d) of rear chamber — positioned at rear-bottom of enclosure
    """
    fig, ax = plt.subplots(figsize=(7, 9))

    # Guard: if coordinates are empty or single point, skip plotting
    if not coordinates or len(coordinates) < 2:
        plt.close(fig)
        return
    if enclosure_dims:
        depth, height = enclosure_dims
        rect = Rectangle(
            (0, 0),
            depth,
            height,
            fill=False,
            edgecolor="#374151",
            linewidth=0.8,
            linestyle="--",
        )
        ax.add_patch(rect)

    # 2. Process segments — draw horn path inside enclosure
    pts = np.array(coordinates) if (coordinates and len(coordinates) >= 2) else np.empty((0, 2))
    n_seg = min(len(conical_segments), max(0, len(pts) - 1))

    if n_seg > 0:
        for i, seg in enumerate(conical_segments[:n_seg]):
            h_start = seg[0]
            h_end = seg[1]
            p1 = pts[i]
            p2 = pts[i + 1]
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length == 0:
                continue
            direction = vec / length
            normal = np.array([-direction[1], direction[0]])
            w1_inner = p1 + normal * (h_start / 2)
            w1_outer = p1 - normal * (h_start / 2)
            w2_inner = p2 + normal * (h_end / 2)
            w2_outer = p2 - normal * (h_end / 2)
            t = i / max(n_seg - 1, 1)
            face_color = colormaps["Blues"](0.15 + 0.45 * t)
            poly = Polygon(
                [w1_inner, w2_inner, w2_outer, w1_outer],
                fill=True, facecolor=face_color, alpha=0.5,
                edgecolor="#374151", linewidth=0.5,
            )
            ax.add_patch(poly)
        # Centerline + throat/mouth markers
        center_pts = pts[:n_seg + 1]
        ax.plot(center_pts[:, 0], center_pts[:, 1],
                color="#dc2626", linestyle="-", alpha=0.5, linewidth=0.4)
        ax.plot(pts[0][0], pts[0][1], "o", color="#dc2626", markersize=3, zorder=4)
        ax.plot(pts[n_seg][0], pts[n_seg][1], "s", color="#dc2626", markersize=3, zorder=4)

    # 4. Draw Driver if provided
    if driver_coord:
        dx, dy = driver_coord

        d_height = 0.166
        d_depth = 0.0776
        mag_depth = 0.018
        cone_depth = 0.040
        mag_height = 0.090

        driver_poly = Polygon(
            [
                (dx, dy - d_height / 2),
                (dx + cone_depth, dy - mag_height / 2),
                (dx + d_depth - mag_depth, dy - mag_height / 2),
                (dx + d_depth, dy - mag_height / 2),
                (dx + d_depth, dy + mag_height / 2),
                (dx + d_depth - mag_depth, dy + mag_height / 2),
                (dx + cone_depth, dy + mag_height / 2),
                (dx, dy + d_height / 2),
            ],
            fill=True,
            facecolor="#6b7280",
            edgecolor="#1f2937",
            linewidth=0.6,
            zorder=5,
        )
        ax.add_patch(driver_poly)
        ax.plot(
            dx,
            dy,
            "+",
            color="#1f2937",
            markersize=7,
            markeredgewidth=0.8,
            label="Driver",
        )

        if (
            throat_chamber_side is not None
            and throat_chamber_side > 0
            and enclosure_dims is not None
        ):
            depth, height = enclosure_dims
            chamber_side = min(throat_chamber_side, depth, height)
            chamber_y0 = max(0.0, dy - chamber_side / 2.0)
            chamber_y0 = min(chamber_y0, height - chamber_side)

            if dx <= depth / 2.0:
                chamber_x0 = 0.0
            else:
                chamber_x0 = depth - chamber_side

            chamber = Rectangle(
                (chamber_x0, chamber_y0),
                chamber_side,
                chamber_side,
                fill=True,
                facecolor="#f59e0b",
                alpha=0.18,
                edgecolor="#b45309",
                linewidth=0.8,
                linestyle=":",
                zorder=3,
            )
            ax.add_patch(chamber)
            ax.text(
                chamber_x0 + chamber_side / 2.0,
                chamber_y0 + chamber_side / 2.0,
                "TC",
                ha="center",
                va="center",
                fontsize=7,
                color="#92400e",
                zorder=6,
            )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.invert_xaxis()

    _apply_style(ax, xlabel="Depth (m)", ylabel="Height (m)")
    ax.set_title(title, fontsize=10, fontweight="medium")
    handles, labels = ax.get_legend_handles_labels()
    if any(l for l in labels):
        ax.legend(fontsize=7, framealpha=0.6, edgecolor="none", loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_waterfall(
    csd_freqs: np.ndarray,
    csd_times_ms: np.ndarray,
    csd_db: np.ndarray,
    output_path: Path,
    title: str = "Cumulative Spectral Decay",
    f_min: float = 20.0,
    f_max: float = 2000.0,
) -> None:
    """
    Generate a 3D waterfall plot of the Cumulative Spectral Decay.
    Shows how resonances ring over time.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Trim frequency range
    f_mask = (csd_freqs >= f_min) & (csd_freqs <= f_max)
    freqs_plot = csd_freqs[f_mask]
    db_plot = csd_db[:, f_mask]

    # Normalize: 0 dB = peak of first slice (full window)
    ref_db = np.max(db_plot[0])
    db_plot = db_plot - ref_db

    # Plot each time slice as a filled line
    log_freqs = np.log10(freqs_plot)
    for i in range(len(csd_times_ms)):
        t = csd_times_ms[i]
        z = db_plot[i]
        # Clip floor
        z = np.maximum(z, -40.0)
        ax.plot(
            log_freqs,
            [t] * len(log_freqs),
            z,
            color=colormaps["viridis"](1.0 - i / max(len(csd_times_ms) - 1, 1)),
            linewidth=0.6,
            alpha=0.8,
        )

    # Format log frequency axis
    tick_freqs = [f for f in _FREQ_TICKS if f_min <= f <= f_max]
    ax.set_xticks([np.log10(f) for f in tick_freqs])
    ax.set_xticklabels([_fmt_hz(f, None) for f in tick_freqs])

    ax.set_xlabel("Frequency (Hz)", fontsize=7, labelpad=8)
    ax.set_ylabel("Time (ms)", fontsize=7, labelpad=8)
    ax.set_zlabel("dB", fontsize=7, labelpad=5)
    ax.set_zlim(-40, 5)
    ax.tick_params(labelsize=6, width=0.3, length=2)
    ax.set_title(title, fontsize=10, fontweight="medium")
    ax.view_init(elev=30, azim=-60)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.95)
    # Note: plt.tight_layout() is skipped for 3D axes — it emits warnings
    # due to z-axis label overflow; subplots_adjust above provides sufficient margin.
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_throat_adapter_profile(
    profile_data: dict,
    output_path: Path,
    title: str = "Throat Adapter Profile",
) -> None:
    """
    Plot the throat adapter area profile as a 2D schematic.

    Parameters
    ----------
    profile_data :
        Dictionary returned by ``throat_adapter_profile()`` with keys:
        ``x`` (m), ``area`` (m²), ``diam`` (m), ``A0`` (m² input area),
        ``Ap1`` (m² output area).
    output_path :
        Where to save the PNG.
    title :
        Plot title.
    """
    x = profile_data["x"]
    area = profile_data["area"]
    diam = profile_data["diam"]
    A0 = profile_data["A0"]
    Ap1 = profile_data["Ap1"]
    lpt = x[-1] if len(x) > 0 else 0.0

    d0 = 2.0 * math.sqrt(A0 / math.pi)
    d1 = 2.0 * math.sqrt(Ap1 / math.pi)

    fig, (ax_area, ax_diam) = plt.subplots(2, 1, figsize=(9, 5))
    fig.suptitle(title, fontsize=11, fontweight="medium", y=0.98)

    # Area plot
    ax_area.fill_between(x * 1000.0, area * 10000.0, alpha=0.25, color="#2563eb")
    ax_area.plot(x * 1000.0, area * 10000.0, color="#2563eb", linewidth=1.2)
    ax_area.axhline(
        A0 * 10000.0,
        color="#9ca3af",
        linestyle="--",
        linewidth=0.6,
        label=f"A0 = {A0 * 10000.0:.2f} cm²",
    )
    ax_area.axhline(
        Ap1 * 10000.0,
        color="#dc2626",
        linestyle="--",
        linewidth=0.6,
        label=f"Ap1 = {Ap1 * 10000.0:.2f} cm²",
    )
    ax_area.set_ylabel("Area (cm²)", fontsize=8)
    ax_area.set_xlabel("Position along adapter (mm)", fontsize=8)
    ax_area.legend(fontsize=7, framealpha=0.6, edgecolor="none")
    ax_area.tick_params(labelsize=7)
    ax_area.grid(True, alpha=0.2)
    for spine in ax_area.spines.values():
        spine.set_linewidth(0.4)

    # Diameter plot
    ax_diam.plot(x * 1000.0, diam * 1000.0, color="#7c3aed", linewidth=1.2)
    ax_diam.axhline(
        d0 * 1000.0,
        color="#9ca3af",
        linestyle="--",
        linewidth=0.6,
        label=f"D1 = {d0 * 1000.0:.1f} mm",
    )
    ax_diam.axhline(
        d1 * 1000.0,
        color="#dc2626",
        linestyle="--",
        linewidth=0.6,
        label=f"D2 = {d1 * 1000.0:.1f} mm",
    )
    ax_diam.set_ylabel("Diameter (mm)", fontsize=8)
    ax_diam.set_xlabel("Position along adapter (mm)", fontsize=8)
    ax_diam.legend(fontsize=7, framealpha=0.6, edgecolor="none")
    ax_diam.tick_params(labelsize=7)
    ax_diam.grid(True, alpha=0.2)
    for spine in ax_diam.spines.values():
        spine.set_linewidth(0.4)

    # Annotation: length label
    mid_x = lpt * 500.0
    mid_d = (d0 + d1) / 2.0 * 1000.0
    fig.text(
        0.77,
        0.72,
        f"Lpt = {lpt * 1000.0:.1f} mm",
        fontsize=7,
        color="#374151",
        ha="left",
    )

    plt.tight_layout(h_pad=1.5)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_impulse_step(
    time_ms: np.ndarray,
    impulse: np.ndarray,
    step: np.ndarray,
    output_path: Path,
    title: str = "Time Domain Response",
) -> None:
    """
    Plot impulse response and step response side by side.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
    fig.suptitle(title, fontsize=11, fontweight="medium", y=0.98)

    ax1.plot(time_ms, impulse, color=_COLORS["spl"], linewidth=0.6)
    _apply_style(ax1, xlabel="Time (ms)", ylabel="Amplitude (Pa)")
    ax1.set_title("Impulse Response", fontsize=8)

    ax2.plot(time_ms, step, color=_COLORS["horn"], linewidth=0.6)
    _apply_style(ax2, xlabel="Time (ms)", ylabel="Amplitude (Pa\u00b7s)")
    ax2.set_title("Step Response", fontsize=8)

    plt.tight_layout(h_pad=1.5)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
