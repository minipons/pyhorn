#!/usr/bin/env python3
"""
bem_generate_calibration.py — Generate BEM radiation calibration files.

This script creates a JSON calibration file that maps frequency (Hz) to complex
radiation impedance Z_rad = R_rad + j·X_rad for a specific horn mouth geometry.

These calibration files are used by pyhorn's BEM radiation model
(mouth_radiation="bem") to replace the Levine/Inglis analytical piston model
with physics-based BEM-computed values.

Workflow
========
1. Run your BEM solver (bempp-cl, BEMPPSolver, horn-simulation) for the horn mouth
2. Export the throat-surface integrated radiation impedance vs frequency
3. Run this script to convert BEM output → pyhorn calibration JSON
4. Set mouth_radiation="bem" and bem_calibration_path in your horn.yaml

BEM Output Format
=================
The BEM solver should compute radiation impedance by:
  Z_rad = (1/S_throat) · ∫ p_throat(x) dS  /  U_cone
where:
  - S_throat = throat surface area
  - p_throat = acoustic pressure on the throat surface (from BEM solution)
  - U_cone = diaphragm volume velocity

Expected columns in BEM CSV output:
  Freq (Hz), Z_real (Pa·s/m³), Z_imag (Pa·s/m³)

Note: BEM radiation impedance is in acoustic ohms (Pa·s/m³), NOT electrical ohms.
The Levine/Inglis model also uses acoustic ohms, so no unit conversion is needed.

Calibration File Format
=======================
{
  "metadata": {
    "geometry_type": "exponential",
    "mouth_area_m2": 0.01,
    "throat_area_m2": 0.0005,
    "mouth_radius_m": 0.0564,
    "path_length_m": 0.3,
    "baffle": "unflanged",        # "infinite_baffle", "half_space", "unflanged"
    "solver": "bempp-cl 3.x",
    "source_file": "path/to/bem_output.csv",
    "notes": "..."
  },
  "freqs": [20.0, 25.0, 31.6, ...],
  "z_real": [0.1, 0.15, 0.22, ...],
  "z_imag": [0.05, 0.06, 0.08, ...]
}

Usage
=====
::

    python scripts/bem_generate_calibration.py \\
        --input bem_output.csv \\
        --output tests/benchmarks/bem/exponential_horn/calibration/mouth_z.json \\
        --mouth-area 0.01 \\
        --throat-area 0.0005 \\
        --baffle unflanged \\
        --solver bempp-cl \\
        --notes "Exponential horn mouth radiation, 56mm radius"

"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_bem_csv(path: Path) -> tuple[list[float], list[float], list[float]]:
    """
    Load BEM CSV output and return (freqs, z_real, z_imag).

    Expected columns: Freq (Hz), Z_real (Pa·s/m³), Z_imag (Pa·s/m³)
    """
    freqs, z_real, z_imag = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            freq_col = row.get("Freq (Hz)") or row.get("Freq") or row.get("freq")
            zr_col = row.get("Z_real (Pa·s/m³)") or row.get("Z_real") or row.get("z_real") or row.get("Zreal")
            zi_col = row.get("Z_imag (Pa·s/m³)") or row.get("Z_imag") or row.get("z_imag") or row.get("Zimag")

            if not freq_col:
                continue

            freqs.append(float(freq_col))
            z_real.append(float(zr_col) if zr_col else 0.0)
            z_imag.append(float(zi_col) if zi_col else 0.0)

    return freqs, z_real, z_imag


def validate_impedance(freqs, z_real, z_imag) -> list[str]:
    """
    Validate BEM radiation impedance data and return list of warnings.
    """
    warnings = []

    import numpy as np
    freqs = np.array(freqs)
    z_real = np.array(z_real)
    z_imag = np.array(z_imag)

    if len(freqs) < 5:
        warnings.append(f"Very few frequency points ({len(freqs)}). Need at least 5.")

    if np.any(z_real < -1e-9):
        neg_r = np.sum(z_real < -1e-9)
        warnings.append(f"Negative radiation resistance at {neg_r} points. Physical sanity check failed.")

    if np.any(z_imag < -1e-6 * np.abs(z_real)):
        neg_x = np.sum(z_imag < -1e-6 * np.abs(z_real))
        warnings.append(f"Heavily negative radiation reactance at {neg_x} points.")

    for i in range(1, len(freqs)):
        if freqs[i] <= freqs[i-1]:
            warnings.append(f"Non-monotonic frequency at index {i}: {freqs[i-1]} → {freqs[i]}")

    return warnings


def generate_calibration(
    input_path: Path,
    output_path: Path,
    mouth_area_m2: float,
    throat_area_m2: float,
    geometry_type: str = "exponential",
    mouth_radius_m: float | None = None,
    path_length_m: float | None = None,
    baffle: str = "unflanged",
    solver: str = "unknown",
    notes: str = "",
    source_file: str = "",
) -> None:
    """
    Generate a pyhorn BEM calibration JSON file from BEM solver output.
    """
    freqs, z_real, z_imag = load_bem_csv(input_path)

    if not freqs:
        raise ValueError(f"No data loaded from {input_path}")

    warnings = validate_impedance(freqs, z_real, z_imag)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    import numpy as np
    a_mouth = np.sqrt(mouth_area_m2 / np.pi)
    if mouth_radius_m is None:
        mouth_radius_m = a_mouth

    metadata = {
        "geometry_type": geometry_type,
        "mouth_area_m2": mouth_area_m2,
        "throat_area_m2": throat_area_m2,
        "mouth_radius_m": float(mouth_radius_m),
        "path_length_m": float(path_length_m) if path_length_m else None,
        "baffle": baffle,
        "solver": solver,
        "source_file": str(source_file) or str(input_path),
        "notes": notes,
        "units": "acoustic ohms (Pa·s/m³)",
        "n_points": len(freqs),
        "freq_range_hz": [float(min(freqs)), float(max(freqs))],
    }

    calibration = {
        "metadata": metadata,
        "freqs": [float(f) for f in freqs],
        "z_real": [float(z) for z in z_real],
        "z_imag": [float(z) for z in z_imag],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"Calibration file written: {output_path}")
    print(f"  {len(freqs)} frequency points")
    print(f"  {min(freqs):.1f} – {max(freqs):.0f} Hz")
    print(f"  mouth_area = {mouth_area_m2*1e4:.1f} cm²")
    print(f"  baffle = {baffle}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pyhorn BEM calibration file from BEM solver CSV output."
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Path to BEM solver CSV output"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Path for output calibration JSON file"
    )
    parser.add_argument(
        "--mouth-area", required=True, type=float,
        help="Mouth area in m² (e.g. 0.01 for 100 cm²)"
    )
    parser.add_argument(
        "--throat-area", required=True, type=float,
        help="Throat area in m² (e.g. 0.0005 for 5 cm²)"
    )
    parser.add_argument(
        "--mouth-radius", type=float, default=None,
        help="Mouth radius in m (derived from mouth_area if not given)"
    )
    parser.add_argument(
        "--path-length", type=float, default=None,
        help="Horn path length in m"
    )
    parser.add_argument(
        "--geometry-type", default="exponential",
        choices=["exponential", "conical", "hyperbolic", "tractrix", "custom"],
        help="Horn geometry type"
    )
    parser.add_argument(
        "--baffle", default="unflanged",
        choices=["infinite_baffle", "half_space", "unflanged", "custom"],
        help="Baffle condition used in BEM simulation"
    )
    parser.add_argument(
        "--solver", default="bempp-cl",
        help="BEM solver name and version"
    )
    parser.add_argument(
        "--notes", default="",
        help="Additional notes for the calibration file"
    )

    args = parser.parse_args()
    generate_calibration(
        input_path=args.input,
        output_path=args.output,
        mouth_area_m2=args.mouth_area,
        throat_area_m2=args.throat_area,
        geometry_type=args.geometry_type,
        mouth_radius_m=args.mouth_radius,
        path_length_m=args.path_length,
        baffle=args.baffle,
        solver=args.solver,
        notes=args.notes,
        source_file=str(args.input),
    )


if __name__ == "__main__":
    main()
