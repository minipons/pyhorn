import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

def _format_csv_value(value: Any) -> Any:
    """Format numeric CSV values with at most two decimal places."""
    if isinstance(value, (bool, np.bool_)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return v
        rounded = round(v, 2)
        if rounded == 0.0:
            rounded = 0.0
        if rounded.is_integer():
            return str(int(rounded))
        return f"{rounded:.2f}".rstrip("0").rstrip(".")
    return value


def export_to_csv(
    freqs: np.ndarray,
    spl_responses: Dict[str, np.ndarray],
    output_path: Path,
    cone_acceleration: Optional[np.ndarray] = None,
    particle_velocity_throat: Optional[np.ndarray] = None,
    particle_velocity_mouth: Optional[np.ndarray] = None,
    particle_velocity_port: Optional[np.ndarray] = None,
    futtrup_gdlimit_ms: Optional[np.ndarray] = None,
) -> None:
    """Export frequency and SPL data to a CSV file.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency points in Hz.
    spl_responses : Dict[str, np.ndarray]
        SPL response arrays keyed by label (e.g. "total", "horn", "direct").
    output_path : Path
        Destination CSV file path.
    cone_acceleration : Optional[np.ndarray]
        Cone acceleration in m/s² (peak). If provided, adds a
        ``Cone_Acceleration_ms2`` column.
    particle_velocity_throat : Optional[np.ndarray]
        Particle velocity at throat in m/s. If provided, adds a
        ``Particle_Velocity_Throat_ms`` column.
    particle_velocity_mouth : Optional[np.ndarray]
        Particle velocity at mouth in m/s. If provided, adds a
        ``Particle_Velocity_Mouth_ms`` column.
    particle_velocity_port : Optional[np.ndarray]
        Particle velocity at port in m/s. If provided, adds a
        ``Particle_Velocity_Port_ms`` column.
    """
    labels = list(spl_responses.keys())
    header = ["Frequency_Hz"] + [f"SPL_dB_{label}" for label in labels]
    if cone_acceleration is not None:
        header.append("Cone_Acceleration_ms2")
    if particle_velocity_throat is not None:
        header.append("Particle_Velocity_Throat_ms")
    if particle_velocity_mouth is not None:
        header.append("Particle_Velocity_Mouth_ms")
    if particle_velocity_port is not None:
        header.append("Particle_Velocity_Port_ms")
    if futtrup_gdlimit_ms is not None:
        header.append("Futtrup_GDlimit_ms")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, freq in enumerate(freqs):
            row = [freq] + [spl_responses[label][i] for label in labels]
            if cone_acceleration is not None:
                row.append(cone_acceleration[i])
            if particle_velocity_throat is not None:
                row.append(particle_velocity_throat[i])
            if particle_velocity_mouth is not None:
                row.append(particle_velocity_mouth[i])
            if particle_velocity_port is not None:
                row.append(particle_velocity_port[i])
            if futtrup_gdlimit_ms is not None:
                row.append(futtrup_gdlimit_ms[i])
            writer.writerow([_format_csv_value(v) for v in row])


def export_to_json(
    freqs: np.ndarray,
    spl_responses: Dict[str, np.ndarray],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    cone_acceleration: Optional[np.ndarray] = None,
    particle_velocity_throat: Optional[np.ndarray] = None,
    particle_velocity_mouth: Optional[np.ndarray] = None,
    particle_velocity_port: Optional[np.ndarray] = None,
    futtrup_gdlimit_ms: Optional[np.ndarray] = None,
) -> None:
    """Export frequency and SPL data to a JSON file.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency points in Hz.
    spl_responses : Dict[str, np.ndarray]
        SPL response arrays keyed by label.
    output_path : Path
        Destination JSON file path.
    metadata : Optional[Dict[str, Any]]
        Optional metadata dict to embed in the JSON.
    cone_acceleration : Optional[np.ndarray]
        Cone acceleration in m/s² (peak). If provided, adds a
        ``cone_acceleration`` field (m/s² array).
    particle_velocity_throat : Optional[np.ndarray]
        Particle velocity at throat in m/s. If provided, adds a
        ``particle_velocity_throat`` field.
    particle_velocity_mouth : Optional[np.ndarray]
        Particle velocity at mouth in m/s. If provided, adds a
        ``particle_velocity_mouth`` field.
    particle_velocity_port : Optional[np.ndarray]
        Particle velocity at port in m/s. If provided, adds a
        ``particle_velocity_port`` field.
    """
    data: Dict[str, Any] = {
        "frequencies": freqs.tolist(),
        "responses": {label: spl.tolist() for label, spl in spl_responses.items()},
    }
    if cone_acceleration is not None:
        data["cone_acceleration"] = cone_acceleration.tolist()
    if particle_velocity_throat is not None:
        data["particle_velocity_throat"] = particle_velocity_throat.tolist()
    if particle_velocity_mouth is not None:
        data["particle_velocity_mouth"] = particle_velocity_mouth.tolist()
    if particle_velocity_port is not None:
        data["particle_velocity_port"] = particle_velocity_port.tolist()
    if futtrup_gdlimit_ms is not None:
        data["futtrup_gdlimit_ms"] = futtrup_gdlimit_ms.tolist()
    if metadata:
        data["metadata"] = metadata

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def export_to_frd(
    freqs: np.ndarray,
    spl_db: np.ndarray,
    phase_deg: np.ndarray,
    output_path: Path,
) -> None:
    """Export frequency response data to a standard .frd file (REW/ARTA format).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency points in Hz (typically log-spaced).
    spl_db : np.ndarray
        Sound pressure level in dB at each frequency.
    phase_deg : np.ndarray
        Phase in degrees at each frequency.
    output_path : Path
        Destination .frd file path.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("!FRD1.0\n")
        f.write("Frequency(Hz)  Magnitude(dB)  Phase(deg)\n")
        for i in range(len(freqs)):
            f.write(f"{freqs[i]:.4f}  {spl_db[i]:.4f}  {phase_deg[i]:.4f}\n")
