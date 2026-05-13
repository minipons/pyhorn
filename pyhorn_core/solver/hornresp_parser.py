"""Parse Hornresp .txt project files into pyhorn HornGeometry + DriverSpecs.

## Purpose

This module bridges pyhorn and the legacy [Hornresp](https://www.hornresp.net/)
loudspeaker simulation tool. It reads Hornresp's plain-text export files and
converts them into pyhorn's native `HornGeometry` and `DriverSpecs` dataclasses,
enabling:

- **Validation** — run a pyhorn simulation with the same inputs as Hornresp
  and compare outputs to verify numerical correctness.
- **Migration** — import existing Hornresp projects without manual re-entry.
- **Regression testing** — the pyhorn test suite uses Hornresp reference data
  (exported via this module) as the golden standard.

## Hornresp File Formats

Hornresp exports two distinct plain-text file types:

| File kind | Format | Contents |
|-----------|--------|----------|
| Driver `.txt` | Flat `KEY = value` lines | T/S parameters (Re, Sd, Bl, Cms, …) |
| Project `.txt` | Pipe-section headers (`\\|NAME\\|`) + `KEY = value` lines | Full system geometry + embedded driver section |

Both formats use units adapted from Hornresp's Windows heritage (cm², cm, g, mH,
mm/N, litres). The parser strips units before converting to SI.

## Public API

- `parseHornrespProject(path)` → `Dict[str, str]` — raw key/value dict from a
  Hornresp project `.txt` file.
- `parseHornrespDriver(path)` → `Dict[str, str]` — raw key/value dict from a
  Hornresp driver `.txt` file.
- `hornresp_driver_to_specs(params)` → `DriverSpecs` — convert a driver params
  dict to pyhorn's `DriverSpecs` (SI units, derived T/S computed).
- `hornresp_project_to_driver_specs(params)` → `DriverSpecs` — extract driver
  section from a project params dict.
- `hornresp_project_to_geometry(params)` → `HornGeometry` — convert a project
  params dict to pyhorn's `HornGeometry`.
- `load_hornresp_project(project_path, driver_path=None)` →
  `(HornGeometry, DriverSpecs)` — highest-level convenience loader.

## Architecture Position

This module lives in `pyhorn_core/solver/` (not `config/`) because it is
concerned with **importing external reference data**, not with pyhorn's own
file format. It is a leaf dependency: it imports config dataclasses only,
but nothing in `config/` imports back from `solver/`. The solver layer receives
already-constructed domain objects; it never calls these parser functions
directly.
"""

import math
import re
from pathlib import Path
from typing import Dict, Optional

from pyhorn_core.config.driver_models import DriverSpecs
from pyhorn_core.config.horn_models import HornGeometry

# ─────────────────────────────────────────────────────────────────────────────
# Raw parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_key_value_line(line: str) -> Optional[tuple[str, str]]:
    """Return (key, value) from 'Key = value' or 'Key = value unit' lines."""
    line = line.strip()
    m = re.match(r"^([^=]+)=\s*(.+)$", line)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def parseHornrespProject(txt_path: str | Path) -> Dict[str, str]:
    """Parse a Hornresp project .txt file into a flat key→value dict.

    Handles pipe-section headers (|NAME|) and standalone KEY = VALUE lines.
    Duplicated keys (e.g. S2, S3 appearing twice) retain only the first-seen
    value; all values are preserved under key+'__all'.
    """
    params: Dict[str, str] = {}
    # Store all values for duplicate keys
    multi: Dict[str, list[str]] = {}

    with open(txt_path) as f:
        for raw in f:
            line = raw.rstrip()
            if line.startswith("|") or not line.strip():
                continue
            kv = parse_key_value_line(line)
            if kv is None:
                continue
            key, val = kv
            if key in params:
                # Subsequent occurrence: track in multi, do NOT overwrite params (keep first)
                multi.setdefault(key, []).append(val)
            else:
                # First occurrence: store in params and initialise multi tracking
                params[key] = val
                multi[key] = []

    for key, vals in multi.items():
        # Only create __all for keys with ≥2 occurrences
        if vals:  # vals contains subsequent occurrences only
            params[key + "__all"] = [params[key]] + vals

    return params


def parseHornrespDriver(txt_path: str | Path) -> Dict[str, str]:
    """Parse a Hornresp driver .txt file (flat key=value, one per line)."""
    params: Dict[str, str] = {}
    with open(txt_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            kv = parse_key_value_line(line)
            if kv is not None:
                params[kv[0]] = kv[1]
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Value coercion helpers
# ─────────────────────────────────────────────────────────────────────────────


def _f(text: str) -> float:
    """Parse a float from Hornresp format.

    Hornresp stores Cms in "N/m × 10^-6" — e.g. 1500.0E-06 means 1500 × 10^-6 N/m.
    Mmd is stored in grams. All other values are standard floats.

    Units (ohms, Hz, cm^2, cm, l, mH, g, kg/s, N/A, mm/N) are stripped
    before conversion so that "7.80 ohms", "40 cm^2", "49.6 Hz" all parse correctly.
    """
    import re

    m = re.search(r"[+-]?[\d.]+[eE+-]*\d*", text.strip())
    if m is None:
        raise ValueError(f"Cannot parse float from Hornresp value: {text!r}")
    return float(m.group(0))


# ─────────────────────────────────────────────────────────────────────────────
# Driver specs
# ─────────────────────────────────────────────────────────────────────────────


def hornresp_driver_to_specs(params: Dict[str, str]) -> DriverSpecs:
    """Convert Hornresp driver params to DriverSpecs (SI units).

    Hornresp stores Cms in mm/N and Mmd in grams — both need conversion to SI.
    fs, Qts, Qms, Qes, Vas are derived from the raw values.
    """
    re_ = _f(params["Re"])
    sd_ = _f(params["Sd"]) / 10000.0  # cm² → m²
    bl_ = _f(params["Bl"])  # N/A  (no conversion)
    le_ = _f(params["Le"]) / 1000.0  # mH → H
    rms_ = _f(params["Rms"])  # kg/s
    cms_ = _f(params["Cms"]) * 1e-3  # mm/N → m/N
    mms_ = _f(params["Mmd"]) / 1000.0  # g → kg

    fs_ = 1.0 / (2.0 * math.pi * cms_ * mms_) if cms_ > 0 and mms_ > 0 else 0.0
    qms_ = rms_ / (2.0 * math.pi * fs_ * mms_) if fs_ > 0 else 0.0
    qes_ = (bl_**2) / (re_ * 2.0 * math.pi * fs_ * mms_) if fs_ > 0 and re_ > 0 else 0.0
    qts_ = qes_ * qms_ / (qes_ + qms_) if (qes_ + qms_) > 0 else 0.0

    # Vas = ρ × c² × Cms × Sd²  (ρ = 1.21 kg/m³, c = 343 m/s)
    vas_ = 1.21 * 343.0**2 * cms_ * (sd_**2) if sd_ > 0 else 0.0

    xmax_ = _f(params.get("Xmax", "0")) / 1000.0  # mm → m

    return DriverSpecs(
        fs=fs_,
        qts=qts_,
        qes=qes_,
        qms=qms_,
        vas=vas_,
        re=re_,
        sd=sd_,
        bl=bl_,
        cms=cms_,
        rms=rms_,
        le=le_,
        mms=mms_,
        xmax=xmax_,
        voltage=2.83,
    )


def hornresp_project_to_driver_specs(params: Dict[str, str]) -> DriverSpecs:
    """Extract driver T/S from a Hornresp project params dict.

    Reads the TRADITIONAL DRIVER PARAMETER VALUES section keys:
    Sd, Bl, Cms, Rms, Mmd, Le, Re, Nd, Xmax.
    """
    driver_keys = {"Sd", "Bl", "Cms", "Rms", "Mmd", "Le", "Re", "Nd", "Xmax"}
    driver_params = {k: v for k, v in params.items() if k in driver_keys}
    if not driver_params:
        return DriverSpecs(
            fs=0.0,
            qts=0.0,
            qes=0.0,
            qms=0.0,
            vas=0.0,
            re=0.0,
            bl=0.0,
            mms=0.0,
            cms=0.0,
            rms=0.0,
            sd=0.0,
            le=0.0,
            xmax=0.0,
        )
    return hornresp_driver_to_specs(driver_params)


# ─────────────────────────────────────────────────────────────────────────────
# Hornresp project → HornGeometry
# ─────────────────────────────────────────────────────────────────────────────


def _first(key: str, params: Dict[str, str], default: str = "0") -> float:
    """Return the first-seen float value for key (non-__all variant).

    For keys that appear multiple times in the Hornresp file (S2, S3, S4),
    the first occurrence is the one that matters for the horn parameters;
    subsequent ones are often placeholder zeroes in other sections.
    """
    return _f(params.get(key, default))


def _first_nonzero(key: str, params: Dict[str, str]) -> float:
    """Return the first non-zero float value for key."""
    for k, v in params.items():
        if k == key and _f(v) != 0.0:
            return _f(v)
    return 0.0


def hornresp_project_to_geometry(params: Dict[str, str]) -> HornGeometry:
    """Convert Hornresp project params to HornGeometry (SI units).

    Conversions applied:
      S1, S2, AT, Atc  : cm² → m²
      Hyp, Lrc          : cm  → m
      Vrc               : litres → m³
      Vtc               : cm³ → m³
    """
    throat_area_m2 = _f(params.get("S1", "0")) / 10000.0
    mouth_area_m2 = _f(params.get("S2", "0")) / 10000.0
    path_length_m = _f(params.get("Hyp", "0")) / 100.0
    t = _f(params.get("T", "1"))
    vrc_l = _f(params.get("Vrc", "0"))

    horn = HornGeometry(
        throat_area=throat_area_m2,
        mouth_area=mouth_area_m2,
        path_length=path_length_m,
        enclosure_type="BLH" if vrc_l > 0 else "FLH",
        profile_type="Hyperbolic",
        hyperbolic_t=t,
        n_segments=39,
        vrc=vrc_l / 1000.0,
        lrc=_f(params.get("Lrc", "0")) / 100.0,
        fr_rc=_f(params.get("Fr1", "0")),
        vtc=_f(params.get("Vtc", "0")) / 1_000_000.0,
        atc=_f(params.get("Atc", "0")) / 10000.0,
    )
    return horn


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loaders
# ─────────────────────────────────────────────────────────────────────────────


def load_hornresp_project(
    project_path: str | Path,
    driver_path: str | Path | None = None,
) -> tuple[HornGeometry, DriverSpecs]:
    """Load a Hornresp project .txt and optional driver .txt.

    Driver priority:
      1. driver_path (explicit override)
      2. project file's inlined driver parameter section
      3. zeroed DriverSpecs()
    """
    params = parseHornrespProject(project_path)
    geometry = hornresp_project_to_geometry(params)

    if driver_path is not None:
        specs = hornresp_driver_to_specs(parseHornrespDriver(driver_path))
    else:
        specs = hornresp_project_to_driver_specs(params)

    return geometry, specs
