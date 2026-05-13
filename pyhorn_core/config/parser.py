"""Parse pyhorn YAML / JSON configuration files into domain dataclasses.

## Purpose

This module is the **front-door for all user-supplied configuration** in pyhorn.
Every simulation starts here: project files (`.yaml`), geometry files (`.yaml`),
and standalone driver spec files (`.yaml` or `.json`) are loaded through
`parse_horn_project`, `parse_horn_geometry`, and `parse_driver_specs`
respectively.

The parser is responsible for:

1. **Loading** — reading JSON or YAML from disk.
2. **Composition** — resolving relative paths (e.g. `geometry_path` in a project
   file points to a separate geometry YAML; the parser resolves it relative to
   the project file's directory).
3. **Validation** — rejecting physically impossible parameter combinations
   before they reach the solver, with user-friendly error messages.
4. **Construction** — building nested dataclass instances
   (`HornProject`, `HornGeometry`, `DriverSpecs`, `Section`, `ThroatChamber`,
   `RearChamber`, `VentedBox`, `PassiveRadiator`, `SlavicBox`) from the flat
   dictionary structure returned by the YAML/JSON loader.

## File Types Handled

| Function | File kind | Returns |
|----------|-----------|---------|
| `parse_driver_specs` | Driver YAML/JSON | `DriverSpecs` |
| `parse_horn_geometry` | Geometry YAML/JSON | `HornGeometry` |
| `parse_horn_project` | Project YAML/JSON | `(HornProject, HornGeometry)` |

A **project file** is the top-level entry point. It references a
**geometry file** via `geometry_path`. The geometry file defines the horn
shape (throat/mouth areas, path length, profile type, optional chained sections)
and may be shared across multiple projects.

## Validation Philosophy

Validation errors are raised as `ValueError` (never raw `TypeError` or
`AssertionError`) so callers always get a human-readable message. The rules
enforce loudspeaker physics:

- `fs`, `qts`, `qes`, `qms`, `vas`, `re`, `mms`, `cms`, `sd` must be **positive**.
- `bl`, `rms`, `le`, `xmax`, `voltage` must be **non-negative**.
- `throat_area`, `mouth_area`, `path_length` must be **positive** when the
  scalar format is used (vs. the `sections` format).
- Section `start_area`, `end_area`, `length` must be **positive**.

## Architecture Position

This module lives in `pyhorn_core/config/` alongside the domain models
(`models.py`). It is the only module in `config/` that touches the filesystem;
all other `config/` code operates on in-memory dataclasses.

It is a **low-level dependency** — the CLI, the FastAPI server, and the test
runner all route through these functions. It must not import from
`pyhorn_core/solver/` (no circular dependencies).
"""

import csv
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, cast, Optional
import numpy as np

from .models import DriverSpecs, HornGeometry, HornProject, PassiveRadiator, RearChamber, Section, SlavicBox, ThroatAdapter, ThroatChamber, VentedBox


def _validate_driver_specs(data: Dict[str, Any]) -> None:
    """Validate driver T-S parameters before constructing DriverSpecs.

    Raises ValueError for any physically impossible parameter values.
    """
    _must_be_positive = {
        "fs": data.get("fs"),
        "qts": data.get("qts"),
        "qes": data.get("qes"),
        "qms": data.get("qms"),
        "vas": data.get("vas"),
        "re": data.get("re"),
        "mms": data.get("mms"),
        "cms": data.get("cms"),
        "sd": data.get("sd"),
    }
    for name, value in _must_be_positive.items():
        if value is None:
            raise ValueError(f"Driver parameter '{name}' is required but missing from driver YAML.")
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Driver parameter '{name}' must be a number, got {type(value).__name__!r}"
            )
        if value <= 0:
            raise ValueError(
                f"Driver parameter '{name}' must be positive, got {value!r}. "
                f"A non-positive value is physically impossible for a loudspeaker driver."
            )

    # Non-negative optional parameters
    _must_be_non_negative = {
        "bl": data.get("bl", 0.0),
        "rms": data.get("rms", 0.0),
        "le": data.get("le", 0.0),
        "xmax": data.get("xmax", 0.0),
        "voltage": data.get("voltage", 2.83),
    }
    for name, value in _must_be_non_negative.items():
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Driver parameter '{name}' must be a number, got {type(value).__name__!r}"
            )
        if value < 0:
            raise ValueError(
                f"Driver parameter '{name}' must be non-negative, got {value!r}."
            )


def _validate_horn_geometry(data: Dict[str, Any]) -> None:
    """Validate horn geometry parameters before constructing HornGeometry.

    Raises ValueError for any physically impossible parameter values or missing
    required fields.  Error messages are user-friendly and never expose raw
    Python tracebacks.

    Validation rules:
      - When ``sections`` / ``profile_sections`` is NOT present:
          throat_area, mouth_area, path_length are required and must be positive
          (n_segments is optional and defaults to 100).
      - When sections format IS present: top-level throat/mouth/path are optional;
          each section's start_area, end_area, length must be positive.
      - vrc, lrc, vtc, fr_rc, fr_tc, lpt: must be non-negative when set
      - ap1, atc: must be positive when set (> 0)
      - sections: each section's start_area, end_area, length must be positive
    """
    # --- top-level HornGeometry scalar fields ---

    # When not using the sections format, throat_area / mouth_area / path_length
    # are mandatory UNLESS the geometry is defined via legacy segment formats
    # (coordinates, conical_segments, rectangular_segments) which the solver handles
    # differently.  Only enforce the check when the file has neither sections nor
    # any legacy segment list.
    has_sections = data.get("sections") is not None or data.get(
        "profile_sections"
    ) is not None
    has_legacy_segments = (
        data.get("coordinates") is not None
        or data.get("conical_segments") is not None
        or data.get("rectangular_segments") is not None
    )

    if not has_sections and not has_legacy_segments:
        _required_horn = {
            "throat_area": data.get("throat_area"),
            "mouth_area":  data.get("mouth_area"),
            "path_length": data.get("path_length"),
        }
        for name, value in _required_horn.items():
            if value is None:
                raise ValueError(
                    f"Horn geometry '{name}' is required but missing from the "
                    f"geometry YAML. Either add '{name}' as a top-level field, "
                    f"or use the 'sections' format to define the horn profile."
                )
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Horn geometry '{name}' must be a number, got {type(value).__name__!r}."
                )
            if value <= 0:
                raise ValueError(
                    f"Horn geometry '{name}' must be positive, got {value!r}. "
                    f"A non-positive value is physically impossible."
                )

    _must_be_positive = {
        "throat_area": data.get("throat_area"),
        "mouth_area":  data.get("mouth_area"),
        "path_length": data.get("path_length"),
        "n_segments":  data.get("n_segments"),
    }
    for name, value in _must_be_positive.items():
        if value is None:
            # Only enforce when the field is absent; HornGeometry defaults to 0
            # which will fail the <= 0 check below.
            pass
        elif not isinstance(value, (int, float)):
            raise ValueError(
                f"Horn geometry '{name}' must be a number, got {type(value).__name__!r}."
            )
        elif value <= 0:
            raise ValueError(
                f"Horn geometry '{name}' must be positive, got {value!r}. "
                f"A non-positive value is physically impossible."
            )

    _must_be_non_negative = {
        "vrc":    data.get("vrc",    0.0),
        "lrc":    data.get("lrc",    0.0),
        "vtc":    data.get("vtc",    0.0),
        "fr_rc":  data.get("fr_rc",  0.0),
        "fr_tc":  data.get("fr_tc",  0.0),
        "lpt":    data.get("lpt",    0.0),
        "ang":    data.get("ang",    6.283185307),
    }
    for name, value in _must_be_non_negative.items():
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Horn geometry '{name}' must be a number, got {type(value).__name__!r}."
            )
        elif value < 0:
            raise ValueError(
                f"Horn geometry '{name}' must be non-negative, got {value!r}."
            )

    # atc and ap1 must be > 0 when explicitly set (they default to 0)
    for name in ("atc", "ap1"):
        val = data.get(name)
        if val is not None and val != 0:
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"Horn geometry '{name}' must be a number, got {type(val).__name__!r}."
                )
            elif val <= 0:
                raise ValueError(
                    f"Horn geometry '{name}' must be positive when set, got {val!r}."
                )

    # --- sections ---
    raw_sections = data.get("sections") or data.get("profile_sections")
    if raw_sections:
        for i, raw in enumerate(raw_sections):
            for field in ("start_area", "end_area", "length"):
                val = raw.get(field)
                if val is None:
                    raise ValueError(
                        f"sections[{i}].{field} is required but missing."
                    )
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        f"sections[{i}].{field} must be a number, got {type(val).__name__!r}."
                    )
                if val <= 0:
                    raise ValueError(
                        f"sections[{i}].{field} must be positive, got {val!r}."
                    )


def parse_driver_specs(filepath: Path | str) -> DriverSpecs:
    """Parses driver specs from a JSON or YAML file."""
    filepath = Path(filepath)
    data = _load_file(filepath)
    _validate_driver_specs(data)
    # Handle frequency-dependent sensitivity_db: [[freq_hz, delta_db], ...] list
    if "sensitivity_db" in data:
        sd = data["sensitivity_db"]
        if isinstance(sd, list):
            # Convert list of [freq, value] pairs to numpy (N, 2) array for interpolation
            data["sensitivity_db"] = np.array(sd, dtype=float)
        # else: scalar float, pass through directly
    # Handle measured spl_response: either a CSV path (relative to driver YAML)
    # or an inline list of [freq, db] pairs.
    if "spl_response" in data:
        sr = data["spl_response"]
        if isinstance(sr, str):
            csv_path = (filepath.parent / sr).resolve() if not Path(sr).is_absolute() else Path(sr)
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"spl_response CSV not found: {csv_path}\n"
                    f"  Driver YAML: {filepath}\n"
                    f"  Check the 'spl_response' path is correct."
                )
            rows = []
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row or row[0].startswith("#"):
                        continue
                    rows.append((float(row[0]), float(row[1])))
            data["spl_response"] = np.array(rows, dtype=float)
        elif isinstance(sr, list):
            data["spl_response"] = np.array(sr, dtype=float)
    return DriverSpecs(**data)


def parse_horn_project(project_path: Path | str) -> tuple[HornProject, HornGeometry]:
    """Parse a horn project file and return (project_metadata, horn_geometry).

    The project YAML references a geometry YAML (resolved relative to the project
    file's directory). Project-level fields — driver_coord, name, notes,
    fold_plot_segments — are applied as overrides on top of the parsed geometry.

    Args:
        project_path: Path to the .yaml project file.

    Returns:
        (HornProject, HornGeometry) tuple. The HornGeometry has project
        overrides applied (driver_coord, fold_plot_segments).
    """
    project_path = Path(project_path)
    raw = _load_file(project_path)

    # 'geometry_path' (or legacy 'geometry') is the path to the geometry YAML.
    # If absent but the YAML contains geometry fields directly, use the project
    # file itself as the geometry (single-file project format).
    geom_rel = raw.pop("geometry_path", raw.pop("geometry", None))
    if geom_rel:
        geom_path = (project_path.parent / geom_rel).resolve()
        if not geom_path.exists():
            raise FileNotFoundError(
                f"Geometry file not found: {geom_path}\n"
                f"  Project file: {project_path}\n"
                f"  Check that the 'geometry_path' path in your project YAML is correct "
                f"and the file exists."
            )
        horn = parse_horn_geometry(geom_path)
    else:
        # Inline geometry — project YAML contains geometry fields directly
        horn = parse_horn_geometry(raw)

    # Build HornProject from remaining project-only fields.
    # 'geometry_path' was already popped from raw — put it back so it survives
    # into the dataclass (it IS a HornProject field).
    raw["geometry_path"] = geom_rel
    proj_fields = HornProject.__dataclass_fields__.keys()
    proj_data = {k: v for k, v in raw.items() if k in proj_fields}
    # Convert nested dicts to dataclass instances
    if "throat_chamber" in proj_data and isinstance(proj_data["throat_chamber"], dict):
        proj_data["throat_chamber"] = ThroatChamber(**proj_data["throat_chamber"])
    if "rear_chamber" in proj_data and isinstance(proj_data["rear_chamber"], dict):
        proj_data["rear_chamber"] = RearChamber(**proj_data["rear_chamber"])
    if "vented_box" in proj_data and isinstance(proj_data["vented_box"], dict):
        proj_data["vented_box"] = VentedBox(**proj_data["vented_box"])
    if "passive_radiator" in proj_data and isinstance(proj_data["passive_radiator"], dict):
        pr_data = proj_data["passive_radiator"].copy()
        # Handle shorthand `sp` key → sp1
        if "sp" in pr_data:
            pr_data["sp1"] = float(pr_data.pop("sp"))
        proj_data["passive_radiator"] = PassiveRadiator(**pr_data)
    if "slavbas" in proj_data and isinstance(proj_data["slavbas"], dict):
        sb_data = proj_data["slavbas"].copy()
        proj_data["slavbas"] = SlavicBox(**sb_data)
    proj = HornProject(**proj_data)

    # Apply throat adapter from project YAML (if present)
    if "throat_adapter" in raw and isinstance(raw["throat_adapter"], dict):
        ta = raw["throat_adapter"]
        if "ap1" in ta:
            horn.ap1 = float(ta["ap1"])
        if "lpt" in ta:
            horn.lpt = float(ta["lpt"])

    # Apply project overrides to HornGeometry
    if proj.driver_coord:
        horn.override_driver_coord(cast(tuple[float, float], tuple(proj.driver_coord)))
    if proj.fold_plot_segments:
        horn.override_folded_plot_segments([tuple(s) for s in proj.fold_plot_segments])

    # Apply rear chamber
    if proj.rear_chamber:
        rc = proj.rear_chamber
        horn.rear_chamber = rc
        if rc.vrc > 0:
            horn.vrc = rc.vrc
        if rc.lrc > 0:
            horn.lrc = rc.lrc
        if rc.fr_rc > 0:
            horn.fr_rc = rc.fr_rc

    # Apply vented box (bass reflex)
    if proj.vented_box:
        horn.vented_box = proj.vented_box

    # Apply passive radiator
    if proj.passive_radiator:
        horn.passive_radiator = proj.passive_radiator

    # Apply Slavic rear chamber (aperiodic box)
    if proj.slavbas:
        horn.slavbas = proj.slavbas

    # Apply throat chamber
    if proj.throat_chamber:
        tc = proj.throat_chamber
        if tc.vtc > 0:
            horn.vtc = tc.vtc
        if tc.atc > 0:
            horn.atc = tc.atc
        if tc.fr_tc > 0:
            horn.fr_tc = tc.fr_tc

    # Apply project-level width override
    if proj.width is not None:
        horn.width = proj.width
    # Apply project-level enclosure_dims override (for 2D schematic)
    if proj.enclosure is not None:
        horn.enclosure_dims = tuple(proj.enclosure)
    # Apply project-level sensitivity_db calibration for dB/W/m SPL
    if proj.sensitivity_db is not None:
        horn.sensitivity_db = proj.sensitivity_db
    return proj, horn


def _parse_sections(data: Dict[str, Any]) -> Optional[List[Section]]:
    """Parse chained profile sections from geometry data into a list of Section dataclasses.

    Accepts two YAML key names:
      - ``sections``        — the preferred key
      - ``profile_sections`` — legacy/alternative key with identical structure

    If neither key is present (legacy format), returns None.
    """
    raw_sections = data.get("sections")
    if raw_sections is None:
        # Support 'profile_sections' as an alias for 'sections'
        raw_sections = data.get("profile_sections")
    if raw_sections is None:
        return None
    sections = []
    for raw in raw_sections:
        sections.append(
            Section(
                name=str(raw["name"]),
                profile_type=str(raw["profile_type"]),
                length=float(raw["length"]),
                start_area=float(raw["start_area"]),
                end_area=float(raw["end_area"]),
                hyperbolic_t=float(raw["hyperbolic_t"])
                if raw.get("hyperbolic_t") is not None
                else None,
                fr1=float(raw["fr1"]) if raw.get("fr1") is not None else 0.0,
                tal1=float(raw["tal1"]) if raw.get("tal1") is not None else 0.0,
            )
        )
    return sections


def parse_horn_geometry(filepath: Path | str | dict) -> HornGeometry:
    """Parses horn geometry from a JSON or YAML file, or from a dict directly."""
    if isinstance(filepath, dict):
        data = filepath
        # Filter to only HornGeometry fields when called with inline project data
        horn_fields = HornGeometry.__dataclass_fields__.keys()
        data = {k: v for k, v in data.items() if k in horn_fields}
    else:
        filepath = Path(filepath)
        data = _load_file(filepath)
        # Filter out non-geometry fields (e.g. 'name', 'notes') before passing to HornGeometry
        horn_fields = HornGeometry.__dataclass_fields__.keys()
        data = {k: v for k, v in data.items() if k in horn_fields}
    _validate_horn_geometry(data)

    # Handle chained profile sections (new format)
    sections = _parse_sections(data)
    if sections is not None:
        data["sections"] = sections
        # Remove 'profile_sections' key so it doesn't get passed to HornGeometry
        # (it was only used as an alias for 'sections')
        data.pop("profile_sections", None)

    # Handle optional lists/tuples conversion if necessary
    if "segments" in data:
        data["segments"] = [tuple(seg) for seg in data["segments"]]
    if "bends" in data and data["bends"]:
        data["bends"] = [tuple(bend) for bend in data["bends"]]
    if "conical_segments" in data and data["conical_segments"]:
        data["conical_segments"] = [tuple(seg) for seg in data["conical_segments"]]
    if "rectangular_segments" in data and data["rectangular_segments"]:
        data["rectangular_segments"] = [
            tuple(seg) for seg in data["rectangular_segments"]
        ]
    if "coordinates" in data and data["coordinates"]:
        data["coordinates"] = [tuple(coord) for coord in data["coordinates"]]
    if "enclosure_dims" in data and data["enclosure_dims"]:
        data["enclosure_dims"] = tuple(data["enclosure_dims"])
    if "driver_coord" in data and data["driver_coord"]:
        data["driver_coord"] = tuple(data["driver_coord"])

    # Strip private keys (e.g. _center_offset, _throat_center) before passing to
    # HornGeometry — these are test metadata, not geometry fields.
    for key in list(data.keys()):
        if key.startswith("_"):
            del data[key]

    # Handle throat adapter — nested dict or top-level ap1/lpt fields
    throat_adapter_data = data.pop("throat_adapter", None)
    if throat_adapter_data and isinstance(throat_adapter_data, dict):
        from pyhorn_core.config.models import ThroatAdapter
        data["ap1"] = throat_adapter_data.get("ap1", data.get("ap1", 0.0))
        data["lpt"] = throat_adapter_data.get("lpt", data.get("lpt", 0.0))
        if "type" in throat_adapter_data:
            data["throat_adapter_type"] = str(throat_adapter_data["type"])

    # Handle vented box — nested dict → VentedBox dataclass
    # (also needed when parse_horn_geometry is called directly with a project YAML
    # that includes vented_box fields, as in the FastAPI /simulate endpoint)
    vented_box_data = data.get("vented_box")
    if vented_box_data and isinstance(vented_box_data, dict):
        from pyhorn_core.config.models import VentedBox
        data["vented_box"] = VentedBox(**vented_box_data)

    # Handle Slavic rear chamber — nested dict → SlavicBox dataclass
    slavbas_data = data.get("slavbas")
    if slavbas_data and isinstance(slavbas_data, dict):
        from pyhorn_core.config.models import SlavicBox
        data["slavbas"] = SlavicBox(**slavbas_data)

    # Handle rear chamber — nested dict → RearChamber dataclass
    # Supports rear_chamber: {vrc: ..., lrc: ..., fr_rc: ..., fr_tuning: ..., chamber_type: "vented"|"sealed"}
    rear_chamber_data = data.get("rear_chamber")
    horn_rear_chamber = None
    if rear_chamber_data and isinstance(rear_chamber_data, dict):
        from pyhorn_core.config.models import RearChamber
        horn_rear_chamber = RearChamber(**rear_chamber_data)
        data.pop("rear_chamber", None)  # HornGeometry has no rear_chamber field

    try:
        horn = HornGeometry(**data)
        if horn_rear_chamber is not None:
            horn.rear_chamber = horn_rear_chamber
            # Also populate top-level vrc/lrc/fr_rc fields so the
            # orchestrator's `if horn.vrc > 0` guard (which reads
            # horn.vrc, NOT horn.rear_chamber.vrc) works correctly.
            if horn_rear_chamber.vrc > 0:
                horn.vrc = horn_rear_chamber.vrc
            if horn_rear_chamber.lrc > 0:
                horn.lrc = horn_rear_chamber.lrc
            if horn_rear_chamber.fr_rc > 0:
                horn.fr_rc = horn_rear_chamber.fr_rc
        return horn
    except TypeError as e:
        # Catch remaining missing-field TypeErrors (e.g. an unknown field snuck
        # through, or a required field has no default).  Re-raise as ValueError
        # so callers always get a user-friendly message, never a raw traceback.
        raise ValueError(
            f"Failed to construct HornGeometry from {filepath}: {e}\n"
            f"Hint: check that all field names in your geometry YAML are correct "
            f"and match the pyhorn schema (see pyhorn documentation)."
        ) from e


def _load_file(filepath: Path) -> Dict[str, Any]:
    """Helper to load JSON or YAML files."""
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    if filepath.suffix.lower() == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    elif filepath.suffix.lower() in (".yaml", ".yml"):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"YAML parse error in {filepath}: {e}. "
                    f"Check for missing colons, incorrect indentation, or unquoted special characters."
                ) from e
    else:
        raise ValueError(
            f"Unsupported file extension: {filepath.suffix}. Use .json or .yaml"
        )
