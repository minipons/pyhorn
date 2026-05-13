"""Tests for pyhorn_core/solver/resize.py — Resize Wizard."""
import math
import tempfile
from pathlib import Path

import pytest
import yaml

from pyhorn_core.config.models import DriverSpecs, HornGeometry
from pyhorn_core.solver.resize import ResizeWizard, apply_resize


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_driver() -> DriverSpecs:
    """Fostex FE166NV2 T-S parameters (SI units)."""
    return DriverSpecs(
        fs=43.0,
        qts=0.21,
        qes=0.23,
        qms=2.5,
        vas=0.0214,        # m³
        re=6.4,            # Ohms
        bl=7.18,           # N/A
        mms=0.0055,        # kg
        cms=2.5e-4,        # m/N
        rms=0.59,          # kg/s
        sd=0.0133,         # m²
        voltage=2.83,
        le=0.00065,       # H
        xmax=0.003,       # m
    )


@pytest.fixture
def base_horn() -> HornGeometry:
    """Simple straight FLH horn geometry."""
    return HornGeometry(
        throat_area=0.004,   # m²
        mouth_area=0.04,    # m²
        path_length=0.5,    # m
        enclosure_type="FLH",
        vrc=0.01,          # m³
        vtc=0.0001,         # m³
        lpt=0.02,           # m
        ap1=0.004,          # m²
    )


@pytest.fixture
def folded_horn() -> HornGeometry:
    """Folded horn with rectangular_segments and coordinates."""
    return HornGeometry(
        enclosure_type="BLH",
        width=0.18,         # m — fixed cabinet width (should NOT scale)
        throat_area=0.004,
        mouth_area=0.04,
        path_length=0.5,
        rectangular_segments=[
            (0.18, 0.02, 0.18, 0.025, 0.05),
            (0.18, 0.025, 0.18, 0.03, 0.05),
            (0.18, 0.03, 0.18, 0.04, 0.05),
        ],
        coordinates=[
            (0.0, 0.0),
            (0.05, 0.02),
            (0.10, 0.025),
        ],
        enclosure_dims=(0.4, 0.6),
        driver_coord=(0.01, 0.123),
    )


# ── Geometry scalar field tests ─────────────────────────────────────────────

def test_area_scales_with_factor_squared(base_horn, base_driver):
    """Throat and mouth areas multiply by resize_factor²."""
    horn, _ = apply_resize(base_horn, base_driver, resize_factor=1.5)
    assert horn.throat_area == pytest.approx(base_horn.throat_area * 1.5**2)
    assert horn.mouth_area == pytest.approx(base_horn.mouth_area * 1.5**2)

    horn2, _ = apply_resize(base_horn, base_driver, resize_factor=0.8)
    assert horn2.throat_area == pytest.approx(base_horn.throat_area * 0.8**2)
    assert horn2.mouth_area == pytest.approx(base_horn.mouth_area * 0.8**2)


def test_length_scales_with_factor(base_horn, base_driver):
    """Horn path_length multiplies by resize_factor."""
    horn, _ = apply_resize(base_horn, base_driver, resize_factor=2.0)
    assert horn.path_length == pytest.approx(base_horn.path_length * 2.0)

    horn2, _ = apply_resize(base_horn, base_driver, resize_factor=0.5)
    assert horn2.path_length == pytest.approx(base_horn.path_length * 0.5)


def test_lpt_and_ap1_scale(base_horn, base_driver):
    """Throat adapter lpt × factor, ap1 × factor²."""
    horn, _ = apply_resize(base_horn, base_driver, resize_factor=1.5)
    assert horn.lpt == pytest.approx(base_horn.lpt * 1.5)
    assert horn.ap1 == pytest.approx(base_horn.ap1 * 1.5**2)


def test_chamber_volumes_scale_by_factor_cubed(base_horn, base_driver):
    """vrc and vtc multiply by resize_factor³."""
    horn, _ = apply_resize(base_horn, base_driver, resize_factor=1.5)
    assert horn.vrc == pytest.approx(base_horn.vrc * 1.5**3)
    assert horn.vtc == pytest.approx(base_horn.vtc * 1.5**3)


def test_lrc_scales_with_factor(base_horn, base_driver):
    """lrc (rear chamber acoustic length) multiplies by resize_factor."""
    horn, _ = apply_resize(base_horn, base_driver, resize_factor=1.5)
    assert horn.lrc == pytest.approx(base_horn.lrc * 1.5)


def test_atc_scales_with_factor_squared(base_horn, base_driver):
    """atc (throat chamber cross-sectional area) multiplies by resize_factor²."""
    horn, _ = apply_resize(base_horn, base_driver, resize_factor=1.5)
    assert horn.atc == pytest.approx(base_horn.atc * 1.5**2)


def test_width_unchanged(folded_horn, base_driver):
    """Cabinet width is a design constraint — it must NOT scale."""
    horn, _ = apply_resize(folded_horn, base_driver, resize_factor=2.0)
    assert horn.width == pytest.approx(folded_horn.width)


def test_rectangular_segments_scale_correctly(folded_horn, base_driver):
    """Rectangular segments: heights × factor² (area dims), widths unchanged, length × factor."""
    horn, _ = apply_resize(folded_horn, base_driver, resize_factor=1.5)

    orig = folded_horn.rectangular_segments
    scaled = horn.rectangular_segments

    assert len(scaled) == len(orig)
    for i, (orig_seg, scaled_seg) in enumerate(zip(orig, scaled)):
        # Width stays same (fixed cabinet width)
        assert scaled_seg[0] == pytest.approx(orig_seg[0])
        assert scaled_seg[2] == pytest.approx(orig_seg[2])
        # Height × factor² (area dimension)
        assert scaled_seg[1] == pytest.approx(orig_seg[1] * 1.5**2)
        assert scaled_seg[3] == pytest.approx(orig_seg[3] * 1.5**2)
        # Length × factor
        assert scaled_seg[4] == pytest.approx(orig_seg[4] * 1.5)


def test_coordinates_scale_with_factor(folded_horn, base_driver):
    """Coordinates (x, y) both scale by resize_factor."""
    horn, _ = apply_resize(folded_horn, base_driver, resize_factor=2.0)

    for (ox, oy), (hx, hy) in zip(folded_horn.coordinates, horn.coordinates):
        assert hx == pytest.approx(ox * 2.0)
        assert hy == pytest.approx(oy * 2.0)


def test_enclosure_dims_scale_with_factor(folded_horn, base_driver):
    """Enclosure dims (depth, height) both scale by resize_factor."""
    horn, _ = apply_resize(folded_horn, base_driver, resize_factor=1.5)

    assert horn.enclosure_dims[0] == pytest.approx(folded_horn.enclosure_dims[0] * 1.5)
    assert horn.enclosure_dims[1] == pytest.approx(folded_horn.enclosure_dims[1] * 1.5)


def test_driver_coord_scales_with_factor(folded_horn, base_driver):
    """Driver coordinate (x, y) scales by resize_factor."""
    horn, _ = apply_resize(folded_horn, base_driver, resize_factor=1.5)

    assert horn.driver_coord[0] == pytest.approx(folded_horn.driver_coord[0] * 1.5)
    assert horn.driver_coord[1] == pytest.approx(folded_horn.driver_coord[1] * 1.5)


# ── Driver tests ──────────────────────────────────────────────────────────────

def test_sd_scales_with_factor_squared(base_horn, base_driver):
    """Driver piston area Sd multiplies by resize_factor²."""
    _, driver = apply_resize(base_horn, base_driver, resize_factor=1.5, adjust_sd=True)
    assert driver.sd == pytest.approx(base_driver.sd * 1.5**2)


def test_sd_not_scaled_when_disabled(base_horn, base_driver):
    """Driver Sd is unchanged when adjust_sd=False."""
    _, driver = apply_resize(base_horn, base_driver, resize_factor=2.0, adjust_sd=False)
    assert driver.sd == pytest.approx(base_driver.sd)


def test_re_unchanged_by_default(base_horn, base_driver):
    """Driver Re is unchanged by default (same driver, size doesn't change Re)."""
    _, driver = apply_resize(base_horn, base_driver, resize_factor=2.0)
    assert driver.re == pytest.approx(base_driver.re)


def test_re_scales_when_adjust_re_true(base_horn, base_driver):
    """When adjust_re=True, Re multiplies by resize_factor².

    Use this when swapping to a different driver sized for the scaled horn.
    For the same driver in a larger/smaller horn, leave adjust_re=False (default).
    """
    _, driver = apply_resize(base_horn, base_driver, resize_factor=2.0, adjust_re=True)
    assert driver.re == pytest.approx(base_driver.re * 2.0**2)


def test_mmd_unchanged(base_horn, base_driver):
    """Driver moving mass (Mms) is unchanged by resize — it's a material property."""
    _, driver = apply_resize(base_horn, base_driver, resize_factor=1.5)
    assert driver.mms == pytest.approx(base_driver.mms)


def test_other_ts_unchanged(base_horn, base_driver):
    """All other T-S parameters (BL, CMS, RMS, VAS, fs, Qts…) are unchanged."""
    _, driver = apply_resize(base_horn, base_driver, resize_factor=2.0)
    assert driver.bl == pytest.approx(base_driver.bl)
    assert driver.cms == pytest.approx(base_driver.cms)
    assert driver.rms == pytest.approx(base_driver.rms)
    assert driver.vas == pytest.approx(base_driver.vas)
    assert driver.fs == pytest.approx(base_driver.fs)
    assert driver.qts == pytest.approx(base_driver.qts)


# ── ResizeWizard dataclass ─────────────────────────────────────────────────────

def test_resize_wizard_defaults():
    """ResizeWizard has the right defaults."""
    rw = ResizeWizard(resize_factor=1.5)
    assert rw.resize_factor == 1.5
    assert rw.adjust_sd is True
    assert rw.adjust_re is False


# ── Error handling ────────────────────────────────────────────────────────────

def test_resize_factor_zero_raises(base_horn, base_driver):
    """resize_factor=0 must raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        apply_resize(base_horn, base_driver, resize_factor=0.0)


def test_resize_factor_negative_raises(base_horn, base_driver):
    """Negative resize_factor must raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        apply_resize(base_horn, base_driver, resize_factor=-1.0)


# ── Round-trip YAML test ───────────────────────────────────────────────────────

def test_resize_round_trip_yaml(base_horn, base_driver):
    """Read project YAML, resize, write, read back → correct values."""
    # Build a minimal geometry YAML dict
    geo_data = {
        "enclosure_type": "FLH",
        "throat_area": 0.004,
        "mouth_area": 0.04,
        "path_length": 0.5,
        "vrc": 0.01,
        "vtc": 0.0001,
        "lpt": 0.02,
        "ap1": 0.004,
        "rectangular_segments": [
            [0.18, 0.02, 0.18, 0.025, 0.05],
        ],
        "coordinates": [[0.0, 0.0], [0.05, 0.02]],
        "enclosure_dims": [0.4, 0.6],
        "driver_coord": [0.01, 0.123],
    }
    driver_data = {
        "fs": 43.0, "qts": 0.21, "qes": 0.23, "qms": 2.5,
        "vas": 0.0214, "re": 6.4, "bl": 7.18, "mms": 0.0055,
        "cms": 2.5e-4, "rms": 0.59, "sd": 0.0133,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        geo_path = Path(tmpdir) / "horn.yaml"
        driver_path = Path(tmpdir) / "driver.yaml"

        yaml.safe_dump(geo_data, geo_path.open("w"))
        yaml.safe_dump(driver_data, driver_path.open("w"))

        # Parse
        from pyhorn_core.config.parser import parse_driver_specs, parse_horn_geometry
        horn_parsed = parse_horn_geometry(geo_path)
        driver_parsed = parse_driver_specs(driver_path)

        # Resize
        factor = 1.5
        resized_geo, resized_driver = apply_resize(
            horn_parsed, driver_parsed, factor,
            adjust_sd=True, adjust_re=False
        )

        # Write
        out_geo_path = Path(tmpdir) / "resized_horn.yaml"
        out_driver_path = Path(tmpdir) / "resized_driver.yaml"

        yaml.safe_dump({
            "enclosure_type": resized_geo.enclosure_type,
            "throat_area": resized_geo.throat_area,
            "mouth_area": resized_geo.mouth_area,
            "path_length": resized_geo.path_length,
            "vrc": resized_geo.vrc,
            "vtc": resized_geo.vtc,
            "lpt": resized_geo.lpt,
            "ap1": resized_geo.ap1,
            "rectangular_segments": resized_geo.rectangular_segments,
            "coordinates": resized_geo.coordinates,
            "enclosure_dims": resized_geo.enclosure_dims,
            "driver_coord": resized_geo.driver_coord,
        }, out_geo_path.open("w"))

        yaml.safe_dump({
            "fs": resized_driver.fs, "qts": resized_driver.qts,
            "qes": resized_driver.qes, "qms": resized_driver.qms,
            "vas": resized_driver.vas, "re": resized_driver.re,
            "bl": resized_driver.bl, "mms": resized_driver.mms,
            "cms": resized_driver.cms, "rms": resized_driver.rms,
            "sd": resized_driver.sd,
        }, out_driver_path.open("w"))

        # Read back
        horn_back = parse_horn_geometry(out_geo_path)
        driver_back = parse_driver_specs(out_driver_path)

        # Verify
        assert horn_back.throat_area == pytest.approx(geo_data["throat_area"] * factor**2)
        assert horn_back.mouth_area == pytest.approx(geo_data["mouth_area"] * factor**2)
        assert horn_back.path_length == pytest.approx(geo_data["path_length"] * factor)
        assert horn_back.vrc == pytest.approx(geo_data["vrc"] * factor**3)
        assert horn_back.lpt == pytest.approx(geo_data["lpt"] * factor)
        assert horn_back.ap1 == pytest.approx(geo_data["ap1"] * factor**2)

        # Rectangular segment: height × factor²
        assert horn_back.rectangular_segments[0][1] == pytest.approx(
            geo_data["rectangular_segments"][0][1] * factor**2
        )
        assert horn_back.rectangular_segments[0][4] == pytest.approx(
            geo_data["rectangular_segments"][0][4] * factor
        )

        # Coordinates × factor
        assert horn_back.coordinates[0][0] == pytest.approx(geo_data["coordinates"][0][0] * factor)
        assert horn_back.coordinates[0][1] == pytest.approx(geo_data["coordinates"][0][1] * factor)

        # Driver
        assert driver_back.sd == pytest.approx(driver_data["sd"] * factor**2)
        assert driver_back.re == pytest.approx(driver_data["re"])  # unchanged
        assert driver_back.mms == pytest.approx(driver_data["mms"])  # unchanged


def test_conical_segments_scale(base_horn, base_driver):
    """conical_segments: linear dims × factor, length × factor."""
    horn = HornGeometry(
        enclosure_type="FLH",
        conical_segments=[
            (0.05, 0.06, 0.1, 5000.0),
            (0.06, 0.08, 0.1, 5000.0),
        ],
        width=0.15,  # fixed
    )
    resized, _ = apply_resize(horn, base_driver, resize_factor=2.0)

    assert resized.conical_segments[0][0] == pytest.approx(0.05 * 2.0)  # dim_start
    assert resized.conical_segments[0][1] == pytest.approx(0.06 * 2.0)  # dim_end
    assert resized.conical_segments[0][2] == pytest.approx(0.1 * 2.0)    # length
    assert resized.conical_segments[0][3] == pytest.approx(5000.0)        # fr unchanged


def test_legacy_segments_scale(base_horn, base_driver):
    """Legacy segments: length × factor, area × factor²."""
    horn = HornGeometry(
        enclosure_type="FLH",
        segments=[
            (0.1, 0.005, 5000.0),
            (0.1, 0.01, 5000.0),
        ],
    )
    resized, _ = apply_resize(horn, base_driver, resize_factor=1.5)

    assert resized.segments[0][0] == pytest.approx(0.1 * 1.5)     # length
    assert resized.segments[0][1] == pytest.approx(0.005 * 1.5**2) # area
    assert resized.segments[0][2] == pytest.approx(5000.0)          # fr unchanged


def test_bends_scale(base_horn, base_driver):
    """Bends: both area values × factor²."""
    horn = HornGeometry(
        enclosure_type="FLH",
        bends=[(0.005, 0.008), (0.008, 0.015)],
    )
    resized, _ = apply_resize(horn, base_driver, resize_factor=1.5)

    assert resized.bends[0] == (0.005 * 1.5**2, 0.008 * 1.5**2)
    assert resized.bends[1] == (0.008 * 1.5**2, 0.015 * 1.5**2)


def test_resize_factor_one_is_identity(base_horn, base_driver):
    """resize_factor=1 should return identical geometry and driver."""
    horn, driver = apply_resize(base_horn, base_driver, resize_factor=1.0)

    assert horn.throat_area == pytest.approx(base_horn.throat_area)
    assert horn.mouth_area == pytest.approx(base_horn.mouth_area)
    assert horn.path_length == pytest.approx(base_horn.path_length)
    assert horn.vrc == pytest.approx(base_horn.vrc)
    assert driver.sd == pytest.approx(base_driver.sd)
    assert driver.re == pytest.approx(base_driver.re)
    assert driver.mms == pytest.approx(base_driver.mms)
