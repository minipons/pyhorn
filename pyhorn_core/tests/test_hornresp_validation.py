"""Validate pyhorn simulation against Hornresp using the Hiro BLH design.

test1.txt contains the complete Hiro design: geometry + driver T/S parameters.
The test loads the horn geometry from test1.txt, runs pyhorn's simulation,
and (once the response CSV is exported from Hornresp) asserts convergence.

Export reference data from Hornresp:
  File → Export response data → CSV → tests/hornresp_data/hiro/response.csv
  Columns: frequency, spl, re_z, im_z
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorn_core.config.models import DriverSpecs, HornGeometry
from pyhorn_core.solver.hornresp_parser import load_hornresp_project
from pyhorn_core.solver.models import horn_response


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA = Path(__file__).parent / "hornresp_data"
TEST1 = DATA / "test1.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Hornresp reference loader (skips if CSV not yet exported)
# ─────────────────────────────────────────────────────────────────────────────

def _load_reference(name: str):
    csv_path = DATA / name / "response.csv"
    if not csv_path.exists():
        return None
    import csv as csv_lib
    freq, spl, re_z, im_z = [], [], [], []
    with open(csv_path) as f:
        for row in csv_lib.DictReader(f):
            freq.append(float(row["frequency"]))
            spl.append(float(row["spl"]))
            re_z.append(float(row["re_z"]))
            im_z.append(float(row["im_z"]))
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Ref:
        freq: np.ndarray; spl: np.ndarray; re_z: np.ndarray; im_z: np.ndarray
    return Ref(freq=np.array(freq), spl=np.array(spl),
                re_z=np.array(re_z), im_z=np.array(im_z))


# ─────────────────────────────────────────────────────────────────────────────
# Driver: FE166NV2 (rated specs — pyhorn's canonical values)
# ─────────────────────────────────────────────────────────────────────────────

FE166NV2 = DriverSpecs(
    fs=49.6, qts=0.27, qes=0.28, qms=7.88,
    vas=0.0369, re=7.8, bl=7.79,
    mms=0.00699, cms=0.001472, rms=0.277,
    sd=0.01327, le=0.0008, xmax=0.001,
    voltage=2.83,
)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry from test1.txt
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def hiro_geometry() -> HornGeometry:
    geo, _ = load_hornresp_project(TEST1)
    return geo


# ─────────────────────────────────────────────────────────────────────────────
# Simulation result
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def hiro_result(hiro_geometry) -> "SimulationResult":
    freq = np.linspace(20.0, 500.0, 481)
    return horn_response(freq, FE166NV2, hiro_geometry)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry checks
# ─────────────────────────────────────────────────────────────────────────────

class TestHiroGeometry:
    """Verify test1.txt → HornGeometry conversion is correct."""

    def test_throat_40_cm2(self, hiro_geometry):
        assert hiro_geometry.throat_area == pytest.approx(40.0 / 10000, rel=1e-3)

    def test_mouth_300_cm2(self, hiro_geometry):
        assert hiro_geometry.mouth_area == pytest.approx(300.0 / 10000, rel=1e-3)

    def test_path_1527_m(self, hiro_geometry):
        assert hiro_geometry.path_length == pytest.approx(1.527, rel=1e-3)

    def test_hyperbolic_t_0_3(self, hiro_geometry):
        assert hiro_geometry.hyperbolic_t == pytest.approx(0.30)

    def test_blh_enclosure(self, hiro_geometry):
        assert hiro_geometry.enclosure_type.upper() == "BLH"

    def test_rear_chamber_3_24_L(self, hiro_geometry):
        assert hiro_geometry.vrc == pytest.approx(3.24e-3, rel=1e-3)

    def test_throat_chamber_88_cm3(self, hiro_geometry):
        assert hiro_geometry.vtc == pytest.approx(88.0e-6, rel=1e-3)

    def test_throat_chamber_atc_250_cm2(self, hiro_geometry):
        assert hiro_geometry.atc == pytest.approx(250.18 / 10000, rel=1e-3)

    def test_profile_hyperbolic(self, hiro_geometry):
        assert hiro_geometry.profile_type == "Hyperbolic"


# ─────────────────────────────────────────────────────────────────────────────
# Simulation sanity checks (always run)
# ─────────────────────────────────────────────────────────────────────────────

class TestHiroDiagnostics:
    """Sanity checks on pyhorn's hiro simulation."""

    def test_spl_in_reasonable_range(self, hiro_result):
        band = (hiro_result.freqs >= 80) & (hiro_result.freqs <= 300)
        mean_spl = float(np.mean(hiro_result.spl[band]))
        assert 70 < mean_spl < 115

    def test_impedance_peak_near_fs(self, hiro_result):
        z_mag = np.abs(hiro_result.impedance)
        peak_idx = int(np.argmax(z_mag[:80]))
        peak_freq = float(hiro_result.freqs[peak_idx])
        assert 35 < peak_freq < 75

    def test_spl_rolloff_below_fs(self, hiro_result):
        below = (hiro_result.freqs >= 20) & (hiro_result.freqs <= 40)
        above = (hiro_result.freqs >= 80) & (hiro_result.freqs <= 150)
        spl_below = float(np.mean(hiro_result.spl[below]))
        spl_above = float(np.mean(hiro_result.spl[above]))
        assert spl_below < spl_above - 5


# ─────────────────────────────────────────────────────────────────────────────
# Convergence against Hornresp reference
# ─────────────────────────────────────────────────────────────────────────────

class TestHiroConvergence:
    """Compare pyhorn output against Hornresp reference CSV.

    Once response.csv is exported from Hornresp, these tests activate.
    """

    @pytest.fixture
    def reference(self):
        return _load_reference("hiro")

    def test_spl_within_1dB(self, hiro_result, reference):
        if reference is None:
            pytest.skip("No response.csv for hiro — export from Hornresp first")
        aligned = np.interp(
            hiro_result.freqs, reference.freq, reference.spl,
            left=np.nan, right=np.nan,
        )
        valid = ~np.isnan(aligned)
        err = aligned[valid] - hiro_result.spl[valid]
        max_err = float(np.max(np.abs(err)))
        rms_err = float(np.sqrt(np.mean(err ** 2)))
        assert max_err <= 1.0, (
            f"SPL max error {max_err:.2f} dB (rms={rms_err:.2f} dB)"
        )

    def test_spl_within_2dB(self, hiro_result, reference):
        if reference is None:
            pytest.skip("No response.csv for hiro")
        aligned = np.interp(
            hiro_result.freqs, reference.freq, reference.spl,
            left=np.nan, right=np.nan,
        )
        valid = ~np.isnan(aligned)
        err = aligned[valid] - hiro_result.spl[valid]
        max_err = float(np.max(np.abs(err)))
        assert max_err <= 2.0, f"SPL max error {max_err:.2f} dB"

    def test_impedance_within_1ohm(self, hiro_result, reference):
        if reference is None:
            pytest.skip("No response.csv for hiro")
        ref_z = np.sqrt(reference.re_z ** 2 + reference.im_z ** 2)
        aligned = np.interp(
            hiro_result.freqs, reference.freq, ref_z,
            left=np.nan, right=np.nan,
        )
        valid = ~np.isnan(aligned)
        z_err = aligned[valid] - np.abs(hiro_result.impedance[valid])
        max_err = float(np.max(np.abs(z_err)))
        assert max_err <= 1.0, f"|Z| max error {max_err:.2f} ohm"
