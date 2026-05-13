"""
Unit tests for filter_schematic.py — ASCII schematic generation.

Covers:
  - generate_schematic() with all presets and types
  - compute_filter_schematic() with FilterBand lists and PassiveFilter instances
  - PRESETS dict
  - Edge cases and error handling
"""

import math
import numpy as np
import pytest

from pyhorn_core.solver.filter_schematic import (
    FilterBand,
    PassiveComponent,
    PassiveFilter,
    PRESETS,
    compute_filter_schematic,
    generate_schematic,
)


# ─── TestPresetsDict ─────────────────────────────────────────────────────────

class TestPresetsDict:
    def test_all_presets_present(self):
        expected = {"le_cleach", "2way_xover", "3way_xover", "peaking_eq", "highshelf", "lowshelf"}
        assert set(PRESETS.keys()) == expected

    def test_preset_structure(self):
        for name, p in PRESETS.items():
            assert "filter_type" in p
            assert "description" in p
            assert isinstance(p["description"], str)

    def test_le_cleach_defaults(self):
        p = PRESETS["le_cleach"]
        assert p["filter_type"] == "le_cleach"
        assert p["default_fc"] == 80.0
        assert p["default_q"] == 0.7
        assert p["default_r_load"] == 8.0

    def test_2way_xover_defaults(self):
        p = PRESETS["2way_xover"]
        assert p["filter_type"] == "lr2_crossover_2way"
        assert p["default_fc"] == 3000.0
        assert p["default_q"] == 0.707

    def test_3way_xover_defaults(self):
        p = PRESETS["3way_xover"]
        assert p["filter_type"] == "lr2_crossover_3way"
        assert p["default_fc1"] == 400.0
        assert p["default_fc2"] == 4000.0

    def test_peaking_eq_defaults(self):
        p = PRESETS["peaking_eq"]
        assert p["filter_type"] == "peaking_eq"
        assert p["default_fc"] == 2500.0
        assert p["default_q"] == 1.4
        assert p["default_gain_db"] == 3.0

    def test_highshelf_defaults(self):
        p = PRESETS["highshelf"]
        assert p["filter_type"] == "highshelf"
        assert p["default_fc"] == 4000.0
        assert p["default_q"] == 0.707
        assert p["default_gain_db"] == -3.0

    def test_lowshelf_defaults(self):
        p = PRESETS["lowshelf"]
        assert p["filter_type"] == "lowshelf"
        assert p["default_fc"] == 200.0
        assert p["default_q"] == 0.707
        assert p["default_gain_db"] == 3.0


# ─── TestGenerateSchematicPresets ─────────────────────────────────────────────

class TestGenerateSchematicPresets:
    def test_le_cleach_preset(self):
        asc = generate_schematic(preset="le_cleach")
        assert isinstance(asc, str)
        assert "Le Cléac'h" in asc
        assert "High-Pass" in asc

    def test_2way_xover_preset(self):
        asc = generate_schematic(preset="2way_xover")
        assert isinstance(asc, str)
        assert "LR2" in asc
        assert "2-Way" in asc

    def test_3way_xover_preset(self):
        asc = generate_schematic(preset="3way_xover")
        assert isinstance(asc, str)
        assert "LR2" in asc
        assert "3-Way" in asc

    def test_peaking_eq_preset(self):
        asc = generate_schematic(preset="peaking_eq")
        assert isinstance(asc, str)
        assert "Peaking" in asc

    def test_highshelf_preset(self):
        asc = generate_schematic(preset="highshelf")
        assert isinstance(asc, str)
        assert "High-Shelf" in asc

    def test_lowshelf_preset(self):
        asc = generate_schematic(preset="lowshelf")
        assert isinstance(asc, str)
        assert "Low-Shelf" in asc


# ─── TestGenerateSchematicTypes ───────────────────────────────────────────────

class TestGenerateSchematicTypes:
    def test_type_le_cleach(self):
        asc = generate_schematic(type="le_cleach", fc=100.0, q=0.71, r_load=8.0)
        assert "Le Cléac'h" in asc
        assert "100.0 Hz" in asc or "100.00 Hz" in asc

    def test_type_lr2_crossover_2way(self):
        asc = generate_schematic(type="lr2_crossover_2way", fc=2000.0, r_load=8.0)
        assert "LR2" in asc
        assert "2-Way" in asc

    def test_type_lr2_crossover_3way(self):
        asc = generate_schematic(type="lr2_crossover_3way", fc1=400.0, fc2=4000.0, r_load=8.0)
        assert "LR2" in asc
        assert "3-Way" in asc

    def test_type_peaking_eq(self):
        asc = generate_schematic(type="peaking_eq", fc=1000.0, q=1.0, gain_db=3.0)
        assert "Peaking" in asc

    def test_type_highshelf(self):
        asc = generate_schematic(type="highshelf", fc=5000.0, q=0.707, gain_db=-6.0)
        assert "High-Shelf" in asc

    def test_type_lowshelf(self):
        asc = generate_schematic(type="lowshelf", fc=100.0, q=0.707, gain_db=6.0)
        assert "Low-Shelf" in asc

    def test_type_lowpass(self):
        asc = generate_schematic(type="lowpass", fc=500.0, q=0.707, order=2, r_load=8.0)
        assert "Low-Pass" in asc

    def test_type_highpass(self):
        asc = generate_schematic(type="highpass", fc=800.0, q=0.707, order=2, r_load=8.0)
        assert "High-Pass" in asc

    def test_type_bandpass(self):
        asc = generate_schematic(type="bandpass", fc=1000.0, q=1.0)
        assert "Band-Pass" in asc


# ─── TestGenerateSchematicOverrides ───────────────────────────────────────────

class TestGenerateSchematicOverrides:
    def test_preset_overrides_fc(self):
        # Override default fc=80 with fc=200
        asc = generate_schematic(preset="le_cleach", fc=200.0)
        assert "200.0 Hz" in asc or "200.00 Hz" in asc

    def test_preset_overrides_q(self):
        asc = generate_schematic(preset="le_cleach", q=1.0)
        assert "1.00" in asc  # Q appears in the schematic

    def test_preset_overrides_gain_db(self):
        asc = generate_schematic(preset="peaking_eq", gain_db=6.0)
        assert "+6.0 dB" in asc or "6.0 dB" in asc


# ─── TestGenerateSchematicErrors ──────────────────────────────────────────────

class TestGenerateSchematicErrors:
    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            generate_schematic(preset="nonexistent_preset")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown filter type"):
            generate_schematic(type="not_a_real_type")

    def test_no_preset_no_type_raises(self):
        with pytest.raises(ValueError, match="Either preset= or type="):
            generate_schematic()


# ─── TestComputeFilterSchematic ───────────────────────────────────────────────

class TestComputeFilterSchematic:
    def test_single_filter_band_le_cleach(self):
        bands = [FilterBand(type="le_cleach", frequency=80.0, q=0.7)]
        asc = compute_filter_schematic(bands)
        assert "Le Cléac'h" in asc

    def test_single_filter_band_lowpass(self):
        bands = [FilterBand(type="lowpass", frequency=500.0, q=0.707)]
        asc = compute_filter_schematic(bands)
        assert "Low-Pass" in asc

    def test_single_filter_band_highpass(self):
        bands = [FilterBand(type="highpass", frequency=2000.0, q=0.707)]
        asc = compute_filter_schematic(bands)
        assert "High-Pass" in asc

    def test_single_filter_band_peaking_eq(self):
        bands = [FilterBand(type="peaking_eq", frequency=2500.0, q=1.4, gain_db=3.0)]
        asc = compute_filter_schematic(bands)
        assert "Peaking" in asc

    def test_single_filter_band_highshelf(self):
        bands = [FilterBand(type="highshelf", frequency=4000.0, q=0.707, gain_db=-3.0)]
        asc = compute_filter_schematic(bands)
        assert "High-Shelf" in asc

    def test_single_filter_band_lowshelf(self):
        bands = [FilterBand(type="lowshelf", frequency=200.0, q=0.707, gain_db=3.0)]
        asc = compute_filter_schematic(bands)
        assert "Low-Shelf" in asc

    def test_all_bands_disabled(self):
        bands = [
            FilterBand(type="lowpass", frequency=500.0, enabled=False),
            FilterBand(type="highpass", frequency=500.0, enabled=False),
        ]
        asc = compute_filter_schematic(bands)
        assert asc == "No filter bands enabled."

    def test_2way_crossover(self):
        bands = [
            FilterBand(type="lowpass", frequency=1000.0),
            FilterBand(type="highpass", frequency=1000.0),
        ]
        asc = compute_filter_schematic(bands)
        assert "LOW-PASS" in asc
        assert "HIGH-PASS" in asc

    def test_3way_crossover(self):
        bands = [
            FilterBand(type="lowpass", frequency=400.0),
            FilterBand(type="highpass", frequency=400.0),
            FilterBand(type="bandpass", frequency=2000.0),
        ]
        asc = compute_filter_schematic(bands)
        assert "LOW-PASS" in asc
        assert "HIGH-PASS" in asc
        assert "BAND-PASS" in asc or "Band-Pass" in asc

    def test_passive_filter_instance(self):
        filt = PassiveFilter(
            topology="series",
            components=[PassiveComponent("R", 8.0)],
        )
        asc = compute_filter_schematic(filt)
        assert isinstance(asc, str)
        assert "Passive" in asc

    def test_passive_filter_with_r_load(self):
        filt = PassiveFilter(
            topology="series",
            components=[PassiveComponent("R", 4.0)],
        )
        asc = compute_filter_schematic(filt, r_load=8.0)
        assert "8.0" in asc  # r_load=8.0 appears in schematic

    def test_empty_bands_raises(self):
        # No bands at all → uses [] path
        asc = compute_filter_schematic([])
        assert asc == "No filter bands enabled."


# ─── TestFilterBand ───────────────────────────────────────────────────────────

class TestFilterBand:
    def test_defaults(self):
        fb = FilterBand(type="lowpass", frequency=1000.0)
        assert fb.q == 1.0
        assert fb.gain_db == 0.0
        assert fb.order == 2
        assert fb.enabled is True

    def test_explicit_fields(self):
        fb = FilterBand(
            type="peaking_eq",
            frequency=2500.0,
            q=2.0,
            gain_db=6.0,
            order=4,
            enabled=False,
        )
        assert fb.type == "peaking_eq"
        assert fb.frequency == 2500.0
        assert fb.q == 2.0
        assert fb.gain_db == 6.0
        assert fb.order == 4
        assert fb.enabled is False


# ─── TestSchematicContent ─────────────────────────────────────────────────────

class TestSchematicContent:
    def test_le_cleach_contains_parameters(self):
        asc = generate_schematic(preset="le_cleach", fc=80.0, q=0.7, r_load=8.0)
        assert "80.0 Hz" in asc
        assert "0.70" in asc
        assert "8.0" in asc

    def test_lr2_2way_contains_fc(self):
        asc = generate_schematic(preset="2way_xover", fc=3000.0, r_load=8.0)
        assert "3000.0 Hz" in asc or "3.00 kHz" in asc

    def test_lr2_3way_contains_both_fcs(self):
        asc = generate_schematic(preset="3way_xover", fc1=400.0, fc2=4000.0, r_load=8.0)
        assert "400.0 Hz" in asc or "400.00 Hz" in asc
        assert "4000.0 Hz" in asc or "4.00 kHz" in asc

    def test_peaking_eq_contains_gain(self):
        asc = generate_schematic(preset="peaking_eq", fc=2500.0, q=1.4, gain_db=3.0)
        assert "+3.0 dB" in asc or "3.0 dB" in asc

    def test_schematic_returns_string(self):
        for preset in PRESETS:
            asc = generate_schematic(preset=preset)
            assert isinstance(asc, str)
            assert len(asc) > 10  # non-trivial output
