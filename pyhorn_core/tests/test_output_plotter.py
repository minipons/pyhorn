"""Unit tests for pyhorn.output.plotter."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from pyhorn_core.output.plotter import (
    plot_horn_2d_folded,
    plot_horn_3d,
    plot_impulse_step,
    plot_simulation_results,
    plot_waterfall,
)


class TestPlotSimulationResults:
    """Tests for plot_simulation_results()."""

    @pytest.fixture
    def sim_result(self):
        from pyhorn_core.solver.models import SimulationResult

        freqs = np.array(
            [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0, 10000.0, 20000.0]
        )
        return SimulationResult(
            freqs=freqs,
            spl=np.array([60.0, 75.0, 82.0, 88.0, 92.0, 95.0, 94.0, 91.0, 88.0]),
            impedance=np.array([7.8, 8.1, 9.5, 12.0, 18.0, 35.0, 22.0, 12.0, 9.0]),
            excursion=np.array([0.5, 1.2, 2.1, 3.5, 5.0, 4.2, 1.8, 0.9, 0.4]),
            direct_spl=np.array([55.0, 68.0, 75.0, 80.0, 82.0, 80.0, 75.0, 70.0, 65.0]),
            horn_spl=np.array([58.0, 72.0, 79.0, 85.0, 90.0, 93.0, 92.0, 89.0, 86.0]),
            group_delay=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.5, 3.0, 2.0, 1.0]),
        )

    def test_creates_png_file(self, sim_result, tmp_path):
        """Should create a PNG output file."""
        out = tmp_path / "sim.png"
        plot_simulation_results(sim_result, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_accepts_custom_title(self, sim_result, tmp_path):
        """Custom title should not cause an error."""
        out = tmp_path / "sim.png"
        plot_simulation_results(sim_result, out, title="My Custom Horn")
        assert out.exists()

    def test_target_spl_adds_axhline(self, sim_result, tmp_path):
        """target_spl parameter should not cause an error."""
        out = tmp_path / "sim.png"
        plot_simulation_results(sim_result, out, target_spl=90.0)
        assert out.exists()

    def test_target_impedance(self, sim_result, tmp_path):
        """target_impedance parameter should not cause an error."""
        out = tmp_path / "sim.png"
        plot_simulation_results(sim_result, out, target_impedance=8.0)
        assert out.exists()

    def test_target_excursion(self, sim_result, tmp_path):
        """target_excursion parameter should not cause an error."""
        out = tmp_path / "sim.png"
        plot_simulation_results(sim_result, out, target_excursion=5.0)
        assert out.exists()

    def test_result_without_optional_fields(self, tmp_path):
        """Result with only required fields should still plot."""
        from pyhorn_core.solver.models import SimulationResult

        freqs = np.array([100.0, 1000.0, 10000.0])
        result = SimulationResult(
            freqs=freqs,
            spl=np.array([80.0, 90.0, 85.0]),
            impedance=np.array([8.0, 30.0, 10.0]),
            excursion=np.array([1.0, 3.0, 1.0]),
        )
        out = tmp_path / "sim_min.png"
        plot_simulation_results(result, out)
        assert out.exists()

    def test_result_with_ib_spl(self, tmp_path):
        """Result with ib_spl should not cause an error."""
        from pyhorn_core.solver.models import SimulationResult

        freqs = np.array([100.0, 1000.0])
        result = SimulationResult(
            freqs=freqs,
            spl=np.array([80.0, 90.0]),
            impedance=np.array([8.0, 30.0]),
            excursion=np.array([1.0, 3.0]),
            ib_spl=np.array([78.0, 88.0]),
        )
        out = tmp_path / "sim_ib.png"
        plot_simulation_results(result, out)
        assert out.exists()


class TestPlotHorn3d:
    """Tests for plot_horn_3d()."""

    @pytest.fixture
    def simple_segments(self):
        """Two simple conical segments."""
        return [
            (0.05, 0.005, 0.01),  # (dim_start, dim_end, length)
            (0.005, 0.02, 0.05),  # (dim_start, dim_end, length)
        ]

    def test_creates_png_file(self, simple_segments, tmp_path):
        """Should create a PNG output file."""
        out = tmp_path / "horn3d.png"
        plot_horn_3d(simple_segments, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_segments_returns_early(self, simple_segments, tmp_path):
        """ "Empty segment list should return without raising and not create a file."""
        out = tmp_path / "horn3d_empty.png"
        plot_horn_3d([], out)  # should not raise
        # Function returns early without creating output
        assert not out.exists()

    def test_width_parameter_sets_rectangular_horn(self, simple_segments, tmp_path):
        """width parameter should be accepted without error."""
        out = tmp_path / "horn3d_w.png"
        plot_horn_3d(simple_segments, out, width=0.2)
        assert out.exists()

    def test_width_profile_parameter(self, simple_segments, tmp_path):
        """width_profile parameter should be accepted without error."""
        out = tmp_path / "horn3d_wp.png"
        plot_horn_3d(simple_segments, out, width_profile=[0.2, 0.15, 0.2])
        assert out.exists()


class TestPlotHorn2dFolded:
    """Tests for plot_horn_2d_folded()."""

    @pytest.fixture
    def simple_coords(self):
        """Simple 2D centerline coordinates."""
        return [
            (0.1, 0.2),
            (0.2, 0.2),
            (0.2, 0.3),
        ]

    @pytest.fixture
    def simple_conical_segments(self):
        """Simple conical segments matching the coordinates."""
        return [
            (0.05, 0.06, 0.1),  # (h_start, h_end, length)
            (0.06, 0.08, 0.1),
        ]

    def test_creates_png_file(self, simple_coords, simple_conical_segments, tmp_path):
        """Should create a PNG output file."""
        out = tmp_path / "horn2d.png"
        plot_horn_2d_folded(simple_conical_segments, simple_coords, None, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_with_enclosure_dims(
        self, simple_coords, simple_conical_segments, tmp_path
    ):
        """enclosure_dims should be accepted without error."""
        out = tmp_path / "horn2d_box.png"
        plot_horn_2d_folded(simple_conical_segments, simple_coords, (0.3, 0.4), out)
        assert out.exists()

    def test_with_driver_coord(self, simple_coords, simple_conical_segments, tmp_path):
        """driver_coord should be accepted without error."""
        out = tmp_path / "horn2d_driver.png"
        plot_horn_2d_folded(
            simple_conical_segments,
            simple_coords,
            (0.3, 0.4),
            out,
            driver_coord=(0.05, 0.3),
        )
        assert out.exists()

    def test_with_throat_chamber_overlay(
        self, simple_coords, simple_conical_segments, tmp_path
    ):
        """throat_chamber_side should draw without error when driver/enclosure are provided."""
        out = tmp_path / "horn2d_chamber.png"
        plot_horn_2d_folded(
            simple_conical_segments,
            simple_coords,
            (0.3, 0.4),
            out,
            driver_coord=(0.0, 0.2),
            throat_chamber_side=0.08,
        )
        assert out.exists()

    def test_empty_coordinates_returns_early(self, simple_conical_segments, tmp_path):
        """Empty coordinates should return without raising and not create a file."""
        out = tmp_path / "horn2d_empty.png"
        plot_horn_2d_folded(simple_conical_segments, [], None, out)
        # Function returns early without creating output
        assert not out.exists()

    def test_custom_title(self, simple_coords, simple_conical_segments, tmp_path):
        """Custom title should not cause an error."""
        out = tmp_path / "horn2d_title.png"
        plot_horn_2d_folded(
            simple_conical_segments,
            simple_coords,
            None,
            out,
            title="My Folded Horn",
        )
        assert out.exists()


class TestPlotWaterfall:
    """Tests for plot_waterfall()."""

    @pytest.fixture
    def simple_csd_data(self):
        """Simple CSD data."""
        csd_freqs = np.linspace(100, 2000, 50)
        csd_times_ms = np.linspace(0, 20, 10)
        csd_db = np.random.uniform(-80, -20, size=(10, 50))
        return csd_freqs, csd_times_ms, csd_db

    def test_creates_png_file(self, simple_csd_data, tmp_path):
        """Should create a PNG output file."""
        out = tmp_path / "waterfall.png"
        plot_waterfall(*simple_csd_data, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_title(self, simple_csd_data, tmp_path):
        """Custom title should not cause an error."""
        out = tmp_path / "waterfall_title.png"
        plot_waterfall(*simple_csd_data, out, title="My Waterfall")
        assert out.exists()

    def test_custom_freq_range(self, simple_csd_data, tmp_path):
        """f_min and f_max should not cause an error."""
        out = tmp_path / "waterfall_range.png"
        plot_waterfall(*simple_csd_data, out, f_min=200.0, f_max=1500.0)
        assert out.exists()


class TestPlotImpulseStep:
    """Tests for plot_impulse_step()."""

    @pytest.fixture
    def simple_time_data(self):
        """Simple impulse/step data."""
        time_ms = np.linspace(0, 20, 500)
        impulse = np.exp(-time_ms / 3) * np.sin(2 * np.pi * 100 * time_ms / 1000)
        step = np.cumsum(impulse) * (time_ms[1] - time_ms[0])
        return time_ms, impulse, step

    def test_creates_png_file(self, simple_time_data, tmp_path):
        """Should create a PNG output file."""
        out = tmp_path / "impulse.png"
        plot_impulse_step(*simple_time_data, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_title(self, simple_time_data, tmp_path):
        """Custom title should not cause an error."""
        out = tmp_path / "impulse_title.png"
        plot_impulse_step(*simple_time_data, out, title="My Impulse Response")
        assert out.exists()
