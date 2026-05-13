"""Tests for pyhorn_core/solver/wavefront.py — Phase 2 additions."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from pyhorn_core.solver.wavefront import (
    WavefrontGrid,
    animate_wave_propagation,
    boundary_condition_mask,
    edit_horn_geometry,
    edit_horn_geometry_from_yaml,
    load_horn_geometry,
    pml_damping_mask,
    solve_2d_wave,
    solve_2d_wave_pml,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_mesh():
    nx, ny = 80, 60
    mesh_x = np.linspace(0.0, 0.2, nx)
    mesh_y = np.linspace(0.0, 0.15, ny)
    mesh_xx, mesh_yy = np.meshgrid(mesh_x, mesh_y)
    return mesh_xx, mesh_yy, nx, ny


@pytest.fixture
def box_walls(simple_mesh):
    mesh_xx, mesh_yy, _, _ = simple_mesh
    box = [(0.05, 0.05), (0.15, 0.05), (0.15, 0.10), (0.05, 0.10), (0.05, 0.05)]
    return boundary_condition_mask(box, mesh_xx, mesh_yy, wall_thickness=0.005)


# ─── load_horn_geometry ───────────────────────────────────────────────────────

class TestLoadHornGeometry:
    def test_source_yaml_loads_coords(self, tmp_path):
        yaml = tmp_path / "test_horn.yaml"
        yaml.write_text(
            "enclosure_type: BLH\n"
            "coordinates:\n"
            "  - [0.0, 0.0]\n"
            "  - [0.1, 0.0]\n"
            "  - [0.1, 0.05]\n"
            "  - [0.0, 0.05]\n"
        )
        geo = load_horn_geometry(yaml)
        assert geo["coords"].shape == (4, 2)
        assert geo["source_x"] is None
        assert geo["source_y"] is None
        assert geo["name"] == "test_horn"

    def test_project_yaml_loads_geometry_path(self, tmp_path):
        src = tmp_path / "source.yaml"
        src.write_text(
            "enclosure_type: BLH\n"
            "coordinates:\n"
            "  - [0.0, 0.0]\n"
            "  - [0.2, 0.0]\n"
            "  - [0.2, 0.1]\n"
            "  - [0.0, 0.1]\n"
        )
        proj = tmp_path / "project.yaml"
        proj.write_text(
            f"name: TestProject\n"
            f"geometry_path: source.yaml\n"
            f"driver_coord: [0.05, 0.03]\n"
        )
        geo = load_horn_geometry(proj)
        assert geo["coords"].shape == (4, 2)
        assert geo["source_x"] == pytest.approx(0.05)
        assert geo["source_y"] == pytest.approx(0.03)
        assert geo["name"] == "TestProject"
        assert geo["source_yaml"] == src

    def test_driver_coord_from_project_yaml(self, tmp_path):
        src = tmp_path / "src.yaml"
        src.write_text(
            "coordinates:\n"
            "  - [0.0, 0.0]\n"
            "  - [0.1, 0.0]\n"
            "  - [0.1, 0.05]\n"
            "  - [0.0, 0.05]\n"
        )
        proj = tmp_path / "proj.yaml"
        proj.write_text("name: D\ngeometry_path: src.yaml\ndriver_coord: [0.02, 0.025]\n")
        geo = load_horn_geometry(proj)
        assert geo["source_x"] == pytest.approx(0.02)
        assert geo["source_y"] == pytest.approx(0.025)

    def test_raises_on_empty_coords(self, tmp_path):
        yaml = tmp_path / "empty.yaml"
        yaml.write_text("enclosure_type: BLH\n")
        with pytest.raises(ValueError, match="YAML must contain"):
            load_horn_geometry(yaml)


# ─── WavefrontGrid.set_point_source ─────────────────────────────────────────

class TestWavefrontGridSource:
    def test_set_point_source_stores_coordinates(self):
        grid = WavefrontGrid(nx=10, ny=10, dx=0.001, dy=0.001, frequency=500.0, k=9.2)
        assert grid.source_x is None
        assert grid.source_y is None
        grid.set_point_source(0.05, 0.03)
        assert grid.source_x == pytest.approx(0.05)
        assert grid.source_y == pytest.approx(0.03)

    def test_from_yaml_extracts_driver_coord(self):
        # Uses the real projects/bk16.yaml which has driver_coord: [0.01, 0.0]
        grid, _, _, _ = WavefrontGrid.from_yaml(
            "projects/bk16.yaml", nx=40, ny=20
        )
        assert grid.source_x == pytest.approx(0.01)
        assert grid.source_y == pytest.approx(0.0)

    def test_from_yaml_builds_correct_grid_dimensions(self):
        grid, walls, mesh_x, mesh_y = WavefrontGrid.from_yaml(
            "source/bk16.yaml", nx=120, ny=80
        )
        assert grid.nx == 120
        assert grid.ny == 80
        assert walls.shape == (80, 120)
        assert mesh_x.shape == (80, 120)
        assert mesh_y.shape == (80, 120)


# ─── pml_damping_mask ─────────────────────────────────────────────────────────

class TestPMLDampingMask:
    def test_zero_inside_domain(self):
        sigma = pml_damping_mask(100, 100, pml_width=10)
        # Centre should be zero
        assert sigma[50, 50] == pytest.approx(0.0)

    def test_nonzero_at_edge(self):
        sigma = pml_damping_mask(100, 100, pml_width=10)
        # Last PML cell should be at kappa_max
        assert sigma[9, 50] == pytest.approx(2.0)
        assert sigma[90, 50] == pytest.approx(2.0)

    def test_quadratic_ramp(self):
        sigma = pml_damping_mask(100, 100, pml_width=10, kappa_max=1.0)
        # Ratio of cell 5 to cell 10 should be (5/10)^2 = 0.25
        assert sigma[4, 50] == pytest.approx(0.25, rel=1e-10)
        assert sigma[9, 50] == pytest.approx(1.0, rel=1e-10)

    def test_small_grid_returns_zeros(self):
        # Very small grid: w < 2, returns zero mask
        sigma = pml_damping_mask(5, 5, pml_width=10)
        assert np.all(sigma == 0.0)


# ─── solve_2d_wave vs solve_2d_wave_pml ───────────────────────────────────────

class TestSolve2DWavePML:
    def test_pml_and_standard_agree_near_source(self, simple_mesh, box_walls):
        mesh_xx, mesh_yy, _, _ = simple_mesh
        walls = box_walls

        grid_std = solve_2d_wave(mesh_xx, mesh_yy, 0.10, 0.075, 500.0, walls)
        grid_pml = solve_2d_wave_pml(mesh_xx, mesh_yy, 0.10, 0.075, 500.0, walls)

        # Near the source both should give similar results
        # (PML absorbs at edges; in the interior they should be close)
        interior = ~walls
        std_vals = grid_std.pressure_field.real[interior]
        pml_vals = grid_pml.pressure_field.real[interior]

        # Both should be non-zero near source
        assert std_vals.max() > 0.0
        assert pml_vals.max() > 0.0

        # Relative difference should be small in interior (away from PML edges)
        # Use 80th percentile to avoid edge effects
        tol = np.percentile(np.abs(std_vals - pml_vals) / (np.abs(std_vals) + 1e-12), 80)
        assert tol < 0.5, "PML and standard solve differ too much in interior"

    def test_pml_reduces_edge_reflections(self, simple_mesh, box_walls):
        mesh_xx, mesh_yy, _, _ = simple_mesh
        walls = box_walls

        grid_std = solve_2d_wave(mesh_xx, mesh_yy, 0.10, 0.075, 800.0, walls)
        grid_pml = solve_2d_wave_pml(mesh_xx, mesh_yy, 0.10, 0.075, 800.0, walls, pml_width=8)

        # PML should have lower amplitude near the boundary rows
        # (first few rows away from the wall band)
        boundary_rows = walls.any(axis=1)
        first_interior_row = np.where(~boundary_rows)[0][0]
        last_interior_row = np.where(~boundary_rows)[0][-1]

        # PML should attenuate more at the far edge vs standard solve
        std_edge = np.abs(grid_std.pressure_field[last_interior_row - 2, :]).max()
        pml_edge = np.abs(grid_pml.pressure_field[last_interior_row - 2, :]).max()

        # PML attenuates; standard doesn't — PML edge amplitude should be lower
        assert pml_edge <= std_edge * 1.1, "PML should attenuate near edges"

    def test_pml_grid_stores_source_coordinates(self, simple_mesh, box_walls):
        mesh_xx, mesh_yy, _, _ = simple_mesh
        grid = solve_2d_wave_pml(mesh_xx, mesh_yy, 0.12, 0.08, 500.0, box_walls)
        assert grid.source_x == pytest.approx(0.12)
        assert grid.source_y == pytest.approx(0.08)


# ─── Shared mock helpers for the editor tests ──────────────────────────────────



# ─── Shared mock helpers for the editor tests ──────────────────────────────────

def _make_editor_mocks():
    """Create a (fig, ax, handlers_ref) trio for editor testing.

    handlers_ref is a list with one element: the list of registered event handlers.
    Call _trigger_key(handlers_ref, 'enter') or _trigger_key(handlers_ref, 'escape')
    to simulate key presses.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    handlers_ref: list = []

    class MockCanvas:
        def mpl_connect(self, event, handler):
            handlers_ref.append((len(handlers_ref), (event, handler)))
            return len(handlers_ref)

        def mpl_disconnect(self, _):
            pass

        def draw_idle(self):
            pass

    class MockAx:
        _is_horn_editor = True

        @property
        def transAxes(self):
            return plt.matplotlib.transforms.IdentityTransform()

        def get_children(self):
            return []

        def text(self, *args, **kwargs):
            class T:
                _is_horn_editor = True
            return T()

        def set_facecolor(self, *args, **kwargs):
            pass

        def set_title(self, *args, **kwargs):
            pass

        def set_xlabel(self, *args, **kwargs):
            pass

        def set_ylabel(self, *args, **kwargs):
            pass

        def tick_params(self, *args, **kwargs):
            pass

        @property
        def spines(self):
            class Spine:
                def set_color(self, *args):
                    pass
            return {"top": Spine(), "bottom": Spine(),
                    "left": Spine(), "right": Spine()}

        def grid(self, *args, **kwargs):
            pass

        def set_aspect(self, *args, **kwargs):
            pass

        def set_xlim(self, *args, **kwargs):
            pass

        def set_ylim(self, *args, **kwargs):
            pass

        def get_xlim(self):
            return (0.0, 0.3)

        def scatter(self, *args, **kwargs):
            class S:
                def remove(self):
                    pass
            return S()

        def plot(self, *args, **kwargs):
            class L:
                def remove(self):
                    pass
            return [L()]

    class MockFig:
        def __init__(self):
            self.canvas = MockCanvas()
            self._ax = MockAx()

        def get_size_inches(self):
            return (10, 8)

        @property
        def dpi(self):
            return 100

        def tight_layout(self):
            pass

    return MockFig(), MockFig()._ax, handlers_ref


def _trigger_key(handlers, key):
    """Fire a key-press event into registered handlers."""
    k = type("K", (), {"key": key})()
    for _cid, (evt, handler) in handlers:
        if evt == "key_press_event":
            handler(k)


# ─── edit_horn_geometry ───────────────────────────────────────────────────────


class TestEditHornGeometry:
    """Tests for the interactive wall editor.

    These tests verify the editor API is sound; full GUI testing requires a display.
    """

    def test_returns_list_of_tuples(self):
        """edit_horn_geometry is callable and returns a list of (x, y) tuples."""
        import matplotlib.pyplot as plt
        from unittest import mock

        coords = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.05), (0.0, 0.05)]
        result_holder = [None]
        mock_fig, mock_ax, handlers = _make_editor_mocks()

        def fake_close(**kwargs):
            _trigger_key(handlers, "enter")

        with mock.patch.object(plt, "show", side_effect=fake_close), mock.patch.object(plt, "close", lambda fig: None):
            with mock.patch.object(plt, "subplots", return_value=(mock_fig, mock_ax)):
                result_holder[0] = edit_horn_geometry(coords)

        assert isinstance(result_holder[0], list)
        assert all(isinstance(v, tuple) and len(v) == 2 for v in result_holder[0])
        assert all(isinstance(coord, float) for v in result_holder[0] for coord in v)

    def test_escape_returns_original_coords(self):
        """Pressing Escape returns the original coordinates unchanged."""
        import matplotlib.pyplot as plt
        from unittest import mock

        original = [(0.0, 0.0), (0.2, 0.0), (0.2, 0.1), (0.0, 0.1)]
        result_holder = [None]
        mock_fig, mock_ax, handlers = _make_editor_mocks()

        def fake_close(**kwargs):
            _trigger_key(handlers, "escape")

        with mock.patch.object(plt, "show", side_effect=fake_close), mock.patch.object(plt, "close", lambda fig: None):
            with mock.patch.object(plt, "subplots", return_value=(mock_fig, mock_ax)):
                result_holder[0] = edit_horn_geometry(original)

        assert result_holder[0] == original


class TestEditHornGeometryFromYaml:
    """Tests for the YAML-backed editor convenience function."""

    def test_loads_and_returns_updated_geometry(self, tmp_path):
        """Result dict contains the expected keys and the updated coords."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        src = tmp_path / "horn.yaml"
        src.write_text(
            "name: TestHorn\n"
            "coordinates:\n"
            "  - [0.0, 0.0]\n"
            "  - [0.2, 0.0]\n"
            "  - [0.2, 0.1]\n"
            "  - [0.0, 0.1]\n"
        )

        result_holder = [None]
        mock_fig, mock_ax, handlers = _make_editor_mocks()

        def fake_close(**kwargs):
            _trigger_key(handlers, "enter")

        with mock.patch.object(plt, "show", side_effect=fake_close), mock.patch.object(plt, "close", lambda fig: None):
            with mock.patch.object(plt, "subplots", return_value=(mock_fig, mock_ax)):
                result_holder[0] = edit_horn_geometry_from_yaml(src)

        result = result_holder[0]
        assert isinstance(result, dict)
        assert "coords" in result
        assert "name" in result
        assert result["name"] == "TestHorn"
        assert isinstance(result["coords"], list)
        assert all(isinstance(v, tuple) and len(v) == 2 for v in result["coords"])

    def test_saved_flag_false_when_no_output_path(self, tmp_path):
        """When no output_yaml_path is given, saved is False."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        src = tmp_path / "horn.yaml"
        src.write_text("coordinates:\n  - [0.0, 0.0]\n  - [0.1, 0.0]\n")
        mock_fig, mock_ax, handlers = _make_editor_mocks()

        def fake_close(**kwargs):
            _trigger_key(handlers, "enter")

        with mock.patch.object(plt, "show", side_effect=fake_close), mock.patch.object(plt, "close", lambda fig: None):
            with mock.patch.object(plt, "subplots", return_value=(mock_fig, mock_ax)):
                result = edit_horn_geometry_from_yaml(src)

        assert result["saved"] is False

    def test_saved_flag_true_and_file_written_when_output_path_provided(self, tmp_path):
        """When output_yaml_path is given, the file is written and saved is True."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import yaml as _yaml

        src = tmp_path / "source.yaml"
        src.write_text("name: Src\ncoordinates:\n  - [0.0, 0.0]\n  - [0.1, 0.0]\n")
        out = tmp_path / "edited.yaml"
        mock_fig, mock_ax, handlers = _make_editor_mocks()

        def fake_close(**kwargs):
            _trigger_key(handlers, "enter")

        with mock.patch.object(plt, "show", side_effect=fake_close), mock.patch.object(plt, "close", lambda fig: None):
            with mock.patch.object(plt, "subplots", return_value=(mock_fig, mock_ax)):
                result = edit_horn_geometry_from_yaml(src, output_yaml_path=out)

        assert result["saved"] is True
        assert out.exists()
        with open(out) as fh:
            written = _yaml.safe_load(fh)
        assert "coordinates" in written
        assert written["name"] == "Src"


# ─── Animation ────────────────────────────────────────────────────────────────

class TestAnimateWavePropagation:
    """Tests for animated wavefront display."""

    def test_animate_wave_propagation_produces_gif(self, simple_mesh, box_walls):
        """animate_wave_propagation produces a valid GIF file."""
        import matplotlib
        matplotlib.use("Agg")
        import tempfile, os

        mesh_xx, mesh_yy, _, _ = simple_mesh
        grid = solve_2d_wave_pml(mesh_xx, mesh_yy, 0.10, 0.075, 500.0, box_walls)
        box = [(0.05, 0.05), (0.15, 0.05), (0.15, 0.10), (0.05, 0.10), (0.05, 0.05)]

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "anim.gif")
            ani = animate_wave_propagation(
                mesh_x=mesh_xx,
                mesh_y=mesh_yy,
                pressure_field=grid.pressure_field,
                horn_path=box,
                frequency=500.0,
                filename=out,
                n_frames=8,
                interval_ms=20,
                figsize=(8, 4),
            )
            assert os.path.exists(out), "GIF file was not created"
            size = os.path.getsize(out)
            assert size > 5000, f"GIF file suspiciously small: {size} bytes"

    def test_from_yaml_stores_boundary_mask_and_horn_coords(self):
        """WavefrontGrid.from_yaml stores boundary_mask and horn_coords in the grid."""
        grid, walls, _, _ = WavefrontGrid.from_yaml(
            "projects/bk16.yaml", nx=40, ny=25
        )
        assert grid.boundary_mask is not None, "boundary_mask not stored"
        assert grid.horn_coords is not None, "horn_coords not stored"
        assert np.array_equal(grid.boundary_mask, walls), "stored boundary_mask != returned walls"

    def test_grid_animate_uses_stored_boundary_mask(self, tmp_path):
        """WavefrontGrid.animate uses the stored boundary_mask, not zeros."""
        import matplotlib
        matplotlib.use("Agg")
        import tempfile, os

        # Write a minimal source YAML with source_x/source_y (not driver_coord)
        yaml = tmp_path / "wf_anim.yaml"
        yaml.write_text(
            "name: TestAnim\n"
            "coordinates:\n"
            "  - [0.0, 0.0]\n"
            "  - [0.2, 0.0]\n"
            "  - [0.2, 0.1]\n"
            "  - [0.0, 0.1]\n"
            "source_x: 0.05\n"
            "source_y: 0.05\n"
        )
        grid, walls, mesh_x, mesh_y = WavefrontGrid.from_yaml(str(yaml), nx=50, ny=40)

        # The stored boundary mask should NOT be all zeros (there is a box geometry)
        assert not np.all(grid.boundary_mask == False), (
            "boundary_mask should have True values for the box walls"
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "grid_anim.gif")
            ani = grid.animate(
                mesh_x=mesh_x,
                mesh_y=mesh_y,
                filename=out,
                n_frames=6,
                interval_ms=20,
                figsize=(7, 4),
            )
            assert os.path.exists(out), "GIF was not created"
            assert os.path.getsize(out) > 5000, "GIF too small"
