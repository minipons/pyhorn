# Validation Matrix

## Physics Changes
- Radiation only:
  - `pyhorn_core/tests/test_physics_radiation.py`
- Calibration only:
  - `pyhorn_core/tests/test_orchestrators_helpers.py`
- BLH / SPL total / direct-horn summing:
  - `tests/benchmarks/hornresp_gdb1/test_hirob_benchmark.py` (uses canonical HiroB fixture/reference files under `tests/benchmarks/hornresp/hirob/`)
  - targeted tests in `pyhorn_core/tests/test_solver_models.py`

## Config Changes
- `pyhorn_core/tests/test_config_models.py`
- `pyhorn_core/tests/test_config_parser.py`

## Benchmark Fixture Changes
- `tests/benchmarks/hornresp_gdb1/test_hirob_benchmark.py` for the HiroB benchmark harness
- `tests/benchmarks/hornresp_gdb1/test_hornresp_benchmark.py`

## CLI Changes
- `pyhorn_cli/tests/test_cli_commands.py`

## Rule Of Thumb
Run the narrowest behavior-scoped test first. Only widen to broader regression suites after the touched slice is green.
