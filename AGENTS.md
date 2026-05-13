# Repo Operating Map

## Start Here
- SPL or Hornresp mismatch: start in `pyhorn_core/pyhorn_physics/orchestrators.py`, `pyhorn_core/pyhorn_physics/calibration.py`, and the dedicated benchmark fixture under `tests/benchmarks/hornresp/`.
- Radiation or mouth-boundary issue: start in `pyhorn_core/pyhorn_physics/radiation.py` and `pyhorn_core/tests/test_physics_radiation.py`.
- Config/schema issue: start in `pyhorn_core/config/driver_models.py`, `pyhorn_core/config/horn_models.py`, `pyhorn_core/config/chamber_models.py`, and `pyhorn_core/config/parser.py`.
- CLI issue: start in `pyhorn_cli/cli/` and validate with focused CLI tests only.

## Source Of Truth
- Production speaker fixtures: `drivers/` and `projects/`.
- Benchmark-only fixtures: `tests/benchmarks/hornresp/**/fixture/`.
- Reference exports from external tools: `tests/benchmarks/hornresp/**/reference/`.
- Compatibility facades: `pyhorn_core/config/models.py` and `pyhorn_core/solver/models.py`.

## Guardrails
- Do not point Hornresp regression tests at productized fixtures in `drivers/` or `projects/`.
- Keep measured `spl_response` and voicing layers out of benchmark fixtures unless the benchmark explicitly tests them.
- Prefer editing the focused config modules over `pyhorn_core/config/models.py`; that file is a shim.
- Prefer adding new narrow tests over growing `pyhorn_core/tests/test_solver_models.py` further.

## Fast Validation
- Benchmark-only: `python -m pytest tests/benchmarks/hornresp_gdb1/test_hirob_benchmark.py -q --no-header --no-cov`
- Radiation-only: `python -m pytest pyhorn_core/tests/test_physics_radiation.py -q --no-header --no-cov`
- Config-only: `python -m pytest pyhorn_core/tests/test_config_models.py pyhorn_core/tests/test_config_parser.py -q --no-header --no-cov`

## Known Hotspots
- `pyhorn_core/pyhorn_physics/orchestrators.py`: public solver facade, still large.
- `pyhorn_core/tests/test_solver_models.py`: legacy catch-all regression file; avoid adding new unrelated coverage here.
- `drivers/FE166NV2.yaml` and `projects/hirob.yaml`: production fixtures, not benchmark fixtures.
