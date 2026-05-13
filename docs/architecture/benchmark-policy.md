# Benchmark Policy

## Separate Fixture Classes
- Production fixtures live in `drivers/` and `projects/`.
- Benchmark fixtures live in `tests/benchmarks/hornresp/**/fixture/`.
- External reference data lives in `tests/benchmarks/hornresp/**/reference/`.

## Hornresp Comparison Rules
- Compare Hornresp against `result.spl_power_based` unless the benchmark explicitly targets coherent microphone summation.
- Treat `result.spl` as a diagnostic when analyzing BLH mouth/direct interference notches.
- Keep measured `spl_response` and product voicing layers out of benchmark fixtures by default.

## Update Policy
- If physics changes require recalibration, update the benchmark fixture first.
- Do not silently reuse production YAMLs for regression tests.
- Keep comparison scripts next to the benchmark they exercise.
