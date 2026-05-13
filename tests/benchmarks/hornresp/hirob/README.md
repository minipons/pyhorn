# HiroB Hornresp Benchmark Fixture

This directory is the canonical benchmark home for the HiroB Hornresp comparison.

## Layout
- `fixture/driver.yaml`: benchmark-safe driver fixture
- `fixture/horn.yaml`: benchmark-safe horn fixture
- `reference/hornresp_spl.csv`: Hornresp exported SPL curve
- `reference/hornresp_params.txt`: Hornresp exported parameter dump

## Policy
- Do not point HiroB benchmark code at `drivers/FE166NV2.yaml` or `projects/hirob.yaml`.
- Production fixtures may include voicing layers and presentation-oriented settings.
- Benchmark fixtures should stay minimal and reproducible.
