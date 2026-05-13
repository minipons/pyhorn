# Benchmark Analysis: HiroB Hornresp Reference

Date: 2026-05-13

## Scope

This note compares the current pyhorn HiroB benchmark setup against the Hornresp reference files:

- `tests/benchmarks/hornresp/hirob/reference/hornresp_spl.csv`
- `tests/benchmarks/hornresp/hirob/reference/hornresp_params.txt`

Legacy helper scripts may still live under `tests/benchmarks/hornresp_gdb1/`, but the canonical HiroB benchmark fixture and reference data now live under `tests/benchmarks/hornresp/hirob/`.

The goal was to answer a narrow question: is the remaining HiroB mismatch caused by horn solver logic, or is the benchmark fixture itself no longer aligned with the Hornresp run it is supposed to reproduce?

## Reference vs Current Fixture

### Hornresp parameter dump

The Hornresp export describes this reference run:

- `Ang = 0.5 x Pi`
- `Eg = 2.83`
- `S1 = 80.00`
- `S2 = 800.00`
- `Hyp = 143.00`
- `F12 = 49.80`
- `T = 0.70`
- `Sd = 132.00`
- `Mmd = 6.04`
- `Le = 0.80`
- `Re = 7.80`
- `Vrc = 4.70`
- `Lrc = 10.00`
- `Fr = 2000.00`
- `Vtc = 0.00`
- `Atc = 0.00`
- `Lossy Inductance Model Flag = 0`
- `Semi-Inductance Model Flag = 0`
- `Damping Model Flag = 0`

### What pyhorn currently loads for HiroB

Running the current benchmark path (`parse_horn_project(projects/hirob.yaml)` + `parse_driver_specs(drivers/FE166NV2.yaml)`) loads:

- `ap1 = 132.0 cm^2`
- `rear_chamber.chamber_type = sealed`
- `rear_chamber.vrc = 4.7 L`
- `rear_chamber.lrc = 10.0 cm`
- `fr_rc = 2000`
- `ang = 1.5708 sr`
- `throat_area = 80.0 cm^2`
- `mouth_area = 800.0 cm^2`
- `path_length = 143.0 cm`
- `driver.sensitivity_db` loaded from `drivers/FE166NV2.yaml`
- `driver.spl_response` enabled from `drivers/spl.csv`
- `driver.lossy_le = true`

## Immediate Fixture Mismatches

These mismatches showed up before touching the solver:

| Area                    | Hornresp reference                                                                       | Current pyhorn fixture                                                            | Finding                                                                                                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rear chamber model      | BLH reference with `Vrc=4.7`, `Lrc=10`, `Fr=2000`, no throat chamber                     | `projects/hirob.yaml` uses `rear_chamber.chamber_type: sealed`                    | This conflicts with the repo's own CRIT-1 note that BLH rear chambers should use `coupling` when matching Hornresp-style BLH behavior.                                 |
| Throat adapter          | Hornresp dump exposes `AT = 1.84`, but the exact mapping to pyhorn `ap1` is not explicit | `projects/hirob.yaml` sets `ap1 = 0.0132 m^2` (132 cm^2)                          | The current HiroB fixture is making a strong throat-adapter assumption that is not justified by the dump as written.                                                   |
| Lossy Le                | Hornresp flags show `Lossy Inductance Model Flag = 0`                                    | `drivers/FE166NV2.yaml` sets `lossy_le: true`                                     | This is a benchmark mismatch, although the measured impact turned out to be negligible in this case.                                                                   |
| Measured driver SPL     | Hornresp export is a pure simulation result                                              | `drivers/FE166NV2.yaml` enables `spl_response: spl.csv`                           | This is a pyhorn-only post-processing layer and it materially changes `result.spl` in the upper bands.                                                                 |
| Sensitivity calibration | Hornresp dump does not include pyhorn-specific `sensitivity_db` offsets                  | Both `drivers/FE166NV2.yaml` and `projects/hirob.yaml` contain calibration tables | Only the driver-level table is actually used by the current benchmark harness. The project-level `sensitivity_db` in `projects/hirob.yaml` is inert in this code path. |

## Variant Sweep

I ran the HiroB reference CSV against several local variants of the current fixture.

Metrics below are deltas vs Hornresp CSV after interpolating pyhorn onto the Hornresp frequency grid.

| Variant                                                                                       | `result.spl` mean / std / maxabs | `result.spl_power_based` mean / std / maxabs |
| --------------------------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------- |
| Current project + current driver                                                              | `+5.01 / 9.60 / 30.18 dB`        | `-4.35 / 12.18 / 72.04 dB`                   |
| Rear chamber forced to `coupling`                                                             | `+4.81 / 9.70 / 30.18 dB`        | `-4.62 / 12.13 / 72.72 dB`                   |
| `ap1 = 0`                                                                                     | `+5.95 / 9.30 / 30.18 dB`        | `-3.72 / 12.49 / 63.13 dB`                   |
| `coupling + ap1 = 0`                                                                          | `+5.75 / 9.38 / 30.18 dB`        | `-3.97 / 12.38 / 61.34 dB`                   |
| `coupling + ap1 = 0 + no sensitivity_db`                                                      | `+6.93 / 8.95 / 30.20 dB`        | `-1.11 / 13.54 / 60.84 dB`                   |
| Current horn, but no `spl_response` and no `lossy_le`                                         | `+1.01 / 11.31 / 32.82 dB`       | `-5.27 / 11.05 / 35.13 dB`                   |
| Hornresp-like guess: `coupling + ap1 = 0 + no sensitivity_db + no spl_response + no lossy_le` | `+4.89 / 11.27 / 36.68 dB`       | `-0.91 / 11.38 / 31.15 dB`                   |

## Findings

### 1. This is not one bug; the benchmark fixture has drifted.

The current HiroB setup mixes at least four pyhorn-specific layers on top of the nominal Hornresp geometry:

- a rear chamber mode choice (`sealed` vs `coupling`)
- a throat-adapter assumption (`ap1 = 132 cm^2`)
- a measured free-air driver SPL override (`spl_response`)
- a driver-level `sensitivity_db` correction table

That means the current benchmark no longer isolates the core horn solver.

### 2. The measured `spl_response` override is the dominant cause of the HF pressure-SPL mismatch.

With the current fixture:

- `result.spl` is only `+0.23 dB` high at `1986.7 Hz`
- but `+8.68 dB` high at `5033.4 Hz`

When only `spl_response` is removed:

- the 5 kHz `result.spl` jumps from `89.17 dB` to `100.05 dB`
- while `result.spl_power_based` stays near the Hornresp curve (`80.98 dB` to `81.61 dB` vs `80.49 dB` reference)

Interpretation: the measured direct-radiator override is pulling the displayed total SPL toward the manufacturer free-air cone curve, not toward the Hornresp system curve. That makes sense for product voicing, but it contaminates the Hornresp benchmark.

### 3. `lossy_le` does not explain the current HiroB mismatch.

Switching `lossy_le` off changed the HiroB numbers only negligibly in this sweep. The mismatch is real at the fixture level, but it is not a first-order contributor here.

### 4. The low-frequency `spl_power_based` curve is not currently a stable full-band benchmark target.

Baseline example:

- At `19.9 Hz`, Hornresp is `72.59 dB`
- `result.spl` is `72.01 dB` (`-0.58 dB`)
- `result.spl_power_based` is `41.34 dB` (`-31.25 dB`)

This pattern repeated across variants: the acoustic-power-based curve collapses at the low end because mouth radiation resistance is tiny there. So `spl_power_based` is useful for HF/MF normalization work, but it is not a good single full-band comparator for this CSV in its current form.

### 5. Forcing `coupling` helps locally around 50-100 Hz, but it does not solve the benchmark on its own.

Examples:

- At `50.3 Hz`, `spl_power_based` improved from `+1.77 dB` to `+0.11 dB`
- At `100 Hz`, `result.spl` improved from `-0.54 dB` to `+0.01 dB`

But the overall summary barely moved. This says the chamber model choice matters, but it is not the only source of drift.

### 6. The repo comment claiming HiroB calibration used `ap1 = 0` is not supported by the current CSV when judging total SPL.

Setting `ap1 = 0` made the 50-100 Hz region worse, not better:

- `50.3 Hz`: `result.spl` moved from `+4.37 dB` to `+9.69 dB`
- `100 Hz`: `result.spl` moved from `-0.54 dB` to `+2.74 dB`

So either:

- the comment in `drivers/FE166NV2.yaml` is stale,
- the current HiroB CSV is not the one used for that calibration,
- or the Hornresp `AT` parameter is being mapped incorrectly in pyhorn.

### 7. There is still a real HF normalization gap after stripping away pyhorn-specific layers.

Using the closest simple Hornresp-like variant I could build locally (`coupling + ap1 = 0 + no sensitivity_db + no spl_response + no lossy_le`):

- `result.spl_power_based` is still `+5.12 dB` high in `1-5 kHz`
- and `+6.92 dB` high in `5-20 kHz`

That is consistent with the existing CRIT-3 story: even after removing fixture drift, pyhorn still has a residual HF normalization gap relative to Hornresp.

### 8. The project-level `sensitivity_db` table in `projects/hirob.yaml` is currently dead configuration for this harness.

`compare_hirob.py` loads:

- geometry from `parse_horn_project(projects/hirob.yaml)`
- driver from `parse_driver_specs(drivers/FE166NV2.yaml)`

`DriverSpecs` owns `sensitivity_db`; `HornGeometry` does not. So the `sensitivity_db` block embedded in `projects/hirob.yaml` is not actually wired into this benchmark path.

## Working Interpretation

The current HiroB benchmark is mixing three distinct questions:

1. Does the horn TMM core reproduce the Hornresp geometry?
2. Does pyhorn's `dB/W/m` normalization reproduce Hornresp's reference level?
3. Do pyhorn-specific presentation layers (`spl_response`, `sensitivity_db`) produce a more realistic-looking loudspeaker response?

Those questions need different fixtures.

Right now the benchmark uses one CSV while the pyhorn side includes extra correction layers that Hornresp never had. That is why the benchmark feels "not perfect": it is not comparing like with like.

## Recommended Benchmark Split

### A. Geometry/physics benchmark

Use a Hornresp-like fixture:

- disable `spl_response`
- disable `lossy_le`
- disable `sensitivity_db`
- explicitly choose the rear chamber mode intended for the HiroB reference
- explicitly document how Hornresp `AT` maps to pyhorn `ap1`

Compare:

- `result.spl` below about `200 Hz`
- `result.spl_power_based` in MF/HF after LF normalization is fixed

### B. Productized response benchmark

Keep the current pyhorn extras:

- `spl_response`
- driver calibration table
- any voicing layers

But do not call that a direct Hornresp benchmark; it is a calibrated pyhorn presentation curve.

## Bottom Line

The main issue is benchmark-fixture drift, not a single isolated solver defect.

Most important concrete points:

- the current HiroB benchmark path does not actually match the Hornresp setup documented in `hornresp_params_hirob.txt`
- `spl_response` is the biggest reason `result.spl` diverges from the Hornresp CSV above 1-2 kHz
- `spl_power_based` should not be used as a full-band reference curve until the LF power normalization issue is addressed
- even after stripping away pyhorn-only layers, a residual `~5-7 dB` HF gap remains, so there is still a genuine CRIT-3-style normalization problem underneath the fixture drift