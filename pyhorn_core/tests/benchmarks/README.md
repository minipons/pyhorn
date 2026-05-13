# Hornresp vs pyhorn — Unit Conversion Reference

Hornresp uses **cgs/cm/g/L** units, while pyhorn uses **SI m/kg/m³**.

## Conversion Table

| Quantity | Hornresp | pyhorn (SI) | Conversion |
|----------|----------|-------------|------------|
| Area (S1, S2, Sd) | cm² | m² | ÷ 10,000 |
| Length (L12, Lrc) | cm | m | ÷ 100 |
| Volume (Vrc, Vtc, Vas) | L | m³ | ÷ 1000 |
| Driver mass (Mmd) | g | kg | ÷ 1000 |
| Compliance (Cms) | mm/N | m/N | ÷ 1000 |
| Voice coil inductance (Le) | mH | H | ÷ 1000 |
| Radiation angle (Ang) | sr | sr | same (dimensionless) |
| Frequency (F12, fs) | Hz | Hz | same |
| Resistance (Re) | Ω | Ω | same |
| Force factor (BL) | N/A | N/A | same |
| Mechanical resistance (Rms) | kg/s | kg/s | same |
| Voltage | V | V | same |

## Worked Examples from FLH Reference

```
Hornresp:  S1 = 40 cm²   → pyhorn: throat_area = 0.004 m²   (40 / 10000)
Hornresp:  S2 = 300 cm²  → pyhorn: mouth_area  = 0.03  m²   (300 / 10000)
Hornresp:  L12 = 152.7 cm→ pyhorn: path_length = 1.527 m     (152.7 / 100)
Hornresp:  Vrc = 3.24 L  → pyhorn: vrc         = 0.00324 m³ (3.24 / 1000)
Hornresp:  Vtc = 88 cm³  → pyhorn: vtc         = 0.000088 m³ (88 / 1e6)
Hornresp:  Lrc = 12 cm   → pyhorn: lrc         = 0.12 m      (12 / 100)
Hornresp:  Mmd = 6.12 g  → pyhorn: mms         = 0.00612 kg  (6.12 / 1000)
Hornresp:  Cms = 1.47E-03 mm/N → pyhorn: cms = 1.47E-03 m/N (already SI)
Hornresp:  Le  = 0.80 mH  → pyhorn: le          = 0.00080 H    (0.80 / 1000)
```

## Key Formula

```
area_m2  = area_cm2  / 10_000
vol_m3   = vol_L     / 1_000
mass_kg  = mass_g    / 1_000
length_m = length_cm / 100
```

## Geometric Tolerance Notes

### Mmd discrepancy: 6.12g (Hornresp) vs 6.99g (FE166NV2 datasheet)

**Finding: Intentional — NOT a discrepancy.**

The Hornresp reference benchmark deliberately uses **Mmd = 6.12g**, not the FE166NV2 datasheet value of 6.99g. This 12% difference is by design: Hornresp's internal TMM uses its own moving mass value, and the reference YAML matches Hornresp's value exactly.

The driver YAML explicitly documents this in its header:

```yaml
# Mmd=6.12g per Hornresp reference (not FE166NV2 spec Mmd=6.99g)
mms: 0.00612   # kg — moving mass (Hornresp Mmd, converted from 6.12g)
```

The test `test_mmd_matches_hornresp` in `test_vs_hornresp.py` explicitly asserts that pyhorn uses the Hornresp Mmd value (6.12g), confirming this is correct behavior.

**Root cause:** Not a measurement error, wrong driver, or bug. Hornresp and pyhorn both use Mmd=6.12g to ensure a valid apples-to-apples comparison. The FE166NV2 datasheet value (6.99g) is simply not used in this benchmark.

---

### Lrc discrepancy: 12.0 cm (Hornresp) vs `lrc: 0.18` (reported issue)

**Finding: Misunderstanding — pyhorn uses the correct value.**

The Hornresp reference specifies **Lrc = 12.0 cm**. The pyhorn benchmark YAML correctly converts this to `lrc: 0.12` (0.12 m = 12 cm). The value `lrc: 0.18` appearing in the BACKLOG issue was a misreading — the actual benchmark geometry YAML uses `lrc: 0.12`, matching Hornresp's 12.0 cm exactly.

Evidence from `hornresp_reference_flh.yaml`:

```yaml
#   Lrc (rear chamber depth): 12.0 cm = 0.12 m
lrc: 0.12
```

The unit conversion table above also documents the correct conversion: `Lrc = 12 cm → pyhorn: lrc = 0.12 m (12 / 100)`.

**Root cause:** No discrepancy exists. Both Hornresp and pyhorn use Lrc = 12 cm. The "0.18" in the issue was an error in reading the pyhorn value.
