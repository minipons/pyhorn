# GdB1 Hornresp vs pyhorn — Discrepancy Investigation Test Plan

**Date:** 2026-05-04
**Status:** Active

## Observed Discrepancy Summary

| Band | Mean Δ (pyhorn − Hornresp) | Direction |
|------|---------------------------|-----------|
| 10–200 Hz | −11 to −15 dB | pyhorn too LOW |
| 200–2000 Hz | −4 dB | pyhorn slightly low |
| 2–5 kHz | +5 dB | pyhorn too HIGH |
| 5–20 kHz | +14 to +19 dB | pyhorn way too HIGH |

**Overall:** mean −1.6 dB, std 13 dB. Systematic LF deficit, HF excess.

---

## Hypothesis A: Mms / Mmd Discrepancy

**Observation:** Hornresp uses Mmd=6.04g (diaphragm mass only). pyhorn FE166NV2 YAML uses Mms=6.99g (effective moving mass including voice coil). Difference: 16%.

**Why it matters:** Mms affects:
- Free-air resonance (fs ∝ 1/√Mms)
- Low-frequency response shape
- Driver excursion at LF (higher Mms → more excursion → potentially more coupling to horn)

**Test A1:** Run with Mms=6.04g (Hornresp value) — expect LF improvement
**Test A2:** Run with Mms=6.99g (FE166NV2 spec) — baseline

```yaml
# In driver YAML, test with:
mms: 0.00604   # Hornresp value
# vs
mms: 0.00699   # FE166NV2 spec
```

---

## Hypothesis B: Voltage / Sensitivity Reference Mismatch

**Observation:** Hornresp Eg=2.83V, Rg=0Ω. This equals 1W into 8Ω (P=E²/R = 2.83²/8 = 1.0W). SPL is referenced to 1W/m.

**Why it matters:** If pyhorn's `voltage: 2.83` is treated as driving a fixed impedance load differently, or if the sensitivity reference is computed as dB/2.83V/1m instead of dB/W/m, that would shift the curve by up to ~9 dB (8Ω reference).

**Test B1:** Compare SPL at 1 kHz (middle of range) between:
- Hornresp: ~105 dB
- pyhorn with voltage=2.83: what value?

The raw difference at mid-band is ~−4 dB. If we can confirm this is a constant offset (not frequency-dependent), it's a sensitivity reference issue.

**Test B2:** Try voltage=1.0 (1W equivalent) — does it shift the whole curve down by ~9 dB?

```yaml
# Try:
voltage: 1.0   # 1W reference instead of 2.83V
```

---

## Hypothesis C: Radiation Angle (Ang) Mapping

**Observation:** Hornresp Ang=0.5π (half-space). pyhorn default Ang may differ.

**Test C1:** Verify pyhorn is using Ang=π/2 (half-space, 2π steradians). If pyhorn uses a different convention, HF directivity would be affected.

**Test C2:** Try Ang=π (full space) and observe HF change.

---

## Hypothesis D: Rear Chamber Vrc / Lrc Mapping

**Observation:** Hornresp: Vrc=5.0L, Lrc=15cm. pyhorn YAML uses same values.

**Why it matters:** The rear chamber acts as a sealed box in series with the horn throat. If Vrc is interpreted differently (acoustic volume vs physical volume), LF response changes.

**Test D1:** Sensitivity sweep — vary Vrc from 2L to 10L in 1L steps. Plot how Fb (system resonance) shifts.

**Test D2:** Sensitivity sweep — vary Lrc from 5cm to 25cm. Plot how Vrc effective acoustic length changes.

---

## Hypothesis E: Throat Chamber Vtc / Atc Mapping

**Observation:** Hornresp: Vtc=160cm³, Atc=80cm². pyhorn YAML uses same.

**Why it matters:** The throat chamber is a sealed volume at the horn throat. Wrong Vtc would shift the mid-bass response.

**Test E1:** Verify Vtc=0.00016 m³ (160cm³) is being used correctly
**Test E2:** Sensitivity sweep on Vtc ±50%

---

## Hypothesis F: Mouth Radiation Model

**Observation:** pyhorn is consistently too high above 2 kHz (+5 to +19 dB).

**Why it matters:** Mouth radiation impedance (where the horn opens to free air) is modeled differently. At high frequencies, the mouth acts as a source of radiation, and the model matters a lot.

**Test F1:** Check if pyhorn uses the Levine/Inglis radiation impedance model (same as Hornresp). If not, this would explain HF divergence.

**Test F2:** At frequencies where the horn mouth is large relative to wavelength, both tools should converge. The divergence above 5 kHz where mouth is ~λ/4 suggests a radiation impedance model difference.

---

## Hypothesis G: Voltage → LF Disconnect (CRITICAL FINDING)

**Observation:** Sweeping voltage from 0.5V to 2.83V changes HF by ~15 dB but LF stays locked at **−12 dB regardless**. This means voltage is only affecting the HF path, not the LF loading region.

**Why it matters:** In a BLH, the driver is loaded by the horn at LF. If voltage changes aren't affecting LF output, something is wrong with how the voltage source couples to the horn-loaded driver in the TMM.

**Test G1 (PRIORITY 1):** Investigate why voltage doesn't affect LF. Check the transfer matrix multiplication — does the voltage source (Eg) only appear in the electrical impedance calculation, not in the acoustic pressure calculation?

**Test G2:** Verify the driving point impedance at the diaphragm is being used to compute the pressure transfer function correctly at LF.

---

## Hypothesis H: Throat Adapter ap1 Has Zero Effect (BUG?)

**Observation:** Sweeping ap1 from 20cm² to 200cm² produces **zero change** in SPL. The throat adapter parameter is completely inactive.

**Why it matters:** If ap1 isn't wired into the TMM, the horn throat boundary condition is wrong.

**Test H1:** Check if `ap1` and `lpt` are actually used in `throat_adapter_tmatrix()` or the horn response computation.

**Test H2:** Compare Hornresp WITH throat adapter vs WITHOUT to understand its effect in Hornresp.

---

## Hypothesis I: Vrc Sensitivity

**Observation:** Increasing Vrc from 5L to 20L improves LF from −12 to −10 dB (2 dB improvement). The Hornresp value of 5L is already small.

**Why it matters:** A larger rear chamber reduces the system Q but doesn't fundamentally change horn loading. The persistent −12 dB LF deficit even at 20L suggests the issue isn't just Vrc.

**Test I1:** Try Vrc=50L, 100L — does LF ever converge to 0 delta?

---

## Systematic Test Execution

### Actual Sweep Results

**Mms sweep (V=2.83):**
| Mms | mean Δ | LF | HF |
|-----|--------|----|----|
| 6.99g | −2.15 | −12.11 | +13.10 |
| 6.04g | −1.61 | −12.12 | +14.41 |
| 5.50g | −1.27 | −12.13 | +15.25 |

→ **Mms has negligible effect** — not the cause of LF or HF issues.

**Voltage sweep (Mms=6.04g):**
| Voltage | mean Δ | LF | HF |
|---------|--------|----|----|
| 2.83 | −1.61 | −12.12 | +14.41 |
| 1.00 | −10.64 | −21.16 | +5.38 |
| 0.50 | −16.66 | −27.18 | −0.64 |

→ **Voltage controls HF (and overall level) but NOT LF.** LF stays at −12 dB regardless. This is the critical bug.

**Vrc sweep:**
| Vrc | LF |
|------|-----|
| 1L | −12.12 |
| 5L (Hornresp) | −12.12 |
| 10L | −10.68 |
| 20L | −9.98 |

→ Larger Vrc helps LF slightly but Hornresp's 5L is already optimal for the model.

**ap1 sweep:** ALL values produce identical SPL — **ap1 is completely inactive (BUG).**

### Priority Fix Order

| Priority | Issue | Action |
|----------|-------|--------|
| 🔴 P0 | LF stuck at −12 dB regardless of voltage | Investigate voltage coupling in TMM — why doesn't voltage affect LF? |
| 🔴 P0 | `ap1` has zero effect | Verify throat adapter TMM is actually being applied |
| 🟡 P1 | HF excess (+14 dB at V=2.83) | Understand voltage → HF scaling relationship |
| 🟢 P2 | Vrc/Lrc optimization | Fine-tune rear chamber for best LF |
| 🟢 P3 | End correction | Verify end correction is applied |

### Target
```
All decades: mean Δ within ±2 dB, std < 3 dB
```
