"""
CRIT-3 Physics Analysis — HF SPL Excess vs Hornresp

Root cause: voltage sensitivity reference mismatch between pyhorn and Hornresp.

The delta between V=0.5 and V=2.83 is exactly 15.06 dB everywhere in HF:
    20*log10(2.83/0.5) = 15.06 dB ✓ (voltage coupling is correct)

However:
    pyhorn V=0.5 ≈ Hornresp V=0.5  (nearly correct, ~−0.6 dB)
    pyhorn V=2.83 >> Hornresp V=2.83  (+14 to +28 dB excess)

This means the ABSOLUTE calibration of pyhorn at V=2.83 is ~15 dB too high
compared to Hornresp's dB/W/m reference.

Key observations from gdb1 geometry (throat=80cm², mouth=600cm², path=1.53m):

    Decade         pyhorn V=0.5  pyhorn V=2.83   Expected delta
    2000–5000 Hz:  83.5 dB       98.6 dB         15.06 dB ✓
    5000–10000 Hz: 75.2 dB       90.2 dB         15.06 dB ✓
   10000–20000 Hz: 67.9 dB       82.9 dB         15.06 dB ✓

Hornresp reference (from compare.py, V=2.83):
    2000–5000 Hz:  ~84 dB   (pyhorn: 98.6 dB → +14.6 dB excess)
   10000–20000 Hz: ~64 dB   (pyhorn: 82.9 dB → +18.9 dB excess)

Hypothesis: Hornresp normalizes HF SPL to the driver's sensitivity rating
(dB/W/m at 1m, 1W input), while pyhorn uses a raw voltage-to-acoustic
conversion without sensitivity normalization.

The Levine/Inglis piston radiation impedance model (Morse & Ingard §9.3, 1968)
is correct in structure — the +14 to +28 dB excess cannot be explained
by a simple 2× factor in radiation resistance (would only give +3 dB).

Likely cause: Hornresp applies an additional HF directivity or efficiency
correction to the raw TMM output. pyhorn's Levine/Inglis model is accurate
for the piston impedance itself but may not include Hornresp's HF sensitivity
normalization (possibly related to the 1W/1m reference calibration).

Investigation paths:
1. Check Hornresp's "sensitivity" setting and how it affects HF absolute level
2. Compare pyhorn's V=0.5 output against Hornresp's dB/W/m reference directly
3. The ratio pyhorn_V=2.83 / Hornresp_V=2.83 ≈ 15 dB in HF → sensitivity offset
"""
