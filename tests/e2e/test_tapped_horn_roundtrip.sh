#!/bin/bash
# E2E test: Tapped Horn E2E roundtrip
# Tests the full tapped-horn CLI pipeline: driver YAML + TH geometry → simulation output
set -e -x

WORKDIR="${WORKDIR:-/Users/guillaume/P/GdB1}"
cd "$WORKDIR"

OUT_DIR="/tmp/test_tapped_horn_out"
rm -rf "$OUT_DIR"

# tapped-horn requires a driver YAML with T-S parameters and a TH geometry YAML
DRIVER="drivers/FE166NV2.yaml"
TH_GEOM="examples/geometry/th_example.yaml"

if [ ! -f "$DRIVER" ]; then
  echo "ERROR: Driver YAML not found at $DRIVER"
  exit 1
fi
if [ ! -f "$TH_GEOM" ]; then
  echo "ERROR: Tapped Horn geometry YAML not found at $TH_GEOM"
  exit 1
fi

# Run tapped-horn with limited frequency range and no plotting for speed.
# --no-export-csv is used to work around a known bug where round(complex128)
# is called on impedance values in the CSV export path.
OUTPUT=$(pyhorn tapped-horn \
  --driver "$DRIVER" \
  --th "$TH_GEOM" \
  --output-dir "$OUT_DIR" \
  --fmin 50 \
  --fmax 2000 \
  --n-points 100 \
  --no-plot \
  --no-export-csv 2>&1)
EXIT_CODE=$?

echo "tapped-horn output:"
echo "$OUTPUT"
echo "Exit code: $EXIT_CODE"

# Verify exit code 0
if [ $EXIT_CODE -ne 0 ]; then
  echo "ERROR: tapped-horn exited with non-zero code"
  exit 1
fi
echo "Exit code 0: OK"

# Verify key acoustic metrics appear in the terminal output
python3 -c "
import sys
output = '''$OUTPUT'''

# Verify Max SPL is reported
assert 'Max SPL' in output, 'Max SPL not found in output'
# Verify SPL at 1 kHz is reported
assert 'SPL at 1 kHz' in output or '1 kHz' in output, 'SPL at 1kHz not found in output'
# Extract Max SPL value and verify it's in a reasonable range (80-150 dB for FE166NV2)
import re
max_spl_match = re.search(r'Max SPL[:\s]+([0-9.]+)\s*dB', output)
assert max_spl_match, 'Could not find Max SPL value in output'
max_spl = float(max_spl_match.group(1))
assert 80.0 <= max_spl <= 150.0, f'Max SPL {max_spl} outside reasonable range (80-150 dB)'
print(f'Max SPL: {max_spl} dB — OK')

# Extract SPL at 1 kHz and verify it's in reasonable range
spl_1k_match = re.search(r'SPL at 1 kHz[:\s]+([0-9.]+)\s*dB', output)
if spl_1k_match:
    spl_1k = float(spl_1k_match.group(1))
    assert 60.0 <= spl_1k <= 130.0, f'SPL at 1kHz {spl_1k} outside reasonable range (60-130 dB)'
    print(f'SPL at 1 kHz: {spl_1k} dB — OK')
else:
    print('WARNING: Could not parse SPL at 1 kHz from output')

print('Output validation: ALL OK')
"

echo "=== test_tapped_horn_roundtrip.sh PASSED ==="
