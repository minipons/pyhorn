#!/bin/bash
# E2E test: basic calculate command
set -e -x

WORKDIR="${WORKDIR:-/Users/guillaume/P/GdB1}"
cd "$WORKDIR"

OUTDIR="/tmp/test_calculate_out"
rm -rf "$OUTDIR"

# Run the calculate command
pyhorn calculate \
  -d pyhorn_core/tests/benchmarks/hornresp_reference_driver.yaml \
  -h pyhorn_core/tests/benchmarks/hornresp_reference_flh.yaml \
  --no-plot \
  --no-plot-3d \
  --no-export-json \
  -o "$OUTDIR"

echo "Exit code: $?"

# Find response.csv (CLI creates a subdirectory with horn/project name)
CSV=$(find "$OUTDIR" -name "response.csv" | head -1)
if [ -z "$CSV" ]; then
  echo "ERROR: response.csv not found in $OUTDIR"
  find "$OUTDIR" -type f
  exit 1
fi
echo "response.csv found: $CSV"

# Verify Frequency column is present
if ! head -1 "$CSV" | tr ',' '\n' | grep -qi "frequency"; then
  echo "ERROR: Frequency column not found in CSV header"
  head -1 "$CSV"
  exit 1
fi
echo "Frequency column: OK"

# Verify SPL column exists and has values in reasonable range
SPL_COL=$(head -1 "$CSV" | tr ',' '\n' | grep -ni "spl" | head -1 | cut -d: -f1 || true)
if [ -z "$SPL_COL" ]; then
  echo "ERROR: SPL column not found in CSV"
  head -1 "$CSV"
  exit 1
fi
echo "SPL column index: $SPL_COL"

# Check that at least some SPL values are in the 60-120 dB range
REASONABLE=$(tail -n +2 "$CSV" | cut -d, -f"$SPL_COL" | grep -E '^[0-9]+\.?[0-9]*$' | awk '$1 >= 60 && $1 <= 120' | wc -l | tr -d ' ')
TOTAL_NUMERIC=$(tail -n +2 "$CSV" | cut -d, -f"$SPL_COL" | grep -E '^[0-9]+\.?[0-9]*$' | wc -l | tr -d ' ')
echo "SPL values in 60-120 dB range: $REASONABLE / $TOTAL_NUMERIC total numeric values"
if [ "$REASONABLE" -gt 0 ]; then
  echo "PASS: SPL values in reasonable range found"
else
  echo "FAIL: No SPL values in 60-120 dB range"
  tail -5 "$CSV"
  exit 1
fi

echo "=== test_basic_calculate.sh PASSED ==="
