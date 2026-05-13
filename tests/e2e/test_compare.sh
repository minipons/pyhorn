#!/bin/bash
# E2E test: compare command
set -e -x

WORKDIR="${WORKDIR:-/Users/guillaume/P/GdB1}"
cd "$WORKDIR"

OUTDIR="/tmp/test_compare_out"
rm -rf "$OUTDIR"

# Run compare with the same horn twice (should produce overlay)
pyhorn compare \
  pyhorn_core/tests/benchmarks/hornresp_reference_flh.yaml \
  pyhorn_core/tests/benchmarks/hornresp_reference_flh.yaml \
  -d pyhorn_core/tests/benchmarks/hornresp_reference_driver.yaml \
  -o "$OUTDIR"

echo "Exit code: $?"

# Verify PNG was created
PNG=$(find "$OUTDIR" -name "*.png" | head -1)
if [ -z "$PNG" ]; then
  echo "ERROR: No PNG file found in $OUTDIR"
  ls -la "$OUTDIR"
  exit 1
fi
echo "PNG found: $PNG"

# Verify PNG has non-zero file size
SIZE=$(stat -f%z "$PNG" 2>/dev/null || stat -c%s "$PNG" 2>/dev/null)
if [ "$SIZE" -eq 0 ]; then
  echo "ERROR: PNG file is empty"
  exit 1
fi
echo "PNG size: $SIZE bytes — OK"

echo "=== test_compare.sh PASSED ==="
