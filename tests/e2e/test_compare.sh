#!/bin/bash
# E2E test: compare command
set -e -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$WORKDIR"

OUTDIR="/tmp/test_compare_out"
rm -rf "$OUTDIR"

# Run compare with the same horn twice (should produce overlay)
pyhorn compare \
  tests/benchmarks/hornresp/hirob/fixture/horn.yaml \
  tests/benchmarks/hornresp/hirob/fixture/horn.yaml \
  -d drivers/FE166NV2.yaml \
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
