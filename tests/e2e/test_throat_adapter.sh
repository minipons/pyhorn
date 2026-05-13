#!/bin/bash
# E2E test: throat-adapter command
set -e -x

WORKDIR="${WORKDIR:-/Users/guillaume/P/GdB1}"
cd "$WORKDIR"

# Run throat-adapter and capture output
OUTPUT=$(pyhorn throat-adapter --d1 50 --d2 100 2>&1)
echo "$OUTPUT"
EXIT=$?

echo "Exit code: $EXIT"

# Verify exit code 0
if [ "$EXIT" -ne 0 ]; then
  echo "ERROR: non-zero exit code"
  exit 1
fi

# Verify output has ap1: key
if ! echo "$OUTPUT" | grep -q "ap1:"; then
  echo "ERROR: ap1: not found in output"
  exit 1
fi
echo "ap1: found: OK"

# Verify output has lpt: key
if ! echo "$OUTPUT" | grep -q "lpt:"; then
  echo "ERROR: lpt: not found in output"
  exit 1
fi
echo "lpt: found: OK"

echo "=== test_throat_adapter.sh PASSED ==="
