#!/bin/bash
# E2E test: hornresp import
# Task specified: --throat-area 0.0044 --mouth-area 0.08 --path-length 1.5 --cutoff 50
# These flag names don't exist; correct flags are --s1 --s2 --l12 --f12.
# The named values are geometrically inconsistent with F12=50Hz for exp profile,
# so we use --s1 44 --s2 800 --f12 52.78 (computed consistent value) and
# omit --l12 to let hornresp solve it.
set -e -x

WORKDIR="${WORKDIR:-/Users/guillaume/P/GdB1}"
cd "$WORKDIR"

# Capture stdout (contains YAML)
OUTPUT=$(pyhorn hornresp \
  --profile-type exp \
  --s1 44 \
  --s2 800 \
  --f12 52.78 \
  2>&1)
EXIT=$?

echo "Exit code: $EXIT"

# Verify exit code 0
if [ "$EXIT" -ne 0 ]; then
  echo "ERROR: non-zero exit code"
  exit 1
fi

# Verify output contains YAML with sections format
if ! echo "$OUTPUT" | grep -q "sections:"; then
  echo "ERROR: sections: not found in output"
  exit 1
fi
echo "sections: found: OK"

# Extract YAML block and validate
YAML_BLOCK=$(echo "$OUTPUT" | sed -n '/^enclosure_type:/,$p')
python3 -c "
import yaml, sys
yaml.safe_load(sys.stdin)
" <<< "$YAML_BLOCK" || {
  echo "ERROR: output is not valid YAML"
  exit 1
}
echo "Valid YAML with sections format: OK"

echo "=== test_hornresp_import.sh PASSED ==="
