#!/bin/bash
# E2E test: chamber-wizard command
set -e -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$WORKDIR"

OUT="/tmp/test_chamber_wizard_out.yaml"
rm -f "$OUT"

# Run chamber-wizard with reference driver (non-interactive)
pyhorn chamber-wizard \
  --driver drivers/FE166NV2.yaml \
  --no-interactive \
  -o "$OUT"

echo "Exit code: $?"

# Verify output YAML exists
if [ ! -f "$OUT" ]; then
  echo "ERROR: output YAML not found at $OUT"
  exit 1
fi
echo "Output YAML exists: OK"

# Verify valid YAML
python3 -c "import yaml; yaml.safe_load(open('$OUT'))" 2>/dev/null || {
  echo "ERROR: Output is not valid YAML"
  exit 1
}
echo "Valid YAML: OK"

# Verify has rear_chamber: with vrc: and lrc:
python3 -c "
import yaml
with open('$OUT') as f:
    data = yaml.safe_load(f)
rc = data.get('rear_chamber', {})
if 'rear_chamber' not in data:
    print('ERROR: rear_chamber: key not found')
    exit(1)
if 'vrc' not in rc:
    print('ERROR: vrc: not found in rear_chamber')
    exit(1)
if 'lrc' not in rc:
    print('ERROR: lrc: not found in rear_chamber')
    exit(1)
print('rear_chamber: vrc: lrc: all present: OK')
print(f'vrc={rc[\"vrc\"]}, lrc={rc[\"lrc\"]}')
"

echo "=== test_chamber_wizard.sh PASSED ==="
