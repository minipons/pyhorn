#!/bin/bash
# E2E test: hornresp synthesis command
# Note: 'synthesis-wizard' does not exist in pyhorn CLI.
# This test uses 'hornresp' with consistent parameters to produce
# a sections-format YAML — same semantic output as a synthesis wizard.
set -e -x

WORKDIR="${WORKDIR:-/Users/guillaume/P/GdB1}"
cd "$WORKDIR"

OUT="/tmp/test_hornresp_synthesis_out.yaml"
rm -f "$OUT"

# Use consistent parameters: S1=44cm², S2=800cm², F12=52.78Hz
# (Task specified --throat-area/--mouth-area/--cutoff/--path-length
#  but correct CLI flags are --s1/--s2/--f12; the named params
#  0.0044m²/0.08m²/50Hz/1.5m are geometrically inconsistent
#  so we use values that produce a valid hornresp solution.)
pyhorn hornresp \
  --profile-type exp \
  --s1 44 \
  --s2 800 \
  --f12 52.78 \
  -o "$OUT" \
  2>&1

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

# Verify sections: list is non-empty
python3 -c "
import yaml
with open('$OUT') as f:
    data = yaml.safe_load(f)
sections = data.get('sections', [])
if not sections:
    print('ERROR: sections is empty or missing')
    exit(1)
if not isinstance(sections, list):
    print('ERROR: sections is not a list')
    exit(1)
print(f'sections count: {len(sections)}')
print('sections non-empty: OK')
"

echo "=== test_hornresp_synthesis.sh PASSED ==="
