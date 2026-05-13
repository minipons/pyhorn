#!/bin/bash
# E2E test: auto-segment command
set -e -x

WORKDIR="${WORKDIR:-/Users/guillaume/P/GdB1}"
cd "$WORKDIR"

OUT="/tmp/test_auto_segment_out.yaml"
rm -f "$OUT"

# Use bk16.json from source/
pyhorn auto-segment \
  -i source/bk16.json \
  -o "$OUT" \
  --n-segments 16

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

# Verify has sections: key with non-empty list
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
print('sections: OK')
"

echo "=== test_auto_segment.sh PASSED ==="
