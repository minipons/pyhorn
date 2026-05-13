#!/bin/bash
# E2E test: synthesis-wizard CLI command
# Tests the full synthesis-wizard pipeline: driver YAML → geometry YAML
set -e -x

WORKDIR="${WORKDIR:-/Users/guillaume/P/GdB1}"
cd "$WORKDIR"

OUT="/tmp/test_synthesis_wizard_out.yaml"
rm -f "$OUT"

# synthesis-wizard requires a driver YAML with T-S parameters
# Use the benchmark driver (FE166NV2-like) that already exists in the repo
DRIVER="pyhorn_core/tests/benchmarks/hornresp_reference_driver.yaml"
if [ ! -f "$DRIVER" ]; then
  echo "ERROR: Driver YAML not found at $DRIVER"
  exit 1
fi

# Run synthesis-wizard targeting f3=50 Hz
pyhorn synthesis-wizard \
  --driver "$DRIVER" \
  --f3 50.0 \
  --f7 8000.0 \
  --qts-alignment 0.55 \
  --output "$OUT" \
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

# Verify required top-level keys
python3 -c "
import yaml
with open('$OUT') as f:
    text = f.read()
# The geometry section starts after 'Synthesised geometry' marker
marker = '# ── Synthesised geometry'
idx = text.find(marker)
if idx == -1:
    print('ERROR: could not find Synthesised geometry marker')
    exit(1)
geo_text = text[idx:]
geo = yaml.safe_load(geo_text)

# Validate required fields
assert 'sections' in geo, 'sections missing'
assert isinstance(geo['sections'], list), 'sections must be a list'
assert len(geo['sections']) > 0, 'sections must not be empty'
sec = geo['sections'][0]
for field in ('name', 'profile_type', 'length', 'start_area', 'end_area'):
    assert field in sec, f'section missing {field}'
assert sec['length'] > 0, 'length must be > 0'
assert sec['start_area'] > 0, 'start_area must be > 0'
assert sec['end_area'] > 0, 'end_area must be > 0'
assert sec['end_area'] >= sec['start_area'], 'mouth must be >= throat (expanding horn)'
print('sections block: OK')
print(f'section name: {sec[\"name\"]}')
print(f'profile_type: {sec[\"profile_type\"]}')
print(f'length: {sec[\"length\"]:.4f} m')
print(f'throat area: {sec[\"start_area\"]*1e4:.2f} cm²')
print(f'mouth area: {sec[\"end_area\"]*1e4:.2f} cm²')
"

echo "=== test_synthesis_wizard.sh PASSED ==="
