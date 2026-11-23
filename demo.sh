#!/usr/bin/env bash
# Demo: onboard 4 factories, inspect the model, then replace one factory's
# zero-point correction by version number and watch correction + model +
# per-factory errors take effect together (and a stale-version conflict
# leave the current version untouched).
#
# Also prints the scalable-LOFO diagnostics: exact-integer summary-op count
# (linear in the number of factories) and matrix factorisations (one per
# fold, reused for all three lambdas).  scripts/demo_constant_pressure.py
# then shows follow-up corrections making one fold's corrected pressure a
# single constant, so that fold drops the pressure feature on its own.
set -euo pipefail

BASE="${BASE_URL:-http://localhost:8000}"
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
fmt() { if command -v jq >/dev/null 2>&1; then jq .; else cat; echo; fi; }
field() { "$PY" -c "import json,sys; print(json.load(sys.stdin)$1)"; }

echo "=== 1. onboard 4 factories -> POST /versions"
CREATE=$(curl -sS -X POST "$BASE/versions" -H 'Content-Type: application/json' \
  --data @demo/create_version.json)
echo "$CREATE" | fmt
V=$(echo "$CREATE" | field "['version']")
echo "$CREATE" | "$PY" -c "
import json,sys
t=json.load(sys.stdin)['training']
print('-> summary processing ops:', t['summary_ops'],
      '(legacy per-fold re-merge:', t['legacy_summary_ops'], '),',
      'matrix factorisations:', t['matrix_factorizations'])
print('-> current version: $V; per-fold feature diagnostics above')
"

echo "=== 2. predict for F3 with raw readings -> POST /predict"
curl -sS -X POST "$BASE/predict" -H 'Content-Type: application/json' \
  --data @demo/predict_f3.json | fmt

echo "=== 3. replace F3's zero-point correction (0 -> 500) based on version $V"
echo "    -> POST /versions/$V/correction"
UPDATED=$(curl -sS -X POST "$BASE/versions/$V/correction" \
  -H 'Content-Type: application/json' --data @demo/correction_update.json)
echo "$UPDATED" | fmt
echo "$UPDATED" | "$PY" -c "
import json,sys
t=json.load(sys.stdin)['training']
print('-> retrained with the same linear work:', t['summary_ops'], 'ops,',
      t['matrix_factorizations'], 'factorisations')
"

echo "=== 4. current version now: lambda + per-factory errors"
curl -sS "$BASE/versions/current" | fmt

echo "=== 5. predict again for F3 (uses the new version)"
curl -sS -X POST "$BASE/predict" -H 'Content-Type: application/json' \
  --data @demo/predict_f3.json | fmt

echo "=== 6. conflict: replay the update against stale version $V -> HTTP 409,"
echo "    current version kept"
curl -sS -o /dev/null -w "HTTP %{http_code}\n" -X POST \
  "$BASE/versions/$V/correction" -H 'Content-Type: application/json' \
  --data @demo/correction_update.json
curl -sS "$BASE/versions/current" | "$PY" -c \
  "import json,sys; j=json.load(sys.stdin); print('current version:', j['version'], '| F3 correction:', j['corrections']['F3'])"

echo "=== 7. corrections that make one fold's corrected pressure constant"
"$PY" scripts/demo_constant_pressure.py
