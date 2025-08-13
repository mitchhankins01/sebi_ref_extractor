#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

timestamp=$(date +%Y%m%d-%H%M%S)
RUN_DIR="runs/${timestamp}"
mkdir -p "$RUN_DIR"
INDEX_MD="${RUN_DIR}/index.md"
echo "# Batch run ${timestamp}" > "$INDEX_MD"
echo "" >> "$INDEX_MD"

shopt -s nullglob
for pdf in samples/*.pdf; do
  base=$(basename "$pdf")
  slug=$(echo "$base" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g')
  OUT_DIR="${RUN_DIR}/${slug}"
  mkdir -p "$OUT_DIR"

  echo "Processing: $base -> $slug" | sed 's/.*/&/'
  # Run extractor and capture stdout for counts
  set +e
  out=$(python extract_references.py \
    --pdf "$pdf" \
    --out "${OUT_DIR}/out.json" \
    --csv "${OUT_DIR}/out.csv" \
    --md  "${OUT_DIR}/out.md" \
    --external-only \
    --guess-titles \
    --debug-titles \
    --summary 2>&1)
  status=$?
  set -e

  counts_line=$(echo "$out" | awk '/^Summary \(external-only=/{flag=1;next} /^$/{flag=0} flag') || true
  if [[ -z "$counts_line" ]]; then
    counts_line="(no summary)"
  fi

  # Count cited titles
  cited=0
  if [[ -f "${OUT_DIR}/out.csv" ]]; then
    cited=$(python - "${OUT_DIR}/out.csv" <<'PY'
import csv,sys
from pathlib import Path
p=Path(sys.argv[1])
rows=0
with p.open() as f:
  r=csv.DictReader(f)
  for row in r:
    if (row.get('cited_title') or '').strip():
      rows+=1
print(rows)
PY
)
  fi

  echo "- ${base}" >> "$INDEX_MD"
  echo '```' >> "$INDEX_MD"
  echo "$counts_line" >> "$INDEX_MD"
  echo "cited_title rows: ${cited}" >> "$INDEX_MD"
  echo '```' >> "$INDEX_MD"
  echo "" >> "$INDEX_MD"

  if [[ $status -ne 0 ]]; then
    echo "Error processing ${base}. Continuing." >> "$INDEX_MD"
  fi
done

echo "Batch complete. See ${INDEX_MD}"


