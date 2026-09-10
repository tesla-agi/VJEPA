#!/usr/bin/env bash
# Build docs/VJEPA_report.pdf from docs/report.md
set -euo pipefail
cd "$(dirname "$0")/.."

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT_HTML="docs/report.html"
OUT_PDF="docs/VJEPA_report.pdf"

pandoc docs/report.md \
  --standalone --embed-resources \
  --css=docs/style.css \
  --toc --toc-depth=2 \
  --metadata title-prefix="" \
  --resource-path=docs:.:runs/vjepa:runs/vjepa/viz \
  -o "$OUT_HTML"

"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=10000 \
  --print-to-pdf="$OUT_PDF" "$OUT_HTML" 2>/dev/null || true

if [ -f "$OUT_PDF" ]; then
  echo "wrote $OUT_PDF  ($(du -h "$OUT_PDF" | cut -f1))"
else
  echo "PDF failed; HTML is at $OUT_HTML" >&2; exit 1
fi
