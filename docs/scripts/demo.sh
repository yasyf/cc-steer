#!/usr/bin/env bash
# Regenerates docs/assets/demo.png from a real run of `uvx cc-steer stats`.
# Requires freeze (https://github.com/charmbracelet/freeze) and a scanned
# ~/.cc-steer/feedback.db. Output shows counts only — no transcript text.
set -euo pipefail

cd "$(dirname "$0")/../.."

out="$(mktemp -t cc-steer-demo)"
trap 'rm -f "$out"' EXIT

printf '$ uvx cc-steer stats\n' >"$out"
uvx cc-steer stats >>"$out"

freeze "$out" \
  --language ansi \
  --theme github-dark \
  --background "#0d1117" \
  --window \
  --padding 24 \
  --font.family "JetBrains Mono" \
  --font.size 28 \
  --output docs/assets/demo.png
