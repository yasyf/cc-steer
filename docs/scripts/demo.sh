#!/usr/bin/env bash
# Regenerates docs/assets/demo.png from a real run of `uvx cc-steer stats`.
# Requires freeze (https://github.com/charmbracelet/freeze) and a scanned
# ~/.cc-steer/feedback.db. Output shows counts only — no transcript text.
set -euo pipefail

cd "$(dirname "$0")/../.."

raw="$(mktemp -t cc-steer-demo)"
out="${raw}.ansi"
trap 'rm -f "$raw" "$out"' EXIT

printf '$ uvx cc-steer stats\n' >"$raw"
env -u UV_EXCLUDE_NEWER uvx cc-steer stats >>"$raw"

# click.echo emits no ANSI codes, so colorize the key: value layout with bat.
bat --plain --color=always --language yaml "$raw" >"$out"

freeze "$out" \
  --language ansi \
  --theme github-dark \
  --background "#0d1117" \
  --window \
  --padding 24 \
  --font.family "JetBrains Mono" \
  --font.size 28 \
  --output docs/assets/demo.png
