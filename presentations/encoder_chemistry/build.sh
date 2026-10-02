#!/usr/bin/env bash
set -euo pipefail
report_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$report_dir/../.." && pwd)"
"$repo_dir/.venv/bin/python" "$report_dir/make_encoder_chemistry_figures.py"
bash "$report_dir/build_diagrams.sh"
bash "$report_dir/../shared/build_deck.sh" "$report_dir" encoder_chemistry
"$repo_dir/.venv/bin/python" "$report_dir/inspect_encoder_chemistry.py"
"$repo_dir/.venv/bin/python" "$report_dir/package_source.py"
