#!/usr/bin/env bash
set -euo pipefail
report_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$report_dir/../.." && pwd)"
"$repo_dir/.venv/bin/python" "$report_dir/make_figures.py"
"$repo_dir/.venv/bin/python" "$report_dir/prepare_results.py"
bash "$report_dir/build_diagrams.sh"
bash "$report_dir/../shared/build_deck.sh" "$report_dir" diffusion_chemistry
"$repo_dir/.venv/bin/python" "$report_dir/inspect_diffusion_chemistry.py"
"$repo_dir/.venv/bin/python" "$report_dir/package_source.py"
