#!/usr/bin/env bash
set -euo pipefail
deck_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$deck_dir/../.." && pwd)"
"$repo_root/.venv/bin/python" "$deck_dir/make_figures.py"
exec bash "$deck_dir/../shared/build_deck.sh" "$deck_dir" mist_modality
