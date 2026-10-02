#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/figures"
export TEXMFCACHE="$PWD/../../shared/build/tex-cache"
mkdir -p ../build/diagrams "$TEXMFCACHE"
if ! lualatex -interaction=nonstopmode -halt-on-error -output-directory=../build/diagrams \
    native_token_sae.tex > ../build/diagrams/compile.txt; then
  tail -n 40 ../build/diagrams/compile.txt
  exit 1
fi
cp ../build/diagrams/native_token_sae.pdf native_token_sae.pdf
cp native_token_sae.pdf native_token_sae_slide.pdf
pdftoppm -r 200 -singlefile -png native_token_sae.pdf ../build/diagrams/native_token_sae >/dev/null 2>&1
