#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/figures"
export TEXMFCACHE="$PWD/../../shared/build/tex-cache"
mkdir -p ../build/training-diagram "$TEXMFCACHE"
if ! lualatex -interaction=nonstopmode -halt-on-error -output-directory=../build/training-diagram training_schemes.tex > ../build/training-diagram/compile.txt; then
  tail -n 40 ../build/training-diagram/compile.txt
  exit 1
fi
cp ../build/training-diagram/training_schemes.pdf training_schemes.pdf
cp training_schemes.pdf training_schemes_slide.pdf
