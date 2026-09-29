#!/usr/bin/env bash
set -euo pipefail
cd "$1"
document="$2"
export TEXINPUTS="../shared//:${TEXINPUTS:-}"
export TEXMFCACHE="$PWD/../shared/build/tex-cache"
mkdir -p "$TEXMFCACHE"
mkdir -p build
for pass in 1 2; do
  if ! lualatex -interaction=nonstopmode -halt-on-error -output-directory=build \
      "$document.tex" > "build/compile-pass${pass}.txt"; then
    tail -n 40 "build/compile-pass${pass}.txt"
    exit 1
  fi
done
cp "build/$document.pdf" "$document.pdf"
printf 'Built %s/%s.pdf\n' "$PWD" "$document"
