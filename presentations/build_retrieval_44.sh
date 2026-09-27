#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

build_dir=build/retrieval_44
mkdir -p "$build_dir"
lualatex --version > "$build_dir/toolchain.txt"
for pass in 1 2; do
  if ! lualatex -interaction=nonstopmode -halt-on-error \
      -output-directory="$build_dir" retrieval_44.tex \
      > "$build_dir/compile-pass${pass}.txt"; then
    tail -n 50 "$build_dir/compile-pass${pass}.txt"
    exit 1
  fi
done
cp "$build_dir/retrieval_44.pdf" retrieval_44.pdf
printf 'Built %s/retrieval_44.pdf\n' "$PWD"
