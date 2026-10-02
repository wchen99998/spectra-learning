#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/figures"
export TEXMFCACHE="$PWD/../../shared/build/tex-cache"
mkdir -p ../build/diagrams "$TEXMFCACHE"
for diagram in checkpoint_lineage token_sae_interventions metadata_paths; do
  output="../build/diagrams/$diagram"
  mkdir -p "$output"
  if ! lualatex -interaction=nonstopmode -halt-on-error -output-directory="$output" \
      "$diagram.tex" > "$output/compile.txt"; then
    tail -n 45 "$output/compile.txt"
    exit 1
  fi
  cp "$output/$diagram.pdf" "$diagram.pdf"
  if ! lualatex -interaction=nonstopmode -halt-on-error -output-directory="$output" \
      -jobname="${diagram}_slide" "\\def\\SlideDiagram{1}\\input{$diagram.tex}" > "$output/compile-slide.txt"; then
    tail -n 45 "$output/compile-slide.txt"
    exit 1
  fi
  cp "$output/${diagram}_slide.pdf" "${diagram}_slide.pdf"
  pdfinfo "$diagram.pdf" > "$output/pdfinfo.txt"
  pdffonts "$diagram.pdf" > "$output/fonts.txt"
  pdfimages -list "$diagram.pdf" > "$output/images.txt"
  pdftotext -layout "$diagram.pdf" "$output/text.txt"
  pdftoppm -r 200 -singlefile -png "$diagram.pdf" "$output/render" >/dev/null 2>&1
  pdfinfo "${diagram}_slide.pdf" > "$output/pdfinfo-slide.txt"
  pdffonts "${diagram}_slide.pdf" > "$output/fonts-slide.txt"
  pdftotext -layout "${diagram}_slide.pdf" "$output/text-slide.txt"
  pdftoppm -r 200 -singlefile -png "${diagram}_slide.pdf" "$output/render-slide" >/dev/null 2>&1
done
