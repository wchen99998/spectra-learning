#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
lualatex -interaction=nonstopmode -halt-on-error -output-directory=build novogaia_spectra.tex > build/compile-pass1.txt
lualatex -interaction=nonstopmode -halt-on-error -output-directory=build novogaia_spectra.tex > build/compile-pass2.txt
cp build/novogaia_spectra.pdf novogaia_spectra.pdf
printf 'Built %s/novogaia_spectra.pdf\n' "$PWD"
