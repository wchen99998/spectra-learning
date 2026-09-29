#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
for deck in spectra retrieval_44 retrieval_44_overview retrieval_failures periodic_semantics periodic_semantics_overview; do
  bash "$root/$deck/build.sh"
done
