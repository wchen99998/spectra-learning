#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname "$0")/../periodic_semantics/build_training_diagram.sh"
exec bash "$(dirname "$0")/../shared/build_deck.sh" "$(dirname "$0")" periodic_semantics_overview
