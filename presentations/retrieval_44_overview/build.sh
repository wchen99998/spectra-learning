#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/../shared/build_deck.sh" "$(dirname "$0")" retrieval_44_overview
