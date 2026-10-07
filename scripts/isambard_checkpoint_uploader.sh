#!/bin/bash
# Copies finished Isambard training checkpoints to GCS from a login node:
# every 5 min, each run under $LOCAL_ROOT (written by
# isambard_gh200_train.sbatch with CKPT_LOCAL_ROOT) goes to $REMOTE_ROOT/<run>.
# Compute nodes never talk to GCS; their shared NAT can drop to KB/s, while
# login nodes have their own, faster route.
#
# Login nodes kill a user's processes when the SSH session ends
# (KillUserProcesses=yes, no cron/tmux/linger), so the uploader lives inside
# an SSH session that a workstation keeps open:
#   scripts/isambard_checkpoint_uploader.sh run               # on a login node
#   scripts/isambard_checkpoint_uploader.sh keepalive <ssh args...>
#                                     # on a workstation; reconnects on drops
# Reconnecting needs a valid clifton certificate; an open session survives
# the certificate's expiry.
set -euo pipefail

case ${1:-} in
  run)
    LOCAL_ROOT=${LOCAL_ROOT:-/projects/u6rb/spectra-checkpoints}
    REMOTE_ROOT=${REMOTE_ROOT:-gs://metal-repeater-411410-spectra-checkpoints/isambard}
    PYTHON=${PYTHON:-$HOME/spectra-mrl/.venv-cu12/bin/python}
    export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/spectra-isambard-ckpt.json}
    echo "$(date -u +%FT%TZ) uploader on $(hostname)" >&2
    exec nice -n 10 "$PYTHON" "$(cd "$(dirname "$0")" && pwd)/gcs_checkpoint_sync.py" \
      push --watch 300 --local-root "$LOCAL_ROOT" --remote-root "$REMOTE_ROOT"
    ;;
  keepalive)
    shift
    REMOTE_SCRIPT=${REMOTE_SCRIPT:-/scratch/u6rb/wuhao.u6rb/spectra-dual-final/scripts/isambard_checkpoint_uploader.sh}
    while true; do
      ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=5 -o BatchMode=yes \
        "$@" "$REMOTE_SCRIPT run" || true
      echo "$(date -u +%FT%TZ) ssh session ended; reconnecting in 60 s" >&2
      sleep 60
    done
    ;;
  *) echo "usage: $0 run | keepalive <ssh args...>" >&2; exit 2 ;;
esac
