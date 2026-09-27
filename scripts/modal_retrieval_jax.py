"""Run the one supported retrieval recipe on the existing Modal data volume."""

import os
from pathlib import Path
import subprocess
import sys

import modal

from scripts.modal_trainable_jepa_glacier import image, volume

PROJECT = Path(__file__).resolve().parents[1]
image = image.add_local_file(PROJECT / "configs/retrieval_4602.json", "/root/spectra-learning/configs/retrieval_4602.json")
app = modal.App("retrieval-4602")


@app.function(image=image, cpu=16, memory=196608, ephemeral_disk=524288,
              gpu="H100", timeout=86400, volumes={"/results": volume})
def run(stage: str, out: str, head_checkpoint: str = "", resume: bool = False):
    command = [sys.executable, "scripts/run_retrieval_jax.py", "--inputs", "configs/retrieval_4602.json",
               "--stage", stage, "--out", out, "--session-hours", "22"]
    if head_checkpoint:
        command += ["--head-checkpoint", head_checkpoint]
    if resume:
        command.append("--resume")
    try:
        subprocess.run(command, cwd="/root/spectra-learning", env=os.environ.copy(), check=True)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(stage: str, out: str, head_checkpoint: str = "", resume: bool = False):
    run.remote(stage, out, head_checkpoint, resume)
