"""Embed and retrieve the out-of-distribution MGF spectra on the retrieval data volume."""

import os
from pathlib import Path
import subprocess
import sys

import modal

from scripts.modal_trainable_jepa_glacier import image, volume

PROJECT = Path(__file__).resolve().parents[1]
image = (image.add_local_file(PROJECT / "configs/retrieval_4602.json", "/root/spectra-learning/configs/retrieval_4602.json")
         .add_local_dir(PROJECT / "data/test_mgf", "/root/spectra-learning/data/test_mgf"))
app = modal.App("ood-mgf-retrieval")


@app.function(image=image, cpu=16, memory=196608, gpu="H100", timeout=4 * 3600, volumes={"/results": volume})
def run():
    try:
        subprocess.run([sys.executable, "scripts/ood_mgf_retrieval.py"], cwd="/root/spectra-learning",
                       env=os.environ.copy(), check=True)
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    run.remote()
