"""Run scripts/analyze_retrieval_reliance.py on Modal: sharded score ablations, gradients, drift.

    .venv/bin/modal run --detach scripts/modal_retrieval_reliance.py --model dual100m_tree_joint --what score
    .venv/bin/modal run --detach scripts/modal_retrieval_reliance.py --model ftjepa1p8b_joint --what grad
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

from scripts.modal_trainable_jepa_glacier import image, volume

PROJECT = Path(__file__).resolve().parents[1]
image = image.add_local_file(PROJECT / "configs/retrieval_4602.json", "/root/spectra-learning/configs/retrieval_4602.json")
app = modal.App("retrieval-reliance")
ROOT = "/results/retrieval_jepa_reliance_20261007"
FINGERPRINTS = "/results/pairfield/fingerprints"
MODELS = {
    "ftjepa1p8b_joint": ("/root/spectra-learning/configs/retrieval_4602.json",
                         "/results/retrieval_selected_4602_20260924/run/best.msgpack", []),
    "dual100m_tree_joint": ("/results/pairfield/inputs_dual100m.json",
                            f"{ROOT}/ckpt/dual100m_tree_joint/best.msgpack", ["--pair-field", "--tree"]),
    "dual100m_tree_head": ("/results/pairfield/inputs_dual100m.json",
                           f"{ROOT}/ckpt/dual100m_tree_head/best.msgpack", ["--pair-field", "--tree"]),
}


def _run(model: str, mode: str, extra: list[str], out: str):
    volume.reload()
    inputs, checkpoint, flags = MODELS[model]
    command = [sys.executable, "-X", "faulthandler", "scripts/analyze_retrieval_reliance.py", "--mode", mode,
               "--inputs", inputs, "--checkpoint", checkpoint, "--fingerprints", FINGERPRINTS,
               "--out", out, *flags, *extra]
    print(" ".join(command), flush=True)
    try:
        env = {**os.environ, "JAX_PLATFORMS": "cpu"} if mode == "drift" else None
        subprocess.run(command, cwd="/root/spectra-learning", check=True, env=env)
    finally:
        volume.commit()


@app.function(image=image, gpu=["H100", "H200", "B200"], cpu=8, memory=262144, timeout=14400,
              volumes={"/results": volume})
def score(model: str, split: str, shard: int, num_shards: int):
    _run(model, "score", ["--split", split, "--shard", str(shard), "--num-shards", str(num_shards)],
         f"{ROOT}/{model}/score_{split}")


@app.function(image=image, gpu=["H100", "H200", "B200"], cpu=8, memory=393216, timeout=14400,
              volumes={"/results": volume})
def grad(model: str, batches: int):
    _run(model, "grad", ["--grad-batches", str(batches)], f"{ROOT}/{model}/grad")


@app.function(image=image, cpu=8, memory=131072, timeout=7200, volumes={"/results": volume})
def drift(model: str):
    _run(model, "drift", [], f"{ROOT}/{model}/drift")


@app.function(image=image, cpu=4, memory=32768, timeout=1800, volumes={"/results": volume})
def merge(model: str, split: str):
    volume.reload()
    subprocess.run([sys.executable, "scripts/analyze_retrieval_reliance.py", "--mode", "merge",
                    "--out", f"{ROOT}/{model}/score_{split}"], cwd="/root/spectra-learning", check=True)
    volume.commit()


@app.local_entrypoint()
def main(model: str, what: str = "score", split: str = "val", shards: int = 8, batches: int = 4):
    if what == "score":
        list(score.starmap([(model, split, i, shards) for i in range(shards)]))
        merge.remote(model, split)
    elif what == "grad":
        grad.remote(model, batches)
    elif what == "drift":
        drift.remote(model)
    else:
        raise ValueError(what)
