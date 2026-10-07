"""Pair-field retrieval on Modal: one process per GPU in a single 8-GPU container.

Stage inputs once (``prep``), then run ``scripts/train_pairfield_retrieval.py``
stages with the same arguments as on Isambard:

    .venv/bin/modal run --detach scripts/modal_pairfield_retrieval.py::prep \\
        --inputs /results/pairfield/inputs_dual100m.json --run-id dual-100m-n32n64-gh200x16-20261003 --step 300000
    .venv/bin/modal run --detach scripts/modal_pairfield_retrieval.py::train \\
        --out /results/pairfield/runs/<name>/head \\
        --args "--inputs /results/pairfield/inputs_dual100m.json --fingerprints /results/pairfield/fingerprints --stage head --pair-field --tree"
"""

import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import modal

from scripts.modal_trainable_jepa_glacier import image, volume

GPUS = 8
PROJECT = Path(__file__).resolve().parents[1]
image = image.add_local_file(PROJECT / "configs/retrieval_4602.json", "/root/spectra-learning/configs/retrieval_4602.json")
app = modal.App("pairfield-retrieval")


def _commit_every(seconds: float, stop: threading.Event):
    while not stop.wait(seconds):
        volume.commit()


@app.function(image=image, cpu=16, memory=131072, timeout=7200, volumes={"/results": volume})
def prep(inputs: str, run_id: str, step: int, reference: str = "/root/spectra-learning/configs/retrieval_4602.json"):
    volume.reload()
    subprocess.run([sys.executable, "scripts/prepare_compact_dual_inputs.py", "--inputs", inputs,
                    "--reference-inputs", reference, "--run-id", run_id, "--step", str(step)],
                   cwd="/root/spectra-learning", check=True)
    volume.commit()


RESOURCES = dict(cpu=64, memory=1048576, ephemeral_disk=524288, timeout=86400, volumes={"/results": volume})


@app.function(image=image, gpu=[f"B200:{GPUS}", f"H200:{GPUS}", f"H100:{GPUS}"], **RESOURCES)
def train(out: str, args: str):
    _train(out, args)


def _train(out: str, args: str):
    volume.reload()
    command = [sys.executable, "-X", "faulthandler", "scripts/train_pairfield_retrieval.py",
               *shlex.split(args), "--out", out]
    if os.path.exists(os.path.join(out, "last.msgpack")) and "--resume" not in command:
        command.append("--resume")
    print(" ".join(command), flush=True)
    stop = threading.Event()
    threading.Thread(target=_commit_every, args=(600, stop), daemon=True).start()
    processes = []
    for rank in range(GPUS):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(rank), "PF_NUM_PROCESSES": str(GPUS),
               "PF_PROCESS_ID": str(rank), "PF_COORDINATOR": "127.0.0.1:12355",
               "XLA_PYTHON_CLIENT_PREALLOCATE": "true", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.92",
               "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"}
        log = None if rank == 0 else open(f"/tmp/rank{rank}.log", "w")
        processes.append(subprocess.Popen(command, cwd="/root/spectra-learning", env=env,
                                          stdout=log, stderr=subprocess.STDOUT if log else None))
    try:
        codes = [None] * GPUS
        while any(code is None for code in codes):
            for rank, process in enumerate(processes):
                if codes[rank] is None:
                    try:
                        codes[rank] = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            if any(code not in (None, 0) for code in codes):
                break
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        stop.set()
        volume.commit()
    if any(code != 0 for code in codes):
        for rank in range(1, GPUS):
            if codes[rank] not in (None, 0):
                print(f"--- rank {rank} ---\n" + Path(f"/tmp/rank{rank}.log").read_text()[-4000:], flush=True)
        raise RuntimeError(f"Ranks exited with {codes}")


@app.function(image=image, cpu=32, memory=262144, timeout=7200, volumes={"/results": volume})
def smoke_cpu(out: str, args: str):
    """Single-process CPU pipeline test (``--smoke``) that leaves the GPU slots free."""
    volume.reload()
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    # XLA:CPU has no bf16 x bf16 -> f32 dot, so the smoke test builds the encoder in f32.
    wrapper = (
        "import functools, runpy, sys, jax.numpy as jnp\n"
        "import spectra_learning.models.model_jax as m\n"
        "build = m.build_peak_set_encoder_jax\n"
        "m.build_peak_set_encoder_jax = lambda *a, **k: build(*a, **{**k, 'compute_dtype': jnp.float32})\n"
        "sys.argv = sys.argv[1:]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n")
    try:
        subprocess.run([sys.executable, "-X", "faulthandler", "-c", wrapper, "scripts/train_pairfield_retrieval.py",
                        *shlex.split(args), "--smoke", "--out", out], cwd="/root/spectra-learning", env=env, check=True)
    finally:
        volume.commit()
