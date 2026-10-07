"""Run scripts/export_mgf_tokens.py parity + sharded GPU export on the retrieval data volume.

Upload `prepared_inputs.npz` (stage prepare, run locally) to `/<out>/` on the volume first:

    modal volume put spectra-massspecgym-maccs-probe <local>/prepared_inputs.npz /<out>/prepared_inputs.npz
    modal run scripts/modal_export_mgf_tokens.py --out <out> --rows <N>
"""

import subprocess
import sys

from scripts.modal_trainable_jepa_glacier import image, volume

import modal

app = modal.App("export-mgf-tokens")
CHECKPOINT = "/results/checkpoints/ftjepa_1p8b_step1190000"
MSG = {
    "--msg-tsv": "/results/jepa_fps_ftjepa_1p8b_step1190000_flare_raw/msg15/MassSpecGym1.5.tsv",
    "--msg-prepared": "/results/massspecgym_maccs_mz_order_metadata_n128_v2/prepared_massspecgym.npz",
    "--msg-cached-tokens": "/results/jepa_fps_ftjepa_1p8b_step1190000_flare_raw/covariance_embeddings.float16.npy",
}


def _run(stage: str, out: str, *extra: str) -> None:
    command = [sys.executable, "scripts/export_mgf_tokens.py", "--stage", stage, "--out", f"/results/{out}",
               "--jepa-metadata", f"{CHECKPOINT}/metadata/metadata",
               "--jepa-checkpoint", f"{CHECKPOINT}/snapshot", *extra]
    subprocess.run(command, cwd="/root/spectra-learning", check=True)


@app.function(image=image, cpu=8, memory=65536, timeout=3600, volumes={"/results": volume})
def parity(out: str):
    _run("parity", out, "--msg-tsv", MSG["--msg-tsv"], "--msg-prepared", MSG["--msg-prepared"])


@app.function(image=image, cpu=8, memory=65536, gpu="H100", timeout=6 * 3600,
              volumes={"/results": volume}, max_containers=8)
def shard(out: str, start: int, stop: int):
    try:
        extra = ["--msg-prepared", MSG["--msg-prepared"], "--msg-cached-tokens", MSG["--msg-cached-tokens"]] if start == 0 else []
        _run("shard", out, "--start", str(start), "--stop", str(stop), *extra)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(out: str, rows: int, shards: int = 8, smoke: bool = False):
    parity.remote(out)
    if smoke:
        shard.remote(out, 0, 256)
        return
    size = -(-rows // shards // 128) * 128
    bounds = [(s, min(s + size, rows)) for s in range(0, rows, size)]
    list(shard.starmap([(out, s, e) for s, e in bounds]))
    print({"parts": bounds})
