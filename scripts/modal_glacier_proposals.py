"""Separate frozen GLACIER preprocessing image; downstream training stays JAX."""

import json
from pathlib import Path

import modal


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "67ba0f202305d4bed3134991dd4562fe1edb1763"
app = modal.App("glacier-frozen-proposals")
volume = modal.Volume.from_name("spectra-massspecgym-maccs-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgomp1", "libstdc++6")
    .uv_pip_install("torch==2.6.0+cu124", "torchvision==0.21.0+cu124",
                    index_url="https://download.pytorch.org/whl/cu124")
    .uv_pip_install("dgl==2.5.0+cu124", "torchdata==0.9.0",
                    find_links="https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html")
    .uv_pip_install("torch-scatter==2.1.2+pt26cu124", "torch-sparse==0.6.18+pt26cu124",
                    find_links="https://data.pyg.org/whl/torch-2.6.0+cu124.html")
    .uv_pip_install("numpy==2.2.6", "rdkit==2025.3.6", "pandas==2.2.3", "scipy==1.15.3",
                    "pytorch-lightning==2.6.6", "LinSATNet==0.1.3", "pygmtools==0.6.0",
                    "pathos", "einops", "h5py", "matplotlib", "seaborn", "omegaconf",
                    "scikit-learn", "platformdirs", "PyYAML", "torch-geometric==2.7.0")
    .run_commands("git clone https://github.com/coleygroup/ms-pred.git /opt/ms-pred",
                  f"git -C /opt/ms-pred checkout --detach {SOURCE_COMMIT}")
    .apt_install("libxrender1", "libxext6")
    .env({"PYTHONPATH": "/opt/ms-pred/src:/root/spectra-learning/scripts",
          "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
          "DGLBACKEND": "pytorch"})
    .add_local_file(PROJECT / "scripts/generate_glacier_proposals.py",
                    "/root/spectra-learning/scripts/generate_glacier_proposals.py")
)


@app.function(image=image, cpu=16, memory=98304, ephemeral_disk=524288,
              gpu="H100", timeout=86400, volumes={"/results": volume})
def generate(arguments: str) -> dict:
    from generate_glacier_proposals import generate as run, parser
    try:
        return run(parser().parse_args(json.loads(arguments)), commit=volume.commit)
    finally:
        volume.commit()


@app.function(image=image, cpu=16, memory=98304, timeout=86400, volumes={"/results": volume})
def preflight(arguments: str) -> dict:
    from generate_glacier_proposals import preflight_mapping
    try:
        return preflight_mapping(**json.loads(arguments))
    finally:
        volume.commit()


@app.local_entrypoint()
def main(arguments: str, upload_release: bool = False, cpu_preflight: bool = False) -> None:
    if upload_release:
        source = Path("/tmp/GLACIER_checkpoint_20260918")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(source / "best.ckpt", "/glacier_release_20260918/best.ckpt")
            batch.put_file(source / "provenance.json", "/glacier_release_20260918/provenance.json")
    print(json.dumps((preflight if cpu_preflight else generate).remote(arguments), indent=2))
