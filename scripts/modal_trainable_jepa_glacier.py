"""Shared pinned JEPA GPU image used by retrieval and archived SALT transfer."""

from pathlib import Path

import modal


PROJECT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("spectra-massspecgym-maccs-probe")
image = (
    modal.Image.from_registry("python:3.14.7-slim-bookworm")
    .apt_install("libgomp1", "libstdc++6")
    .uv_pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("jax[cuda13]==0.11.0", "flax==0.12.8", "optax==0.2.8",
                    "orbax-checkpoint==0.12.4", "numpy==2.5.2", "scikit-learn==1.9.0",
                    "rdkit==2026.3.5", "pyarrow==25.0.1", "ml-collections==1.1.0",
                    "PyYAML>=6.0.3", "google-cloud-storage>=3.13.1", "h5py>=3.16.0",
                    "huggingface-hub>=1.27.0", "gcsfs>=2026.7.0", "jaxtyping>=0.3.11")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root/spectra-learning",
          "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OPENBLAS_NUM_THREADS": "16",
          "OMP_NUM_THREADS": "16"})
    .add_local_dir(PROJECT / "scripts", "/root/spectra-learning/scripts", ignore=["**/__pycache__/**"])
    .add_local_dir(PROJECT / "spectra_learning", "/root/spectra-learning/spectra_learning", ignore=["**/__pycache__/**"])
)
