# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch-based deep learning framework for pretraining JEPA-style models on continuous mass spectrometry peak sets. The pipeline streams HDF5 MassIVE shards and MassSpecGym datasets, preprocesses raw peaks in collators, and trains a masked latent prediction model with teacher targets. During training, a periodic MSG linear probe evaluates learned representations on molecular property regression and MACCS fingerprint prediction.

## Commands

### Training
```bash
.venv/bin/python train.py --config configs/pretrain.py --workdir experiments/my_run
```

### Running tests
```bash
# All tests
.venv/bin/python -m pytest

# Single test file
.venv/bin/python -m pytest tests/test_pretrain.py

# Single test class or method
.venv/bin/python -m pytest tests/test_pretrain.py::BlockJEPATests::test_forward_loss_is_finite
```

### Data preparation (standalone)
```bash
.venv/bin/python -m spectra_learning.data.download \
    --config configs/pretrain.py \
    --artifact-dir data/artifacts
```

## Architecture

### Training Flow

`train.py:_train` dispatches to the configured training task:
1. Data flows from `GemsDataModule`, which resolves the HDF5 shard manifest and applies peak preprocessing on the fly.
2. The training collator produces masked-context JEPA batches with `peak_*`, `context_mask`, and `target_masks`.
3. The compiled forward pass (`torch.compile` with `reduce-overhead` + CUDA graphs) runs the batch through encoder -> masked latent predictor -> JEPA losses.
4. During training, `run_msg_probe` trains fixed linear probes on frozen mean-pooled readouts.

### Model (PeakSetJEPA in `models/model.py`)

- **PeakSetEncoder**: raw scalar peak features (`mz`, `intensity`, `log1p(intensity)`) -> Fourier/MLP embedder -> PairMixer blocks with pair features -> LayerNorm.
- **Targets / Predictor**: shared encoder target states supervise masked-token prediction; predictor maps visible context tokens to target-space latents.

### Masked Training Batch (`spectra_learning/data/gems/collate.py`)

`GemsBatchCollator` applies runtime preprocessing to raw 128-peak spectra:
- precursor m/z filtering
- minimum intensity filtering
- optional precursor-window exclusion
- top-k selection to `num_peaks`
- final ordering and normalization
- block mask sampling for context and targets

### Batch Contract

Training batches contain:
- `peak_mz`, `peak_intensity`: float32 [B, N]
- `peak_valid_mask`, `context_mask`: bool [B, N]
- `target_masks`: bool [B, K, N]
- `precursor_mz`: float32 [B]

### Configuration System

`ml_collections.ConfigDict` configs live in `configs/` and are loaded dynamically.
The canonical pretraining base is `configs/pretrain.py`; current scale-specific
configs derive from it.

### Data Pipeline (`spectra_learning/data/gems/`)

HDF5-shard based with auto-download from HuggingFace. `GemsDataModule` uses h5py as the HDF5 dataset backend, then owns the PyTorch `DataLoader`, chunk-aware batch sampling, distributed rank partitioning, resume offset, and peak preprocessing policy.

## Code Style

- Use PyTorch directly. Do not use training frameworks.
- Avoid defensive programming and try-catch clauses
- Prefer simple code over complicated solutions
- Always use Context7 MCP for library/API documentation
- Check `pyproject.toml` for library versions
- Use local `.venv` environment
- Python 3.12+ type hints (e.g., `list[str]`, `dict[str, int]`)
- Package manager: uv

## Notebook Notes

- Legacy notebooks that reference `FourierFeatures` are historical analyses.

## Key Dependencies

- PyTorch 2.12.0 (CUDA 13.0)
- ml-collections, rdkit, wandb, huggingface_hub
