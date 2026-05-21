# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch-based deep learning framework for pretraining SIGReg models on continuous mass spectrometry peak sets. The pipeline ingests raw peak lists from GeMS and MassSpecGym datasets, preprocesses them into native shard artifacts, and trains a masked latent prediction model with JEPA-style teacher targets plus SIGReg regularization. During training, a periodic MSG linear probe evaluates learned representations on molecular property regression and MACCS fingerprint prediction.

## Commands

### Training
```bash
python train.py --config configs/gems_a_50_mask.py --workdir experiments/my_run
```

### Running tests
```bash
# All tests
python -m pytest tests/

# Single test file
python -m pytest tests/test_pretrain.py

# Single test class or method
python -m pytest tests/test_pretrain.py::SIGRegForwardTests::test_forward_loss_is_finite
```

### Data preparation (standalone)
```bash
python input_pipeline.py configs/gems_a_dataset.py
```

## Architecture

### Training Flow

`train.py:train_and_evaluate` orchestrates the full pipeline:
1. Data flows from `GemsNativeDataModule`, which loads native GeMS shard artifacts and applies peak preprocessing on the fly.
2. The training collator produces masked-context JEPA batches with `peak_*`, `context_mask`, and `target_masks`.
3. The compiled forward pass (`torch.compile` with `reduce-overhead` + CUDA graphs) runs the batch through encoder -> masked latent predictor -> JEPA losses.
4. During training, `run_msg_probe` trains fixed linear probes on frozen `mean + cls` readouts.

### Model (PeakSetSIGReg in `models/model.py`)

- **PeakSetEncoder**: raw scalar peak features (`mz`, `intensity`, `log1p(intensity)`) -> MLP embedder -> N non-causal TransformerBlocks -> RMSNorm.
- **Encoder**: raw scalar peak features (`mz`, `intensity`) -> Fourier/MLP embedder -> non-causal Transformer blocks.
- **Targets / Predictor**: shared encoder target states supervise masked-token prediction; predictor maps visible context tokens to target-space latents.
- **SIGReg**: optional regularizer on learned representations.

### Masked Training Batch (`input_pipeline.py`)

`input_pipeline.py` applies runtime preprocessing to raw 128-peak spectra:
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

`ml_collections.ConfigDict` configs in `configs/`. Each config file is self-contained and loaded dynamically via importlib. Key config: `configs/gems_a_50_mask.py`.

### Data Pipeline (`spectra_learning/data/gems/`)

Native-shard based with auto-download from HuggingFace. `GemsNativeDataModule` memmaps raw peak spectra, preprocesses peaks in the PyTorch collator, and builds DataLoaders directly.

### Key Aliases

`PeakSetJEPA = PeakSetSIGReg` (historical alias in `models/model.py:463`)

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

- PyTorch 2.11.0 (CUDA 13.0)
- ml-collections, rdkit, wandb, huggingface_hub
