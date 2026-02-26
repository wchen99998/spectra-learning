# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch-based deep learning framework for pretraining SIGReg (Strict SIGmoid Regularization) models on continuous mass spectrometry peak sets. The pipeline ingests raw peak lists from GeMS and MassSpecGym datasets, normalizes them into continuous features via a tf.data pipeline, and trains a two-view self-supervised model with MSE invariance + BCS (Batched Characteristic Slicing) regularization losses. After pretraining, a final attentive probe evaluates learned representations on adduct classification, precursor bin prediction, and instrument type identification.

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

`train.py:MAELightningModule` orchestrates the full pipeline:
1. Data flows from tf.data TFRecords through `TfLightningDataModule` (round-robin interleaving GeMS + MassSpecGym datasets)
2. Two-view augmentation happens in the tf.data pipeline (`_augment_sigreg_batch_tf`), producing `fused_*` tensors that stack view1 (masked+jittered) and view2 (unmasked+jittered) along the batch dimension
3. The compiled forward pass (`torch.compile` with `max-autotune` + CUDA graphs) runs the fused batch through encoder -> PMA pooling -> projector -> BCS+invariance loss
4. After pretraining completes, `_run_final_attentive_probe` trains a lightweight multi-task probe on frozen encoder features

### Model (PeakSetSIGReg in `models/model.py`)

- **PeakSetEncoder**: raw scalar peak features (`mz`, `intensity`, `log1p(intensity)`) -> MLP embedder -> N non-causal TransformerBlocks -> RMSNorm. Uses mass-aware RoPE on m/z only.
- **PMA Pooling**: Multihead cross-attention with learned seed queries (`pool_query`) that attend to peak embeddings, producing a fixed-size representation regardless of valid peak count.
- **Projector**: 3-layer MLP (Linear -> RMSNorm -> SiLU) x2, maps pooled embeddings to lower-dim space for the loss.
- **BCSLoss** (`models/losses.py`): Projects both views via random slicing directions, tests Gaussianity using Epps-Pulley characteristic function distance. Combined loss = MSE(z1, z2) + lambda * BCS.

### Two-View Augmentation (`input_pipeline.py`)

The TF implementation in `input_pipeline.py` runs augmentation in the data pipeline for training.
- **Global view**: Full-spectrum (no masking), jitter on valid peaks
- **Local views**: Local masking + jitter with original valid/padding layout preserved

### Batch Contract

Training batches contain fused stacked tensors:
- `fused_mz`, `fused_intensity`: float32 [V*B, N]
- `fused_valid_mask`, `fused_masked_positions`, `fused_padding_mask`: bool [V*B, N]
- `peak_padding_mask`: bool [B, N]

Raw (pre-augmentation) batches: `peak_mz` [B, N], `peak_intensity` [B, N], `peak_valid_mask` [B, N], `precursor_mz` [B].

### Configuration System

`ml_collections.ConfigDict` configs in `configs/`. Shared defaults in `configs/_defaults.py` (`apply_training_defaults`, `apply_final_probe_defaults`, `apply_tune_defaults`). Configs loaded dynamically via importlib. Key config: `configs/gems_a_50_mask.py`.

### Data Pipeline (`input_pipeline.py`)

TFRecord-based with auto-download from HuggingFace. Processing chain: parse -> filter precursor mz -> filter peak mz range -> filter min intensity -> topk -> optional neutral loss -> compact sort -> normalize -> batch -> augment. `TfLightningDataModule` wraps tf.data datasets as PyTorch IterableDatasets with stateful resume support.

### Key Aliases

`PeakSetJEPA = PeakSetSIGReg` (historical alias in `models/model.py:463`)

## Code Style

- Use PyTorch and PyTorch Lightning exclusively
- Avoid defensive programming and try-catch clauses
- Prefer simple code over complicated solutions
- Always use Context7 MCP for library/API documentation
- Check `pyproject.toml` for library versions
- Use local `.venv` environment
- Python 3.12+ type hints (e.g., `list[str]`, `dict[str, int]`)
- Package manager: uv

## Notebook Notes

- Legacy notebooks that reference `FourierFeatures` are historical analyses from before the mass-aware RoPE migration.
- Current evaluation notebook for the active architecture is `notebooks/mass_aware_rope_evaluation.ipynb` (executed copy saved via nbconvert as `notebooks/mass_aware_rope_evaluation.executed.ipynb`).

## Key Dependencies

- PyTorch 2.10.0 (CUDA 13.0)
- Lightning 2.5.5
- TensorFlow CPU 2.19.0 (tf.data pipeline only, GPU disabled)
- ml-collections, rdkit, wandb, huggingface_hub
