# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch-based deep learning framework for pretraining and fine-tuning JEPA (Joint-Embedding Predictive Architecture) models on continuous mass spectrometry peak sets. The pipeline ingests raw peak lists, normalizes them into continuous features, and trains a JEPA model with prediction + BCS (Batched Characteristic Slicing) regularization losses. Fine-tuning supports downstream tasks like fingerprint prediction and adduct classification.

## Commands

### Training (JEPA pretraining)
```bash
python train.py --config configs/gems_a_50_mask.py --workdir experiments/my_run
```


## Architecture

### Core Components

- **input_pipeline.py**: Data loading, TFRecord creation/loading, continuous peak normalization, Lightning DataModule. Emits `peak_mz`, `peak_intensity`, `peak_valid_mask`, `precursor_mz` tensors.
- **models/model.py**: PeakSetJEPA model (encoder + mask sampler + predictor + losses). Encoder is a non-causal transformer on set-style peak features.
- **models/losses.py**: BCSLoss (Batched Characteristic Slicing via Epps-Pulley), squared_prediction_loss.
- **networks/transformer_torch.py**: TransformerBlock, Attention, FeedForward primitives with optional rotary embeddings.
- **train.py**: JEPA pretraining LightningModule with prediction + BCS regularization losses.

### JEPA Model Structure

1. **PeakSetEncoder**: MLP embedder (mz, intensity, precursor -> D) + non-causal Transformer blocks + RMSNorm
2. **PeakMaskSampler**: Samples fixed-size context/target index sets with valid-peak priority
3. **PeakSetPredictor**: Embeds target queries, concatenates with context embeddings, applies non-causal self-attention
4. **Losses**: MSE prediction loss + BCSLoss (Epps-Pulley Gaussianity test on random 1-D slices)

### Batch Contract

Each training batch contains:
- `peak_mz`: float32 [B, N] - normalized m/z values (/ 1000)
- `peak_intensity`: float32 [B, N] - scaled intensity values
- `peak_valid_mask`: bool [B, N] - valid peak mask
- `precursor_mz`: float32 [B] - normalized precursor m/z
- Probe metadata when available: `fingerprint`, `adduct_id`, `instrument_type_id`, etc.

### Configuration System

Uses `ml_collections.ConfigDict`. Configs are Python modules in `configs/` loaded via importlib:
```python
cfg.model_type = "jepa_peak_set"
cfg.num_peaks = 60
cfg.model_dim = 768
cfg.num_layers = 20
cfg.num_heads = 12
cfg.jepa_target_ratio = 0.4
cfg.jepa_bcs_num_slices = 256
cfg.jepa_bcs_lambda = 10.0
cfg.learning_rate = 3e-4
```

### Data Sources

- **GeMS**: `roman-bushuiev/GeMS` (HuggingFace)
- **MassSpecGym**: `roman-bushuiev/MassSpecGym` (HuggingFace)

### WandB Integration

Run counter in `.wandb_run_counter` incremented after each run for unique naming. Enable via `cfg.enable_wandb = True`.

## Code Style

- Use PyTorch and PyTorch Lightning exclusively
- Avoid defensive programming and try-catch clauses
- Prefer simple code over complicated solutions
- Check `pyproject.toml` for library versions
- Use local `.venv` environment
- Python 3.12+ type hints (e.g., `list[str]`, `dict[str, int]`)

## Key Dependencies

- PyTorch 2.10.0 (CUDA 13.0)
- PyTorch Lightning 2.5.5
- TensorFlow 2.19.0 CPU (for tf.data pipeline)
- ml-collections, rdkit, wandb, huggingface_hub
- Package manager: uv
