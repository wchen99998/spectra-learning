# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch-based deep learning framework for pretraining and fine-tuning BERT-style masked language models on mass spectrometry peak lists and molecular data. Handles spectral data tokenization, training with PyTorch Lightning, and fine-tuning on downstream tasks like precursor m/z and adduct prediction.

## Commands

### Training (pretraining BERT)
```bash
python train.py --config configs/gems_a_50_mask.py --workdir experiments/my_run
```

### Fine-tuning on Precursor Prediction
```bash
python finetune.py \
    --config configs/gems_a_50_mask_finetune.py \
    --workdir experiments/ft_run \
    --model experiments/pretrain_run/checkpoints/last.ckpt
```

### Fine-tuning Multitask (Adduct + Precursor)
```bash
python finetune_adduct_precursor.py \
    --config configs/gems_a_50_mask_finetune_adduct_precursor.py \
    --workdir experiments/ft_adduct_run \
    --model experiments/pretrain_run/checkpoints/last.ckpt
```

## Architecture

### Core Components

- **input_pipeline.py**: Data loading, TFRecord creation/loading, tokenization logic, Lightning DataModule
- **models/bert_torch.py**: BERT model with token/segment embeddings, TransformerStack encoder, output heads (lm_head, precursor_head, retention_head)
- **networks/transformer_torch.py**: TransformerBlock, Attention, FeedForward primitives with rotary embeddings
- **train.py**: Pretraining LightningModule with MLM loss, LR scheduling, profiling callbacks
- **finetune.py / finetune_adduct_precursor.py**: Fine-tuning pipelines for downstream tasks

### Tokenization Scheme

- Special tokens: [PAD]=0, [CLS]=1, [SEP]=2, [MASK]=3
- m/z tokens: floor(mz) + 4 (range [4, 1004] for m/z up to 1000)
- Intensity tokens: log-scaled binning into 32 bins
- Precursor m/z: separate binned range with offset

### Configuration System

Uses `ml_collections.ConfigDict`. Configs are Python modules in `configs/` loaded via importlib:
```python
cfg.model_dim = 768
cfg.num_layers = 20
cfg.num_heads = 12
cfg.mask_ratio = 0.15
cfg.learning_rate = 3e-4
cfg.enable_wandb = True
```

### Data Sources

- **GeMS**: `roman-bushuiev/GeMS` (HuggingFace)
- **MassSpecGym**: `roman-bushuiev/MassSpecGym` (HuggingFace)
- **GeMS 2M Formula**: Google Cloud Storage (requires `key.json` credentials)

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
