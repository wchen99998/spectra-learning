import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ml_collections import config_dict

from spectra_learning.config import config_to_dict, load_config
from spectra_learning.data.gems.settings import GemsDataConfig
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.training.configuration import CONFIG_FILENAME, save_config


def test_load_config_applies_overrides() -> None:
    config = load_config(
        "configs/ar_spectra_coarse_to_fine.py",
        {
            "training_max_steps": 12,
            "wandb_kwargs": {"name": "test-run", "tags": ["test"]},
        },
    )

    assert config.training_max_steps == 12
    assert config.config_path == "configs/ar_spectra_coarse_to_fine.py"
    assert config.ar_attention_kernel == "pallas"
    assert config.ar_gelu_approximation == "quick"
    assert config.wandb_kwargs.to_dict() == {
        "name": "test-run",
        "tags": ["test"],
    }


def test_serialized_config_can_be_replayed_as_overrides() -> None:
    path = "configs/300m_pairmixer_dense_adamw.py"
    expected = config_to_dict(load_config(path))

    replayed = load_config(path, expected)

    assert config_to_dict(replayed) == expected
    assert isinstance(replayed.grouped_peak_isotope_charges, tuple)


def test_current_pretraining_config_records_model_and_data_defaults() -> None:
    config = load_config("configs/100m_pairmixer_dense_adamw.py")

    model_keys = set(asdict(PeakSetJEPASettings.from_config(config)))
    data_keys = set(asdict(GemsDataConfig.from_config(config)))

    assert model_keys - set(config) == {
        "pairmixer_fast_encoder_max_visible_tokens",
        "pairmixer_fast_max_visible_tokens",
    }
    assert data_keys - set(config) == {"jepa_intensity_aware_mask_config"}


def test_model_settings_copy_typed_config_values_without_coercion() -> None:
    config = {
        "attention_mlp_multiple": 3,
        "activation_checkpoint_modules": ("encoder",),
    }

    settings = PeakSetJEPASettings.from_config(config)

    assert type(settings.attention_mlp_multiple) is int
    assert settings.attention_mlp_multiple == config["attention_mlp_multiple"]
    assert settings.activation_checkpoint_modules == ("encoder",)


def test_saved_config_uses_the_canonical_serialization(tmp_path: Path) -> None:
    config = config_dict.ConfigDict()
    config.path = tmp_path
    config.scalar = np.int64(7)
    config.shape = (2, 3)

    expected = {
        "path": str(tmp_path),
        "scalar": 7,
        "shape": [2, 3],
    }
    assert config_to_dict(config) == expected

    save_config(config, tmp_path)

    assert json.loads((tmp_path / CONFIG_FILENAME).read_text()) == expected


def test_saved_config_includes_derived_fastmixer_capacity(tmp_path: Path) -> None:
    config = load_config("configs/100m_pairmixer_dense_adamw.py")

    save_config(config, tmp_path)

    saved = json.loads((tmp_path / CONFIG_FILENAME).read_text())
    assert saved["resolved_pairmixer_fast_max_visible_tokens"] == 41
    assert saved["resolved_pairmixer_fast_encoder_max_visible_tokens"] == 17
    assert config_to_dict(config) == saved
