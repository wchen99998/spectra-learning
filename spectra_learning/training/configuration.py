import json

from ml_collections import config_dict

from spectra_learning.config import config_to_dict
from spectra_learning.models.fastmixer_capacity import (
    pairmixer_fast_mae_encoder_visible_tokens,
    pairmixer_fast_required_visible_tokens,
)
from spectra_learning.training.storage import StoragePath, storage_join, write_text


CONFIG_FILENAME = "config.json"


def save_config(config: config_dict.ConfigDict, workdir: StoragePath) -> None:
    finalize_config(config)
    write_text(
        storage_join(workdir, CONFIG_FILENAME),
        json.dumps(config_to_dict(config), indent=2, sort_keys=True) + "\n",
    )


def finalize_config(config: config_dict.ConfigDict) -> None:
    block_type = str(config.get("pairmixer_block_type", "dense")).lower()
    if block_type not in {"fastmixer", "fastmixer-dense"}:
        return
    config.resolved_pairmixer_fast_max_visible_tokens = (
        pairmixer_fast_required_visible_tokens(config)
    )
    config.resolved_pairmixer_fast_encoder_max_visible_tokens = (
        pairmixer_fast_mae_encoder_visible_tokens(config)
        if str(config.get("training_mode", "jepa")).lower() == "mae"
        else pairmixer_fast_required_visible_tokens(config)
    )
