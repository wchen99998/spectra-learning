"""Deterministic input pipeline."""

from typing import Any

from ml_collections import config_dict

import input_pipeline_gems_set


def get_num_train_steps(config: config_dict.ConfigDict) -> int:
    """Calculates the total number of training steps."""
    if config.num_train_steps > 0:
        return config.num_train_steps
    raise NotImplementedError()


def create_datasets(
    config: config_dict.ConfigDict, seed: int
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Create data loaders for training and evaluation.

    Args:
      config: Configuration to use.
      seed: Seed for shuffle and random operations in the training dataset.

    Returns:
      A tuple with the training dataset loader, the evaluation dataset
      loader, and a dictionary of other infos.
    """
    info = {}

    if config.dataset == "gems_a":
        train_dataset, eval_dataset, gems_info = (
            input_pipeline_gems_set.create_gems_set_datasets(config, seed)
        )
        info.update(gems_info)
        config.dataset_info = dict(info)
        config.vocab_size = info["vocab_size"]
        config.max_length = info["pair_sequence_length"]
        config.pad_token_id = info["special_tokens"]["[PAD]"]
        config.cls_token_id = info["special_tokens"]["[CLS]"]
        config.sep_token_id = info["special_tokens"]["[SEP]"]
        config.mask_token_id = info["special_tokens"]["[MASK]"]
        config.precursor_bins = info["precursor_bins"]
        config.precursor_offset = info["precursor_offset"]
        return train_dataset, eval_dataset, info
    else:
        raise NotImplementedError(
            "Only gems_a (peak set) dataset is supported."
        )
