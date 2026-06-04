from ml_collections import config_dict

from configs.medium_pairformer_encoder import get_config as get_medium_config


OUTPUT_WORKDIR = (
    "gs://metal-repeater-411410-spectra-checkpoints/"
    "pairformer_medium_pairnorm"
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_medium_config()

    cfg.encoder_apply_final_pair_norm = True
    cfg.output_workdir = OUTPUT_WORKDIR
    cfg.run_name_suffix = "mae-medium-pairformer-pairnorm-92m"

    return cfg
