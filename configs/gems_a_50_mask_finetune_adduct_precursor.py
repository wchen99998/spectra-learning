from configs.gems_a_50_mask import get_config as _get_pretrain_config


def get_config():
    cfg = _get_pretrain_config()

    cfg.finetune_task_name = "adduct_precursor"
    cfg.finetune_data_source = "gems_formula_2m"

    # Multitask loss weights
    cfg.adduct_loss_weight = 1.0
    cfg.precursor_loss_weight = 1.0
    cfg.adduct_label_smoothing = 0.0
    cfg.formula_token_offset = 4

    # GeMS 2M dataset defaults
    cfg.gcp_key_path = "./key.json"
    cfg.gems_formula_tfrecord_dir = "data/gems_formula_tfrecord"
    cfg.gems_formula_raw_csv_path = (
        "data/gems_formula/raw/GeMS_2m_combined_formula_identifications.csv"
    )
    cfg.gems_formula_gcs_uri = (
        "gs://main-novogaia-bucket/gems/GeMS_2m_combined_formula_identifications.csv"
    )
    cfg.gems_formula_column_name = "molecularFormula"
    cfg.gems_adduct_column_name = "adduct"
    cfg.gems_formula_split_seed = 42
    cfg.gems_formula_num_shards = 16
    cfg.gems_formula_drop_remainder = False

    # Finetune optimizer defaults
    cfg.finetune_learning_rate = 1e-4
    cfg.finetune_weight_decay = 1e-4

    # Finetune schedule / eval cadence
    cfg.num_epochs = 20
    cfg.val_check_interval = 1.0
    cfg.checkpoint_every_steps = 2000

    # Logging
    cfg.wandb_project = "token-mass-spec-finetuning"
    cfg.wandb_run_name_prefix = "gems_a_50_mask_finetune_adduct_precursor"

    return cfg
