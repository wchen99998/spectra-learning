from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.300m_pairmixer_dense_adamw").get_config()

    cfg.device_backend = "jax"
    cfg.dataloader_output_format = "numpy"
    cfg.dataloader_pin_memory = False
    cfg.dataloader_multiprocessing_context = "spawn"
    cfg.artifact_dir = "data/massive_v2_ms2_t095_l080_sharded_10gb"
    cfg.gems_hdf5_repo_id = (
        "novogaia/massive-v2-ms2-t095-l080-sharded-10gb"
    )
    cfg.gems_hdf5_revision = "de80d280d319f0b9a8825956b13d8dc7d9ab1eb1"

    cfg.model_dim = 1536
    cfg.encoder_num_layers = 25
    cfg.encoder_num_heads = 12
    cfg.encoder_use_cls_token = False
    cfg.feature_mlp_hidden_dim = 3072
    cfg.encoder_fourier_mlp_hidden_dim = 3072
    cfg.learning_rate = 3e-4
    cfg.weight_decay = 0.1
    cfg.optimizer_state_dtype = "fp32"
    cfg.optimizer = "muon"

    cfg.pairmixer_pair_dim = 640
    cfg.pairmixer_pair_feature_hidden_dim = 1280
    cfg.mae_loss_weight = 1.0
    cfg.mae_intensity_loss_weight = 0.0
    cfg.distogram_loss_weight = 0.0

    cfg.jepa_context_fraction_schedule = (0.35, 0.55, 0.75)
    cfg.jepa_target_fraction_schedule = (0.50, 0.30, 0.10)
    cfg.jepa_mask_schedule_steps = (250_000, 350_000)
    cfg.gradient_accumulation_steps = 8
    cfg.gradient_accumulation_steps_schedule = (8, 16, 16)
    cfg.activation_checkpoint_mode = "none"
    cfg.training_max_steps = 300_000
    cfg.checkpoint_every_steps = 50_000
    cfg.max_duration_hours = 71.0
    cfg.wandb_resume_from_env = False
    cfg.wandb_resume_id = ""
    cfg.wandb_kwargs = {}

    cfg.predictor_dim = 1536
    cfg.masked_latent_predictor_num_layers = 5
    cfg.masked_latent_predictor_num_heads = 12

    cfg.run_name_suffix = (
        "jax-mae-massive-v2-10gb-1b-200m-v6e-d1536-p640-l25-h12-nomassprior-"
        "feat3072-fmlp3072-xattnrope1536-l5-h12-bs2048-ga8-16-16-"
        "muon-no-cls-fp32state-lr3e-4-wd1e-1-"
        "random-mask65-45-25-target50-30-10-intensityorder-fullctx-"
        "noac-noprobe-val500x10k"
    )

    return cfg
