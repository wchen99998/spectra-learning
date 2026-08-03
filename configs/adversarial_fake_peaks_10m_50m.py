from ml_collections import config_dict

from configs.fake_peak_discriminator_50m import get_config as get_discriminator_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_discriminator_config()

    cfg.training_task = "adversarial_fake_peak"
    del cfg.generator_source_checkpoint
    del cfg.generator_checkpoint_path
    del cfg.generator_top_k
    del cfg.generator_temperature
    del cfg.generator_compile_mode

    cfg.generator_model = config_dict.ConfigDict(
        {
            "training_mode": "jepa",
            "use_ema_teacher": False,
            "mae_loss_weight": 0.0,
            "jepa_mae_loss_weight": 0.0,
            "masked_token_loss_weight": 0.0,
            "target_projector_dim": -1,
            "model_dim": 256,
            "encoder_num_layers": 5,
            "encoder_num_heads": 4,
            "encoder_apply_final_pair_norm": False,
            "feature_mlp_hidden_dim": 512,
            "encoder_fourier_mlp_hidden_dim": 512,
            "pairmixer_pair_dim": 128,
            "pairmixer_pair_feature_hidden_dim": 256,
            "predictor_dim": 256,
            "masked_latent_predictor_num_heads": 4,
        }
    )
    cfg.generator_learning_rate = 3e-4
    cfg.generator_min_learning_rate = 3e-5
    cfg.generator_warmup_steps = 250
    cfg.generator_mz_smooth_l1_beta = 0.0005
    cfg.generator_intensity_smooth_l1_beta = 0.1
    cfg.generator_mz_loss_weight = 1.0
    cfg.generator_intensity_loss_weight = 1.0
    cfg.generator_adversarial_loss_weight = 1e-5
    cfg.generator_adversarial_warmup_steps = 250

    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-adversarial-fake-peaks"
    cfg.wandb_kwargs = {
        "name": "adversarial-fake-peaks-g10m-d51m-scratch-2xh100-2p5k"
    }
    return cfg
