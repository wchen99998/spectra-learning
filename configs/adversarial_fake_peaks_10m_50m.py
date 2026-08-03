from ml_collections import config_dict

from configs.mz_token_ablation import get_config as get_base_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_base_config()

    cfg.training_task = "adversarial_fake_peak"
    cfg.device_backend = "torch"
    cfg.encoder_num_layers = 13
    cfg.batch_size = 512
    cfg.gradient_accumulation_steps = 8
    cfg.compile_mode = "none"

    cfg.discriminator_device = "cuda:0"
    cfg.generator_device = "cuda:1"
    cfg.fake_peak_detection_loss_weight = 1.0
    cfg.fake_peak_reconstruction_loss_weight = 1.0
    cfg.fake_peak_intensity_reconstruction_loss_weight = 1.0

    cfg.generator_model = config_dict.ConfigDict(
        {
            "training_mode": "mae",
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
    cfg.learning_rate = 1e-4
    cfg.min_learning_rate = 1e-5
    cfg.generator_learning_rate = 3e-4
    cfg.generator_min_learning_rate = 3e-5
    cfg.generator_warmup_steps = 250
    cfg.generator_gumbel_temperature = 1.0
    cfg.generator_mz_loss_weight = 1.0
    cfg.generator_intensity_loss_weight = 1.0
    cfg.generator_adversarial_loss_weight = 1e-4
    cfg.generator_adversarial_warmup_steps = 250

    cfg.checkpoint_every_steps = 500
    cfg.val_num_steps = 32
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-adversarial-fake-peaks"
    cfg.wandb_kwargs = {
        "name": "adversarial-categorical-g10m-d51m-scratch-2xh100-2p5k"
    }
    return cfg
