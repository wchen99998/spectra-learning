from ml_collections import config_dict

from configs.mz_token_ablation import get_config as get_base_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_base_config()

    cfg.training_task = "adversarial_fake_peak"
    cfg.device_backend = "torch"
    cfg.encoder_num_layers = 18
    cfg.batch_size = 1_280
    cfg.gradient_accumulation_steps = 16
    cfg.training_max_steps = 50_000
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
            "encoder_num_layers": 17,
            "encoder_apply_final_pair_norm": False,
            "predictor_dim": 280,
        }
    )
    cfg.learning_rate = 1e-4
    cfg.min_learning_rate = 1e-5
    cfg.warmup_steps = 1_000
    cfg.generator_learning_rate = 3e-4
    cfg.generator_min_learning_rate = 3e-5
    cfg.generator_warmup_steps = 2_000
    cfg.generator_gumbel_temperature = 1.0
    cfg.generator_mz_loss_weight = 1.0
    cfg.generator_intensity_loss_weight = 1.0
    cfg.generator_adversarial_loss_weight = 0.25
    cfg.generator_adversarial_start_step = 2_000
    cfg.generator_adversarial_warmup_steps = 10_000
    cfg.generator_adversarial_grad_max_ratio = 0.1

    cfg.log_every_n_steps = 25
    cfg.checkpoint_every_steps = 1_000
    cfg.val_every_n_steps = 500
    cfg.val_num_steps = 64
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-adversarial-fake-peaks"
    cfg.wandb_kwargs = {
        "name": (
            "adversarial-equal50m-masked-peaks-adv-gradcap01-ramp10k-50k"
        )
    }
    return cfg
