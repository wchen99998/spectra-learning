from importlib import import_module

from ml_collections import config_dict


GENERATOR_RUN = "1b-xattn-rope-muon-v6e4x8-b2048-ga8-16-16-3d-20260731-013822"


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.mz_token_ablation").get_config()

    cfg.training_task = "fake_peak"
    cfg.device_backend = "torch"
    cfg.encoder_num_layers = 13
    cfg.batch_size = 512
    cfg.gradient_accumulation_steps = 8
    cfg.compile_mode = "none"

    cfg.discriminator_device = "cuda:0"
    cfg.generator_device = "cuda:1"
    cfg.generator_source_checkpoint = (
        "gs://metal-repeater-411410-spectra-checkpoints/skypilot/"
        f"{GENERATOR_RUN}/checkpoints/orbax/300000"
    )
    cfg.generator_checkpoint_path = (
        "data/frozen_generators/1b_xattn_rope_step300000_fp32.pt"
    )
    cfg.generator_top_k = 32
    cfg.generator_temperature = 1.0
    cfg.generator_compile_mode = "default"
    cfg.fake_peak_detection_loss_weight = 1.0
    cfg.fake_peak_reconstruction_loss_weight = 1.0
    cfg.fake_peak_intensity_reconstruction_loss_weight = 1.0

    cfg.checkpoint_every_steps = 500
    cfg.val_num_steps = 32
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-fake-peaks"
    cfg.wandb_kwargs = {
        "name": "fake-peak-disc-51m-frozen-1b-step300k-2xh100-2p5k"
    }
    return cfg
