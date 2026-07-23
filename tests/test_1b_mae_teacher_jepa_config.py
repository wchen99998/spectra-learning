from spectra_learning.config import load_config
from spectra_learning.data.gems.mask_schedule import jepa_mask_stage


def test_1b_mae_teacher_jepa_uses_target_only_loss_and_latest_teacher() -> None:
    config = load_config("configs/1b_mae_teacher_jepa.py")

    assert config.device_backend == "jax"
    assert config.dataloader_output_format == "numpy"
    assert config.dataloader_pin_memory is False
    assert config.dataloader_persistent_workers is False
    assert config.training_mode == "mae_teacher_jepa"
    assert config.training_max_steps == 300_000
    assert config.batch_size == 1024
    assert config.learning_rate == 3e-4
    assert config.min_learning_rate == 3e-6
    assert config.warmup_steps == 10_000
    assert tuple(config.gradient_accumulation_steps_schedule) == (4, 8, 8)
    assert config.checkpoint_every_steps == 500
    assert config.jax_checkpoint_max_to_keep == 5
    assert config.activation_checkpoint_mode == "full"
    assert config.max_duration_hours is None
    assert config.frozen_teacher_checkpoint_path.endswith(
        "/checkpoints/orbax/300000"
    )
    assert config.masked_token_loss_weight == 1.0
    assert config.contrastive_loss_weight == 0.0
    assert config.online_probe_loss_weight == 0.0
    assert config.mae_loss_weight == 0.0
    assert config.mae_intensity_loss_weight == 0.0
    assert config.jepa_mae_loss_weight == 0.0
    assert config.distogram_loss_weight == 0.0
    assert config.latent_pair_loss_weight == 0.0


def test_1b_mae_teacher_jepa_changes_masking_at_100k_intervals() -> None:
    config = load_config("configs/1b_mae_teacher_jepa.py")

    assert jepa_mask_stage(config, 99_999, 300_000).index == 0
    assert jepa_mask_stage(config, 100_000, 300_000).index == 1
    assert jepa_mask_stage(config, 199_999, 300_000).index == 1
    assert jepa_mask_stage(config, 200_000, 300_000).index == 2
    assert jepa_mask_stage(config, 299_999, 300_000).index == 2
