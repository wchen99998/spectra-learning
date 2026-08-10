from pathlib import Path

from spectra_learning.config import load_config


def test_1b_muon_no_cls_copies_the_1b_config_instead_of_importing_it() -> None:
    source = Path("configs/1b_pairmixer_dense_muon_no_cls.py").read_text()

    assert "configs.1b_pairmixer_dense_adamw" not in source


def test_1b_muon_no_cls_matches_control_training_settings() -> None:
    config = load_config("configs/1b_pairmixer_dense_muon_no_cls.py")

    assert config.optimizer == "muon"
    assert not config.encoder_use_cls_token
    assert config.training_max_steps == 300_000
    assert config.learning_rate == 3e-4
    assert config.min_learning_rate == 6e-6
    assert config.warmup_steps == 5_000
    assert config.batch_size == 4_096
    assert config.jax_mesh_devices == "64"
    assert config.batch_size // int(config.jax_mesh_devices) == 64
    assert config.gradient_accumulation_steps == 8
    assert (
        config.batch_size
        // int(config.jax_mesh_devices)
        // config.gradient_accumulation_steps
        == 8
    )
    assert tuple(config.gradient_accumulation_steps_schedule) == (8, 16, 16)
    assert tuple(config.jepa_context_fraction_schedule) == (0.35, 0.55, 0.75)
    assert tuple(config.jepa_target_fraction_schedule) == (0.5, 0.3, 0.1)
    assert tuple(config.jepa_mask_schedule_steps) == (250_000, 350_000)
    assert config.checkpoint_every_steps == 50_000
    assert config.mae_loss_weight == 0.7
    assert config.distogram_loss_weight == 0.3
    assert not config.wandb_resume_from_env
    assert config.wandb_resume_id == ""
    assert config.wandb_kwargs.to_dict() == {}
    assert "learning_rate_schedule_start_step" not in config
    assert "warmup_start_learning_rate" not in config
    assert config.device_backend == "jax"
    assert config.dataloader_output_format == "numpy"
    assert not config.dataloader_pin_memory
    assert config.dataloader_multiprocessing_context == "spawn"
    assert (
        config.gems_hdf5_repo_id
        == "novogaia/massive-v2-ms2-t095-l080-sharded-10gb"
    )
    assert (
        config.gems_hdf5_revision
        == "de80d280d319f0b9a8825956b13d8dc7d9ab1eb1"
    )
    assert "peak_filtering" not in config
    assert "grouped_peak_shoulder_da" not in config
    assert "grouped_peak_isotope_charges" not in config
