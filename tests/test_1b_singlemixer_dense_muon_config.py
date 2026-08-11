from pathlib import Path

from spectra_learning.config import load_config
from spectra_learning.data.gems.mask_schedule import jepa_mask_stages
from spectra_learning.models.settings import PeakSetJEPASettings


CONFIG_PATH = "configs/1b_singlemixer_dense_muon.py"


def test_1b_singlemixer_muon_config_is_standalone() -> None:
    source = Path(CONFIG_PATH).read_text()

    assert "import_module" not in source
    assert "runtime_config" not in source
    assert "configs." not in source


def test_1b_singlemixer_muon_training_settings() -> None:
    config = load_config(CONFIG_PATH)
    settings = PeakSetJEPASettings.from_config(config)

    assert config.optimizer == "muon"
    assert config.encoder_use_cls_token
    assert config.num_peaks == 63
    assert config.num_peaks + int(config.encoder_use_cls_token) == 64
    assert config.distogram_loss_weight == 0.0
    assert config.mae_loss_weight == 1.0
    assert settings.encoder_use_cls_token
    assert settings.distogram_loss_weight == 0.0
    assert config.training_max_steps == 300_000
    assert config.learning_rate == 3e-4
    assert config.min_learning_rate == 6e-6
    assert config.warmup_steps == 5_000
    assert config.batch_size == 4_096
    assert config.jax_mesh_devices == "64"
    assert config.batch_size // int(config.jax_mesh_devices) == 64
    assert config.gradient_accumulation_steps == 2
    assert (
        config.batch_size
        // int(config.jax_mesh_devices)
        // config.gradient_accumulation_steps
        == 32
    )
    assert config.jepa_context_fraction == 0.60
    assert config.jepa_target_fraction == 0.25
    assert "gradient_accumulation_steps_schedule" not in config
    assert "jepa_context_fraction_schedule" not in config
    assert "jepa_target_fraction_schedule" not in config
    assert "jepa_mask_schedule_steps" not in config
    stages = jepa_mask_stages(config)
    assert len(stages) == 1
    assert stages[0].context_fraction == 0.60
    assert stages[0].target_fraction == 0.25
    assert stages[0].gradient_accumulation_steps == 2
    assert config.max_duration_hours == 47.0
    assert config.checkpoint_every_steps == 50_000
    assert not config.wandb_resume_from_env
    assert config.wandb_resume_id == ""
    assert config.wandb_kwargs.to_dict() == {}
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
    assert "singlemixer" in config.run_name_suffix
    assert "disto" not in config.run_name_suffix
    assert "no-cls" not in config.run_name_suffix


def test_1b_singlemixer_muon_omits_removed_settings() -> None:
    config = load_config(CONFIG_PATH)

    assert "learning_rate_schedule_start_step" not in config
    assert "warmup_start_learning_rate" not in config
    assert "peak_filtering" not in config
    assert "grouped_peak_shoulder_da" not in config
    assert "grouped_peak_isotope_charges" not in config
