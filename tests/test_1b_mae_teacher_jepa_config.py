import pytest

from spectra_learning.config import load_config
from spectra_learning.data.gems.mask_schedule import jepa_mask_stage
from spectra_learning.models.fastmixer_capacity import pairmixer_fast_stage_capacities
from spectra_learning.models.settings import PeakSetJEPASettings


@pytest.mark.parametrize("config_path", (
    "configs/1b_mae_teacher_jepa.py", "configs/1p8b_mae_teacher_jepa.py",
))
def test_mae_teacher_jepa_uses_full_1p8b_frozen_teacher(config_path: str) -> None:
    config = load_config(config_path)

    assert config.device_backend == "jax"
    assert config.dataloader_output_format == "numpy"
    assert config.dataloader_pin_memory is False
    assert config.dataloader_persistent_workers is False
    assert config.gems_hdf5_shuffle_seed == 67
    assert config.training_mode == "mae_teacher_jepa"
    assert config.num_peaks == 127
    assert config.encoder_use_cls_token is True
    assert config.num_peaks + int(config.encoder_use_cls_token) == 128
    assert config.peak_ordering == "mz"
    assert config.training_max_steps == 2_000_000
    assert config.batch_size == (1024 if "1p8b_mae" in config_path else 2048)
    assert config.optimizer == "adam"
    assert config.b1 == 0.9
    assert config.weight_decay == 0.0
    assert config.optimizer_state_dtype == "bf16"
    assert config.learning_rate == 3e-4
    assert config.min_learning_rate == 3e-6
    assert config.warmup_steps == 10_000
    assert config.gradient_accumulation_steps == 8
    assert tuple(config.gradient_accumulation_steps_schedule) == (8,)
    assert config.checkpoint_every_steps == 500
    assert config.jax_checkpoint_max_to_keep == 5
    assert config.activation_checkpoint_mode == "full"
    assert config.max_duration_hours is None
    assert config.frozen_teacher_checkpoint_path.endswith(
        "/checkpoints/orbax/666000"
    )
    assert "1p8b-sm-hd256-adam-v6e64-b6144-ga1-666k" in (
        config.frozen_teacher_checkpoint_path
    )
    teacher_config = load_config(config.frozen_teacher_config_path)
    student = PeakSetJEPASettings.from_config(config)
    teacher = PeakSetJEPASettings.from_config(teacher_config)
    assert config.gems_hdf5_repo_id == teacher_config.gems_hdf5_repo_id
    assert config.gems_hdf5_revision == teacher_config.gems_hdf5_revision
    assert config.gems_hdf5_revision == "b098155fccb7335b752f2690bf16d2479d69bf5b"
    assert student.encoder_metadata_schema == "massive_v2_acquisition_v1"
    assert student.encoder_metadata_schema == teacher.encoder_metadata_schema
    assert student.encoder_metadata_conditioning == "adaln_zero"
    assert student.encoder_metadata_conditioning == teacher.encoder_metadata_conditioning
    assert student.encoder_metadata_condition_dim == teacher.encoder_metadata_condition_dim
    assert teacher.model_dim == 2_048
    assert teacher.num_peaks == 127
    assert teacher.encoder_use_cls_token is True
    assert config.spectrum_metadata_dropout_probability == 0.10
    assert config.masked_token_loss_weight == 1.0
    assert config.contrastive_loss_weight == 0.0
    assert config.online_probe_loss_weight == 0.0
    assert config.mae_loss_weight == 0.0
    assert config.mae_intensity_loss_weight == 0.0
    assert config.jepa_mae_loss_weight == 0.0
    assert config.distogram_loss_weight == 0.0
    assert config.latent_pair_loss_weight == 0.0


@pytest.mark.parametrize("config_path", (
    "configs/1b_mae_teacher_jepa.py", "configs/1p8b_mae_teacher_jepa.py",
))
def test_mae_teacher_jepa_keeps_teacher_masking_for_entire_run(config_path: str) -> None:
    config = load_config(config_path)
    teacher = load_config(config.frozen_teacher_config_path)

    assert config.jepa_mask_strategy == teacher.jepa_mask_strategy == ["random"]
    assert config.jepa_context_fraction == teacher.jepa_context_fraction == 0.60
    assert config.jepa_target_fraction == teacher.jepa_target_fraction == 0.25
    assert tuple(config.jepa_mask_schedule_steps) == ()
    assert pairmixer_fast_stage_capacities(config) == (
        pairmixer_fast_stage_capacities(teacher)[0],
    )
    initial = jepa_mask_stage(config, 0, config.training_max_steps)
    assert initial.context_fraction == teacher.jepa_context_fraction
    assert initial.target_fraction == teacher.jepa_target_fraction
    assert initial.gradient_accumulation_steps == config.gradient_accumulation_steps
    for step in (99_999, 100_000, 199_999, 200_000, 1_999_999):
        assert jepa_mask_stage(config, step, config.training_max_steps) == initial


def test_1p8b_mae_teacher_jepa_matches_teacher_backbone_and_inputs() -> None:
    config = load_config("configs/1p8b_mae_teacher_jepa.py")
    teacher_config = load_config(config.frozen_teacher_config_path)
    student = PeakSetJEPASettings.from_config(config)
    teacher = PeakSetJEPASettings.from_config(teacher_config)
    assert student.model_dim == student.predictor_dim == 2048
    assert student.encoder_num_layers == 25
    assert student.encoder_num_heads == 8
    for key in teacher_config:
        if key.startswith(("encoder_", "pairmixer_", "feature_", "peak_")):
            assert config[key] == teacher_config[key], key
    for key in (
        "num_peaks", "max_precursor_mz", "precursor_peak_exclusion_window_da",
        "min_peak_intensity", "attention_mlp_multiple", "norm_eps",
        "masked_latent_predictor_num_layers", "masked_latent_predictor_num_heads",
        "spectrum_metadata_dropout_probability",
    ):
        assert config.get(key) == teacher_config.get(key), key
    assert student.feature_mlp_hidden_dim == teacher.feature_mlp_hidden_dim == 4096
    assert "-1p8b-student-" in config.run_name_suffix
    assert config.jax_zero1_optimizer is True
    assert config.jax_zero1_mode == "per_leaf"
    assert "-bf16state-zero1-perleaf-" in config.run_name_suffix
    assert "-bs1024-" in config.run_name_suffix


def test_periodic_teacher_starts_from_linked_jepa_encoder_with_same_recipe() -> None:
    config = load_config("configs/1p8b_periodic_teacher_jepa.py")
    baseline = load_config("configs/1p8b_mae_teacher_jepa.py")

    assert config.frozen_teacher_update_every_steps == 50_000
    assert baseline.frozen_teacher_update_every_steps == 0
    assert config.frozen_teacher_config_path == "configs/1p8b_mae_teacher_jepa.py"
    assert config.frozen_teacher_checkpoint_path == (
        "gs://metal-repeater-411410-spectra-checkpoints/skypilot/"
        "1p8b-ftjepa-t1p8b-s666k-perleaf-b1024-v6e32-20260905-115335/"
        "checkpoints/orbax/1190000"
    )
    for key in (
        "device_backend", "training_mode", "model_dim", "predictor_dim",
        "encoder_num_layers", "encoder_num_heads", "batch_size",
        "gradient_accumulation_steps", "optimizer", "learning_rate",
        "optimizer_state_dtype", "jax_zero1_mode", "jepa_context_fraction",
        "jepa_target_fraction", "jepa_mask_strategy", "training_max_steps",
    ):
        assert config[key] == baseline[key], key
