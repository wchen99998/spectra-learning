from pathlib import Path

from spectra_learning.config import load_config
from spectra_learning.models.settings import PeakSetJEPASettings


CONFIG_PATH = "configs/1b_singlemixer_dense_muon_metadata_adaln.py"


def test_metadata_adaln_config_is_standalone() -> None:
    source = Path(CONFIG_PATH).read_text()
    assert "import_module" not in source
    assert "runtime_config" not in source
    assert "configs." not in source


def test_metadata_adaln_config_preserves_training_settings() -> None:
    config = load_config(CONFIG_PATH)
    baseline = load_config("configs/1b_singlemixer_dense_muon.py")
    settings = PeakSetJEPASettings.from_config(config)

    assert settings.encoder_metadata_schema == "massive_v2_acquisition_v1"
    assert settings.encoder_metadata_conditioning == "adaln_zero"
    assert settings.encoder_metadata_condition_dim == 256
    assert config.spectrum_metadata_dropout_probability == 0.10
    assert config.gems_hdf5_revision == "b098155fccb7335b752f2690bf16d2479d69bf5b"
    assert config.peak_ordering == "mz"
    assert "mzorder" in config.run_name_suffix
    for key in (
        "training_max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "optimizer",
        "learning_rate",
        "jepa_context_fraction",
        "jepa_target_fraction",
        "jax_mesh_devices",
    ):
        assert config[key] == baseline[key]


def test_metadata_adaln_parameter_count_formula() -> None:
    baseline_parameters = 940_416_464
    model_dim = 1_536
    condition_dim = 256
    layers = 25
    metadata_embedder = 26 * condition_dim + condition_dim
    metadata_embedder += condition_dim * condition_dim + condition_dim
    block_modulations = layers * (
        condition_dim * (6 * model_dim) + 6 * model_dim
    )
    final_modulation = condition_dim * (2 * model_dim) + 2 * model_dim
    removed_legacy_projection = 2 * model_dim
    removed_affine_pre_norms = layers * 2 * model_dim

    assert (
        baseline_parameters
        - removed_legacy_projection
        - removed_affine_pre_norms
        + metadata_embedder
        + block_modulations
        + final_modulation
        == 1_000_411_600
    )
