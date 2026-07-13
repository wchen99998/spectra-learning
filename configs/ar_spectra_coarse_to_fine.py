from ml_collections import config_dict

from spectra_learning.config.defaults import runtime_config


def get_config() -> config_dict.ConfigDict:
    cfg = runtime_config()

    cfg.seed = 0
    cfg.training_task = "ar_spectra"
    cfg.device_backend = "jax"
    cfg.device = "auto"
    cfg.num_epochs = 6.0
    cfg.training_max_steps = 250_000
    cfg.autocast_dtype = "bf16"
    cfg.learning_rate = 3e-4
    cfg.min_learning_rate = None
    cfg.warmup_steps = 0
    cfg.jax_profile_start_step = cfg.get_ref("warmup_steps")
    cfg.optimizer = "adamw"
    cfg.weight_decay = 0.01
    cfg.b1 = 0.9
    cfg.b2 = 0.95
    cfg.grad_clip_norm = 1.0
    cfg.log_every_n_steps = 20
    cfg.throughput_warmup_steps = 25
    cfg.val_every_n_steps = 500.0
    cfg.val_num_steps = 50
    cfg.checkpoint_every_steps = 1000
    cfg.checkpoint_every_n_steps = 1000
    cfg.msg_probe_every_n_steps = -1.0
    cfg.msg_probe_at_final_step = False
    cfg.jax_mesh_devices = "32"
    cfg.jax_checkpoint_max_to_keep = None
    cfg.jax_enable_async_checkpointing = True
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-finalrun"
    cfg.wandb_kwargs = {
        "name": "ar-spectra-jax-rope-multilevel-100m-bs2048-bf16-250k",
    }

    cfg.artifact_dir = "data/massive_v1_ms2_100m_stratified_x16"
    cfg.gems_hdf5_repo_id = "novogaia/massive-v1-ms2-100m-stratified-x16"
    cfg.gems_hdf5_revision = "main"
    cfg.gems_hdf5_manifest = "fdataloader_shards.json"
    cfg.gems_hdf5_spectrum_dataset = "spectrum"
    cfg.gems_hdf5_precursor_dataset = "precursor_mz"
    cfg.gems_hdf5_rows_per_block = 0

    cfg.batch_size = 2048
    cfg.gradient_accumulation_steps = 1
    cfg.drop_remainder = True
    cfg.dataloader_num_workers = 8
    cfg.dataloader_pin_memory = True
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_multiprocessing_context = ""
    cfg.dataloader_output_format = "torch"

    cfg.num_peaks = 128
    cfg.max_precursor_mz = 1000.0
    cfg.min_peak_intensity = 1e-4
    cfg.peak_drop_min_intensity = 1e-4
    cfg.peak_filtering = "grouped"
    cfg.grouped_peak_shoulder_da = 0.02
    cfg.grouped_peak_isotope_charges = (1, 2, 3)
    cfg.peak_ordering = "mz"
    cfg.precursor_peak_exclusion_window_da = 0.0

    cfg.ar_max_num_peaks = 128
    cfg.ar_mz_max = 1000.0
    cfg.ar_precursor_mz_max = 1000.0
    cfg.ar_mz_bin_widths = (50.0, 25.0, 5.0, 1.0)
    cfg.ar_residual_bins = 100
    cfg.ar_intensity_bins = 101
    cfg.ar_collision_energy_bins = 101
    cfg.ar_charge_bins = 22

    cfg.ar_model_dim = 1024
    cfg.ar_num_layers = 8
    cfg.ar_num_heads = 8
    cfg.ar_mlp_multiple = 4.0
    cfg.ar_dropout = 0.0
    cfg.ar_rope_base = 10_000.0
    cfg.ar_attention_kernel = "pallas"
    cfg.ar_splash_block_size = 128
    cfg.ar_gelu_approximation = "quick"

    return cfg
