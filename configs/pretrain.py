from ml_collections import config_dict

from spectra_learning.config.defaults import runtime_config


def get_config() -> config_dict.ConfigDict:
    """Return the canonical peak-set pretraining defaults."""
    cfg = runtime_config()

    cfg.training_task = "pretrain"

    # Dataset
    cfg.artifact_dir = "data/massive_v1_ms2_100m_stratified_x16"
    cfg.gems_hdf5_repo_id = "novogaia/massive-v1-ms2-100m-stratified-x16"
    cfg.gems_hdf5_revision = "7ff47061cbde23e4cdd113378dcfb489e86b32c4"
    cfg.gems_hdf5_manifest = "fdataloader_shards.json"
    cfg.gems_hdf5_spectrum_dataset = "spectrum"
    cfg.gems_hdf5_precursor_dataset = "precursor_mz"
    cfg.gems_hdf5_retention_time_dataset = "RT"
    cfg.gems_hdf5_ms_level_dataset = "MS level"
    cfg.gems_hdf5_rows_per_block = 0
    cfg.nist_murcko_probe_num_repeats = 1
    cfg.batch_size = 512
    cfg.gradient_accumulation_steps = 4
    cfg.shuffle_buffer = 1_000_000
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000
    cfg.min_peak_intensity = 0.0001
    cfg.peak_drop_min_intensity = 0.0001
    cfg.precursor_peak_exclusion_window_da = 0
    cfg.peak_ordering = "mz"
    cfg.seed = 66

    # Encoder
    cfg.num_peaks = 31
    cfg.model_dim = 640
    cfg.encoder_num_layers = 15
    cfg.encoder_num_heads = 10
    cfg.encoder_use_position_embedding = False
    cfg.encoder_apply_final_norm = True
    cfg.encoder_apply_final_pair_norm = True
    cfg.encoder_mz_embedding = "fourier"
    cfg.encoder_mz_token_bin_size = 0.02
    cfg.encoder_mz_token_embedding_dim = 77
    cfg.encoder_fourier_mlp_hidden_dim = 1280
    cfg.encoder_fourier_mlp_num_layers = 4
    cfg.encoder_fourier_num_freqs = 64
    cfg.encoder_fourier_x_max = 1000
    cfg.encoder_fourier_x_min = 0.003
    cfg.feature_mlp_hidden_dim = 1024
    cfg.pairmixer_pair_dim = 384
    cfg.pairmixer_pair_feature_hidden_dim = 768
    cfg.pairmixer_dropout = 0.0
    cfg.pairmixer_use_pair_bias = True
    cfg.pairmixer_use_fourier_features = True
    cfg.pairmixer_fourier_num_freqs = 16
    cfg.pairmixer_fourier_x_min = 0.01
    cfg.pairmixer_relative_fourier_x_min = 0.001
    cfg.pairmixer_relative_fourier_x_max = 1.0
    cfg.attention_mlp_multiple = 4
    cfg.norm_eps = 1e-5

    # Masked latent predictor
    cfg.predictor_dim = 640
    cfg.predictor_dropout = 0.1
    cfg.predictor_apply_final_norm = True
    cfg.masked_latent_predictor_num_layers = 3
    cfg.masked_latent_predictor_num_heads = 10
    cfg.target_projector_dim = -1
    cfg.masked_token_input_mode = "latent_token"
    cfg.masked_mz_sentinel = -1.0

    # MAE masking and reconstruction objectives
    cfg.training_mode = "mae"
    cfg.use_ema_teacher = False
    cfg.jepa_num_target_blocks = 1
    cfg.jepa_mask_strategy = ["intensity_aware", "ragged"]
    cfg.jepa_context_fraction = 0.35
    cfg.jepa_target_fraction = 0.25
    cfg.jepa_block_min_len = 1
    cfg.jepa_mask_lengths = (2, 4, 8, 12)
    cfg.jepa_mask_round_from = 3
    cfg.jepa_target_normalization = "none"
    cfg.masked_token_loss_weight = 0.0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.mae_loss_weight = 1.0
    cfg.distogram_loss_weight = 1.0
    cfg.latent_pair_loss_weight = 0.0
    cfg.latent_pair_target_normalization = "none"
    cfg.jepa_mae_mz_bin_size = 0.5
    cfg.jepa_mae_intensity_bin_size = 0.1
    cfg.jepa_mae_intensity_max = 1.0
    cfg.mae_intensity_loss_weight = 1.0
    cfg.jepa_allow_target_overlap = False
    cfg.jepa_intensity_aware_tau = 0.5
    cfg.jepa_intensity_aware_alpha = 0.75
    cfg.jepa_intensity_aware_beta_context = 0.85
    cfg.jepa_intensity_aware_beta_target = 0.85
    cfg.jepa_intensity_aware_eps_context = 0.04
    cfg.jepa_intensity_aware_eps_target = 0.08
    cfg.jepa_intensity_aware_anchor_keep = 0.60
    cfg.jepa_intensity_aware_min_eff_context = 4.0
    cfg.jepa_intensity_aware_min_eff_target = 1.8
    cfg.jepa_intensity_aware_tail_target_mix = 0.10
    cfg.jepa_intensity_aware_local_gap_da = 1.0
    cfg.jepa_intensity_aware_local_gap_probability = 0.50
    cfg.jepa_intensity_aware_min_unused_mass = 0.09

    # Pooling
    cfg.covariance_pooling_dim = 64

    # Contrastive NIST Murcko training
    cfg.contrastive_temperature = 0.1
    cfg.contrastive_loss_weight = 1.0
    cfg.contrastive_triplet_hard_fraction = None
    cfg.online_probe_loss_weight = 1.0
    cfg.contrastive_batch_size = 128
    cfg.contrastive_covariance_dim = 64
    cfg.contrastive_online_probe_hidden_dim = 1024
    cfg.contrastive_pairs_per_epoch = 30_000
    cfg.contrastive_val_pairs_per_epoch = 10_000
    cfg.contrastive_val_every_n_steps = 1_000
    cfg.contrastive_single_pair_include_diagonal = False
    cfg.contrastive_compile_mode = "none"

    # EMA teacher
    cfg.ema_teacher_momentum_start = 0.996
    cfg.ema_teacher_momentum_mid = None
    cfg.ema_teacher_momentum_final = 0.99925
    cfg.ema_teacher_schedule_peak_fraction = 0.35
    cfg.ema_teacher_schedule = "cosine"
    cfg.frozen_teacher_config_path = None
    cfg.frozen_teacher_checkpoint_path = None

    # Training
    cfg.num_epochs = 8
    cfg.training_max_steps = 300_000
    cfg.autocast_dtype = "bf16"
    cfg.compile_mode = "max-autotune-no-cudagraphs"
    cfg.activation_checkpoint_mode = "selective"
    cfg.activation_checkpoint_every_n_layers = 1
    cfg.activation_checkpoint_modules = ("encoder", "predictor")
    cfg.activation_checkpoint_preserve_rng_state = True
    cfg.device_prefetch_size = 8
    cfg.throughput_warmup_steps = 25
    cfg.log_every_n_steps = 250
    cfg.collapse_metrics_every_n_steps = 0
    cfg.checkpoint_every_steps = 10_000
    cfg.dataloader_num_workers = 8
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = True
    cfg.dataloader_multiprocessing_context = ""
    cfg.dataloader_output_format = "torch"

    # MSG probe
    cfg.msg_probe_early_stopping = True
    cfg.msg_probe_early_stopping_min_delta = 0.0001
    cfg.msg_probe_early_stopping_min_epochs = 20
    cfg.msg_probe_early_stopping_patience = 5
    cfg.msg_probe_every_n_steps = 50_000.
    cfg.val_every_n_steps = cfg.msg_probe_every_n_steps
    cfg.val_num_steps = 64
    cfg.msg_probe_learning_rate = 0.0003
    cfg.msg_probe_mlp_hidden_dim = 256
    cfg.msg_probe_num_epochs = 100
    cfg.msg_probe_pma_num_heads = 8
    cfg.msg_probe_pma_num_seeds = 32
    cfg.msg_probe_select_metric = "msg_probe/val/auc_fluorine"
    cfg.msg_probe_variants = ["single_pair_covariance"]
    cfg.msg_probe_warmup_epochs = 0.5
    cfg.msg_probe_grad_clip_norm = 1.0
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_weight_decay = 0
    cfg.msg_probe_batch_size = 512
    cfg.msg_probe_fingerprint = "maccs"
    cfg.msg_probe_num_repeats = None
    cfg.msg_probe_pairwise_alignment_num_pairs = 0

    # Optimizer
    cfg.learning_rate = 4e-04
    cfg.min_learning_rate = 4e-05
    cfg.warmup_steps = 3_000
    cfg.jax_profile_start_step = cfg.get_ref("warmup_steps")
    cfg.weight_decay = 0.0
    cfg.b2 = 0.95
    cfg.grad_clip_norm = 0.
    cfg.optimizer = "adam"
    cfg.optimizer_fused = True

    # Logging
    cfg.enable_wandb = True
    cfg.wandb_project = "jepa-finalrun"
    cfg.run_name_suffix = "pretrain"

    return cfg
