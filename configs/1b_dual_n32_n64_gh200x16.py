"""Joint single+pair 1B ``dual`` run for 16 GH200 GPUs (4 Isambard nodes).

Recipe from docs/pair_single_dual_20261001.md. One fastmixer encoder carries
single and pair streams with pair width equal to single width. After each
block, pair->single bridges (mean row summary plus edge-gated messages,
rank 64) add to the attention-logit bias. A cross-attention predictor
reconstructs masked m/z from singles (weight 0.7). A pair-only predictor
scores the distogram on the encoder's pair tensor (weight 0.3), so pairs
receive the distance gradient directly. Both heads use 0.1 Da bins.
Acquisition metadata enters as a precursor node: CLS edges carry each peak's
neutral loss and the embedded condition vector, with 10% per-field dropout.

Context expansion: the first stage trains on 31 peaks plus CLS (32 tokens);
from jepa_mask_schedule_steps[0] on, it trains on 63 peaks plus CLS
(64 tokens). The model is built for 63 peaks, so the stage switch keeps every
parameter and optimizer state and only changes data, capacities and graphs.

GPU performance (GH200, 128 spectra per GPU, no gradient accumulation):
``activation_checkpoint_mode='ordered'`` is required, since nnx.remat saves no
memory in multi-GPU programs (see common_jax.activation_checkpoint). Peak
memory is 26 GB at 31 peaks and 73 GB at 63 peaks. One node runs 454 and
137 spectra/s; two nodes keep 93% and 97% of linear scaling. On 16 GPUs that
is about 1.2 s per step at 31 peaks and 4 s per step at 63 peaks.
"""

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Training task and joint single+pair architecture
    cfg.training_task = 'pretrain'
    cfg.training_mode = 'dual'
    cfg.model_dim = 1024
    cfg.pairmixer_pair_dim = 1024
    cfg.encoder_num_layers = 21
    cfg.encoder_num_heads = 8
    cfg.predictor_dim = 1024
    cfg.masked_latent_predictor_num_layers = 4
    cfg.masked_latent_predictor_num_heads = 8
    cfg.pair_predictor_num_layers = 4
    cfg.pair_to_single_rank = 64
    cfg.encoder_use_cls_token = True
    cfg.encoder_apply_final_pair_norm = True
    cfg.predictor_apply_final_norm = True
    cfg.pairmixer_block_type = 'fastmixer'
    cfg.pairmixer_transition_type = 'swiglu'
    cfg.attention_mlp_multiple = 8 / 3
    cfg.predictor_mlp_multiple = 8 / 3
    cfg.norm_eps = 1e-05
    cfg.masked_token_input_mode = 'latent_token'
    cfg.encoder_use_position_embedding = True

    # Single-stream peak embedding
    cfg.encoder_apply_final_norm = True
    cfg.encoder_fourier_mlp_hidden_dim = 2048
    cfg.encoder_fourier_mlp_num_layers = 2
    cfg.encoder_fourier_num_freqs = 16
    cfg.encoder_fourier_x_max = 1000
    cfg.encoder_fourier_x_min = 0.003
    cfg.encoder_mz_embedding = 'fourier'
    cfg.encoder_mz_token_bin_size = 0.02
    cfg.encoder_mz_token_embedding_dim = 77
    cfg.feature_mlp_hidden_dim = 2048

    # Peak-count stages: 31 peaks (+CLS = 32 tokens), then 63 (+CLS = 64).
    cfg.num_peaks = 63
    cfg.num_peaks_schedule = (31, 63)
    cfg.jepa_mask_schedule_steps = (250000,)
    cfg.gradient_accumulation_steps_schedule = (1, 1)

    # Pair features and metadata conditioning
    cfg.encoder_metadata_condition_dim = 256
    cfg.encoder_metadata_conditioning = 'precursor_edge'
    cfg.encoder_metadata_schema = 'massive_v2_acquisition_v1'
    cfg.pairmixer_dropout = 0.0
    cfg.pairmixer_fourier_num_freqs = 16
    cfg.pairmixer_fourier_x_min = 0.01
    cfg.pairmixer_mz_embedding = 'fourier'
    cfg.pairmixer_mz_token_bin_size = 0.1
    cfg.pairmixer_mz_token_embedding_dim = 128
    cfg.pairmixer_pair_feature_hidden_dim = 2048
    cfg.pairmixer_relative_fourier_x_max = 1.0
    cfg.pairmixer_relative_fourier_x_min = 0.001
    cfg.pairmixer_use_pair_bias = True
    cfg.spectrum_metadata_dropout_probability = 0.1

    # Masking, masked m/z and distance supervision
    cfg.mae_loss_weight = 0.7
    cfg.distogram_loss_weight = 0.3
    cfg.distogram_multiscale_weight = 0.0
    cfg.distogram_multiscale_widths = ()
    cfg.jepa_allow_target_overlap = False
    cfg.jepa_block_min_len = 1
    cfg.jepa_context_fraction = 0.6
    cfg.jepa_intensity_aware_alpha = 0.75
    cfg.jepa_intensity_aware_anchor_keep = 0.6
    cfg.jepa_intensity_aware_beta_context = 0.85
    cfg.jepa_intensity_aware_beta_target = 0.85
    cfg.jepa_intensity_aware_eps_context = 0.04
    cfg.jepa_intensity_aware_eps_target = 0.08
    cfg.jepa_intensity_aware_local_gap_da = 1.0
    cfg.jepa_intensity_aware_local_gap_probability = 0.5
    cfg.jepa_intensity_aware_min_eff_context = 4.0
    cfg.jepa_intensity_aware_min_eff_target = 1.8
    cfg.jepa_intensity_aware_min_unused_mass = 0.09
    cfg.jepa_intensity_aware_tail_target_mix = 0.1
    cfg.jepa_intensity_aware_tau = 0.5
    cfg.jepa_mae_intensity_bin_size = 0.1
    cfg.jepa_mae_intensity_max = 1.0
    cfg.jepa_mae_loss_weight = 0.0
    cfg.jepa_mae_mz_bin_size = 0.1
    cfg.jepa_mask_lengths = (2, 4, 8, 12)
    cfg.jepa_mask_round_from = 3
    cfg.jepa_mask_strategy = ['random']
    cfg.jepa_num_target_blocks = 1
    cfg.jepa_target_fraction = 0.25
    cfg.jepa_target_normalization = 'none'
    cfg.latent_pair_loss_weight = 0.0
    cfg.latent_pair_target_normalization = 'none'
    cfg.mae_intensity_loss_weight = 0.0
    cfg.mae_mz_multiscale_weight = 0.0
    cfg.mae_mz_multiscale_widths = ()
    cfg.masked_token_loss_weight = 0.0
    cfg.distogram_pair_dim = None

    # Dataset and input pipeline
    # Isambard: /scratch/u6rb/wuhao.u6rb/spectra-data/massive_v2_ms2_t095_l080_sharded_10gb
    cfg.artifact_dir = 'data/massive_v2_ms2_t095_l080_sharded_10gb'
    cfg.gems_hdf5_manifest = 'manifest.json'
    cfg.gems_hdf5_ms_level_dataset = 'MS level'
    cfg.gems_hdf5_precursor_dataset = 'precursor_mz'
    cfg.gems_hdf5_repo_id = 'novogaia/massive-v2-ms2-t095-l080-sharded-10gb'
    cfg.gems_hdf5_retention_time_dataset = 'RT'
    cfg.gems_hdf5_revision = 'b098155fccb7335b752f2690bf16d2479d69bf5b'
    cfg.gems_hdf5_rows_per_block = 0
    cfg.gems_hdf5_shuffle_seed = 67
    cfg.gems_hdf5_shuffle_window_blocks = 2048
    cfg.gems_hdf5_spectrum_dataset = 'spectrum'
    cfg.seed = 66
    cfg.batch_size = 2048
    cfg.drop_remainder = True
    cfg.max_precursor_mz = 1000
    cfg.min_peak_intensity = 0.0001
    cfg.peak_drop_min_intensity = 0.0001
    cfg.precursor_peak_exclusion_window_da = 0
    cfg.peak_ordering = 'mz'
    cfg.dataloader_multiprocessing_context = 'spawn'
    cfg.dataloader_num_workers = 16
    cfg.dataloader_output_format = 'numpy'
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = False
    cfg.dataloader_prefetch_factor = 2

    # Optimizer and training schedule
    cfg.optimizer = 'adamw'
    cfg.optimizer_fused = True
    cfg.optimizer_state_dtype = 'bf16'
    cfg.b1 = 0.9
    cfg.b2 = 0.95
    cfg.learning_rate = 0.0003
    cfg.min_learning_rate = 3e-05
    cfg.warmup_steps = 10000
    cfg.weight_decay = 0.01
    cfg.grad_clip_norm = 1.0
    cfg.gradient_accumulation_steps = 1
    cfg.num_epochs = 98
    cfg.training_max_steps = 300000
    cfg.max_duration_hours = 23.5
    cfg.autocast_dtype = 'bf16'

    # JAX runtime and activation checkpointing
    cfg.device = 'auto'
    cfg.device_backend = 'jax'
    cfg.jax_checkpoint_max_to_keep = 5
    cfg.jax_compilation_cache_dir = ''
    cfg.jax_compile_stall_threshold_seconds = 0.0
    cfg.jax_distributed_initialize = True
    cfg.jax_enable_async_checkpointing = True
    cfg.jax_enable_compilation_cache = True
    cfg.jax_explain_cache_misses = False
    cfg.jax_grad_all_reduce_dtype = 'bfloat16'
    cfg.jax_log_compiles = False
    cfg.jax_log_update_stats = False
    cfg.jax_mesh_devices = 'all'
    cfg.jax_persistent_cache_min_compile_time_secs = None
    cfg.jax_persistent_cache_min_entry_size_bytes = None
    cfg.jax_profile_dir = ''
    cfg.jax_profile_start_step = 5000
    cfg.jax_profile_steps = 0
    cfg.jax_shard_optimizer = False
    cfg.jax_timing_barriers = False
    cfg.jax_zero1_optimizer = False
    cfg.compile_mode = 'max-autotune-no-cudagraphs'
    cfg.activation_checkpoint_every_n_layers = 1
    cfg.activation_checkpoint_mode = 'ordered'
    cfg.activation_checkpoint_modules = ('encoder', 'predictor', 'distogram')
    cfg.activation_checkpoint_preserve_rng_state = True

    # Validation, checkpoint cadence, and logging
    cfg.val_every_n_steps = 10000.0
    cfg.val_num_steps = 500
    cfg.checkpoint_every_steps = 10000
    cfg.log_every_n_steps = 25
    cfg.collapse_metrics_every_n_steps = 0
    cfg.throughput_warmup_steps = 25
    cfg.training_flops_per_optimizer_step = None
    cfg.training_flops_per_parameter = 6.0
    cfg.training_flops_per_sample = None
    cfg.enable_wandb = True
    cfg.wandb_kwargs = {}
    cfg.wandb_project = 'jepa-finalrun'
    cfg.wandb_resume_from_env = False
    cfg.wandb_resume_id = ''
    cfg.wandb_shared_label = 'train'
    cfg.wandb_shared_mode = False
    cfg.wandb_shared_primary = True
    cfg.wandb_shared_update_finish_state = False
    cfg.run_name_suffix = "-dual-d1024-l21-pred4-n32-n64-p2s64-precursoredge-mz0.1-gh200x16"

    # Inactive teacher, contrastive, and probe settings
    cfg.contrastive_batch_size = 128
    cfg.contrastive_compile_mode = 'none'
    cfg.contrastive_covariance_dim = 64
    cfg.contrastive_loss_weight = 1.0
    cfg.contrastive_online_probe_hidden_dim = 1024
    cfg.contrastive_pairs_per_epoch = 30000
    cfg.contrastive_single_pair_include_diagonal = False
    cfg.contrastive_temperature = 0.1
    cfg.contrastive_triplet_hard_fraction = None
    cfg.contrastive_val_every_n_steps = 1000
    cfg.contrastive_val_pairs_per_epoch = 10000
    cfg.covariance_pooling_dim = 64
    cfg.ema_teacher_momentum_final = 0.99925
    cfg.ema_teacher_momentum_mid = None
    cfg.ema_teacher_momentum_start = 0.996
    cfg.ema_teacher_schedule = 'cosine'
    cfg.ema_teacher_schedule_peak_fraction = 0.35
    cfg.frozen_teacher_checkpoint_path = None
    cfg.frozen_teacher_config_path = None
    cfg.masked_mz_sentinel = -1.0
    cfg.msg_probe_at_final_step = False
    cfg.msg_probe_batch_size = 512
    cfg.msg_probe_early_stopping = True
    cfg.msg_probe_early_stopping_min_delta = 0.0001
    cfg.msg_probe_early_stopping_min_epochs = 20
    cfg.msg_probe_early_stopping_patience = 5
    cfg.msg_probe_every_n_steps = -1.0
    cfg.msg_probe_fingerprint = 'maccs'
    cfg.msg_probe_grad_clip_norm = 1.0
    cfg.msg_probe_learning_rate = 0.0003
    cfg.msg_probe_mlp_hidden_dim = 256
    cfg.msg_probe_num_epochs = 100
    cfg.msg_probe_num_repeats = None
    cfg.msg_probe_pairwise_alignment_num_pairs = 0
    cfg.msg_probe_pma_num_heads = 8
    cfg.msg_probe_pma_num_seeds = 32
    cfg.msg_probe_select_metric = 'msg_probe/val/auc_fluorine'
    cfg.msg_probe_variants = ['single_pair_covariance']
    cfg.msg_probe_warmup_epochs = 0.5
    cfg.msg_probe_warmup_steps = 0
    cfg.msg_probe_weight_decay = 0
    cfg.nist_murcko_probe_num_repeats = 1
    cfg.online_probe_loss_weight = 1.0
    cfg.predictor_dropout = 0.0
    cfg.target_projector_dim = -1
    cfg.use_ema_teacher = False

    return cfg
