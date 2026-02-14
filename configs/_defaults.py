from ml_collections import config_dict


def apply_training_defaults(cfg: config_dict.ConfigDict) -> None:
    """Apply shared training infrastructure defaults.

    Call after model-specific fields (like num_heads) are set.
    Per-experiment configs should override values that differ.
    """
    # Profiler
    cfg.profile_enabled = False
    cfg.profile_wait_steps = 20
    cfg.profile_warmup_steps = 20
    cfg.profile_active_steps = 40
    cfg.profile_repeat = 1
    cfg.profile_record_shapes = True
    cfg.profile_with_stack = True
    cfg.profile_profile_memory = True
    cfg.profile_trace_dir = "profiler"

    # Dataloader
    cfg.non_blocking_device_transfer = True
    cfg.optimizer_capturable = True
    cfg.optimizer_fused = True
    cfg.dataloader_num_workers = 1
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_pin_memory = True

    # Muon optimizer defaults (only used when cfg.optimizer == "muon")
    cfg.muon_lr = None              # Falls back to cfg.learning_rate
    cfg.adamw_lr = None             # Falls back to cfg.learning_rate
    cfg.muon_momentum = 0.95
    cfg.muon_nesterov = True
    cfg.muon_ns_steps = 5
    cfg.muon_weight_decay = None    # Falls back to cfg.weight_decay
    cfg.muon_adjust_lr_fn = "match_rms_adamw"


def apply_final_probe_defaults(cfg: config_dict.ConfigDict) -> None:
    """Apply shared final attentive probe defaults.

    Call after num_heads is set. Override per-experiment values afterward.
    """
    cfg.final_probe_num_epochs = 5
    cfg.final_probe_learning_rate = 1e-4
    cfg.final_probe_weight_decay = 1e-4
    cfg.final_probe_warmup_steps = 100
    cfg.final_probe_feature_source = "projector"
    cfg.final_probe_head_hidden_dim = 512

    cfg.final_probe_num_precursor_bins = 1000
    cfg.final_probe_attention_heads = cfg.num_heads
    cfg.final_probe_loss_weights = [1.0, 1.0, 1.0]
    cfg.final_probe_freeze_backbone = True


def apply_tune_defaults(cfg: config_dict.ConfigDict) -> None:
    """Apply default tune search space definition.

    Each entry is {"param": <config key>, "dist": <distribution>, "args": [...]}.
    Supported distributions: grid, loguniform, uniform, choice, randint, quniform.
    """
    cfg.tune_param_space = [
        # {"param": "learning_rate", "dist": "grid", "args": [1e-4, 4e-4]},
        {"param": "weight_decay", "dist": "grid", "args": [1e-4, 1e-3, 1e-2]},
        {"param": "sigreg_lambda", "dist": "grid", "args": [1.0]},
    ]
