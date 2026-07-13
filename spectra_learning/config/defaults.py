from ml_collections import config_dict


def runtime_config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict(
        {
            "device": "auto",
            "device_backend": "auto",
            "jax_checkpoint_max_to_keep": 5,
            "jax_compilation_cache_dir": "",
            "jax_compile_stall_threshold_seconds": 0.0,
            "jax_distributed_initialize": False,
            "jax_enable_async_checkpointing": True,
            "jax_enable_compilation_cache": True,
            "jax_explain_cache_misses": False,
            "jax_log_compiles": False,
            "jax_log_update_stats": False,
            "jax_mesh_devices": "all",
            "jax_msg_probe_shard_batches": False,
            "jax_persistent_cache_min_compile_time_secs": None,
            "jax_persistent_cache_min_entry_size_bytes": None,
            "jax_profile_dir": "",
            "jax_profile_steps": 0,
            "jax_timing_barriers": False,
            "max_duration_hours": None,
            "training_flops_per_optimizer_step": None,
            "training_flops_per_parameter": 6.0,
            "training_flops_per_sample": None,
            "wandb_kwargs": {},
            "wandb_resume_from_env": True,
            "wandb_resume_id": "",
            "wandb_shared_label": "train",
            "wandb_shared_mode": False,
            "wandb_shared_primary": True,
            "wandb_shared_update_finish_state": False,
        }
    )
