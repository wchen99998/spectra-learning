from typing import Any


DEPRECATED_MSG_PROBE_CONFIG_KEYS = (
    "probe_dataset",
    "msg_probe_tune_metric",
    "msg_probe_sample_size",
    "msg_probe_max_train_samples",
    "msg_probe_max_val_samples",
    "msg_probe_max_test_samples",
    "msg_probe_fingerprint_type",
    "nist_murcko_probe_train_samples",
    "nist_murcko_probe_val_samples",
    "nist_murcko_probe_test_samples",
)


def validate_msg_probe_config(config: Any) -> None:
    deprecated = [key for key in DEPRECATED_MSG_PROBE_CONFIG_KEYS if key in config]
    if deprecated:
        raise ValueError(
            "Deprecated MSG probe config fields: " + ", ".join(deprecated)
        )
