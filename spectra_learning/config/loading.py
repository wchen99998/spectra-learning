import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from ml_collections import config_dict


def load_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> config_dict.ConfigDict:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    assert spec is not None, f"Could not load module spec from {path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = module.get_config()
    if overrides:
        config.update(overrides)
    config.config_path = str(path)
    return config


def config_to_dict(config: config_dict.ConfigDict) -> dict[str, Any]:
    return _serializable(config.to_dict())


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    return str(value)
