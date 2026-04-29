import importlib.util
from pathlib import Path

from ml_collections import config_dict


def load_config(path: str | Path) -> config_dict.ConfigDict:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    assert spec is not None, f"Could not load module spec from {path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()
