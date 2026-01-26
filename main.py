import argparse
import importlib.util
from pathlib import Path


def _disable_tensorflow_gpu():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")


def _load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("run_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def main():
    parser = argparse.ArgumentParser(description="Train/eval entrypoint.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    parser.add_argument("--olddir", default=None, help="Optional checkpoint load dir.")
    args = parser.parse_args()

    _disable_tensorflow_gpu()
    from train_mae import train_and_evaluate

    cfg = _load_config(args.config)
    workdir = Path(args.workdir).expanduser().resolve()
    olddir = None if args.olddir is None else Path(args.olddir).expanduser().resolve()
    train_and_evaluate(cfg, workdir=workdir, olddir=olddir)


if __name__ == "__main__":
    main()
