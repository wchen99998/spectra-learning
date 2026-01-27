from pathlib import Path

import jax
import tensorflow as tf
from absl import app, flags, logging
from ml_collections import config_flags


tf.config.set_visible_devices([], "GPU")

from train_mae import train_and_evaluate

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    None,
    "Path to config file.",
    lock_config=False,
)
_WORKDIR = flags.DEFINE_string("workdir", None, "Output directory.")
_OLDDIR = flags.DEFINE_string(
    "olddir",
    None,
    "Optional checkpoint load dir.",
)
flags.mark_flags_as_required(["config", "workdir"])


def main(_: list[str]) -> None:
    logging.set_verbosity(logging.INFO)
    config = _CONFIG.value

    workdir = Path(_WORKDIR.value).expanduser().resolve()
    olddir = None if _OLDDIR.value is None else Path(_OLDDIR.value).expanduser().resolve()
    train_and_evaluate(config, workdir=workdir, olddir=olddir)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
