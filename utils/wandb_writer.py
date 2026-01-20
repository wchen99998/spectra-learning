"""WandB metric writer implementation."""

from __future__ import annotations

import datetime
import os
from typing import Any, Mapping, Optional

import numpy as np
from clu.metric_writers import interface
import wandb

Array = interface.Array
Scalar = interface.Scalar


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    """Best-effort helper to read attributes from ConfigDict-like objects."""
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def build_wandb_init_kwargs(config: Any | None) -> dict[str, Any]:
    """Build wandb.init kwargs, applying an optional run-name prefix.

    The returned dict respects any user-provided `wandb_kwargs` but, when a
    `wandb_run_name_prefix` is supplied in the config and no explicit run name
    is set, a unique name derived from the prefix, trial, timestamp, and a short
    random suffix is generated.
    """

    if config is None:
        return {}

    wandb_kwargs = dict(_config_get(config, "wandb_kwargs", {}) or {})

    resume_id = os.environ.get("WANDB_RESUME_ID")
    if resume_id:
        wandb_kwargs.setdefault("id", resume_id)
        wandb_kwargs.setdefault("resume", "must")
        wandb_kwargs.pop("name", None)
    else:
        prefix = _config_get(config, "wandb_run_name_prefix")
        if prefix and "name" not in wandb_kwargs:
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M")
            name_parts = [str(prefix).strip(), timestamp]

            wandb_kwargs["name"] = "-".join(filter(None, name_parts))

    return wandb_kwargs


def _to_serialisable_config(value: Any) -> Any:
    """Best-effort conversion of config values into WandB friendly formats."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()

    if isinstance(value, Mapping):
        return {str(k): _to_serialisable_config(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_serialisable_config(v) for v in value]

    # Many ML types implement .tolist()/.item() without inheriting from numpy types.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_serialisable_config(tolist())
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    return str(value)


def config_to_wandb_dict(config: Any | None) -> dict[str, Any]:
    """Convert a config-like object into a WandB friendly dictionary."""
    if config is None:
        return {}

    config_dict: Mapping[str, Any] | None = None
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            config_dict = to_dict()
        except Exception:
            config_dict = None

    if config_dict is None and isinstance(config, Mapping):
        config_dict = dict(config)

    if not config_dict:
        return {}

    serialised = _to_serialisable_config(config_dict)
    if isinstance(serialised, Mapping):
        return dict(serialised)
    return {}


class WandBWriter(interface.MetricWriter):
    """A metric writer that logs to Weights & Biases (wandb)."""

    def __init__(self, project: Optional[str] = None, **wandb_init_kwargs):
        """Initialize the WandB writer.
        
        Args:
            project: WandB project name. If None, will use wandb defaults.
            **wandb_init_kwargs: Additional arguments to pass to wandb.init()
        """
        self._project = project
        self._wandb_init_kwargs = wandb_init_kwargs
        
        # Initialize wandb immediately
        init_kwargs = self._wandb_init_kwargs.copy()
        if self._project is not None:
            init_kwargs['project'] = self._project
        wandb.init(**init_kwargs)
        self._initialized = True

    def log_config(self, config: Any | None):
        """Log the provided configuration to the active wandb run."""
        if not self._initialized:
            return
        config_payload = config_to_wandb_dict(config)
        if not config_payload:
            return
        wandb.config.update(config_payload, allow_val_change=True)

    def write_summaries(
        self, 
        step: int,
        values: Mapping[str, Array],
        metadata: Optional[Mapping[str, Any]] = None
    ):
        """Saves an arbitrary tensor summary.
        
        Note: This method is not implemented for wandb writer yet.
        
        Args:
            step: Step at which the scalar values occurred.
            values: Mapping from tensor keys to tensors.
            metadata: Optional SummaryMetadata.
        """
        raise NotImplementedError("write_summaries is not implemented for WandBWriter yet.")

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        """Write scalar values for the step.

        Args:
            step: Step at which the scalar values occurred.
            scalars: Mapping from metric name to value.
        """        
        # Convert scalars to Python native types for wandb
        wandb_scalars = {}
        for key, value in scalars.items():
            # Convert all values to Python scalars
            try:
                # Try to get item() method first (for numpy scalars/0-d arrays)
                if hasattr(value, 'item'):
                    wandb_scalars[key] = value.item()  # type: ignore
                # Try to convert to float for other numeric types
                elif hasattr(value, '__float__'):
                    wandb_scalars[key] = float(value)
                # Fall back to the original value for basic Python types
                else:
                    wandb_scalars[key] = value
            except (AttributeError, ValueError):
                # For arrays with dimensions, provide helpful error
                if hasattr(value, 'shape') and hasattr(value, 'ndim'):
                    if value.ndim > 0:  # type: ignore
                        raise ValueError(f"Expected scalar for key '{key}', got array with shape {value.shape}")  # type: ignore
                # If all else fails, just use the raw value
                wandb_scalars[key] = value
        
        # Log to wandb with the step
        wandb.log(wandb_scalars, step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        """Write images for the step.

        Args:
            step: Step at which the images occurred.
            images: Mapping from image key to images.
        """
        if not images:
            return

        def _make_wandb_image(
            array_like: Any,
            *,
            caption: Optional[str] = None,
        ) -> wandb.Image:
            arr = np.asarray(array_like)
            if arr.ndim == 2:
                arr = arr[..., None]
            if arr.ndim != 3:
                raise ValueError(
                    "Images must be rank-3 tensors (H, W, C) or sequences thereof; "
                    f"got shape {arr.shape}"
                )
            channels = arr.shape[-1]
            if channels == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif channels == 4:
                arr = arr[..., :3]
            elif channels not in (3,):
                raise ValueError(
                    f"Images must have 1, 3, or 4 channels; got {channels}"
                )
            return wandb.Image(arr, caption=caption)

        def _to_wandb_images(value: Any) -> list[wandb.Image]:
            if value is None:
                return []
            if isinstance(value, wandb.Image):
                return [value]
            if isinstance(value, Mapping):
                if "image" not in value:
                    raise ValueError(
                        "Image mappings must contain an 'image' entry when using dict inputs."
                    )
                return [
                    _make_wandb_image(value["image"], caption=value.get("caption"))
                ]
            if isinstance(value, (list, tuple)):
                result: list[wandb.Image] = []
                for item in value:
                    result.extend(_to_wandb_images(item))
                return result

            arr = np.asarray(value)
            if arr.ndim == 4:
                return [_make_wandb_image(img) for img in arr]
            return [_make_wandb_image(arr)]

        wandb_payload: dict[str, Any] = {}
        for key, value in images.items():
            if value is None:
                continue
            wandb_images = _to_wandb_images(value)
            if not wandb_images:
                continue
            if len(wandb_images) == 1:
                wandb_payload[key] = wandb_images[0]
            else:
                wandb_payload[key] = wandb_images

        if wandb_payload:
            wandb.log(wandb_payload, step=step)

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        """Write videos for the step.
        
        Note: This method is not implemented for wandb writer yet.
        
        Args:
            step: Step at which the videos occurred.
            videos: Mapping from video key to videos.
        """
        raise NotImplementedError("write_videos is not implemented for WandBWriter yet.")

    def write_audios(
        self, 
        step: int, 
        audios: Mapping[str, Array], 
        *, 
        sample_rate: int
    ):
        """Write audios for the step.
        
        Note: This method is not implemented for wandb writer yet.
        
        Args:
            step: Step at which the audios occurred.
            audios: Mapping from audio key to audios.
            sample_rate: Sample rate for the audios.
        """
        raise NotImplementedError("write_audios is not implemented for WandBWriter yet.")

    def write_texts(self, step: int, texts: Mapping[str, Any]):
        """Writes text snippets for the step.

        Args:
            step: Training step associated with the text entries.
            texts: Mapping from name to text snippet or iterable of snippets.
        """
        if not texts:
            return

        wandb_payload: dict[str, Any] = {}
        for key, value in texts.items():
            if value is None:
                continue

            # Normalise value to a list of strings for logging.
            if isinstance(value, str):
                rows = [(0, value)]
            elif isinstance(value, Mapping):
                rows = [(int(k), str(v)) for k, v in value.items()]
            elif isinstance(value, (list, tuple)):
                rows = [(idx, str(v)) for idx, v in enumerate(value)]
            else:
                # Handle numpy/arraylike inputs by flattening first dimension.
                try:
                    iterable = list(value)  # type: ignore[arg-type]
                except TypeError:
                    rows = [(0, str(value))]
                else:
                    rows = [(idx, str(v)) for idx, v in enumerate(iterable)]

            if not rows:
                continue

            table = wandb.Table(columns=["index", "text"], data=rows)
            wandb_payload[key] = table

        if wandb_payload:
            wandb.log(wandb_payload, step=step)

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Optional[Mapping[str, int]] = None
    ):
        """Writes histograms for the step.
        
        Note: This method is not implemented for wandb writer yet.
        
        Args:
            step: Step at which the arrays were generated.
            arrays: Mapping from name to arrays to summarize.
            num_buckets: Number of buckets used to create the histogram.
        """
        raise NotImplementedError("write_histograms is not implemented for WandBWriter yet.")

    def write_hparams(self, hparams: Mapping[str, Any]):
        """Write hyper parameters.

        Args:
            hparams: Flat mapping from hyper parameter name to value.
        """        
        # Update wandb config with hyperparameters
        wandb.config.update(hparams)

    def flush(self):
        """Tells the MetricWriter to write out any cached values."""
        if self._initialized:
            # wandb doesn't have an explicit flush, but we can ensure sync
            pass

    def close(self):
        """Flushes and closes the MetricWriter."""
        if self._initialized:
            wandb.finish()
            self._initialized = False
