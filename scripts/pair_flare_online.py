"""Generate frozen JAX pair tokens for one FLARE batch at a time."""

from functools import partial
import json
from pathlib import Path
import time

import numpy as np


KEYS = ["peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz", "collision_energy",
        "collision_energy_present", "charge", "charge_present", "polarity_id", "instrument_family_id"]


class OnlinePairTokens:
    def __init__(self, root: Path, prepared: Path, checkpoint_metadata: Path,
                 checkpoint_path: Path | None = None):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from ml_collections import config_dict

        from spectra_learning.models.encoder_jax import PairOnlyEncoder
        from spectra_learning.models.settings import PeakSetJEPASettings
        from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch
        from spectra_learning.training.checkpointing_jax import restore_jax_encoder_state

        metadata = json.loads((root / "embedding_metadata.json").read_text())
        checkpoint = metadata["checkpoint"]
        if checkpoint_path is not None:
            copied = json.loads((checkpoint_path / "checkpoint_copy.json").read_text())
            if not copied["all_checks_passed"] or copied["snapshot_uri"] != checkpoint:
                raise ValueError("Local encoder checkpoint must be a verified copy of the pinned snapshot")
            checkpoint = checkpoint_path
        self.shape = (metadata["samples"], metadata["maximum_pair_tokens"], metadata["pair_dim"])
        self.dtype = np.dtype(np.float16)
        self.offsets = np.load(root / "pair_offsets.npy")
        self.source_rows = np.load(root / "source_rows.npy")
        with np.load(prepared) as archive:
            self.data = {key: archive[key] for key in KEYS}
        self.counts = self.data["peak_valid_mask"].sum(-1).astype(np.int64)
        assert np.array_equal(np.diff(self.offsets), self.counts[self.source_rows] ** 2)
        config = config_dict.ConfigDict(json.loads(checkpoint_metadata.read_text())["task_contract"]["config"])
        assert config.num_peaks == 47 and config.training_mode == "pair_distogram"
        assert self.shape[1:] == (47 ** 2, config.model_dim)
        config.activation_checkpoint_mode = "none"
        self.encoder = PairOnlyEncoder(PeakSetJEPASettings.from_config(config),
                                      compute_dtype=jnp.bfloat16, rngs=nnx.Rngs(int(config.seed)))
        state = nnx.as_pure(nnx.state(self.encoder))
        nnx.update(self.encoder, restore_jax_encoder_state(checkpoint, state))
        del state

        @partial(nnx.jit, static_argnames=("capacity",))
        def encode(encoder, batch, *, capacity):
            spectrum_metadata = jax_spectrum_metadata_from_batch(batch, schema=config.encoder_metadata_schema)
            pair, _, valid = encoder(
                batch["peak_mz"], batch["peak_intensity"], valid_mask=batch["peak_valid_mask"],
                visible_mask=batch["peak_valid_mask"], max_visible_tokens=capacity,
                precursor_mz=batch["precursor_mz"], spectrum_metadata=spectrum_metadata,
            )
            return (pair[:, :-1, :-1].astype(jnp.float16), pair[:, -1, -1].astype(jnp.float16),
                    valid[:, :-1].sum(-1))

        self.encode = encode
        self.cached_rows = None
        self.cached_batch = None
        self.requests = self.generated_spectra = self.cache_hits = 0
        self.encoding_seconds = 0.0
        self.last_report = time.monotonic()
        print(f"ONLINE ENCODER restored {checkpoint}; pinned snapshot {metadata['checkpoint']}; no pair-value file opened", flush=True)

    def batch_values(self, rows, cls_tokens=None):
        import jax
        import jax.numpy as jnp
        import torch

        self.requests += 1
        if self.cached_rows is not None and np.array_equal(rows, self.cached_rows):
            self.cache_hits += 1
            return self.cached_batch
        started = time.monotonic()
        source = self.source_rows[rows]
        counts = self.counts[source]
        offsets = np.concatenate([[0], np.cumsum(counts ** 2)])
        values = torch.empty((int(offsets[-1]), self.shape[-1]), dtype=torch.float16, device="cuda")
        cls = torch.empty((len(rows), self.shape[-1]), dtype=torch.float16, device="cuda")
        capacities = np.array([16, 32, 48])
        buckets = capacities[np.searchsorted(capacities, counts + 1)]
        for capacity in capacities:
            positions = np.flatnonzero(buckets == capacity)
            maximum = 16 if capacity == 48 else 32
            for start in range(0, len(positions), maximum):
                take_positions = positions[start:start + maximum]
                take = len(take_positions)
                size = 1 << (take - 1).bit_length()
                inputs = {key: jnp.asarray(np.pad(self.data[key][source[take_positions]],
                    ((0, size - take),) + ((0, 0),) * (self.data[key].ndim - 1))) for key in KEYS}
                pairs, global_tokens, retained = self.encode(self.encoder, inputs, capacity=int(capacity))
                assert np.array_equal(jax.device_get(retained[:take]), counts[take_positions]), "Native peak loss"
                pair_tensor = torch.from_dlpack(pairs)
                cls_tensor = torch.from_dlpack(global_tokens)
                for local, position in enumerate(take_positions):
                    k = int(counts[position])
                    values[offsets[position]:offsets[position + 1]].copy_(pair_tensor[local, :k, :k].reshape(-1, self.shape[-1]))
                    cls[position].copy_(cls_tensor[local])
                # Finish consuming shared buffers before JAX may release them.
                torch.cuda.current_stream().synchronize()
        self.cached_rows, self.cached_batch = np.array(rows, copy=True), (values, cls)
        self.generated_spectra += len(rows)
        self.encoding_seconds += time.monotonic() - started
        if time.monotonic() - self.last_report >= 60:
            print("ONLINE ENCODING " + json.dumps(self.statistics()), flush=True)
            self.last_report = time.monotonic()
        return self.cached_batch

    def statistics(self) -> dict:
        return {"requests": self.requests, "generated_spectra": self.generated_spectra,
                "same_batch_cache_hits": self.cache_hits, "encoding_seconds": self.encoding_seconds,
                "retained_batches": int(self.cached_batch is not None),
                "retained_bytes": 0 if self.cached_batch is None else sum(value.nbytes for value in self.cached_batch)}
