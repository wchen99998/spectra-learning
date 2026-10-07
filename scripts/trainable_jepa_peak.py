"""Train the original JEPA student through the unchanged FLARE peak readout."""

import hashlib
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch
from spectra_learning.probes.massspec.atom_flare_jax import PeakEncoder


def load_spectra(input_root: Path, data_root: Path, metadata_path: Path):
    """Pack raw spectra into the runner's row-indexable peak input interface.

    Channels 0:3 contain m/z, intensity, validity. Per-spectrum precursor and
    acquisition metadata occupy channels 3: only at the first physical slot.
    Provenance binds the complete prepared archive and encoder configuration,
    including intensities and acquisition fields not read by the ion scorer.
    """
    embedding = json.loads((input_root / "embedding_metadata.json").read_text())
    preparation_bytes = (data_root / "preparation_metadata.json").read_bytes()
    preparation = json.loads(preparation_bytes)
    if preparation["tsv_sha256"] != embedding["tsv_sha256"]:
        raise ValueError("Raw spectra and frozen baseline originate from different TSVs")
    metadata_bytes = metadata_path.read_bytes()
    config = json.loads(metadata_bytes)["task_contract"]["config"]
    prepared_path = data_root / "prepared_massspecgym.npz"
    with prepared_path.open("rb") as stream:
        prepared_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    with np.load(prepared_path, allow_pickle=False) as data:
        mask = data["peak_valid_mask"]
        baseline_mask = np.load(input_root / "data/token_valid_mask.npy")
        np.testing.assert_array_equal(mask, baseline_mask[:, :-1])
        fold = np.load(input_root / "data/fold.npy", allow_pickle=True).astype(str)
        np.testing.assert_array_equal(data["split"], np.array([
            {"train": 0, "val": 1, "test": 2}[value] for value in fold]))
        names = ("precursor_mz", "collision_energy", "collision_energy_present",
                 "charge", "charge_present", "polarity_id", "instrument_family_id")
        metadata = np.asarray(jax_spectrum_metadata_from_batch(
            {name: jnp.asarray(data[name]) for name in names},
            schema=config["encoder_metadata_schema"]), dtype=np.float32)
        if metadata.shape[1] != embedding["metadata_input_dim"]:
            raise ValueError("Acquisition metadata differs from the baseline encoder input")
        packed = np.zeros((*mask.shape, 4 + metadata.shape[1]), np.float32)
        packed[..., 0], packed[..., 1], packed[..., 2] = data["peak_mz"], data["peak_intensity"], mask
        packed[:, 0, 3], packed[:, 0, 4:] = data["precursor_mz"], metadata
    return packed, mask, {
        "kind": "trainable_jepa", "shape": list(packed.shape), "dtype": str(packed.dtype),
        "data_root": str(data_root), "embedding_metadata": embedding,
        "preparation_metadata": preparation, "cls_used": False,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "prepared_spectra_sha256": prepared_sha256,
        "preparation_metadata_sha256": hashlib.sha256(preparation_bytes).hexdigest(),
    }


def compact_encode(encoder, packed, num_peaks: int):
    """Encode at the pretraining slot count and scatter back to the input slots.

    The valid slots (at most ``num_peaks`` per row, m/z-ordered) move to the
    first ``num_peaks`` positions, so token positions and the trailing CLS
    position match pretraining; the physical outputs return to their original
    slots and every other slot is zero.
    """
    mask = packed[..., 2].astype(bool)
    index = jnp.argsort(~mask, axis=1, stable=True)[:, :num_peaks]
    compact = jnp.take_along_axis(packed[..., :3], index[..., None], axis=1)
    compact_mask = compact[..., 2].astype(bool)
    inputs = dict(valid_mask=compact_mask, visible_mask=compact_mask,
                  precursor_mz=packed[:, 0, 3], spectrum_metadata=packed[:, 0, 4:])
    tokens = encoder(compact[..., 0], compact[..., 1], **inputs)
    tokens = tokens[:, :-1] * compact_mask[..., None].astype(tokens.dtype)
    rows = jnp.arange(packed.shape[0])[:, None]
    return jnp.zeros((*mask.shape, tokens.shape[-1]), tokens.dtype).at[rows, index].set(tokens)


class TrainableJEPAPeak:
    """Functional NNX encoder bridge with the existing Linen readout interface."""

    def __init__(self, encoder, out_dim: int, dropout: float, model_dim: int,
                 adaptation: str = "full", num_peaks: int | None = None,
                 pair_field: bool = False, tree: bool = False, pair_dim: int | None = None):
        # Match JEPA pretraining: Fourier frequencies named b are fixed buffers,
        # even though the NNX implementation stores them as Param variables.
        self.adaptation = adaptation
        def trainable(path, value):
            if adaptation == "frozen":
                return False
            if adaptation == "lora":
                return isinstance(value, nnx.Param) and path[-1] in ("lora_a", "lora_b")
            return isinstance(value, nnx.Param) and path[-1] != "b"
        self.graphdef, parameters, self.static = nnx.split(encoder, trainable, ...)
        self.initial_parameters = nnx.to_pure_dict(parameters)
        self.pair_field = pair_field
        if pair_field and num_peaks is None:
            raise ValueError("The pair field requires the compacted dual encoder")
        if pair_field:
            from spectra_learning.probes.massspec.pair_field_jax import PairFieldPeakEncoder
            self.readout = PairFieldPeakEncoder(out_dim, dropout, tree=tree)
        else:
            self.readout = PeakEncoder(out_dim, dropout)
        self.model_dim = model_dim
        self.pair_dim = pair_dim or model_dim
        # Pretraining peak count of an encoder fed fewer slots than the
        # 127-slot input; None keeps the original uncompacted path.
        self.num_peaks = num_peaks

    def init(self, key, tokens, *, train=False):
        parameters, self.initial_parameters = self.initial_parameters, None
        shape = (1, tokens.shape[1], self.model_dim)
        if self.pair_field:
            from spectra_learning.probes.massspec.pair_field_jax import pair_edge_mask
            n = tokens.shape[1]
            readout = self.readout.init(key, jnp.zeros(shape), jnp.zeros((1, n + 1, n, self.pair_dim)),
                                        pair_edge_mask(jnp.ones((1, n), bool)), train=train)
        else:
            readout = self.readout.init(key, jnp.zeros(shape), train=train)
        return {"params": {"encoder": parameters, "readout": readout["params"]}}

    def encode(self, parameters, packed, *, frozen=None):
        encoder = nnx.merge(self.graphdef, nnx.State(parameters),
                            self.static if frozen is None else nnx.State(frozen))
        mask = packed[..., 2].astype(bool)
        if self.num_peaks is None:
            tokens = encoder(packed[..., 0], packed[..., 1], valid_mask=mask, visible_mask=mask,
                             precursor_mz=packed[:, 0, 3], spectrum_metadata=packed[:, 0, 4:])
            tokens = tokens[:, :-1]
        elif self.pair_field:
            # Inputs already use the pretraining slots: [B, N] peaks, CLS last.
            from spectra_learning.probes.massspec.pair_field_jax import pair_edge_mask
            if packed.shape[1] != self.num_peaks:
                raise ValueError("The pair field expects inputs compacted to the pretraining slots")
            tokens, pair = encoder.forward_with_pair(
                packed[..., 0], packed[..., 1], valid_mask=mask, visible_mask=mask,
                precursor_mz=packed[:, 0, 3], spectrum_metadata=packed[:, 0, 4:])
            tokens = tokens[:, :-1] * mask[..., None].astype(tokens.dtype)
            cast = lambda value: value.astype(jnp.float16).astype(jnp.float32)
            return cast(tokens), cast(pair[:, :, :-1]), pair_edge_mask(mask)
        else:
            tokens = compact_encode(encoder, packed, self.num_peaks)
        # Retain the baseline's physical-only float16 storage boundary. Casts
        # remain differentiable; no stop_gradient or cached embeddings are used.
        return tokens.astype(jnp.float16).astype(jnp.float32)

    def apply(self, variables, tokens, *, train=False, rngs=None, frozen=None):
        params = variables["params"]
        encoded = self.encode(params["encoder"], tokens, frozen=frozen)
        if not self.pair_field:
            encoded = (encoded,)
        return self.readout.apply({"params": params["readout"]}, *encoded, train=train, rngs=rngs)


def restore_peak(args):
    from ml_collections import config_dict
    import spectra_learning.models.encoder_jax as encoder_module
    from spectra_learning.models.model_jax import build_peak_set_encoder_jax
    from spectra_learning.models.settings import PeakSetJEPASettings
    from spectra_learning.training.checkpointing_jax import restore_jax_encoder_state

    metadata = json.loads(args.jepa_metadata.read_text())
    config = config_dict.ConfigDict(metadata["task_contract"]["config"])
    config.activation_checkpoint_mode = "full"
    config.activation_checkpoint_every_n_layers = 1
    encoder_module.MASSIVE_V2_CONDITION_DIM = 28
    settings = PeakSetJEPASettings.from_config(config)
    # The joint single+pair ``dual`` student keeps the single-stream token
    # interface; it runs at its pretraining peak count plus CLS.
    dual = config.training_mode == "dual"
    num_peaks = settings.num_peaks
    if dual:
        # Run at the peak count the inputs were compacted to: the stage the
        # checkpoint was pretraining in (e.g. 31 peaks before the 63 stage).
        prepared = json.loads((args.input_root / "embedding_metadata.json").read_text())
        num_peaks = int(prepared.get("physical_peak_subset", {}).get("num_peaks", num_peaks))
    encoder = nnx.eval_shape(lambda: build_peak_set_encoder_jax(
        settings, compute_dtype=jnp.bfloat16, pairmixer_block_type=settings.pairmixer_block_type,
        pairmixer_fast_encoder_max_visible_tokens=num_peaks + 1 if dual else None,
        rngs=nnx.Rngs(int(config.seed))))
    if encoder.use_pair_path != dual or not encoder.use_cls_token:
        raise ValueError("Expected the single-stream or dual JEPA student with CLS")
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    template = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
                            nnx.as_pure(nnx.state(encoder)))
    nnx.update(encoder, restore_jax_encoder_state(args.jepa_checkpoint, template))
    adaptation = getattr(args, "encoder_adaptation", "full")
    if adaptation == "lora":
        from retrieval_lora import install_attention_lora
        install_attention_lora(encoder, rank=args.lora_rank, alpha=args.lora_alpha, seed=args.seed)
    pair_field = getattr(args, "pair_field", False)
    if pair_field and not dual:
        raise ValueError("The pair field needs a dual student")
    return TrainableJEPAPeak(encoder, args.d, args.dropout, settings.model_dim, adaptation,
                             num_peaks=num_peaks if dual else None,
                             pair_field=pair_field, tree=getattr(args, "tree", False),
                             pair_dim=settings.pairmixer_pair_dim or settings.model_dim)


def validate_initial_tokens(peak, params, packed, mask, input_root, rows):
    rows = np.asarray(rows[:8])
    expected = np.asarray(np.load(input_root / "covariance_embeddings.float16.npy", mmap_mode="r")[rows], np.float32)
    # Match the cache export batch shape: bf16 GEMM kernels can differ for B=8
    # versus B=128 even when the individual spectra and weights are identical.
    batch_size = json.loads((input_root / "embedding_metadata.json").read_text())["batch_size"]
    batch = np.zeros((batch_size, *packed.shape[1:]), np.float32)
    batch[:len(rows)] = packed[rows]
    actual = np.asarray(jax.jit(peak.encode)(params["encoder"], batch))[:len(rows)]
    delta = actual[mask[rows]] - expected[mask[rows]]
    relative_l2 = float(np.linalg.norm(delta) / np.linalg.norm(expected[mask[rows]]))
    if not np.isfinite(actual).all() or relative_l2 > 0.005:
        raise ValueError(f"Restored JEPA does not reproduce baseline tokens: relative L2={relative_l2}")
    return {"rows": rows.tolist(), "relative_l2": relative_l2,
            "max_absolute_error": float(np.abs(delta).max()), "tolerance_relative_l2": 0.005}
