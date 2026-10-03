"""Direct JAX training and inference for the selected cross-attention recipe."""

from functools import partial

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from spectra_learning.probes.massspec import atom_flare_jax as core
from spectra_learning.probes.massspec import glacier_ion_logmixture_jax as scoring
from spectra_learning.probes.massspec.gine_global_glacier_jax import GlobalGlacierGNN


def scale_by_group_learning_rate(jepa_lr: float, head_lr: float) -> optax.GradientTransformation:
    """Apply the two learning rates after shared Adam moment estimation."""
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        rates = jax.tree.map(lambda _: head_lr, updates)
        rates["peak"]["encoder"] = jax.tree.map(lambda _: jepa_lr, updates["peak"]["encoder"])
        return jax.tree.map(lambda update, rate: -rate * update, updates, rates), state

    return optax.GradientTransformation(init_fn, update_fn)


def init_global_term(dim: int) -> dict:
    """Zero-weight global spectrum-molecule term: the initial score equals the baseline."""
    eye = jnp.eye(dim, dtype=jnp.float32)
    return {"query": eye, "key": eye, "weight": jnp.float32(0.0)}


def global_vectors(G, P, pmask, Z, fmask, gate):
    """Unit spectrum vector [..., d] and gate-weighted unit molecule vectors [..., C, d]."""
    spectrum = jnp.where(pmask[..., None], P.astype(jnp.float32), 0.0).sum(-2)
    spectrum = core.normalize(spectrum @ G["query"])
    weights = jax.nn.softmax(jnp.where(fmask, gate, -1e9), axis=-1) * fmask
    molecule = core.normalize((weights[..., None] * Z.astype(jnp.float32)).sum(-2) @ G["key"])
    return spectrum, molecule


class AuxSpectrumEncoder(nn.Module):
    """Per-spectrum external embedding (e.g. a precomputed second spectrum encoder) -> unit [d]."""

    dim: int = 256
    hidden: int = 496
    dropout: float = 0.1

    @nn.compact
    def __call__(self, features: jax.Array, *, train: bool = False) -> jax.Array:
        x = nn.LayerNorm(epsilon=1e-5, use_fast_variance=False, name="norm")(features.astype(jnp.float32))
        x = jax.nn.gelu(core.Linear(self.hidden, name="linear1")(x), approximate=False)
        x = nn.Dropout(self.dropout)(x, deterministic=not train)
        return core.normalize(core.Linear(self.dim, name="linear2")(x))


def init_aux_term(key, features: int, dim: int, dropout: float) -> dict:
    """Zero-weight auxiliary spectrum-molecule term: the initial score equals the baseline."""
    encoder = AuxSpectrumEncoder(dim, dropout=dropout).init(key, jnp.zeros((1, features)))["params"]
    return {"encoder": encoder, "key": jnp.eye(dim, dtype=jnp.float32), "weight": jnp.float32(0.0)}


def aux_molecule_vectors(A, Z, fmask, gate):
    """Gate-weighted unit molecule vectors [..., d] for the auxiliary term."""
    weights = jax.nn.softmax(jnp.where(fmask, gate, -1e9), axis=-1) * fmask
    return core.normalize((weights[..., None] * Z.astype(jnp.float32)).sum(-2) @ A["key"])


def functions(args):
    if args.train_jepa:
        from trainable_jepa_peak import restore_peak

        peak = restore_peak(args)
    else:
        peak = core.PeakEncoder(args.d, args.dropout)
        if getattr(args, "peak_readout", "mlp") == "attention":
            from spectra_learning.probes.massspec.peak_readout_jax import AttentionPeakReadout
            peak = AttentionPeakReadout(args.d, dropout=args.dropout)
    masked_readout = getattr(args, "peak_readout", "mlp") == "attention"
    # Frozen base weights are explicit JIT arguments, not multi-GB HLO constants
    # or optimizer leaves. The existing full-adaptation path keeps its signature.
    frozen_peak = None
    if getattr(peak, "adaptation", "full") != "full":
        from flax import nnx
        frozen_peak = nnx.to_pure_dict(peak.static)
    gnn = GlobalGlacierGNN(args.hidden, args.d, args.dropout)
    compatibility = core.QueryKeyProjection(args.d)
    agreement = scoring.CrossAttentionAgreement(dim=args.d)
    if args.train_jepa and args.jepa_lr != args.lr:
        optimizer = optax.chain(
            optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0),
            scale_by_group_learning_rate(args.jepa_lr, args.lr),
        )
    else:
        optimizer = optax.adam(args.lr, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)
    if getattr(args, "weight_decay", 0.):
        # Decay weight matrices only; scalar physical widths and biases keep their values.
        optimizer = optax.adamw(args.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=args.weight_decay,
                                mask=lambda params: jax.tree.map(lambda x: x.ndim >= 2, params))

    aux_encoder = AuxSpectrumEncoder(args.d, dropout=args.dropout)

    def loss_fn(params, stats, graph, tokens, pmask, fractions, ik, hn_idx, hn_valid, key, frozen):
        peak_key, gnn_key, aux_key = jax.random.split(key, 3)
        extras = {"frozen": frozen} if args.train_jepa else {}
        if masked_readout:
            extras["mask"] = pmask
        P = peak.apply({"params": params["peak"]}, tokens, train=True, rngs={"dropout": peak_key}, **extras)
        (Z, gate), updated = gnn.apply(
            {"params": params["gnn"], "batch_stats": stats}, graph, train=True,
            rngs={"dropout": gnn_key}, mutable=["batch_stats"],
        )
        P, Z = compatibility.apply({"params": params["compatibility"]}, P, Z)
        B = len(tokens)
        fmask = graph["fragment_mask"]
        coordinates = {"fraction": graph["fragment_mass_frac"], "mass": graph["molecular_mass"]}
        agree = params["agreement"]
        score = scoring.score_matrix(
            P, pmask, Z[:B], fmask[:B], gate[:B], fractions,
            jax.tree.map(lambda value: value[:B], coordinates),
            agree["log_sigma"], agree["log_beta"], chunk_size=args.score_chunk,
            cross_attention=agree["cross_attention"],
        )
        negative_scores = scoring.score_hardneg(
            P, pmask, Z[hn_idx], fmask[hn_idx] & hn_valid[..., None], gate[hn_idx],
            fractions, jax.tree.map(lambda value: value[hn_idx], coordinates),
            agree["log_sigma"], agree["log_beta"], chunk_size=args.score_chunk,
            cross_attention=agree["cross_attention"],
        )
        if "global" in agree:
            G = agree["global"]
            spectrum, molecule = global_vectors(G, P, pmask, Z, fmask, gate)
            score = score + G["weight"] * spectrum @ molecule[:B].T
            negative_scores = negative_scores + G["weight"] * jnp.einsum("bd,bkd->bk", spectrum, molecule[hn_idx])
        if "aux" in agree:
            A = agree["aux"]
            spectrum = aux_encoder.apply({"params": A["encoder"]}, fractions["aux"], train=True,
                                         rngs={"dropout": aux_key})
            spectrum = jnp.where(fractions["aux_valid"][:, None], spectrum, 0.0)
            molecule = aux_molecule_vectors(A, Z, fmask, gate)
            score = score + A["weight"] * spectrum @ molecule[:B].T
            negative_scores = negative_scores + A["weight"] * jnp.einsum("bd,bkd->bk", spectrum, molecule[hn_idx])
        return core.info_nce(score, ik, args.tau, negative_scores, hn_valid), updated["batch_stats"]

    @partial(jax.jit, donate_argnums=(0, 2) if args.train_jepa else ())
    def compiled_train_step(params, stats, opt_state, graph, tokens, pmask, fractions, ik, hn_idx, hn_valid, key, frozen):
        (loss, stats), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            params, stats, graph, tokens, pmask, fractions, ik, hn_idx, hn_valid, key, frozen,
        )
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), stats, opt_state, loss

    def train_step(params, stats, opt_state, graph, tokens, pmask, fractions, ik, hn_idx, hn_valid, key):
        return compiled_train_step(params, stats, opt_state, graph, tokens, pmask, fractions,
                                   ik, hn_idx, hn_valid, key, frozen_peak)

    @jax.jit
    def encode_molecules(params, stats, graph, score_params):
        Z, gate = gnn.apply({"params": params, "batch_stats": stats}, graph, train=False)
        Z = compatibility.apply({"params": score_params}, Z, method=compatibility.project_keys)
        return Z.astype(jnp.float16), gate

    @jax.jit
    def compiled_encode_peaks(params, tokens, score_params, frozen, mask):
        extras = {"frozen": frozen} if args.train_jepa else {}
        if masked_readout:
            extras["mask"] = mask
        P = peak.apply({"params": params}, tokens, train=False, **extras)
        P = compatibility.apply({"params": score_params}, P, method=compatibility.project_queries)
        return P.astype(jnp.float16)

    def encode_peaks(params, tokens, score_params, mask=None):
        return compiled_encode_peaks(params, tokens, score_params, frozen_peak,
                                     mask if masked_readout else None)

    @jax.jit
    def query_score(params, P, pmask, Z, fmask, fp, ff, gate):
        scores = scoring.score_query(
            P.astype(jnp.float32), pmask, Z.astype(jnp.float32), fmask, gate, fp, ff,
            params["log_sigma"], params["log_beta"], cross_attention=params["cross_attention"],
        )
        if "global" in params:
            spectrum, molecule = global_vectors(params["global"], P, pmask, Z, fmask, gate)
            scores = scores + params["global"]["weight"] * molecule @ spectrum
        if "aux" in params:
            A = params["aux"]
            spectrum = aux_encoder.apply({"params": A["encoder"]}, fp["aux"][None], train=False)[0]
            spectrum = jnp.where(fp["aux_valid"], spectrum, 0.0)
            scores = scores + A["weight"] * aux_molecule_vectors(A, Z, fmask, gate) @ spectrum
        return scores

    return peak, gnn, agreement, optimizer, train_step, encode_molecules, encode_peaks, query_score
