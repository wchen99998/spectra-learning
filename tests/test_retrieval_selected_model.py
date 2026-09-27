"""Selected recipe parity against its immutable, validation-selected runtime."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results/global_glacier_crossattn_joint_runtime_20260924"
sys.path.insert(0, str(ROOT / "scripts"))

import train_global_glacier_jax as training
from initialize_glacier_head import restore_head
from atom_flare_data import batch_graphs, build_graph_store
from spectra_learning.probes import massspec
from spectra_learning.probes.massspec import atom_flare_jax as core
from spectra_learning.probes.massspec import glacier_ion_logmixture_jax as scoring
from spectra_learning.probes.massspec.gine_global_glacier_jax import GlobalGlacierGNN


def _load(relative, name):
    manifest = (FROZEN / "source_manifest.json").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == "59e6c1c7dd7ccad6c45fcd9e0df77fbb6d14c2f0470aeae76ca6c4fa87f78984"
    assert hashlib.sha256((FROZEN / relative).read_bytes()).hexdigest() == json.loads(manifest)[relative]
    spec = importlib.util.spec_from_file_location(name, FROZEN / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference():
    if not FROZEN.exists():
        pytest.skip("Frozen selected runtime is an external result artifact; independent selected-recipe checks still run")
    prefix = "spectra_learning/probes/massspec/"
    # Resolve every scoring dependency from the preserved archive, including
    # its unused experimental import, without requiring retired working files.
    with pytest.MonkeyPatch.context() as patch:
        atom = _load(prefix + "atom_flare_jax.py", "_selected_reference_atom")
        patch.setitem(sys.modules, "spectra_learning.probes.massspec.atom_flare_jax", atom)
        patch.setattr(massspec, "atom_flare_jax", atom)
        context = _load(prefix + "atom_global_context_jax.py", "_selected_reference_context")
        patch.setitem(sys.modules, "spectra_learning.probes.massspec.atom_global_context_jax", context)
        semi = _load(prefix + "glacier_semirelaxed_jax.py", "_selected_reference_semi")
        patch.setitem(sys.modules, "spectra_learning.probes.massspec.glacier_semirelaxed_jax", semi)
        fragments = _load(prefix + "glacier_fragments_jax.py", "_selected_reference_fragments")
        patch.setitem(sys.modules, "spectra_learning.probes.massspec.glacier_fragments_jax", fragments)
        patch.setattr(massspec, "glacier_fragments_jax", fragments)
        score = _load(prefix + "glacier_ion_logmixture_jax.py", "_selected_reference_scoring")
        gnn = _load(prefix + "gine_global_glacier_jax.py", "_selected_reference_gnn")
        train = _load("scripts/train_global_glacier_jax.py", "_selected_reference_training")
        train.GlobalGlacierGNN = gnn.GlobalGlacierGNN
    return SimpleNamespace(score=score, gnn=gnn.GlobalGlacierGNN, train=train)


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    store = build_graph_store(["CCCO", "CCNC", "CCCC"], tmp_path_factory.mktemp("selected_graph") / "graphs")
    graph = batch_graphs(store, [0, 1, 2], graph_capacity=4, node_capacity=16,
                         edge_capacity=24, atom_cap=4)
    # Two nontrivial fragments plus padding, including an intact anchor with
    # empty boundary and complete all-atom molecular context.
    member = np.zeros((4, 3, 4), np.float32)
    member[:3, 0] = .25
    member[:3, 1, :2] = .5
    boundary = np.zeros_like(member)
    boundary[:3, 1, 1] = 1
    mask = np.zeros((4, 3), bool)
    mask[:3, :2] = True
    graph.update(member_mean=member, boundary_mean=boundary, fragment_mask=mask,
                 fragment_n_cuts=np.tile([0, 1, 0], (4, 1)).astype(np.int32),
                 fragment_kind=np.tile([0, 1, 0], (4, 1)).astype(np.int32),
                 fragment_mass_frac=np.tile([1, .5, 0], (4, 1)).astype(np.float32),
                 fragment_log_n_atoms=np.tile(np.log([4, 2, 1]), (4, 1)).astype(np.float32),
                 fragment_intensity=np.tile([1, .7, 0], (4, 1)).astype(np.float32),
                 molecular_mass=np.array([74, 73, 58, 0], np.float32))
    return jax.tree.map(jnp.asarray, graph)


def _args(**updates):
    return SimpleNamespace(**(dict(d=8, hidden=16, dropout=.1, train_jepa=False,
        jepa_lr=3e-6, lr=5e-4, tau=.05, score_chunk=1,
        molecule_encoder="gine_global_glacier", similarity="qk", peak_input="raw",
        fragment_readout="ion_logmixture", fragment_cross_attention=True) | updates))


def _equal(actual, expected, *, exact=True):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if exact:
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


def _initialize(functions, args, graph, input_dim=12):
    peak, gnn, agreement = functions[:3]
    tokens = jax.random.normal(jax.random.key(5), (2, 3, input_dim))
    gv = gnn.init(jax.random.key(6), graph)
    peak_params = peak.init(jax.random.key(7), tokens)["params"]
    qk = core.QueryKeyProjection(args.d)
    qk_params = qk.init(jax.random.key(8), jnp.zeros((2, 3, args.d)), jnp.zeros((4, 3, args.d)))["params"]
    agree = agreement.init(jax.random.key(9), jnp.zeros((2, 3)), graph["fragment_mass_frac"],
                           graph["fragment_mask"])["params"]
    # Exercise a trained adapter, not only its zero-output initial identity.
    agree["cross_attention"]["O"] = jax.random.normal(jax.random.key(10), (32, 32)) * .03
    return {"peak": peak_params, "gnn": gv["params"], "compatibility": qk_params,
            "agreement": agree}, gv["batch_stats"], tokens


def _frozen_functions(reference, args):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(massspec, "glacier_ion_logmixture_jax", reference.score)
        return reference.train.functions(args)


def test_selected_initialization_bn_dropout_and_parameter_tree_match_frozen(reference, graph):
    before, after = reference.gnn(16, 8, .1), GlobalGlacierGNN(16, 8, .1)
    key = jax.random.key(1)
    old, new = before.init(key, graph), after.init(key, graph)
    _equal(new, old)
    _equal(after.apply(new, graph), before.apply(old, graph))
    options = dict(train=True, rngs={"dropout": jax.random.key(2)}, mutable=["batch_stats"])
    _equal(after.apply(new, graph, **options), before.apply(old, graph, **options))


def test_production_head_parameter_count_and_agreement_initialization(reference, graph):
    args = _args(d=256, hidden=496)
    params, _, _ = _initialize(training.functions(args), args, graph, input_dim=2048)
    assert sum(x.size for x in jax.tree.leaves(params)) == 4_121_707
    current = scoring.CrossAttentionAgreement(dim=256)
    archived = reference.score.CrossAttentionAgreement(dim=256)
    values = (jnp.zeros((2, 3)), graph["fragment_mass_frac"], graph["fragment_mask"])
    _equal(current.init(jax.random.key(1), *values), archived.init(jax.random.key(1), *values))


@pytest.mark.parametrize("eligible", [True, False])
def test_selected_scores_gradients_and_poisoned_padding_match_frozen(reference, eligible):
    p = jnp.array([[.7, .2], [.1, .9], [jnp.nan, jnp.nan]])
    z = jnp.array([[[.8, .2], [.1, .6], [jnp.nan, jnp.nan]],
                   [[.3, .5], [.9, -.2], [jnp.nan, jnp.nan]], [[jnp.nan, jnp.nan]] * 3])
    pm = jnp.array([True, True, False])
    fm = jnp.array([[True, True, False], [True, True, False], [False] * 3])
    gate = jnp.array([[.2, -.1, -jnp.inf], [-.2, .4, -jnp.inf], [-jnp.inf] * 3])
    q = dict(fraction=jnp.array([.402, .645, jnp.nan]),
             mz=jnp.array([80.003 + scoring.PROTON_MASS, 130.006 + scoring.PROTON_MASS, jnp.nan]),
             eligible=jnp.asarray(eligible), adduct_mass=jnp.float32(scoring.PROTON_MASS))
    c = dict(fraction=jnp.array([[.4, .65, jnp.nan], [.42, .64, jnp.nan], [jnp.nan] * 3]),
             mass=jnp.array([200., 200., jnp.nan]))
    agreement = scoring.CrossAttentionAgreement(dim=2).init(jax.random.key(9), q["fraction"][None], c["fraction"], fm)["params"]
    agreement["cross_attention"]["O"] = .03 * jax.random.normal(jax.random.key(10), (32, 32))
    def evaluate(module):
        def loss(a, peaks, fragments, logits):
            scores = module.score_query(peaks, pm, fragments, fm, logits, q, c,
                a["log_sigma"], a["log_beta"], cross_attention=a["cross_attention"])
            return scores.sum(), scores
        return jax.jit(jax.value_and_grad(loss, (0, 1, 2, 3), has_aux=True))(agreement, p, z, gate)
    old, new = evaluate(reference.score), evaluate(scoring)
    _equal(new, old, exact=False)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(new))
    np.testing.assert_array_equal(new[1][1][~pm], 0)
    np.testing.assert_array_equal(new[1][2][~fm], 0)


def test_selected_adam_step_encoders_and_loss_match_frozen(reference, graph):
    args = _args()
    old, new = _frozen_functions(reference, args), training.functions(args)
    params, stats, tokens = _initialize(new, args, graph)
    pmask = jnp.array([[True, True, False], [True, True, True]])
    fractions = jnp.array([[.3, .5, 0], [.25, .6, 1]])
    query = dict(fraction=fractions, mz=fractions * graph["molecular_mass"][:2, None] + scoring.PROTON_MASS,
                 eligible=jnp.array([True, False]), adduct_mass=jnp.full(2, scoring.PROTON_MASS))
    args_tail = (graph, tokens, pmask, query, jnp.array([1, 2]),
                 jnp.array([[2, 3], [2, 3]]), jnp.array([[True, False], [True, False]]), jax.random.key(11))
    before = old[4](params, stats, old[3].init(params), *args_tail)
    after = new[4](params, stats, new[3].init(params), *args_tail)
    _equal(after, before, exact=False)
    assert int(after[2][0].count) == 1
    for index, values in ((5, (params["gnn"], stats, graph, params["compatibility"])),
                          (6, (params["peak"], tokens, params["compatibility"]))):
        _equal(new[index](*values), old[index](*values))


def test_online_encoder_learning_rate_keeps_the_same_adam_moments(reference):
    params = {"peak": {"encoder": {"w": jnp.array([1., 2.])}, "readout": {"w": jnp.array([3., 4.])}},
              "gnn": {"w": jnp.array([5., 6.])}}
    gradients = jax.tree.map(lambda x: x * .3, params)
    old = optax.chain(optax.scale_by_adam(), reference.train.scale_by_group_learning_rate(3e-6, 1e-5))
    new = optax.chain(optax.scale_by_adam(), training.scale_by_group_learning_rate(3e-6, 1e-5))
    _equal(new.update(gradients, new.init(params), params), old.update(gradients, old.init(params), params))


@pytest.fixture
def checkpoint(tmp_path):
    args = _args(train_jepa=True)
    source = {"params": {"peak": {"w": np.ones((3, 4), np.float32)},
                         "gnn": {"w": np.ones((4, 4), np.float32)},
                         "compatibility": {"query": np.eye(4, dtype=np.float32)},
                         "agreement": {"cross_attention": {"O": np.eye(32, dtype=np.float32)}}},
              "stats": {"mean": np.arange(4, dtype=np.float32)}}
    inputs = dict(kind="raw", embedding_metadata={"checkpoint_step": 1190000})
    graphs, fragments = {"smiles_sha256": "graphs"}, {"smiles_sha256": "graphs", "f_max": 128}
    manifest = dict(args=vars(_args()), peak_input=inputs, graphs=graphs, fragments=fragments)
    path = tmp_path / "best.msgpack"
    path.write_bytes(serialization.to_bytes(source))
    path.with_name("manifest.json").write_text(json.dumps(manifest))
    template = jax.tree.map(np.zeros_like, source)
    encoder = {"w": np.full((7, 3), 5, np.float32)}
    template["params"]["peak"] = {"encoder": encoder, "readout": template["params"]["peak"]}
    return path, source, template, args, inputs, graphs, fragments


def test_restore_preserves_all_head_weights_bn_and_original_encoder(checkpoint):
    path, source, template, args, inputs, graphs, fragments = checkpoint
    params, stats, provenance = restore_head(path, template["params"], template["stats"], args, inputs, graphs, fragments)
    assert params["peak"]["encoder"] is template["params"]["peak"]["encoder"]
    _equal({"params": {**params, "peak": params["peak"]["readout"]}, "stats": stats}, source)
    assert provenance["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["optimizer"] == "fresh"


@pytest.mark.parametrize("problem", ["shape", "dtype", "nonfinite", "encoder", "fragments", "architecture"])
def test_restore_rejects_changed_checkpoint_or_input_identity(checkpoint, problem):
    path, source, template, args, inputs, graphs, fragments = checkpoint
    if problem == "shape":
        source["params"]["peak"]["w"] = np.zeros((4, 4), np.float32)
    elif problem == "dtype":
        source["params"]["peak"]["w"] = source["params"]["peak"]["w"].astype(np.float16)
    elif problem == "nonfinite":
        source["stats"]["mean"][0] = np.nan
    elif problem == "encoder":
        inputs["embedding_metadata"]["checkpoint_step"] = 407000
    elif problem == "fragments":
        fragments["f_max"] = 64
    else:
        args.dropout = .2
    path.write_bytes(serialization.to_bytes(source))
    with pytest.raises(ValueError, match="shape or dtype|nonfinite|provenance|inputs differ|architecture"):
        restore_head(path, template["params"], template["stats"], args, inputs, graphs, fragments)


@pytest.mark.parametrize("eligible", [True, False])
def test_selected_score_and_all_input_gradients_match_independent_scalar_equations(eligible):
    """Enumerate hydrogen shifts and candidate attention without vector wrappers."""
    from jax.scipy.special import logsumexp

    p = jnp.array([[.7, .2], [.1, .9]])
    z = jnp.array([[.8, .2], [.1, .6]])
    gate = jnp.array([.2, -.1])
    fp, ff = jnp.array([.402, .645]), jnp.array([.4, .65])
    mz = jnp.array([80.003 + scoring.PROTON_MASS, 130.006 + scoring.PROTON_MASS])
    query = dict(fraction=fp, mz=mz, eligible=jnp.asarray(eligible), adduct_mass=jnp.float32(scoring.PROTON_MASS))
    candidates = dict(fraction=ff[None], mass=jnp.array([200.]))
    model = scoring.CrossAttentionAgreement(dim=2)
    a = model.init(jax.random.key(31), fp[None], ff[None], jnp.ones((1, 2), bool))["params"]
    a["cross_attention"]["O"] = .03 * jax.random.normal(jax.random.key(32), (32, 32))

    def scalar(params, peaks, fragments, gates):
        weights = params["cross_attention"]
        def kernel(adduct):
            delta = mz[:, None] - (200. * ff[None] + adduct)
            errors = delta[..., None] - jnp.arange(-2, 3) * 1.00782503207
            return jnp.max(jnp.exp(-jnp.square(errors / .01)), axis=-1)
        physical = jnp.maximum(kernel(query["adduct_mass"]), .5 * kernel(scoring.PROTON_MASS))
        fractional = jnp.exp(-jnp.square((fp[:, None] - ff[None]) / jnp.exp(params["log_sigma"])))
        similarity = peaks @ fragments.T + 1.5 * jnp.exp(params["log_beta"]) * jnp.where(eligible, physical, fractional)
        prior = gates - logsumexp(gates)
        def norm(vector):
            return vector / jnp.sqrt(jnp.sum(vector ** 2) / len(vector) + 1e-6)
        values = jnp.stack([norm(fragment @ weights["V"]) for fragment in fragments])
        rows = []
        for peak, scores in zip(peaks, similarity, strict=True):
            u = norm(peak @ weights["U"])
            logits = scores + prior
            assignment = jnp.exp(logits - logsumexp(logits))
            context = jnp.sum(assignment[:, None] * values, axis=0)
            hidden = jnp.tanh(u @ weights["A_q"] + context @ weights["A_c"]
                               + (u * context) @ weights["A_x"] + weights["b"])
            correction = hidden @ weights["O"]
            rows.append(scores + jnp.stack([jnp.dot(correction, v) / jnp.sqrt(32.) for v in values]))
        similarity = jnp.stack(rows)
        forward = jnp.mean(.1 * logsumexp(similarity / .1 + prior[None], axis=-1))
        reverse = jnp.sum(jnp.exp(prior) * jnp.max(similarity, axis=0))
        return .65 * forward + .35 * reverse

    def actual(params, peaks, fragments, gates):
        return scoring.score_query(peaks, jnp.ones(2, bool), fragments[None], jnp.ones((1, 2), bool),
            gates[None], query, candidates, params["log_sigma"], params["log_beta"],
            cross_attention=params["cross_attention"])[0]
    expected = jax.value_and_grad(scalar, (0, 1, 2, 3))(a, p, z, gate)
    observed = jax.value_and_grad(actual, (0, 1, 2, 3))(a, p, z, gate)
    _equal(observed, expected, exact=False)
