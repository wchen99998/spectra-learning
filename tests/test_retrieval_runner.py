"""Selected-recipe inputs, sampling, ranks and checkpoint provenance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from flax import serialization
import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_retrieval_jax as runner


@pytest.fixture
def input_paths(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({name: str(tmp_path / name) for name in runner.INPUT_NAMES}))
    return path


def args_for(input_paths, stage, *extra):
    return runner.recipe_args(["--inputs", str(input_paths), "--out", str(input_paths.parent / "run"),
                               "--stage", stage, *extra])


@pytest.mark.parametrize("stage", ["train", "head", "joint", "eval"])
def test_selected_stage_settings(input_paths, stage):
    extra = ["--head-checkpoint", "source/best.msgpack"] if stage == "joint" else []
    args = args_for(input_paths, stage, *extra)
    assert (args.hidden, args.d, args.gine_layers, args.cross_attention_dim) == (496, 256, 3, 32)
    assert (args.batch, args.negatives, args.negative_bank_size, args.atom_cap) == (128, 8, 0, 160)
    assert (args.seed, args.dropout, args.tau, args.reverse_loss_weight) == (0, .1, .05, 1.)
    assert args.lr == (5e-4 if stage in ("train", "head") else 1e-5)
    assert args.epochs == (60 if stage in ("train", "head") else 8)
    assert args.train_jepa == (stage != "head")
    assert args.jepa_lr == 3e-6
    assert args.fragment_readout == "ion_logmixture" and args.fragment_cross_attention


@pytest.mark.parametrize("stage,extra,message", [
    ("joint", [], "requires --head-checkpoint"),
    ("train", ["--head-checkpoint", "source.msgpack"], "only used by joint"),
    ("head", ["--head-checkpoint", "source.msgpack"], "only used by joint"),
    ("eval", ["--resume"], "loads best.msgpack"),
    ("head", ["--session-hours", "nan"], "finite and nonnegative"),
])
def test_stage_lifecycle_arguments_reject_ambiguity(input_paths, stage, extra, message):
    with pytest.raises(ValueError, match=message):
        args_for(input_paths, stage, *extra)


def test_resume_joint_does_not_require_source_head(input_paths):
    args = args_for(input_paths, "joint", "--resume")
    assert args.head_checkpoint is None and args.resume


def test_no_architecture_or_optimizer_tuning_cli(input_paths):
    with pytest.raises(SystemExit) as exc:
        args_for(input_paths, "head", "--lr", "0.1", "--molecule-encoder", "pairmixer")
    assert exc.value.code == 2


@pytest.fixture
def raw_tokens(tmp_path):
    root = tmp_path / "raw"
    (root / "data").mkdir(parents=True)
    tokens = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    mask = np.array([[True, False, True, True], [True, True, True, True]])
    metadata = {"embedding": "valid_peak_tokens", "embedding_shape": list(tokens.shape),
                "embedding_dim": 4, "embedding_dtype": "float16", "samples": 2}
    np.save(root / "covariance_embeddings.float16.npy", tokens)
    np.save(root / "data/token_valid_mask.npy", mask)
    (root / "embedding_metadata.json").write_text(json.dumps(metadata))
    return root, tokens, mask, metadata


def test_raw_input_preserves_last_physical_peak_and_drops_only_cls_mask(raw_tokens):
    root, expected, mask, metadata = raw_tokens
    tokens, physical_mask, provenance = runner.load_raw_tokens(root)
    np.testing.assert_array_equal(tokens, expected)
    np.testing.assert_array_equal(tokens[:, -1], expected[:, -1])
    np.testing.assert_array_equal(physical_mask, mask[:, :-1])
    assert tokens.dtype == np.float16
    assert provenance["cls_used"] is False
    assert provenance["embedding_metadata"] == metadata


@pytest.mark.parametrize("changed", ["missing_cls", "invalid_cls", "empty_spectrum", "mask_dtype",
                                     "token_dtype", "embedding", "embedding_shape", "embedding_dim",
                                     "embedding_dtype", "samples"])
def test_raw_input_rejects_misaligned_cache(raw_tokens, changed):
    root, tokens, mask, metadata = raw_tokens
    if changed == "missing_cls":
        mask = mask[:, :-1]
    elif changed == "invalid_cls":
        mask[0, -1] = False
    elif changed == "empty_spectrum":
        mask[0, :-1] = False
    elif changed == "mask_dtype":
        mask = mask.astype(np.int32)
    elif changed == "token_dtype":
        tokens = tokens.astype(np.float32)
    elif changed == "embedding_shape":
        metadata[changed] = [2, 4, 4]
    elif changed in ("embedding_dim", "samples"):
        metadata[changed] += 1
    else:
        metadata[changed] = "invalid"
    np.save(root / "covariance_embeddings.float16.npy", tokens)
    np.save(root / "data/token_valid_mask.npy", mask)
    (root / "embedding_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="physical raw tokens"):
        runner.load_raw_tokens(root)


def test_full_pool_sampling_and_shared_decoy_indices_replay_rng(tmp_path):
    bank = [np.arange(11, 36, dtype=np.int32), np.array([11, 12], np.int32),
            np.array([], np.int32)]
    rows, truths = np.arange(3), np.array([10, 10, 40], np.int32)
    rng = np.random.default_rng(0)
    # These draws reproduce the historical preselection consumption before training.
    rng.choice(np.arange(19_429), 5000, replace=False)
    rng.choice(np.arange(17_556), 6000, replace=False)
    saved_rng = json.loads(json.dumps(rng.bit_generator.state))
    graph_ids, indices, valid = runner.sample_negatives(bank, rows, truths, rng, 8)
    replay = np.random.default_rng()
    replay.bit_generator.state = saved_rng
    repeated = runner.sample_negatives(bank, rows, truths, replay, 8)
    for actual, expected in zip(repeated, (graph_ids, indices, valid), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert rng.bit_generator.state == replay.bit_generator.state
    assert graph_ids[:3] == [10, 10, 40]  # Paired duplicates remain distinct instances.
    assert len(set(graph_ids[3:])) == len(graph_ids[3:])
    sampled = np.asarray(graph_ids)[indices[0, valid[0]]]
    assert len(set(sampled)) == 8 and np.any(sampled > 26)  # Beyond first 16 entries.
    np.testing.assert_array_equal(np.asarray(graph_ids)[indices[1, valid[1]]], [11, 12])
    assert valid.sum(axis=1).tolist() == [8, 2, 0]


def test_native_ranks_keep_exact_ties_and_sub_micro_score_margins():
    truth_score = np.float32(.5)
    larger = np.nextafter(truth_score, np.float32(1.))
    scores = np.array([truth_score, larger, truth_score, .1], np.float32)
    original = scores.copy()
    assert runner.ranks_with_ties(scores, 2) == (2, 3, 1)
    assert runner.ranks_with_ties(np.array([.4, .4]), 1) == (1, 2, 0)
    np.testing.assert_array_equal(scores, original)
    metrics = runner.metrics(np.array([1, 1, 3, 7]))
    assert metrics["hit1"] == .5 and metrics["median_rank"] == 2.
    assert metrics["mean_rank"] == 3.
    assert metrics["mrr"] == pytest.approx((2 + 1 / 3 + 1 / 7) / 4)


@pytest.fixture
def provenance(input_paths):
    args = args_for(input_paths, "eval")
    args.out.mkdir()
    args.prepared.mkdir()
    (args.input_root / "msg15").mkdir(parents=True)
    np.savez(args.prepared / "rows.npz", train=[0], val=[1], test=[2])
    (args.input_root / "msg15/MassSpecGym1.5_retrieval_candidates_formula.json").write_text('{"C":["C"]}')
    store = SimpleNamespace(metadata={"graph": "identity"},
                            fragments=SimpleNamespace(metadata={"fragment": "identity"}))
    peak = {"embedding_metadata": {"checkpoint": "original1190000"},
            "metadata_sha256": "config", "prepared_spectra_sha256": "spectra",
            "preparation_metadata_sha256": "preparation"}
    ions = {"query_array_sha256": {"mz": "physical-masses"}}
    identity = runner.input_identity(args, store, peak, ions)
    return args, store, peak, ions, identity


def test_identity_binds_prepared_spectral_inputs_and_candidate_order(provenance):
    args, store, peak, ions, identity = provenance
    for name in ("metadata_sha256", "prepared_spectra_sha256", "preparation_metadata_sha256"):
        changed = {**peak, name: "changed"}
        assert runner.input_identity(args, store, changed, ions) != identity
    pool_path = args.input_root / "msg15/MassSpecGym1.5_retrieval_candidates_formula.json"
    pool_path.write_text('{"C":["N","C"]}')
    changed = runner.input_identity(args, store, peak, ions)
    assert changed["pool_sha256"] != identity["pool_sha256"]
    assert changed["rows_sha256"] == identity["rows_sha256"]


@pytest.mark.parametrize("changed", [None, "rows_sha256", "pool_sha256", "spectrum_inputs", "stage"])
def test_canonical_checkpoint_stage_and_input_provenance(provenance, changed):
    args, _, _, _, identity = provenance
    (args.out / "validation_complete.json").write_text('{"complete":true}')
    saved = {"recipe": runner.RECIPE, "args": {"stage": "joint"}, "input_identity": deepcopy(identity)}
    if changed == "stage":
        saved["args"]["stage"] = "head"
    elif changed:
        saved["input_identity"][changed] = "changed"
    if changed:
        with pytest.raises(ValueError, match="stage or prepared inputs differ"):
            runner.validate_saved_run(saved, identity, args)
    else:
        runner.validate_saved_run(saved, identity, args)
        args.stage = "joint"
        runner.validate_saved_run(saved, identity, args)


def test_canonical_evaluation_requires_completed_selection(provenance):
    args, _, _, _, identity = provenance
    (args.out / "validation_complete.json").write_text('{"complete":false}')
    saved = {"recipe": runner.RECIPE, "args": {"stage": "joint"}, "input_identity": identity}
    with pytest.raises(ValueError, match="Complete validation selection"):
        runner.validate_saved_run(saved, identity, args)


def test_single_stage_checkpoint_evaluation_and_resume(provenance):
    args, _, _, _, identity = provenance
    (args.out / "validation_complete.json").write_text('{"complete":true}')
    saved = {"recipe": runner.RECIPE, "args": {"stage": "train"}, "input_identity": identity}
    runner.validate_saved_run(saved, identity, args)
    args.stage = "train"
    runner.validate_saved_run(saved, identity, args)
    args.stage = "joint"
    with pytest.raises(ValueError, match="stage or prepared inputs differ"):
        runner.validate_saved_run(saved, identity, args)


@pytest.mark.parametrize("changed", [None, "checkpoint", "rows_sha256", "pool_sha256", "graphs",
                                     "spectrum_inputs", "hidden", "stage"])
def test_only_selected_historical_checkpoint_and_inputs_can_be_imported(provenance, monkeypatch, changed):
    args, store, peak, ions, identity = provenance
    payload = b"selected-checkpoint-fixture"
    (args.out / "best.msgpack").write_bytes(payload)
    monkeypatch.setattr(runner, "SELECTED_CHECKPOINT_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(runner, "SELECTED_INPUT_SHA256", {name: identity[name] for name in
                                                        ("rows_sha256", "pool_sha256")})
    saved = {"args": vars(args).copy(), "graphs": store.metadata, "fragments": store.fragments.metadata,
             "peak_input": deepcopy(peak), "ion_inputs": ions}
    if changed == "checkpoint":
        (args.out / "best.msgpack").write_bytes(b"different checkpoint")
    elif changed == "spectrum_inputs":
        identity[changed]["prepared_spectra_sha256"] = "different intensities"
    elif changed == "hidden":
        saved["args"]["hidden"] = 128
    elif changed == "stage":
        args.stage = "joint"
    elif changed:
        identity[changed] = "changed"
    if changed:
        with pytest.raises(ValueError):
            runner.validate_saved_run(saved, identity, args)
    else:
        runner.validate_saved_run(saved, identity, args)


def test_checkpoint_preserves_model_statistics_optimizer_and_training_key(tmp_path):
    state = {"params": {"w": np.array([.25, -.5], np.float32)},
             "stats": {"mean": np.array([.1], np.float32)},
             "opt_state": {"count": np.array(4, np.int32), "moment": np.array([.2], np.float32)},
             "key": jax.random.PRNGKey(7)}
    path = tmp_path / "last.msgpack"
    runner.save_checkpoint(path, state)
    restored = serialization.from_bytes(state, path.read_bytes())
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(jax.random.split(restored["key"]), jax.random.split(state["key"]))
    assert not path.with_suffix(".msgpack.partial").exists()


@pytest.mark.parametrize("stage,adaptation", [("head", "full"), ("train", "full"),
                                             ("train", "lora"), ("train", "frozen")])
def test_main_real_head_update_session_resume_matches_uninterrupted_training(input_paths, monkeypatch, stage, adaptation):
    """Exercise main's initialization, actual gradients, saves and resumed RNGs."""
    from atom_flare_data import build_graph_store
    from glacier_fragment_data import build_fragment_store

    args = args_for(input_paths, stage)
    args.encoder_adaptation = adaptation
    args.__dict__.update(d=8, hidden=8, batch=4, negatives=1, atom_cap=8,
                         node_capacity=64, edge_capacity=128, eval_batch=4,
                         eval_chunk=2, candidate_cap=4, score_chunk=2,
                         epochs=2, session_hours=1e-9)
    args.prepared.mkdir()
    (args.input_root / "data").mkdir(parents=True)
    (args.input_root / "msg15").mkdir()
    store = build_graph_store(["CCO", "CCN", "CCC", "COC"], args.prepared / "graphs")
    proposal = input_paths.parent / "proposals"
    proposal.mkdir()
    np.savez(proposal / "part.npz", graph_ids=np.arange(4), frag_off=np.zeros(5, np.int64),
             mem_off=np.zeros(1, np.int64), atoms=np.zeros(0, np.int16), inten=np.zeros(0, np.float32))
    (proposal / "metadata.json").write_text(json.dumps({
        "smiles_sha256": store.metadata["smiles_sha256"], "completed_shards": ["part.npz"]}))
    store.fragments = build_fragment_store(store, proposal, args.fragments, f_max=2)
    train, val, test = np.arange(4), np.array([4]), np.array([5])
    np.savez(args.prepared / "rows.npz", train=train, val=val, test=test)
    (args.prepared / "selection.json").write_text('{"val_n":1}')
    np.save(args.input_root / "data/fold.npy", np.array(["train"] * 4 + ["val", "test"]))
    (args.input_root / "msg15/MassSpecGym1.5_retrieval_candidates_formula.json").write_text(
        json.dumps({s: store.smiles for s in store.smiles}))
    tokens = np.random.default_rng(8).normal(size=(6, 3, 6)).astype(np.float32)
    if stage == "train":
        from flax import nnx
        import trainable_jepa_peak as bridge
        from spectra_learning.models.encoder_jax import PeakSetEncoder
        from spectra_learning.models.peak_features_jax import PeakFeatureEmbedder

        def restore_tiny_encoder(current):
            encoder = PeakSetEncoder(
                model_dim=16, embedder=PeakFeatureEmbedder(model_dim=16, hidden_dim=16, fourier_num_freqs=4),
                num_layers=2, num_heads=2, num_peaks=3, use_pair_path=False,
                activation_checkpoint_mode="full", rngs=nnx.Rngs(4))
            if adaptation == "lora":
                from retrieval_lora import install_attention_lora
                install_attention_lora(encoder, rank=4, alpha=4., seed=current.seed)
            return bridge.TrainableJEPAPeak(encoder, current.d, current.dropout, 16, adaptation)

        monkeypatch.setattr(bridge, "restore_peak", restore_tiny_encoder)
        tokens = np.abs(tokens)
        tokens[..., 2] = 1
    mask = np.ones((6, 3), bool)
    truth_ids = np.array([0, 1, 2, 3, 0, 1], np.int32)
    smiles = np.array(store.smiles)[truth_ids]
    fractions = {"fraction": np.tile(np.array([.25, .5, 1], np.float32), (6, 1)),
                 "mz": np.tile(np.array([20., 30., 45.], np.float32), (6, 1)),
                 "eligible": np.ones(6, bool), "adduct_mass": np.full(6, 1.0072764, np.float32)}
    candidates = {s: np.arange(4, dtype=np.int32) for s in store.smiles}
    bank = [np.array([g for g in range(4) if g != truth], np.int32) for truth in truth_ids]
    peak_input = {"kind": "raw", "embedding_metadata": {"checkpoint": "fixture"}}
    ions = {"query_array_sha256": {"mz": "fixture"}}
    data = (store, tokens, mask, fractions, smiles, candidates, truth_ids)
    auxiliary = (train, val, test, smiles, truth_ids, bank, peak_input, ions)
    monkeypatch.setattr(runner, "recipe_args", lambda argv: args)
    monkeypatch.setattr(runner, "load_data", lambda current: (data, auxiliary))
    # Reuse the same tiny compiled real model across the three main invocations.
    if stage == "head":
        funcs = runner.functions(args)
        monkeypatch.setattr(runner, "functions", lambda current: funcs)
    actual_evaluate = runner.evaluate
    evaluated_rows = []

    def validation_only(*values):
        evaluated_rows.append(np.asarray(values[2]).copy())
        return actual_evaluate(*values)

    monkeypatch.setattr(runner, "evaluate", validation_only)
    runner.main([])
    first_state = serialization.msgpack_restore((args.out / "last.msgpack").read_bytes())
    first_progress = json.loads((args.out / "progress.json").read_text())
    assert len(first_progress["history"]) == 1
    assert not json.loads((args.out / "validation_complete.json").read_text())["complete"]
    assert int(first_state["opt_state"]["0"]["count"]) == 1
    for group in ("peak", "gnn", "compatibility", "agreement"):
        moments = jax.tree.leaves(first_state["opt_state"]["0"]["mu"][group])
        assert all(np.isfinite(value).all() for value in moments)
        assert any(np.any(value != 0) for value in moments)
    if stage == "train":
        for group in ("encoder", "readout"):
            moments = jax.tree.leaves(first_state["opt_state"]["0"]["mu"]["peak"][group])
            if group == "encoder" and adaptation == "frozen":
                assert not moments
            else:
                assert any(np.any(value != 0) for value in moments)
        assert json.loads((args.out / "manifest.json").read_text())["head_initialization"] is None
        assert not (args.out / "initial_validation.json").exists()
    args.resume, args.session_hours = True, 0.
    runner.main([])
    resumed_state = serialization.msgpack_restore((args.out / "last.msgpack").read_bytes())
    resumed_best = (args.out / "best.msgpack").read_bytes()
    progress = json.loads((args.out / "progress.json").read_text())
    assert [row["epoch"] for row in progress["history"]] == [0, 1]
    assert int(resumed_state["opt_state"]["0"]["count"]) == 2
    assert json.loads((args.out / "validation_complete.json").read_text())["complete"]
    assert not (args.out / "test_ranks.npz").exists()
    hits = [row["validation"]["optimistic"]["hit1"] for row in progress["history"]]
    assert progress["best_epoch"] == int(np.argmax(hits))
    assert any(np.any(a != b) for a, b in zip(jax.tree.leaves(first_state["params"]),
                                             jax.tree.leaves(resumed_state["params"]), strict=True))
    args.out, args.resume = input_paths.parent / "uninterrupted", False
    runner.main([])
    direct_state = serialization.msgpack_restore((args.out / "last.msgpack").read_bytes())
    for direct, resumed in zip(jax.tree.leaves(direct_state), jax.tree.leaves(resumed_state), strict=True):
        np.testing.assert_array_equal(direct, resumed)
    assert (args.out / "best.msgpack").read_bytes() == resumed_best
    direct_progress = json.loads((args.out / "progress.json").read_text())
    assert direct_progress["numpy_rng"] == progress["numpy_rng"]
    assert len(evaluated_rows) == 4
    for rows in evaluated_rows:
        np.testing.assert_array_equal(rows, val)


def test_source_manifest_ignores_generated_modules_without_source_files(monkeypatch):
    from types import ModuleType
    import sys
    module = ModuleType("_retrieval_generated_test_module")
    module.__file__ = str(Path(runner.__file__).resolve().parents[1] / "_classes.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    sources = runner.source_hashes()
    assert "_classes.py" not in sources
    assert "scripts/run_retrieval_jax.py" in sources
