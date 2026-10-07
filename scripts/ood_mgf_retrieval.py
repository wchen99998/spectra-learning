"""Embed out-of-distribution MGF spectra and rank them against hard-negative pools.

Stages, all in one process on one GPU:
1. Parity: the MGF preprocessing path reproduces the prepared MassSpecGym rows,
   the restored step-1190000 student reproduces the cached tokens and CLS, and
   the restored 44.10% retrieval checkpoint reproduces its saved test ranks.
2. Embeddings: original student and joint-adapted retrieval encoder for every
   (spectrum, precursor, acquisition-metadata) variant, plus MSG nearest
   neighbours by CLS cosine.
3. Retrieval: the unchanged scorer over each hard-negative family.
"""

import argparse
import json
import math
from pathlib import Path

from flax import nnx, serialization
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.csv as csv
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from atom_flare_data import GraphStore
from glacier_fragment_data import build_fragment_store
from ion_logmixture_data import PROTON_MASS
import run_retrieval_jax as runner
from spectra_learning.data.spectra import COLLISION_ENERGY_MAX, PEAK_MZ_MAX, preprocess_peak_batch_numpy
from spectra_learning.models.spectrum_metadata import (
    instrument_family_id, jax_spectrum_metadata_from_batch, polarity_id,
)
from spectra_learning.probes.massspec import atom_flare_jax as model
from trainable_jepa_peak import validate_initial_tokens

ACQUISITION = {  # name -> (instrument_type, collision_energy)
    "mgf_only": (None, None), "orbitrap": ("Orbitrap", None),
    "qtof": ("QTOF", None), "qtof_ce25": ("QTOF", 25.0),
}
FAMILIES = ("H1_ring_site", "H2_sequence", "H3_residue_isomer", "H4_pubchem_formula", "X_cross_species")
SUBSAMPLE = 255  # MassSpecGym pools hold at most 256 candidates


def read_mgf(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    header, peaks = {}, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            key, value = line.split("=", 1)
            header[key] = value
        elif line and line[0].isdigit():
            mz, intensity = line.split()[:2]
            peaks.append((float(mz), float(intensity)))
    peaks = np.asarray(peaks, np.float32)
    return peaks[:, 0], peaks[:, 1], float(header["PEPMASS"].split()[0])


def pack(spectra, precursor, energy, instrument, adduct, config) -> np.ndarray:
    """Exactly the MassSpecGym preparation followed by the trainable-JEPA packing."""
    width = max(len(mz) for mz, _ in spectra)
    raw = np.zeros((len(spectra), 2, width), np.float32)
    for row, (mz, intensity) in enumerate(spectra):
        raw[row, 0, :len(mz)], raw[row, 1, :len(intensity)] = mz, intensity
    batch = preprocess_peak_batch_numpy(
        raw, np.asarray(precursor, np.float32), num_peaks=int(config["num_peaks"]),
        peak_drop_min_intensity=float(config["peak_drop_min_intensity"]),
        peak_ordering=str(config["peak_ordering"]), max_precursor_mz=float(config["max_precursor_mz"]),
        precursor_peak_exclusion_window_da=float(config["precursor_peak_exclusion_window_da"]),
        min_peak_intensity=float(config["min_peak_intensity"]))
    energy_value = np.asarray([0.0 if e is None else e for e in energy], np.float32)
    fields = {
        "precursor_mz": batch["precursor_mz"],
        "collision_energy": np.clip(energy_value, 0.0, COLLISION_ENERGY_MAX) / COLLISION_ENERGY_MAX,
        "collision_energy_present": np.asarray([e is not None and e > 0.0 for e in energy], np.float32),
        "charge": np.ones(len(spectra), np.float32), "charge_present": np.ones(len(spectra), np.float32),
        "polarity_id": np.asarray([polarity_id(a) for a in adduct], np.int32),
        "instrument_family_id": np.asarray([instrument_family_id(i) for i in instrument], np.int32),
    }
    metadata = np.asarray(jax_spectrum_metadata_from_batch(
        {k: jnp.asarray(v) for k, v in fields.items()}, schema=config["encoder_metadata_schema"]), np.float32)
    packed = np.zeros((*batch["peak_valid_mask"].shape, 4 + metadata.shape[1]), np.float32)
    packed[..., 0], packed[..., 1] = batch["peak_mz"], batch["peak_intensity"]
    packed[..., 2] = batch["peak_valid_mask"]
    packed[:, 0, 3], packed[:, 0, 4:] = batch["precursor_mz"], metadata
    return packed


def tsv_rows(path: Path, rows: np.ndarray) -> dict:
    table = csv.read_csv(path, parse_options=csv.ParseOptions(delimiter="\t"), convert_options=csv.ConvertOptions(
        include_columns=["mzs", "intensities", "precursor_mz", "collision_energy", "instrument_type", "adduct"]))
    table = table.take(rows).to_pydict()
    table["spectra"] = [(np.fromstring(m, sep=",", dtype=np.float32), np.fromstring(i, sep=",", dtype=np.float32))
                        for m, i in zip(table["mzs"], table["intensities"], strict=True)]
    return table


def make_encoder(peak):
    """Full encoder output [B, N+1, D]: physical tokens followed by the CLS token."""
    @jax.jit
    def encode(parameters, packed):
        encoder = nnx.merge(peak.graphdef, nnx.State(parameters), peak.static)
        mask = packed[..., 2].astype(bool)
        return encoder(packed[..., 0], packed[..., 1], valid_mask=mask, visible_mask=mask,
                       precursor_mz=packed[:, 0, 3], spectrum_metadata=packed[:, 0, 4:])
    return encode


def encode_rows(encode, parameters, packed, batch_size):
    out = []
    for start in range(0, len(packed), batch_size):
        batch = np.zeros((batch_size, *packed.shape[1:]), np.float32)
        chunk = packed[start:start + batch_size]
        batch[:len(chunk)] = chunk
        out.append(np.asarray(encode(parameters, batch), np.float32)[:len(chunk)])
    return np.concatenate(out)


def summarize_embedding(full, mask):
    tokens, cls = full[:, :-1], full[:, -1]
    mean = (tokens * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
    return {"tokens": tokens.astype(np.float16), "cls": cls, "mean": mean,
            "token_norm_mean": np.array([np.linalg.norm(t[m], axis=-1).mean() for t, m in zip(tokens, mask)])}


def cosine(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


def topk_boundary_tie(mz: np.ndarray, intensity: np.ndarray, num_peaks: int) -> bool:
    """Whether the num_peaks-th most intense in-range peak is tied with a dropped one."""
    kept = np.sort(intensity[mz <= PEAK_MZ_MAX])[::-1]
    return bool(len(kept) > num_peaks and kept[num_peaks - 1] == kept[num_peaks])


def expected_subsampled_hit1(beaten_by: int, negatives: int, sample: int = SUBSAMPLE) -> float:
    """P(no candidate scoring >= truth lands in a uniform `sample`-negative subset)."""
    if negatives <= sample:
        return float(beaten_by == 0)
    return math.comb(negatives - beaten_by, sample) / math.comb(negatives, sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("configs/retrieval_4602.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/results/retrieval_selected_4602_20260924/run"))
    parser.add_argument("--mgf-dir", type=Path, default=Path("data/test_mgf"))
    parser.add_argument("--work", type=Path, default=Path("/results/ood_mgf_20261003"))
    parser.add_argument("--parity-rows", type=int, default=256)
    cli = parser.parse_args()
    out = cli.work / "run"
    out.mkdir(parents=True, exist_ok=True)
    args = runner.recipe_args(["--inputs", str(cli.inputs), "--out", str(cli.checkpoint_dir), "--stage", "eval"])
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    config = json.loads(args.jepa_metadata.read_text())["task_contract"]["config"]
    embedding_metadata = json.loads((args.input_root / "embedding_metadata.json").read_text())
    batch_size = embedding_metadata["batch_size"]
    report = {"devices": [str(d) for d in jax.devices()], "jax": jax.__version__}

    # ---- restore the selected model exactly as runner.main does for --stage eval
    data, auxiliary = runner.load_data(args)
    msg_store, msg_tokens, msg_mask, msg_fractions, msg_smiles, _, msg_truth = data
    train, val, test = auxiliary[:3]
    peak, gnn, agreement, _, _, encode_molecules, encode_peaks, query_score = runner.functions(args)
    kp, kg, ka = jax.random.split(jax.random.PRNGKey(args.seed), 4)[1:]
    graph = runner.graph_batch(msg_store, msg_truth[train[:4]].tolist(), args, train=True)
    gp = gnn.init(kg, graph, train=False)
    pp = peak.init(kp, jnp.zeros((1, *msg_tokens.shape[1:]), jnp.float32), train=False)
    ap = agreement.init(ka, jnp.zeros((1, 1)), jnp.ones((1, 1)), jnp.ones((1, 1), dtype=bool))
    params = {"peak": pp["params"], "gnn": gp["params"], "agreement": ap["params"],
              "compatibility": model.QueryKeyProjection(args.d).init(
                  jax.random.PRNGKey(args.seed), jnp.zeros((1, args.d)), jnp.zeros((1, args.d)))["params"]}
    original_encoder = params["peak"]["encoder"]
    report["initial_token_parity"] = validate_initial_tokens(peak, {"encoder": original_encoder}, msg_tokens, msg_mask,
                                                             args.input_root, test)
    state = jax.device_put(serialization.from_bytes(
        {"params": params, "stats": gp["batch_stats"]}, (cli.checkpoint_dir / "best.msgpack").read_bytes()))
    params, stats = state["params"], state["stats"]
    del state, gp, pp, ap
    encoders = {"student_s1190000": original_encoder, "retrieval_adapted": params["peak"]["encoder"]}
    encode = make_encoder(peak)

    # ---- parity 1: MGF packing path == prepared MassSpecGym rows
    saved = np.load(cli.checkpoint_dir / "test_ranks.npz", allow_pickle=False)
    rows = np.asarray(saved["rows"][:cli.parity_rows])
    table = tsv_rows(args.input_root / "msg15/MassSpecGym1.5.tsv", rows)
    mine = pack(table["spectra"], table["precursor_mz"], table["collision_energy"],
                table["instrument_type"], table["adduct"], config)
    reference = np.asarray(msg_tokens[rows], np.float32)
    # np.argpartition breaks top-k intensity ties by CPU-dependent SIMD paths, so a row whose
    # 127th intensity is tied may keep a different (equally intense) peak than the preparation host.
    differing = np.flatnonzero((mine != reference).reshape(len(rows), -1).any(1))
    tie_swaps = [int(r) for r in differing
                 if np.array_equal(mine[r, :, 2:], reference[r, :, 2:])
                 and np.array_equal(np.sort(mine[r, :, 1]), np.sort(reference[r, :, 1]))]
    report["packing_parity"] = {"rows": len(rows), "max_abs_diff": float(np.abs(mine - reference).max()),
                                "exact_rows": len(rows) - len(differing), "topk_tie_swaps": tie_swaps}
    if len(tie_swaps) != len(differing):
        raise ValueError(f"MGF packing differs from prepared rows: {report['packing_parity']}")
    exact = np.setdiff1d(np.arange(len(rows)), tie_swaps)

    # ---- parity 2: CLS of the original student == cached CLS export
    cls_cache = np.load(args.input_root / "cls_embeddings.float16.npy", mmap_mode="r")
    exact = exact[:batch_size]
    full = encode_rows(encode, original_encoder, mine[exact], batch_size)
    expected = np.asarray(cls_cache[rows[exact]], np.float32)
    report["cls_parity"] = {"rows": len(exact), "relative_l2": float(
        np.linalg.norm(full[:, -1].astype(np.float16).astype(np.float32) - expected) / np.linalg.norm(expected))}

    # ---- parity 3: restored checkpoint reproduces the saved test ranks
    result, ranks = runner.evaluate(params, stats, rows, data, args, encode_molecules, encode_peaks, query_score)
    agree = ranks["ranks_stable"] == saved["ranks_stable"][:len(rows)]
    report["rank_parity"] = {"rows": len(rows), "identical_ranks": int(agree.sum()),
                             "hit1_now": result["optimistic"]["hit1"],
                             "hit1_saved": float((saved["ranks_optimistic"][:len(rows)] == 1).mean()),
                             "max_rank_difference": int(np.abs(ranks["ranks_stable"] - saved["ranks_stable"][:len(rows)]).max())}
    print(json.dumps(report, indent=1), flush=True)
    if agree.mean() < 0.95:
        raise ValueError(f"Restored checkpoint does not reproduce saved ranks: {report['rank_parity']}")

    # ---- OOD queries
    candidates = json.loads((cli.work / "candidates.json").read_text())
    names = sorted(candidates["truth"])
    spectra = {name: read_mgf(cli.mgf_dir / f"{name}.mgf") for name in names}
    theoretical = {name: ExactMolWt(Chem.MolFromSmiles(candidates["truth"][name])) + PROTON_MASS for name in names}
    variants = [(name, precursor, acquisition) for name in names
                for precursor in ("pepmass", "theoretical") for acquisition in ACQUISITION]
    packed = pack([spectra[n][:2] for n, _, _ in variants],
                  [spectra[n][2] if p == "pepmass" else theoretical[n] for n, p, _ in variants],
                  [ACQUISITION[a][1] for _, _, a in variants], [ACQUISITION[a][0] for _, _, a in variants],
                  ["[M+H]+"] * len(variants), config)
    mask = packed[..., 2].astype(bool)
    report["queries"] = {name: {"pepmass": spectra[name][2], "theoretical_mh": theoretical[name],
                                "pepmass_error_da": spectra[name][2] - theoretical[name],
                                "raw_peaks": int(len(spectra[name][0])),
                                "valid_peaks_after_preprocessing": int(mask[variants.index((name, "pepmass", "mgf_only"))].sum()),
                                "peaks_above_1000": int((spectra[name][0] > PEAK_MZ_MAX).sum()),
                                "topk_boundary_tie": topk_boundary_tie(*spectra[name][:2], int(config["num_peaks"]))}
                         for name in names}

    # ---- embeddings and their validation
    rng = np.random.default_rng(0)
    reference_rows = np.sort(rng.choice(len(cls_cache), 4096, replace=False))
    reference_cls = np.asarray(cls_cache[reference_rows], np.float32)
    all_cls = np.asarray(cls_cache, np.float32)
    msg_cls_norm = np.linalg.norm(reference_cls, axis=-1)
    token_cache = np.load(args.input_root / "covariance_embeddings.float16.npy", mmap_mode="r")
    sample_rows = reference_rows[:256]
    sample_tokens = np.asarray(token_cache[sample_rows], np.float32)
    sample_norms = np.linalg.norm(sample_tokens, axis=-1)[msg_mask[sample_rows]]
    embeddings, report["embeddings"] = {}, {
        "msg_reference": {"cls_norm_p01_p50_p99": np.percentile(msg_cls_norm, [1, 50, 99]).tolist(),
                          "token_norm_p01_p50_p99": np.percentile(sample_norms, [1, 50, 99]).tolist()}}
    msg_smiles_all = np.asarray(msg_smiles)
    for label, parameters in encoders.items():
        encoded = summarize_embedding(encode_rows(encode, parameters, packed, batch_size), mask)
        embeddings[label] = encoded
        entry = {"finite": bool(all(np.isfinite(v).all() for v in encoded.values())),
                 "cls_norm": dict(zip(map("|".join, variants), np.linalg.norm(encoded["cls"], axis=-1).tolist())),
                 "token_norm_mean": dict(zip(map("|".join, variants), encoded["token_norm_mean"].tolist()))}
        base = [variants.index((n, "pepmass", "mgf_only")) for n in names]
        entry["cls_cosine_between_spectra"] = {f"{a}~{b}": float(cosine(encoded["cls"][[base[i]]], encoded["cls"][[base[j]]])[0, 0])
                                               for i, a in enumerate(names) for j, b in enumerate(names) if i < j}
        entry["mean_cosine_between_spectra"] = {f"{a}~{b}": float(cosine(encoded["mean"][[base[i]]], encoded["mean"][[base[j]]])[0, 0])
                                                for i, a in enumerate(names) for j, b in enumerate(names) if i < j}
        entry["variant_sensitivity"] = {}
        for name, b in zip(names, base):
            for precursor in ("pepmass", "theoretical"):
                for acquisition in ACQUISITION:
                    v = variants.index((name, precursor, acquisition))
                    tokens_a, tokens_b = encoded["tokens"][b].astype(np.float32), encoded["tokens"][v].astype(np.float32)
                    entry["variant_sensitivity"][f"{name}|{precursor}|{acquisition}"] = {
                        "cls_cosine_to_mgf_only": float(cosine(encoded["cls"][[b]], encoded["cls"][[v]])[0, 0]),
                        "token_relative_l2_to_mgf_only": float(np.linalg.norm((tokens_a - tokens_b)[mask[b]])
                                                               / np.linalg.norm(tokens_a[mask[b]]))}
        if label == "student_s1190000":
            similarity = cosine(encoded["cls"][base], all_cls)
            entry["msg_nearest_neighbours_by_cls"] = {}
            for name, scores in zip(names, similarity):
                top = np.argsort(-scores)[:5]
                entry["msg_nearest_neighbours_by_cls"][name] = [
                    {"row": int(r), "cosine": float(scores[r]), "smiles": str(msg_smiles_all[r])} for r in top]
                entry["msg_nearest_neighbours_by_cls"][name + "_cosine_percentiles_p50_p99_max"] = (
                    np.percentile(scores, [50, 99]).tolist() + [float(scores.max())])
        report["embeddings"][label] = entry
    np.savez(out / "embeddings.npz", variants=np.array(["|".join(v) for v in variants]), peak_valid_mask=mask,
             packed_inputs=packed, **{f"{label}_{key}": value for label, encoded in embeddings.items()
                                     for key, value in encoded.items()})

    # ---- retrieval over hard-negative pools
    store = GraphStore(cli.work / "graphs")
    store.fragments = build_fragment_store(store, cli.work / "proposals", cli.work / "fragments",
                                           f_max=128, min_atoms=3, min_frac=0.05)
    if store.fragments.metadata["f_max"] != msg_store.fragments.metadata["f_max"]:
        raise ValueError("OOD fragment cap differs from the training fragments")
    lookup = {s: i for i, s in enumerate(store.smiles)}
    pools, query_rows = {}, []
    for name in names:
        families = dict(candidates["pools"][name])
        families["ALL_same_formula"] = sorted(set().union(*(families[f] for f in FAMILIES[:4])))
        for family, members in families.items():
            # Truth last: stable ranks count every tie against the truth.
            pools[f"{name}|{family}"] = np.array([lookup[s] for s in members] + [lookup[candidates["truth"][name]]], np.int32)
    keys, rows_variant = [], []
    for v, (name, precursor, acquisition) in enumerate(variants):
        for key in pools:
            if key.startswith(name + "|"):
                keys.append(key)
                rows_variant.append(v)
    rows_variant = np.asarray(rows_variant)
    args.candidate_cap = int(-(-max(len(p) for p in pools.values()) // 64) * 64)
    query = {"fraction": np.zeros(packed.shape[:2], np.float32),
             "mz": packed[..., 0] * np.float32(PEAK_MZ_MAX), "eligible": np.ones(len(packed), bool),
             "adduct_mass": np.full(len(packed), PROTON_MASS, np.float32)}
    query = {k: v[rows_variant] for k, v in query.items()}
    truth_ids = np.array([pools[k][-1] for k in keys], np.int32)
    ood = (store, packed[rows_variant], mask[rows_variant], query, np.array(keys), pools, truth_ids)
    scored = []
    runner.evaluate(params, stats, np.arange(len(keys)), ood, args, encode_molecules, encode_peaks, query_score, scored)
    results = []
    for (row, pool_ids, scores), key, v in zip(scored, keys, rows_variant, strict=True):
        name, family = key.split("|")
        opt, stable, _ = runner.ranks_with_ties(scores, len(pool_ids) - 1)
        order = np.argsort(-scores, kind="stable")[:5]
        results.append({"spectrum": name, "precursor": variants[v][1], "acquisition": variants[v][2], "family": family,
                        "candidates": len(pool_ids), "rank_optimistic": opt, "rank_stable": stable,
                        "percentile": 1 - (stable - 1) / max(len(pool_ids) - 1, 1),
                        "expected_hit1_at_256": expected_subsampled_hit1(stable - 1, len(pool_ids) - 1),
                        "truth_score": float(scores[-1]), "best_negative_score": float(np.max(scores[:-1])),
                        "top5": [{"smiles": store.smiles[int(pool_ids[i])], "score": float(scores[i]),
                                  "truth": bool(i == len(pool_ids) - 1)} for i in order]})
        np.save(out / f"scores__{key.replace('|', '__')}__{variants[v][1]}__{variants[v][2]}.npy",
                np.stack([pool_ids.astype(np.float64), scores.astype(np.float64)]))
    report["retrieval"] = results
    report["candidate_cap"] = args.candidate_cap
    (out / "report.json").write_text(json.dumps(report, indent=1))
    for r in results:
        if r["precursor"] == "pepmass" and r["acquisition"] == "mgf_only":
            print(f"{r['spectrum']:13s} {r['family']:20s} n={r['candidates']:5d} rank={r['rank_stable']:5d} "
                  f"pct={r['percentile']:.3f} E[hit1@256]={r['expected_hit1_at_256']:.3f}", flush=True)


if __name__ == "__main__":
    main()
