"""Offline paired comparison with the saved original-FLARE checkpoint reranking."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/local_moljepa_atom_latest'
REFERENCE = Path('/mnt/data/jepa-fps-upstream-agreement/results/ftjepa_1p8b_step530500_agreement_top10')
DATASET = REPO / 'data/massspecgym_benchmark/data/MassSpecGym1.5.tsv'
POOL = Path('/mnt/data/jepa-fps/data/cache/msg15_formula_candidates.json')
POOL_PROVENANCE = REPO / 'results/ftjepa530500_molebert_warmstart_flare/atom_cache_metadata.json'
TSV_SHA256 = '50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a'
POOL_SHA256 = '454636d2346aebb3cd87edc2efc3fa1bf91eb8e0219edff5880f02aa537ac954'
TEST_N = 17556


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def metrics(ranks: np.ndarray) -> dict:
    return {'n': len(ranks), **{f'hit{k}': float((ranks <= k).mean()) for k in (1, 5, 10, 20)},
            'mrr': float((1 / ranks).mean()), 'median_rank': float(np.median(ranks))}


@dataclass
class Reference:
    rows: np.ndarray
    truths: np.ndarray
    ranks: dict[str, np.ndarray]
    counts: np.ndarray
    pools: dict[str, list[str]]
    evidence: dict


def load_reference(reference: Path = REFERENCE, dataset: Path = DATASET,
                   pool_path: Path = POOL, pool_provenance: Path = POOL_PROVENANCE) -> Reference:
    dataset_hash, pool_hash = sha256(dataset), sha256(pool_path)
    if dataset_hash != TSV_SHA256 or pool_hash != POOL_SHA256:
        raise ValueError('Canonical dataset or candidate-pool hash differs from the historical benchmark')
    provenance = read_json(pool_provenance)
    if (provenance['root'] != '/results/jepa_fps_ftjepa_1p8b_step530500_flare_raw'
            or provenance['dataset_tsv_sha256'] != dataset_hash
            or provenance['candidate_pools_sha256'] != pool_hash):
        raise ValueError('Historical candidate-pool provenance does not match the canonical files')
    with dataset.open(newline='') as stream:
        queries = [(i, row['smiles']) for i, row in enumerate(csv.DictReader(stream, delimiter='\t'))
                   if row['fold'] == 'test']
    rows = np.asarray([row for row, _ in queries], dtype=np.int64)
    truths = np.asarray([truth for _, truth in queries])
    with np.load(reference / 'query_ranks.npz', allow_pickle=False) as saved:
        if len(rows) != TEST_N or not np.array_equal(rows, saved['query_rows']):
            raise ValueError('Historical ranks do not contain exactly the canonical test rows in order')
        if not np.array_equal(truths, saved['predictions'][:, 0]):
            raise ValueError('Historical ground-truth SMILES do not match canonical dataset rows')
        predictions = saved['predictions'][:, 1]
        ranks = {kind: saved[kind].copy() for kind in ('stable', 'optimistic')}
    pools = read_json(pool_path)
    pools = {truth: pools[truth] for truth in set(truths)}
    counts = np.asarray([len(pools[truth]) for truth in truths], dtype=np.int64)
    if any(truth not in pools[truth] or prediction not in pools[truth]
           for truth, prediction in zip(truths, predictions, strict=True)):
        raise ValueError('Historical prediction or truth absent from its canonical candidate pool')
    saved_metrics = read_json(reference / 'metrics_retrieval.json')['test']
    for kind, values in ranks.items():
        if not np.issubdtype(values.dtype, np.integer) or not np.all((values >= 1) & (values <= counts)):
            raise ValueError('Historical ranks violate candidate-pool bounds')
        if metrics(values) != saved_metrics[kind]:
            raise ValueError('Historical rank-derived metrics do not reproduce the saved reevaluation')
    if np.any(ranks['optimistic'] > ranks['stable']):
        raise ValueError('Historical tie conventions are inconsistent')
    original = read_json(reference / 'manifest.json')
    if int((ranks['stable'] == 1).sum()) != 5805:
        raise ValueError('Reference is not the original 33.0656% checkpoint reranking')
    evidence = {
        'n': len(rows), 'unique_truth_smiles': len(pools),
        'query_rows_sha256': hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
        'candidate_counts_sha256': hashlib.sha256(counts.astype('<i8').tobytes()).hexdigest(),
        'candidate_count_min': int(counts.min()), 'candidate_count_max': int(counts.max()),
        'dataset_path': str(dataset), 'dataset_sha256': dataset_hash,
        'candidate_pool_path': str(pool_path), 'candidate_pool_sha256': pool_hash,
        'candidate_pool_provenance_path': str(pool_provenance),
        'candidate_pool_provenance_sha256': sha256(pool_provenance),
        'historical_rank_path': str(reference / 'query_ranks.npz'),
        'historical_rank_sha256': sha256(reference / 'query_ranks.npz'),
        'historical_checkpoint_sha256': original['checkpoint_sha256'],
        'canonical_rows_and_truth_smiles_verified': True,
        'historical_prediction_membership_verified': True,
        'candidate_count_basis': 'Reconstructed from hash-verified canonical pools; historical NPZ omits counts.',
        'reference_metrics': {kind: metrics(values) for kind, values in ranks.items()},
        'reference_label': 'Top-10 reevaluation of the original step-530500 FLARE agreement checkpoint',
        'reevaluation_note': 'Hit@1/5/20 match the original run exactly. MRR differs by about 7e-8; '
                             'the original run did not save per-query ranks, so paired ranks use its saved-checkpoint reevaluation.',
    }
    return Reference(rows, truths, ranks, counts, pools, evidence)


def align_records(reference: Reference, records: list[dict]) -> dict[str, np.ndarray]:
    if any(type(record['row']) is not int or type(record['candidates']) is not int for record in records):
        raise ValueError('New query rows and candidate counts must be integers')
    rows = np.asarray([record['row'] for record in records], dtype=np.int64)
    if len(rows) != len(reference.rows) or len(np.unique(rows)) != len(rows):
        raise ValueError('New ranks contain the wrong count or duplicate query rows')
    if not np.array_equal(np.sort(rows), reference.rows):
        raise ValueError('New ranks do not contain exactly the same canonical test rows')
    ordered = [records[i] for i in np.argsort(rows)]
    counts = np.asarray([record['candidates'] for record in ordered], dtype=np.int64)
    if not np.array_equal(counts, reference.counts):
        raise ValueError('New candidate counts differ from the canonical historical pools')
    if any(record['prediction'] not in reference.pools[truth]
           for record, truth in zip(ordered, reference.truths, strict=True)):
        raise ValueError('New prediction is absent from its canonical candidate pool')
    ranks = {}
    for kind in ('stable', 'optimistic'):
        values = np.asarray([record[f'{kind}_rank'] for record in ordered])
        if not np.issubdtype(values.dtype, np.integer) or not np.all((values >= 1) & (values <= counts)):
            raise ValueError('New ranks violate candidate-pool bounds')
        ranks[kind] = values
    if np.any(ranks['optimistic'] > ranks['stable']):
        raise ValueError('New tie conventions are inconsistent')
    return ranks


def paired_hit1_bootstrap(delta: np.ndarray, truths: np.ndarray, repetitions: int, seed: int) -> dict:
    """Paired spectrum bootstrap, plus repeated-spectrum molecule-cluster sensitivity."""
    n = len(delta)
    frequency = np.asarray([(delta == value).sum() for value in (-1, 0, 1)])
    # Multinomial counts are exactly equivalent to resampling the paired spectrum deltas.
    draws = np.random.default_rng(seed).multinomial(n, frequency / n, size=repetitions)
    query_draws = (draws[:, 2] - draws[:, 0]) / n
    unique, groups = np.unique(truths, return_inverse=True)
    cluster_delta = np.bincount(groups, weights=delta)
    cluster_size = np.bincount(groups)
    rng = np.random.default_rng(seed + 1)
    cluster_draws = []
    for start in range(0, repetitions, 128):
        selected = rng.integers(len(unique), size=(min(128, repetitions - start), len(unique)))
        cluster_draws.extend((cluster_delta[selected].sum(1) / cluster_size[selected].sum(1)).tolist())
    return {
        'method': 'paired percentile bootstrap, fixed models and test predictions',
        'repetitions': repetitions, 'query_seed': seed, 'molecule_cluster_seed': seed + 1,
        'query_ci95_percentage_points': (100 * np.quantile(query_draws, [.025, .975])).tolist(),
        'molecule_cluster_ci95_percentage_points': (100 * np.quantile(cluster_draws, [.025, .975])).tolist(),
        'molecule_clusters': len(unique),
        'cluster_definition': 'Exact ground-truth SMILES; resample molecules and retain all their spectra, then compute spectrum-weighted Hit@1.',
        'interpretation': 'Sampling uncertainty conditional on these fitted models; does not measure training-seed uncertainty or remove experimental confounding.',
    }


def compare_ranks(old: np.ndarray, new: np.ndarray, truths: np.ndarray,
                  repetitions: int = 10000, seed: int = 7301) -> dict:
    old_hit, new_hit = old == 1, new == 1
    delta = new_hit.astype(np.int8) - old_hit.astype(np.int8)
    before, after = metrics(old), metrics(new)
    return {
        'reference_metrics': before, 'new_metrics': after,
        'metric_delta_new_minus_reference': {key: after[key] - before[key] for key in before if key != 'n'},
        'hit1': {
            'gains': int((delta == 1).sum()), 'losses': int((delta == -1).sum()),
            'ties': int((delta == 0).sum()), 'both_correct': int((old_hit & new_hit).sum()),
            'both_incorrect': int((~old_hit & ~new_hit).sum()),
            'delta_percentage_points': 100 * float(delta.mean()),
            'bootstrap': paired_hit1_bootstrap(delta, truths, repetitions, seed),
        },
        'rank': {
            'improved': int((new < old).sum()), 'worsened': int((new > old).sum()),
            'unchanged': int((new == old).sum()),
            'mean_change_new_minus_reference': float((new - old).mean()),
            'median_change_new_minus_reference': float(np.median(new - old)),
        },
    }


def compare_results(root: Path = ROOT, reference: Reference | None = None,
                    repetitions: int = 10000, seed: int = 7301) -> dict:
    reference = reference or load_reference()
    stages = {}
    for stage in ('hard', 'agreement'):
        path = root / stage / 'test_ranks.json'
        records = read_json(path)
        ranks = align_records(reference, records)
        summary = read_json(root / stage / 'summary.json')
        for kind in ('stable', 'optimistic'):
            actual = metrics(ranks[kind])
            if any(summary['test'][kind][key] != value for key, value in actual.items()):
                raise ValueError(f'{stage} saved metrics do not match paired ranks')
        stages[stage] = {
            'new_test_ranks_sha256': sha256(path),
            'new_checkpoint_sha256': summary['checkpoint_sha256'],
            'row_alignment_verified': True, 'candidate_counts_verified': True,
            'prediction_pool_membership_verified': True,
            'original_saved_row_order_matches': [record['row'] for record in records] == reference.rows.tolist(),
            **{kind: compare_ranks(reference.ranks[kind], ranks[kind], reference.truths, repetitions, seed)
               for kind in ('stable', 'optimistic')},
        }
    return {
        'reference_alignment': reference.evidence,
        'comparisons': stages,
        'caveat': 'These are paired descriptive comparisons on the same test spectra, not a matched architecture control. '
                  'The architecture, spectrum checkpoint (1190000 versus 530500), and checkpoint-selection validation set '
                  '(5000 versus 2000) differ. The hard-stage comparison also differs in mass-agreement usage. '
                  'Both implementations load the canonical pool in supplied order; saved rank files alone do not record every candidate identity.',
    }


def markdown_lines(result: dict) -> list[str]:
    lines = [
        '## Paired comparison with the original 33% checkpoint', '',
        'All 17,556 test row IDs and ground-truth SMILES match the canonical dataset. Saved candidate counts '
        'match the hash-verified original pools. The reference uses the original checkpoint’s saved top-10 '
        'reevaluation, which reproduced 33.0656% Hit@1 exactly.', '',
        '| New model | Hit@1 gains / losses / ties | Δ Hit@1 (pp) | Paired query 95% CI (pp) | Molecule-cluster 95% CI (pp) | Rank better / worse / same |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for stage, label in (('hard', 'Without mass agreement'), ('agreement', 'With mass agreement')):
        values = result['comparisons'][stage]['stable']
        hit, rank = values['hit1'], values['rank']
        query_ci = hit['bootstrap']['query_ci95_percentage_points']
        cluster_ci = hit['bootstrap']['molecule_cluster_ci95_percentage_points']
        lines.append(f'| {label} | {hit["gains"]:,} / {hit["losses"]:,} / {hit["ties"]:,} '
                     f'| {hit["delta_percentage_points"]:+.4f} '
                     f'| [{query_ci[0]:+.4f}, {query_ci[1]:+.4f}] '
                     f'| [{cluster_ci[0]:+.4f}, {cluster_ci[1]:+.4f}] '
                     f'| {rank["improved"]:,} / {rank["worsened"]:,} / {rank["unchanged"]:,} |')
    repetitions = result['comparisons']['hard']['stable']['hit1']['bootstrap']['repetitions']
    lines += ['', 'Stable ranks are shown; both tie conventions are recorded in `paired_comparison.json`. '
              'A gain is a previously incorrect spectrum becoming rank 1; ties include spectra that remain '
              f'correct and those that remain incorrect. Bootstrap intervals use {repetitions:,} paired resamples with '
              'fixed seeds. Molecule clustering accounts for repeated spectra with the same exact truth SMILES. '
              'These intervals condition on the fitted models and do not measure training-seed variation.',
              '', result['caveat']]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--alignment-only', action='store_true')
    args = parser.parse_args()
    reference = load_reference()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / 'historical_test_alignment.json').write_text(json.dumps(reference.evidence, indent=2) + '\n')
    if args.alignment_only:
        print(json.dumps(reference.evidence, indent=2))
        return
    result = compare_results(args.root, reference)
    (args.root / 'paired_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.root / 'paired_comparison.md').write_text('\n'.join(markdown_lines(result)) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
