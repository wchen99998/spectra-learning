"""Export full JEPA token arrays (127 peak slots + CLS) for every spectrum of an MGF file.

Stages (`--stage`):
    prepare   parse the MGF on the host, apply the MassSpecGym preprocessing contract of the
              encoder config, and write `prepared_inputs.npz` + `rows.tsv` (one row per
              BEGIN IONS block, in file order).
    parity    pack the first MassSpecGym 1.5 rows through the same path and compare them with
              the prepared MassSpecGym arrays the cached harness exports were built from.
    shard     encode rows [start, stop) into a raw float16 part file `_parts/{start}-{stop}.bin`;
              the first shard also checks restore parity against the cached MSG tokens.
    assemble  concatenate the parts into `tokens_f16.npy` and write `token_valid_mask.npy`
              and `embedding_metadata.json` (harness contract: `cls_token_index`).
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

INPUT_NAMES = ('peak_mz', 'peak_intensity', 'peak_valid_mask', 'precursor_mz',
               'collision_energy', 'collision_energy_present', 'charge', 'charge_present',
               'polarity_id', 'instrument_family_id')
ROW_FIELDS = ('TITLE', 'SOURCE', 'SMILES', 'INCHIKEY', 'FORMULA', 'ADDUCT', 'IONMODE', 'CHARGE',
              'PRECURSOR_MZ', 'INSTRUMENT_TYPE', 'COLLISION_ENERGY_1', 'COLLISION_ENERGY_2',
              'COLLISION_ENERGY_3', 'MS_LEVEL', 'FOLD', 'NUM_PEAKS')
CHUNK = 4096


def read_mgf(path: Path):
    """Yield (header, mz, intensity) per BEGIN IONS block, in file order."""
    header, peaks = None, []
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line == 'BEGIN IONS':
                header, peaks = {}, []
            elif line == 'END IONS':
                values = np.asarray(peaks, np.float32).reshape(-1, 2)
                yield header, values[:, 0], values[:, 1]
                header = None
            elif header is None or not line:
                continue
            elif line[0].isdigit():
                mz, intensity = line.split()[:2]
                peaks.append((float(mz), float(intensity)))
            elif '=' in line:
                key, value = line.split('=', 1)
                header[key] = value


def single_collision_energy(header: dict) -> float | None:
    """The collision energy in eV when exactly one finite positive value is reported."""
    values = [float(header.get(f'COLLISION_ENERGY_{i}', 'nan')) for i in (1, 2, 3)]
    finite = [v for v in values if math.isfinite(v) and v > 0.0]
    return finite[0] if len(finite) == 1 else None


def pack(spectra, precursor, energy, charge, instrument, adduct, config) -> dict:
    """MassSpecGym preparation (scripts/modal_massspecgym_maccs_probe.py) for raw peak lists."""
    from spectra_learning.data.spectra import COLLISION_ENERGY_MAX, preprocess_peak_batch_numpy
    from spectra_learning.models.spectrum_metadata import instrument_family_id, polarity_id
    width = max(1, max(len(mz) for mz, _ in spectra))
    raw = np.zeros((len(spectra), 2, width), np.float32)
    for row, (mz, intensity) in enumerate(spectra):
        raw[row, 0, :len(mz)], raw[row, 1, :len(intensity)] = mz, intensity
    batch = preprocess_peak_batch_numpy(
        raw, np.asarray(precursor, np.float32), num_peaks=int(config['num_peaks']),
        peak_drop_min_intensity=float(config['peak_drop_min_intensity']),
        peak_ordering=str(config['peak_ordering']), max_precursor_mz=float(config['max_precursor_mz']),
        precursor_peak_exclusion_window_da=float(config['precursor_peak_exclusion_window_da']),
        min_peak_intensity=float(config['min_peak_intensity']))
    energy_value = np.asarray([0.0 if e is None else e for e in energy], np.float32)
    return {
        'peak_mz': batch['peak_mz'], 'peak_intensity': batch['peak_intensity'],
        'peak_valid_mask': batch['peak_valid_mask'], 'precursor_mz': batch['precursor_mz'],
        'collision_energy': np.clip(energy_value, 0.0, COLLISION_ENERGY_MAX) / COLLISION_ENERGY_MAX,
        'collision_energy_present': np.asarray([e is not None and e > 0.0 for e in energy], np.float32),
        'charge': np.asarray(charge, np.float32), 'charge_present': np.ones(len(spectra), np.float32),
        'polarity_id': np.asarray([polarity_id(a) for a in adduct], np.int32),
        'instrument_family_id': np.asarray([instrument_family_id(i) for i in instrument], np.int32),
    }


def file_sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_config(metadata_path: Path) -> dict:
    return json.loads(metadata_path.read_text())['task_contract']['config']


def prepare(args):
    config = load_config(args.jepa_metadata)
    args.out.mkdir(parents=True, exist_ok=True)
    arrays = {name: [] for name in INPUT_NAMES}
    raw_peak_counts = []
    started = time.monotonic()
    with (args.out / 'rows.tsv').open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(['row', *[f.lower() for f in ROW_FIELDS], 'collision_energy_used_ev', 'raw_peaks'])
        chunk, row = [], 0

        def flush():
            spectra = [(mz, intensity) for _, mz, intensity in chunk]
            energy = [single_collision_energy(h) for h, _, _ in chunk]
            packed = pack(spectra, [float(h['PRECURSOR_MZ']) for h, _, _ in chunk], energy,
                          [abs(int(h.get('CHARGE', '1'))) or 1 for h, _, _ in chunk],
                          [h.get('INSTRUMENT_TYPE', '') for h, _, _ in chunk],
                          [h.get('ADDUCT') or h.get('IONMODE', '') for h, _, _ in chunk], config)
            for name in INPUT_NAMES:
                arrays[name].append(packed[name])
            for (header, mz, _), e in zip(chunk, energy, strict=True):
                writer.writerow([len(raw_peak_counts), *[header.get(f, '') for f in ROW_FIELDS],
                                 '' if e is None else e, len(mz)])
                raw_peak_counts.append(len(mz))
            chunk.clear()

        for record in read_mgf(args.mgf):
            chunk.append(record)
            row += 1
            if len(chunk) == CHUNK:
                flush()
                if row % (CHUNK * 25) == 0:
                    print(json.dumps({'stage': 'prepare', 'rows': row, 'seconds': time.monotonic() - started}), flush=True)
        if chunk:
            flush()
    data = {name: np.concatenate(values) for name, values in arrays.items()}
    np.savez(args.out / 'prepared_inputs.npz', **data)
    summary = {
        'mgf': str(args.mgf), 'mgf_sha256': file_sha256(args.mgf), 'samples': len(raw_peak_counts),
        # preprocessing marks slot 0 valid (with zero intensity) when no peak survives
        'spectra_without_kept_peaks': int((data['peak_intensity'].max(1) == 0).sum()),
        'kept_peaks_mean': float(data['peak_valid_mask'].sum(1).mean()),
        'raw_peaks_over_num_peaks': int((np.asarray(raw_peak_counts) > int(config['num_peaks'])).sum()),
        'collision_energy_present_samples': int(data['collision_energy_present'].sum()),
        'polarity_counts': {str(k): int(v) for k, v in zip(*np.unique(data['polarity_id'], return_counts=True))},
        'instrument_family_counts': {str(k): int(v) for k, v in zip(*np.unique(data['instrument_family_id'], return_counts=True))},
        'charge_counts': {str(k): int(v) for k, v in zip(*np.unique(data['charge'], return_counts=True))},
        'preprocessing': {k: config[k] for k in ('num_peaks', 'peak_ordering', 'peak_drop_min_intensity',
                          'min_peak_intensity', 'precursor_peak_exclusion_window_da', 'max_precursor_mz')},
        'seconds': time.monotonic() - started,
    }
    (args.out / 'preparation_metadata.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def parity(args):
    """The MGF packing path reproduces the prepared MassSpecGym arrays on their own TSV rows."""
    import pyarrow.csv as pacsv
    config = load_config(args.jepa_metadata)
    rows = np.arange(args.parity_rows)
    table = pacsv.read_csv(args.msg_tsv, parse_options=pacsv.ParseOptions(delimiter='\t'),
                           convert_options=pacsv.ConvertOptions(include_columns=[
                               'mzs', 'intensities', 'precursor_mz', 'collision_energy', 'instrument_type', 'adduct']))
    table = table.take(rows).to_pydict()
    spectra = [(np.fromstring(m, sep=',', dtype=np.float32), np.fromstring(i, sep=',', dtype=np.float32))
               for m, i in zip(table['mzs'], table['intensities'], strict=True)]
    mine = pack(spectra, table['precursor_mz'], table['collision_energy'], np.ones(len(rows)),
                table['instrument_type'], table['adduct'], config)
    report = {'rows': len(rows)}
    with np.load(args.msg_prepared, allow_pickle=False) as saved:
        reference = {name: saved[name][rows] for name in INPUT_NAMES}
    for name in INPUT_NAMES:
        report[name] = int((mine[name].reshape(len(rows), -1) != reference[name].reshape(len(rows), -1)).any(1).sum())
    # np.argpartition breaks top-k intensity ties by CPU-dependent SIMD paths, so a row whose
    # 127th intensity is tied may keep a different (equally intense) peak than the preparation host.
    differing = np.flatnonzero((mine['peak_mz'] != reference['peak_mz']).any(1)
                               | (mine['peak_intensity'] != reference['peak_intensity']).any(1))
    tie_swaps = [int(r) for r in differing
                 if np.array_equal(mine['peak_valid_mask'][r], reference['peak_valid_mask'][r])
                 and np.array_equal(np.sort(mine['peak_intensity'][r]), np.sort(reference['peak_intensity'][r]))]
    report['topk_tie_swaps'] = tie_swaps
    report['passed'] = len(tie_swaps) == len(differing) and all(
        v == 0 for k, v in report.items() if k in INPUT_NAMES and k not in ('peak_mz', 'peak_intensity'))
    print(json.dumps({'stage': 'parity', **report}), flush=True)
    if not report['passed']:
        raise ValueError(f'MGF packing differs from the prepared MassSpecGym arrays: {report}')
    return report


def shard(args):
    import jax
    from scripts.export_retrieval_tokens import BATCH_SIZE, build_token_encoder, restore_encoder
    jax.config.update('jax_default_matmul_precision', 'highest')
    config = load_config(args.jepa_metadata)
    with np.load(args.out / 'prepared_inputs.npz', allow_pickle=False) as saved:
        data = {name: saved[name] for name in INPUT_NAMES}
    n = len(data['peak_valid_mask'])
    stop = min(args.stop, n)
    encoder = restore_encoder(config, args.jepa_checkpoint)
    encode = build_token_encoder(config['encoder_metadata_schema'])

    def run(source, rows):
        batch = []
        for name in INPUT_NAMES:
            value = np.zeros((BATCH_SIZE, *source[name].shape[1:]), source[name].dtype)
            value[:len(rows)] = source[name][rows]
            batch.append(value)
        return np.asarray(encode(encoder, *batch), dtype=np.float16)[:len(rows)]

    report = {'start': args.start, 'stop': stop}
    if args.start == 0 and args.msg_prepared is not None:
        with np.load(args.msg_prepared, allow_pickle=False) as saved:
            msg = {name: saved[name][:8] for name in INPUT_NAMES}
        full = run(msg, np.arange(8)).astype(np.float32)
        actual, cls = full[:, :-1], full[:, -1]
        expected = np.load(args.msg_cached_tokens, mmap_mode='r')[:8].astype(np.float32)
        expected_cls = np.load(args.msg_cached_tokens.with_name('cls_embeddings.float16.npy'), mmap_mode='r')[:8]
        valid = msg['peak_valid_mask']
        relative = float(np.linalg.norm(actual[valid] - expected[valid]) / np.linalg.norm(expected[valid]))
        relative_cls = float(np.linalg.norm(cls - expected_cls) / np.linalg.norm(expected_cls.astype(np.float32)))
        report['restore_parity_relative_l2'] = relative
        report['restore_parity_cls_relative_l2'] = relative_cls
        print(json.dumps({'stage': 'restore_parity', 'relative_l2': relative, 'cls_relative_l2': relative_cls}), flush=True)
        if not np.isfinite(relative) or max(relative, relative_cls) > 0.005:
            raise ValueError(f'Restored encoder does not reproduce cached MSG tokens: {relative}, {relative_cls}')
    parts = args.out / '_parts'
    parts.mkdir(exist_ok=True)
    target = parts / f'{args.start:07d}-{stop:07d}.bin'
    started = time.monotonic()
    with (parts / (target.name + '.partial')).open('wb') as stream:
        for begin in range(args.start, stop, BATCH_SIZE):
            end = min(begin + BATCH_SIZE, stop)
            values = run(data, np.arange(begin, end))
            if values.shape[1:] != (int(config['num_peaks']) + 1, int(config['model_dim'])) or not np.isfinite(values).all():
                raise ValueError(f'Unexpected token batch at row {begin}: {values.shape}')
            stream.write(values.tobytes())
            if ((begin - args.start) // BATCH_SIZE + 1) % 256 == 0 or end == stop:
                print(json.dumps({'stage': 'shard', 'rows': end, 'stop': stop,
                                  'seconds': time.monotonic() - started}), flush=True)
    (parts / (target.name + '.partial')).replace(target)
    report.update({'bytes': target.stat().st_size, 'seconds': time.monotonic() - started,
                   'device': jax.devices()[0].device_kind, 'jax_version': jax.__version__})
    (parts / (target.name + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    return report


def assemble(args):
    config = load_config(args.jepa_metadata)
    with np.load(args.out / 'prepared_inputs.npz', allow_pickle=False) as saved:
        mask = saved['peak_valid_mask']
    n, slots, dim = len(mask), mask.shape[1] + 1, int(config['model_dim'])
    row_bytes = slots * dim * 2
    reports = sorted((json.loads(p.read_text()) for p in (args.parts or args.out / '_parts').glob('*.bin.json')),
                     key=lambda r: r['start'])
    if [r['start'] for r in reports] != [0, *[r['stop'] for r in reports[:-1]]] or reports[-1]['stop'] != n:
        raise ValueError(f'Parts do not tile [0, {n}): {[(r["start"], r["stop"]) for r in reports]}')
    parts_dir = args.parts or args.out / '_parts'
    target = args.out / 'tokens_f16.npy'
    with (args.out / 'tokens_f16.npy.partial').open('wb') as stream:
        np.lib.format.write_array_header_1_0(stream, {'descr': '<f2', 'fortran_order': False, 'shape': (n, slots, dim)})
        for report in reports:
            part = parts_dir / f"{report['start']:07d}-{report['stop']:07d}.bin"
            if part.stat().st_size != (report['stop'] - report['start']) * row_bytes:
                raise ValueError(f'Part {part} has the wrong size')
            with part.open('rb') as source:
                while block := source.read(1 << 28):
                    stream.write(block)
    (args.out / 'tokens_f16.npy.partial').replace(target)
    np.save(args.out / 'token_valid_mask.npy', np.concatenate([mask, np.ones((n, 1), bool)], axis=1))
    tokens = np.load(target, mmap_mode='r')
    probe = np.asarray(tokens[np.linspace(0, n - 1, 64).astype(int)], np.float32)
    if not np.isfinite(probe).all():
        raise ValueError('Non-finite tokens in assembled array')
    preparation = json.loads((args.out / 'preparation_metadata.json').read_text())
    shard0 = reports[0]
    metadata = {
        'cls_token_index': slots - 1, 'token_layout': f'{slots - 1} peak slots followed by CLS',
        'peak_slots': slots - 1, 'peak_ordering': config['peak_ordering'],
        'embedding_shape': [n, slots, dim], 'embedding_dtype': 'float16',
        'valid_mask_shape': [n, slots], 'valid_mask_dtype': 'bool', 'samples': n,
        'row_order': 'BEGIN IONS blocks in MGF file order; see rows.tsv (title = SPECTRAVERSE id)',
        'encoder': 'student', 'encoder_runtime': 'JAX', 'jax_version': shard0['jax_version'],
        'device': shard0['device'], 'batch_size': 128,
        'restore_parity_relative_l2_vs_msg_cache_first8': shard0.get('restore_parity_relative_l2'),
        'restore_parity_cls_relative_l2_vs_msg_cache_first8': shard0.get('restore_parity_cls_relative_l2'),
        'collision_energy_policy': 'eV from COLLISION_ENERGY_1..3 when exactly one finite value > 0, else absent',
        'charge_policy': 'abs(CHARGE)', 'polarity_source': 'ADDUCT',
        'extraction_seconds_sum': float(sum(r['seconds'] for r in reports)),
        **{f'preparation_{k}': v for k, v in preparation.items() if k != 'seconds'},
        'status': 'complete', **args.extra_metadata,
    }
    (args.out / 'embedding_metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'parity', 'shard', 'assemble'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jepa-metadata', type=Path, required=True)
    parser.add_argument('--jepa-checkpoint', type=Path)
    parser.add_argument('--mgf', type=Path)
    parser.add_argument('--msg-tsv', type=Path)
    parser.add_argument('--msg-prepared', type=Path, help='prepared_massspecgym.npz used for the cached MSG export')
    parser.add_argument('--msg-cached-tokens', type=Path, help='covariance_embeddings.float16.npy (peak tokens)')
    parser.add_argument('--parity-rows', type=int, default=2048)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=sys.maxsize)
    parser.add_argument('--parts', type=Path, help='directory holding the downloaded part files (assemble)')
    parser.add_argument('--extra-metadata', type=json.loads, default={})
    args = parser.parse_args(argv)
    return {'prepare': prepare, 'parity': parity, 'shard': shard, 'assemble': assemble}[args.stage](args)


if __name__ == '__main__':
    main()
