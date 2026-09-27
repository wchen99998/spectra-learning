"""Frozen continuation tokens in the exact GLACIER input schema and row order."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

ARMS = ('baseline_final_random', 'encoder_aux_final_random')
BATCH_SIZE = 128
VALIDATION_ROWS_SHA256 = '0959c5bbedf2c7efee626c49cf279d869edd20470297bf67420437ca5da0106b'
TOKEN_NAME = 'covariance_embeddings.float16.npy'


def file_sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def parameter_digest(tree) -> str:
    """Match the accepted continuation's path-and-raw-array checkpoint identity."""
    import jax
    digest = hashlib.sha256()
    for key, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        digest.update(str(key).encode())
        digest.update(np.asarray(jax.device_get(value)).tobytes())
    return digest.hexdigest()


def imported_source_provenance(expected_root: Path | None = None) -> tuple[dict, dict]:
    """Pin the helper and model implementation actually selected by PYTHONPATH."""
    from scripts import prepare_atom_flare_jax, trainable_jepa_peak
    import spectra_learning

    paths = {'export_salt_glacier_tokens.py': Path(__file__).resolve(),
             'trainable_jepa_peak.py': Path(trainable_jepa_peak.__file__).resolve(),
             'prepare_atom_flare_jax.py': Path(prepare_atom_flare_jax.__file__).resolve()}
    package_root = Path(spectra_learning.__file__).resolve().parent
    if expected_root is not None:
        expected_root = expected_root.resolve(strict=True)
        for name in ('trainable_jepa_peak.py', 'prepare_atom_flare_jax.py'):
            if paths[name] != expected_root / 'scripts' / name:
                raise ValueError(f'Imported {name} is outside the pinned source tree: {paths[name]}')
        if package_root != expected_root / 'spectra_learning':
            raise ValueError(f'Imported model package is outside the pinned source tree: {package_root}')
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    hashes.update({f'spectra_learning/{path.relative_to(package_root)}': file_sha256(path)
                   for path in sorted(package_root.rglob('*.py'))})
    return hashes, {**{name: str(path) for name, path in paths.items()},
                    'spectra_learning': str(package_root)}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def padded_rows(packed: np.ndarray, rows: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
    if len(rows) > batch_size or np.any(rows < 0) or np.any(rows >= len(packed)):
        raise ValueError('Invalid input rows for the fixed inference batch')
    result = np.zeros((batch_size, *packed.shape[1:]), packed.dtype)
    result[:len(rows)] = packed[rows]
    return result


def validate_row_mapping(input_root: Path, prepared: Path, count: int, *,
                         validation_rows_sha256: str = VALIDATION_ROWS_SHA256) -> np.ndarray:
    """The existing frozen cache and molecular graph selection share exact rows."""
    with np.load(prepared / 'rows.npz', allow_pickle=False) as saved:
        fold = np.load(input_root / 'data/fold.npy', allow_pickle=True).astype(str)
        from scripts.prepare_atom_flare_jax import read_rows
        original_rows = read_rows(input_root)
        smiles, keys = original_rows['smiles'], original_rows['inchikey']
        np.testing.assert_array_equal(original_rows['fold'], fold)
        if len(fold) != count:
            raise ValueError('Prepared spectrum count differs from frozen cache')
        np.testing.assert_array_equal(saved['smiles'].astype(str), smiles)
        np.testing.assert_array_equal(saved['inchikey14'].astype(str), np.array([value[:14] for value in keys]))
        for name in ('train', 'val', 'test'):
            rows = saved[name]
            if rows.ndim != 1 or len(np.unique(rows)) != len(rows) or np.any((rows < 0) | (rows >= count)):
                raise ValueError(f'Invalid prepared {name} rows')
            if not np.all(fold[rows] == name):
                raise ValueError(f'Prepared {name} rows do not match original folds')
        # Validation is the fixed5000-query subset; training/test retain all original rows.
        np.testing.assert_array_equal(saved['train'], np.flatnonzero(fold == 'train'))
        np.testing.assert_array_equal(saved['test'], np.flatnonzero(fold == 'test'))
        if array_sha256(saved['val']) != validation_rows_sha256:
            raise ValueError('Historical validation query order changed')
        return saved['train'].copy()


def restore_continuation_encoder(initial_parameters, checkpoint: Path):
    """Restore only trained encoder leaves, retaining original fixed Fourier buffers."""
    import jax
    import orbax.checkpoint as ocp

    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    target = {'params': {'encoder': jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding), initial_parameters)}}
    restore_args = jax.tree.map(lambda value: ocp.ArrayRestoreArgs(
        sharding=sharding, global_shape=value.shape, dtype=value.dtype), target)
    with ocp.PyTreeCheckpointer() as checkpointer:
        saved = checkpointer.restore(str(checkpoint.resolve()), args=ocp.args.PyTreeRestore(
            item=target, restore_args=restore_args, partial_restore=True))
    return saved['params']['encoder']


def check_initial_parity(encode, parameters, packed, mask, input_root, training_rows):
    rows = np.asarray(training_rows[:8])
    actual = np.asarray(encode(parameters, padded_rows(packed, rows)), np.float16)[:len(rows)]
    expected = np.asarray(np.load(input_root / TOKEN_NAME, mmap_mode='r')[rows])
    if not np.isfinite(actual).all() or actual.dtype != expected.dtype:
        raise ValueError('Invalid original-encoder parity outputs')
    difference = actual.astype(np.float32) - expected.astype(np.float32)
    result = {'rows': rows.tolist(), 'batch_size': BATCH_SIZE,
              'all_slots_bitwise_equal': array_sha256(actual) == array_sha256(expected),
              'valid_slots_bitwise_equal': array_sha256(actual[mask[rows]]) == array_sha256(expected[mask[rows]]),
              'max_absolute_error': float(np.abs(difference).max()),
              'actual_sha256': array_sha256(actual), 'expected_sha256': array_sha256(expected)}
    if not result['all_slots_bitwise_equal']:
        raise ValueError(f'Original encoder does not reproduce frozen cache bitwise: {result}')
    return result


def link_original_inputs(output: Path, input_root: Path) -> None:
    for name in ('data', 'msg15'):
        source = (input_root / name).resolve(strict=True)
        target = output / name
        if target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f'Output {name} points to different original inputs')
        elif target.exists():
            raise ValueError(f'Expected an original-input symlink: {target}')
        else:
            target.symlink_to(source, target_is_directory=True)


class TokenWriter:
    """Sequential NPY publication with a durable, hash-verified resumable prefix."""

    def __init__(self, root: Path, shape: tuple[int, ...], contract: dict, *, resume: bool):
        self.root, self.shape, self.contract = root, shape, contract
        self.partial = root / (TOKEN_NAME + '.partial')
        self.final = root / TOKEN_NAME
        self.progress_path = root / 'export_progress.json'
        self.rows_written, self.chunks = 0, []
        if self.progress_path.exists():
            if not resume:
                raise FileExistsError(self.progress_path)
            progress = json.loads(self.progress_path.read_text())
            if progress['contract'] != contract:
                raise ValueError('Cannot resume an export with changed inputs, parameters, or source')
            self.rows_written, self.chunks = progress['rows_written'], progress['chunks']
            if self.final.exists() and self.rows_written != shape[0]:
                raise ValueError('Published array has an incomplete progress record')
            path = self.final if self.final.exists() else self.partial
            self.handle = path.open('r+b')
            if np.lib.format.read_magic(self.handle) != (1, 0):
                raise ValueError('Unexpected partial NPY version')
            stored_shape, fortran, dtype = np.lib.format.read_array_header_1_0(self.handle)
            self.offset = self.handle.tell()
            if stored_shape != shape or fortran or dtype != np.float16:
                raise ValueError('Incomplete array shape/dtype changed')
            previous = 0
            for chunk in self.chunks:
                if chunk['start'] != previous or not previous < chunk['end'] <= self.rows_written:
                    raise ValueError('Noncontiguous recorded export prefix')
                size = (chunk['end'] - previous) * int(np.prod(shape[1:])) * 2
                payload = self.handle.read(size)
                if len(payload) != size or hashlib.sha256(payload).hexdigest() != chunk['sha256']:
                    raise ValueError(f'Committed token chunk changed at row {previous}')
                previous = chunk['end']
            if previous != self.rows_written:
                raise ValueError('Export progress and token chunk ledger disagree')
            if path == self.partial:
                self.handle.truncate(self.handle.tell())
            elif self.handle.read(1):
                raise ValueError('Published token file has unexpected trailing bytes')
        else:
            if self.partial.exists() or self.final.exists():
                raise FileExistsError('Token array exists without a committed progress record')
            self.handle = self.partial.open('w+b')
            np.lib.format.write_array_header_1_0(self.handle, {'descr': np.dtype(np.float16).str, 'fortran_order': False, 'shape': shape})
            self.offset = self.handle.tell()
            self.flush()

    def append(self, start: int, values: np.ndarray) -> None:
        values = np.asarray(values)
        if (start != self.rows_written or values.dtype != np.float16
                or values.shape[1:] != self.shape[1:] or start + len(values) > self.shape[0]
                or not len(values) or not np.isfinite(values).all()):
            raise ValueError('Token output must be finite float16, sequential, and shape-aligned')
        self.handle.write(memoryview(np.ascontiguousarray(values)).cast('B'))
        self.chunks.append({'start': start, 'end': start + len(values), 'sha256': array_sha256(values)})
        self.rows_written += len(values)

    def flush(self) -> None:
        if self.partial.exists():
            self.handle.flush()
            os.fsync(self.handle.fileno())
        atomic_json(self.progress_path, {'contract': self.contract, 'rows_written': self.rows_written, 'chunks': self.chunks})

    def finish(self) -> dict:
        if self.rows_written != self.shape[0]:
            raise ValueError('Cannot publish an incomplete token cache')
        self.flush()
        self.handle.close()
        if self.partial.exists():
            self.partial.replace(self.final)
        inventory = {'shape': list(self.shape), 'dtype': 'float16', 'order': 'C', 'chunks': self.chunks,
                     'hash_scheme': 'Each SHA256 covers raw C-order float16 array bytes for [start,end); NPY header is excluded.'}
        atomic_json(self.root / 'token_chunks.json', inventory)
        return {'chunks': len(self.chunks), 'raw_array_bytes': int(np.prod(self.shape)) * 2,
                'file_bytes': self.final.stat().st_size, 'chunk_inventory_sha256': file_sha256(self.root / 'token_chunks.json'),
                'hash_scheme': inventory['hash_scheme']}


def export_tokens(input_root: Path, raw_data: Path, jepa_metadata: Path, initial_checkpoint: Path,
                  continuation_dir: Path, output: Path, *, prepared: Path,
                  smoke_rows: int | None = None, resume: bool = False, commit=None,
                  import_source_root: Path | None = None) -> dict:
    """Export one prespecified continuation, or a separate training-only smoke."""
    import jax
    from scripts.trainable_jepa_peak import load_spectra, restore_peak

    started = time.monotonic()
    arm = continuation_dir.name
    if arm not in ARMS or (smoke_rows is not None and smoke_rows < 1):
        raise ValueError('Only the two prespecified original-stream checkpoints are allowed')
    original = json.loads((input_root / 'embedding_metadata.json').read_text())
    if (original['batch_size'] != BATCH_SIZE or original['embedding_shape'][1:] != [127, 2048]
            or original['embedding'] != 'valid_peak_tokens' or original['embedding_dtype'] != 'float16'):
        raise ValueError('Original GLACIER token contract changed')
    execution = json.loads((continuation_dir / 'execution.json').read_text())
    if execution['seed'] != 20260921 or execution['steps'] != 1024 or execution.get('smoke'):
        raise ValueError('Expected the prespecified original1024-update continuation')
    source_hashes, source_paths = imported_source_provenance(import_source_root)
    inputs = [input_root / 'embedding_metadata.json', jepa_metadata, raw_data / 'preparation_metadata.json',
              raw_data / 'prepared_massspecgym.npz', prepared / 'rows.npz', continuation_dir / 'execution.json']
    inputs += [input_root / 'data' / name for name in ('fold.npy', 'token_valid_mask.npy')]
    inputs += [input_root / 'msg15/MassSpecGym1.5_retrieval_candidates_formula.json', input_root / 'msg15/MassSpecGym1.5.tsv']
    input_hashes = {str(path): file_sha256(path) for path in inputs}
    if input_hashes[str(input_root / 'msg15/MassSpecGym1.5.tsv')] != original['tsv_sha256']:
        raise ValueError('Original TSV content changed')
    expected = execution['parameter_sha256']['final_encoder']
    contract = {'arm': arm, 'shape': original['embedding_shape'], 'batch_size': BATCH_SIZE,
                'dtype': 'float16', 'encoder_parameter_sha256': expected, 'initial_checkpoint': str(initial_checkpoint),
                'continuation_checkpoint': str(continuation_dir / 'trainable_checkpoint'),
                'input_provenance': input_hashes, 'source_sha256': source_hashes,
                'source_paths': source_paths,
                'smoke_rows': smoke_rows}
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / 'completion.json'
    if completion_path.exists():
        saved = json.loads(completion_path.read_text())
        if saved['contract'] != contract or saved['embedding_metadata_sha256'] != file_sha256(output / 'embedding_metadata.json'):
            raise ValueError('Completed immutable export does not match requested inputs')
        if (saved['status'] != 'complete' or saved['file_bytes'] != (output / TOKEN_NAME).stat().st_size
                or saved['chunk_inventory_sha256'] != file_sha256(output / 'token_chunks.json')):
            raise ValueError('Completed export inventory changed')
        checked = TokenWriter(output, tuple(original['embedding_shape']), contract, resume=True)
        checked.handle.close()
        for name in ('data', 'msg15'):
            if not (output / name).is_symlink() or (output / name).resolve() != (input_root / name).resolve():
                raise ValueError('Completed export original-input links changed')
        return saved
    jax.config.update('jax_default_matmul_precision', 'highest')
    packed, mask, _ = load_spectra(input_root, raw_data, jepa_metadata)
    train_rows = validate_row_mapping(input_root, prepared, len(packed))
    peak = restore_peak(SimpleNamespace(jepa_metadata=jepa_metadata, jepa_checkpoint=initial_checkpoint, d=256, dropout=.1))
    initial = peak.initial_parameters
    if parameter_digest(initial) != execution['parameter_sha256']['initial_encoder']:
        raise ValueError('Original encoder differs from the continuation initialization')
    encode = jax.jit(peak.encode)
    parity = check_initial_parity(encode, initial, packed, mask, input_root, train_rows)
    print(f'Original-token bitwise parity passed: {arm}', flush=True)
    parameters = restore_continuation_encoder(initial, continuation_dir / 'trainable_checkpoint')
    if parameter_digest(parameters) != expected:
        raise ValueError('Restored continuation encoder hash differs from the accepted endpoint')
    peak.initial_parameters = None
    del initial
    common = {'arm': arm, 'contract': contract, 'encoder_parameter_sha256': expected,
              'initial_token_parity': parity, 'batch_size': BATCH_SIZE,
              'input_provenance': input_hashes, 'source_sha256': source_hashes,
              'source_paths': source_paths,
              'versions': {name: importlib.metadata.version(name) for name in ('jax', 'flax', 'numpy', 'orbax-checkpoint')}}
    if smoke_rows is not None:
        if smoke_rows > len(train_rows):
            raise ValueError('Smoke exceeds the training population')
        rows = train_rows[:smoke_rows]
        pieces = []
        for start in range(0, len(rows), BATCH_SIZE):
            selected = rows[start:start + BATCH_SIZE]
            pieces.append(np.asarray(encode(parameters, padded_rows(packed, selected)), np.float16)[:len(selected)])
        values = np.concatenate(pieces)
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite continuation smoke tokens')
        np.savez_compressed(output / 'smoke_tokens.npz', rows=rows, tokens=values, valid_mask=mask[rows])
        result = {**common, 'status': 'smoke_complete', 'samples': len(rows), 'shape': list(values.shape),
                  'smoke_tokens_sha256': file_sha256(output / 'smoke_tokens.npz'), 'elapsed_seconds': time.monotonic() - started}
        atomic_json(output / 'smoke.json', result)
        if commit is not None:
            commit()
        return result
    shape = tuple(original['embedding_shape'])
    if shape[:2] != packed.shape[:2]:
        raise ValueError('Original cache shape differs from prepared spectra')
    link_original_inputs(output, input_root)
    writer = TokenWriter(output, shape, contract, resume=resume)
    count = len(packed)
    for start in range(writer.rows_written, count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, count)
        values = np.asarray(encode(parameters, padded_rows(packed, np.arange(start, end))), np.float16)[:end - start]
        writer.append(start, values)
        if (start // BATCH_SIZE + 1) % 32 == 0 or end == count:
            writer.flush()
            if commit is not None and ((start // BATCH_SIZE + 1) % 256 == 0 or end == count):
                commit()
            print(f'Exported {arm}: {end}/{count}', flush=True)
    hashes = writer.finish()
    metadata = {**original, 'checkpoint': str(continuation_dir / 'trainable_checkpoint'),
                'checkpoint_role': arm, 'checkpoint_step': None, 'initial_checkpoint': str(initial_checkpoint),
                'initial_checkpoint_step': 1190000, 'continuation_steps': 1024,
                'encoder_parameter_sha256': expected, 'also_saved_final_cls': False,
                'extraction_seconds': time.monotonic() - started, 'total_seconds': time.monotonic() - started,
                'preprocessing_seconds': None, 'source_embedding_root': str(input_root),
                'embedding_complete': True, 'jax_version': jax.__version__,
                'jax_devices': [str(device) for device in jax.devices()]}
    atomic_json(output / 'embedding_metadata.json', metadata)
    result = {**common, 'status': 'complete', 'samples': count, 'shape': list(shape), 'dtype': 'float16',
              'embedding_metadata_sha256': file_sha256(output / 'embedding_metadata.json'), **hashes,
              'elapsed_seconds': time.monotonic() - started,
              'shared_inputs': {name: str((output / name).resolve()) for name in ('data', 'msg15')}}
    atomic_json(completion_path, result)
    (output / 'executed_exporter.py').write_bytes(Path(__file__).read_bytes())
    if commit is not None:
        commit()
    return result


def main(argv: list[str] | None = None) -> dict:
    """Run in an isolated process with the pinned archived tree on PYTHONPATH."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-root', 'raw-data', 'jepa-metadata', 'initial-checkpoint',
                 'continuation-dir', 'output', 'prepared', 'import-source-root'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--smoke-rows', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    result = export_tokens(args.input_root, args.raw_data, args.jepa_metadata,
                           args.initial_checkpoint, args.continuation_dir, args.output,
                           prepared=args.prepared, smoke_rows=args.smoke_rows, resume=args.resume,
                           import_source_root=args.import_source_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == '__main__':
    main()
