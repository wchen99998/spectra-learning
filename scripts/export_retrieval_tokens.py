"""Export a pinned replacement student for the fixed retrieval recipe."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from scripts.export_salt_glacier_tokens import (
    BATCH_SIZE, TokenWriter, atomic_json, array_sha256,
    file_sha256, link_original_inputs, parameter_digest, validate_row_mapping,
)
from spectra_learning.models.model_jax import build_peak_set_encoder_jax
from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.training.checkpointing_jax import restore_jax_encoder_state


INPUT_NAMES = ('peak_mz', 'peak_intensity', 'peak_valid_mask', 'precursor_mz',
               'collision_energy', 'collision_energy_present', 'charge', 'charge_present',
               'polarity_id', 'instrument_family_id')


def restore_encoder(config: dict, checkpoint: Path):
    import spectra_learning.models.encoder_jax as encoder_module
    encoder_module.MASSIVE_V2_CONDITION_DIM = 28
    settings = PeakSetJEPASettings.from_config(config)
    encoder = nnx.eval_shape(lambda: build_peak_set_encoder_jax(
        settings, compute_dtype=jnp.bfloat16, pairmixer_block_type=settings.pairmixer_block_type,
        pairmixer_fast_encoder_max_visible_tokens=None, rngs=nnx.Rngs(int(config['seed']))))
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    template = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
                            nnx.as_pure(nnx.state(encoder)))
    nnx.update(encoder, restore_jax_encoder_state(checkpoint, template))
    return encoder


def build_token_encoder(schema: str):
    # Match the original cache: metadata is computed inside the NNX inference
    # function; physical-token extraction and float16 rounding happen on host.
    @nnx.jit
    def encode(encoder, mz, intensity, valid, precursor, energy, energy_present,
               charge, charge_present, polarity, instrument):
        metadata = jax_spectrum_metadata_from_batch({
            'precursor_mz':precursor, 'collision_energy':energy, 'collision_energy_present':energy_present,
            'charge':charge, 'charge_present':charge_present, 'polarity_id':polarity,
            'instrument_family_id':instrument}, schema=schema)
        return encoder(mz, intensity, valid_mask=valid, visible_mask=valid,
                       precursor_mz=precursor, spectrum_metadata=metadata).astype(jnp.float32)
    return encode


def padded_inputs(data: dict, rows: np.ndarray):
    batch=[]
    for name in INPUT_NAMES:
        source=data[name]
        value=np.zeros((BATCH_SIZE,*source.shape[1:]),source.dtype)
        value[:len(rows)]=source[rows]
        batch.append(value)
    return tuple(batch)


def original_state_sha256(encoder):
    """Use the historical encoder audit's path/shape/dtype/bytes hash format."""
    digest=hashlib.sha256()
    for path,value in nnx.to_flat_state(nnx.as_pure(nnx.state(encoder))):
        value=np.ascontiguousarray(jax.device_get(value))
        header=json.dumps([list(map(str,path)),list(value.shape),str(value.dtype)],separators=(',',':')).encode()
        digest.update(header+b'\0')
        digest.update(memoryview(value.view(np.uint8)).cast('B'))
    return digest.hexdigest()


def validate_encoder_contract(original: dict, replacement: dict) -> dict:
    """Require the same encoder and preprocessing, independent of pretraining loss."""
    core = {"model_dim", "feature_mlp_hidden_dim", "attention_mlp_multiple", "norm_eps",
            "num_peaks", "autocast_dtype"}
    preprocessing = ("peak_ordering", "peak_drop_min_intensity", "min_peak_intensity",
                     "precursor_peak_exclusion_window_da", "max_precursor_mz")
    def contract(config):
        settings = asdict(PeakSetJEPASettings.from_config(config))
        result = {k: v for k, v in settings.items() if k in core or k.startswith(("encoder_", "pairmixer_"))}
        result['preprocessing'] = {k: config[k] for k in preprocessing}
        return json.loads(json.dumps(result))
    old, new = contract(original), contract(replacement)
    differences = {k: [old[k], new[k]] for k in old if old[k] != new[k]}
    if differences:
        raise ValueError(f'Encoder-only comparison changed architecture or preprocessing: {differences}')
    return new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference-inputs', type=Path, required=True)
    parser.add_argument('--checkpoint-selection', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    paths = {k: Path(v) for k, v in json.loads(args.inputs.read_text()).items()}
    reference = {k: Path(v) for k, v in json.loads(args.reference_inputs.read_text()).items()}
    for name in ('prepared', 'fragments', 'peak_fractions', 'jepa_data'):
        if paths[name] != reference[name]:
            raise ValueError(f'Encoder-only comparison changed shared input: {name}')
    metadata = json.loads(paths['jepa_metadata'].read_text())
    original_config = json.loads(reference['jepa_metadata'].read_text())['task_contract']['config']
    encoder_contract = validate_encoder_contract(original_config, metadata['task_contract']['config'])
    selection = json.loads(args.checkpoint_selection.read_text())
    if not selection['all_checks_passed'] or file_sha256(paths['jepa_metadata']) != selection['checkpoint_metadata_sha256']:
        raise ValueError('Staged encoder metadata differs from selected checkpoint')
    copy_record = json.loads((paths['jepa_checkpoint'] / 'checkpoint_copy.json').read_text())
    if not copy_record['all_checks_passed'] or copy_record['objects'] != selection['objects']:
        raise ValueError('Checkpoint copy differs from the pinned object inventory')
    output = paths['input_root']
    output.mkdir(parents=True, exist_ok=True)
    link_original_inputs(output, reference['input_root'])
    original = json.loads((reference['input_root'] / 'embedding_metadata.json').read_text())
    prepared=paths['jepa_data']/'prepared_massspecgym.npz'
    preparation=json.loads((paths['jepa_data']/'preparation_metadata.json').read_text())
    if preparation['tsv_sha256']!=original['tsv_sha256']:
        raise ValueError('Prepared spectra and original token source differ')
    with np.load(prepared,allow_pickle=False) as saved:
        data={name:saved[name] for name in INPUT_NAMES}
        fold=np.load(reference['input_root']/'data/fold.npy',allow_pickle=True).astype(str)
        np.testing.assert_array_equal(saved['split'],np.array([{'train':0,'val':1,'test':2}[x] for x in fold]))
    mask=data['peak_valid_mask']
    np.testing.assert_array_equal(mask,np.load(reference['input_root']/'data/token_valid_mask.npy')[:,:-1])
    training_rows = validate_row_mapping(reference['input_root'], paths['prepared'], len(mask))
    jax.config.update('jax_default_matmul_precision', 'highest')
    original_encoder=restore_encoder(original_config,reference['jepa_checkpoint'])
    encode=build_token_encoder(original_config['encoder_metadata_schema'])
    rows=training_rows[:8]
    actual=np.asarray(encode(original_encoder,*padded_inputs(data,rows)),np.float16)[:len(rows),:-1]
    expected=np.asarray(np.load(reference['input_root']/'covariance_embeddings.float16.npy',mmap_mode='r')[rows])
    valid=mask[rows]
    delta=actual[valid].astype(np.float32)-expected[valid].astype(np.float32)
    relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(expected[valid].astype(np.float32)))
    parity={'rows':rows.tolist(),'batch_size':BATCH_SIZE,'bitwise_equal':array_sha256(actual)==array_sha256(expected),
            'max_absolute_error':float(np.max(np.abs(actual.astype(np.float32)-expected.astype(np.float32)))),
            'actual_sha256':array_sha256(actual),'expected_sha256':array_sha256(expected),
            'valid_relative_l2':relative_l2,'valid_max_absolute_error':float(np.abs(delta).max()),
            'tolerance_relative_l2':.005,'tolerance_source':'scripts/trainable_jepa_peak.py:validate_initial_tokens',
            'original_state_sha256':original_state_sha256(original_encoder)}
    atomic_json(output/'original_encoder_parity.json',parity)
    if not np.isfinite(actual).all() or not np.isfinite(relative_l2) or relative_l2>.005:
        raise ValueError(f'Original export exceeds the accepted pipeline token tolerance: {parity}')
    print(json.dumps({'stage':'original_encoder_parity','passed':True,**parity}),flush=True)
    del original_encoder
    jax.clear_caches()
    encoder=restore_encoder(metadata['task_contract']['config'],paths['jepa_checkpoint'])
    fingerprint = parameter_digest(nnx.as_pure(nnx.state(encoder)))
    shape = tuple(original['embedding_shape'])
    contract = {'shape': list(shape), 'batch_size': BATCH_SIZE, 'dtype': 'float16',
                'checkpoint_selection_sha256': file_sha256(args.checkpoint_selection),
                'student_parameter_sha256': fingerprint, 'encoder_contract': encoder_contract,
                'prepared_spectra_sha256': file_sha256(prepared),
                'exporter_sha256': file_sha256(Path(__file__))}
    writer = TokenWriter(output, shape, contract, resume=args.resume)
    started = time.monotonic()
    try:
        for start in range(writer.rows_written, len(mask), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(mask))
            batch = padded_inputs(data, np.arange(start, end))
            values = np.asarray(encode(encoder, *batch), dtype=np.float16)[:end-start,:-1]
            writer.append(start, values)
            if (start // BATCH_SIZE + 1) % 64 == 0 or end == len(mask):
                writer.flush()
                print(json.dumps({'stage': 'export', 'rows': end, 'total': len(mask),
                                  'seconds': time.monotonic()-started}), flush=True)
        inventory = writer.finish()
    finally:
        writer.handle.close()
    embedding = {**original, 'run_id': selection['run_id'],
                 'run_url': f"https://wandb.ai/iclac/jepa-finalrun/runs/{selection['run_id']}",
                 'checkpoint': str(paths['jepa_checkpoint']), 'checkpoint_role': 'student',
                 'checkpoint_step': selection['selected_checkpoint_step'], 'encoder': 'student',
                 'config_metadata_path': str(paths['jepa_metadata']), 'batch_size': BATCH_SIZE,
                 'data_root': str(paths['jepa_data']), 'embedding_complete': True,
                 'checkpoint_selection_sha256': contract['checkpoint_selection_sha256'],
                 'student_parameter_sha256': fingerprint, 'source_embedding_root': str(reference['input_root']),
                 'extraction_seconds': time.monotonic()-started}
    atomic_json(output / 'embedding_metadata.json', embedding)
    atomic_json(output / 'completion.json', {'complete': True, 'contract': contract,
                'original_encoder_parity': parity, 'training_performed': False, **inventory})
    print(json.dumps({'stage':'export_complete', 'shape':shape, 'seconds':time.monotonic()-started}),flush=True)


if __name__ == '__main__':
    main()
