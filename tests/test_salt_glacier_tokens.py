import hashlib

import numpy as np
import pytest

from scripts.export_salt_glacier_tokens import TokenWriter, padded_rows, restore_continuation_encoder, validate_row_mapping


def test_writer_resume_verifies_prefix_and_never_publishes_partial(tmp_path):
    values = np.arange(30, dtype=np.float16).reshape(5, 3, 2)
    contract = {'arm': 'test', 'encoder_parameter_sha256': 'pinned'}
    writer = TokenWriter(tmp_path, values.shape, contract, resume=False)
    writer.append(0, values[:2]); writer.flush()
    assert not writer.final.exists()
    assert writer.partial.stat().st_size < writer.offset + values.nbytes
    with pytest.raises(ValueError, match='incomplete'):
        writer.finish()
    del writer
    with pytest.raises(ValueError, match='changed inputs'):
        TokenWriter(tmp_path, values.shape, {**contract, 'arm': 'changed'}, resume=True)
    resumed = TokenWriter(tmp_path, values.shape, contract, resume=True)
    assert resumed.rows_written == 2
    resumed.append(2, values[2:])
    result = resumed.finish()
    np.testing.assert_array_equal(np.load(tmp_path / 'covariance_embeddings.float16.npy'), values)
    assert result['raw_array_bytes'] == values.nbytes
    assert result['chunks'] == 2
    # Recovery from a crash after array publication, before the final completion marker.
    recovered = TokenWriter(tmp_path, values.shape, contract, resume=True)
    assert recovered.finish()['chunk_inventory_sha256'] == result['chunk_inventory_sha256']
    corrupted = np.load(tmp_path / 'covariance_embeddings.float16.npy', mmap_mode='r+')
    corrupted[0, 0, 0] = 100; corrupted.flush()
    with pytest.raises(ValueError, match='chunk changed'):
        TokenWriter(tmp_path, values.shape, contract, resume=True)


def test_row_mapping_and_fixed_padding(tmp_path):
    original = tmp_path / 'original'; data = original / 'data'; data.mkdir(parents=True)
    prepared = tmp_path / 'prepared'; prepared.mkdir()
    fold = np.array(['train', 'val', 'train', 'test', 'val', 'test'])
    smiles = np.array(['C', 'CC', 'CCC', 'O', 'CO', 'CCO'])
    keys = np.array([f'{i:014d}-AAAAAAAAAA-N' for i in range(6)])
    np.save(data / 'fold.npy', fold)
    msg = original / 'msg15'; msg.mkdir()
    (msg / 'MassSpecGym1.5.tsv').write_text('smiles\tinchikey\tfold\n' + ''.join(
        f'{a}\t{b}\t{c}\n' for a, b, c in zip(smiles, keys, fold, strict=True)))
    contents = dict(smiles=smiles, inchikey14=np.array([k[:14] for k in keys]),
                    train=np.array([0, 2]), val=np.array([4]), test=np.array([3, 5]))
    np.savez(prepared / 'rows.npz', **contents)
    validation_hash = hashlib.sha256(contents['val'].tobytes()).hexdigest()
    np.testing.assert_array_equal(validate_row_mapping(original, prepared, 6, validation_rows_sha256=validation_hash), [0, 2])
    with pytest.raises(ValueError, match='validation query order'):
        validate_row_mapping(original, prepared, 6, validation_rows_sha256='wrong')
    contents['train'] = np.array([2, 0])
    np.savez(prepared / 'rows.npz', **contents)
    with pytest.raises(AssertionError):
        validate_row_mapping(original, prepared, 6, validation_rows_sha256=validation_hash)
    contents['train'] = np.array([0, 2])
    packed = np.arange(36, dtype=np.float32).reshape(6, 3, 2)
    padded = padded_rows(packed, np.array([2, 0]), batch_size=4)
    np.testing.assert_array_equal(padded[:2], packed[[2, 0]])
    assert not padded[2:].any()
    contents['smiles'] = smiles[::-1]
    np.savez(prepared / 'rows.npz', **contents)
    with pytest.raises(AssertionError):
        validate_row_mapping(original, prepared, 6, validation_rows_sha256=validation_hash)


def test_partial_checkpoint_restores_only_encoder_tree(tmp_path):
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    original = {'block': {'weight': jnp.zeros((2, 3)), 'bias': jnp.ones(3)}}
    changed = {'block': {'weight': jnp.arange(6, dtype=jnp.float32).reshape(2, 3), 'bias': jnp.zeros(3)}}
    checkpoint = tmp_path / 'checkpoint'
    with ocp.PyTreeCheckpointer() as saver:
        saver.save(str(checkpoint), args=ocp.args.PyTreeSave({
            'params': {'encoder': changed, 'predictor': {'unused': jnp.ones((4, 4))}},
            'head': {'unused': jnp.ones(2)}}))
    restored = restore_continuation_encoder(original, checkpoint)
    assert set(restored) == {'block'}
    for key in ('weight', 'bias'):
        np.testing.assert_array_equal(restored['block'][key], changed['block'][key])
    np.testing.assert_array_equal(original['block']['weight'], np.zeros((2, 3)))
