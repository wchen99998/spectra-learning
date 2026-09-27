import pytest

from scripts.export_retrieval_tokens import validate_encoder_contract


@pytest.fixture
def config():
    return {'model_dim': 2048, 'peak_ordering': 'mz', 'peak_drop_min_intensity': True,
            'min_peak_intensity': .0001, 'precursor_peak_exclusion_window_da': 0.,
            'max_precursor_mz': 1000.}


def test_encoder_swap_allows_different_pretraining_objective_and_optimizer(config):
    replacement = {**config, 'learning_rate': 1e-6, 'frozen_teacher_update_every_steps': 50000}
    assert validate_encoder_contract(config, replacement)['model_dim'] == 2048


@pytest.mark.parametrize(('name', 'value'), [('model_dim', 1024), ('peak_ordering', 'intensity'),
                                           ('min_peak_intensity', .1), ('max_precursor_mz', 2000.)])
def test_encoder_swap_rejects_architecture_or_preprocessing_change(config, name, value):
    with pytest.raises(ValueError, match='changed architecture or preprocessing'):
        validate_encoder_contract(config, {**config, name: value})
