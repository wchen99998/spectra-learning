from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_encoder_chemistry_causal import analyze_predictions, select_head_layers


def test_evaluation_support_does_not_enable_untrained_target(tmp_path: Path):
    truth=np.zeros((30,188),np.uint8)
    truth[20:25,0]=1  # Test support alone must not admit a constant training bit.
    truth[::2,1]=1
    np.savez(tmp_path/'labels.npz',maccs=truth[:,:166],binary=truth[:,166:],binary_names=np.array([f'concept{i}' for i in range(22)]))
    np.savez(tmp_path/'inputs.npz',fold=np.array(['train']*20+['test']*10),molecule=np.arange(30).astype(str))
    np.savez(tmp_path/'predictions.npz',rows=np.arange(20,30),logits=np.zeros((1,10,3,188)),variants=np.array(['identity']))
    summary=analyze_predictions(tmp_path/'predictions.npz',tmp_path,resamples=10)
    result=summary['results'][0]
    assert not result['per_target'][0]['supported']
    assert result['per_target'][1]['supported']
    assert result['groups']['all']['targets']==1
    assert result['groups']['all']['delta_balanced_bce']==0


def test_test_effects_cannot_select_head_layers():
    with pytest.raises(ValueError,match='validation'):
        select_head_layers({'folds':['test'],'results':[]})
