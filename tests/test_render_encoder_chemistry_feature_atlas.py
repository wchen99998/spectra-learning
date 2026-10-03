import numpy as np
import pytest

from scripts.render_encoder_chemistry_feature_atlas import physical_activations, select_examples


def test_sparse_tokens_reconstruct_original_physical_slots_without_rank_shifting():
    sparse={'rows':np.array([2,0]),'slots':np.array([3,1]),
            'feature_indices':np.array([[5,7],[7,9]]),'feature_values':np.array([[.2,.8],[.4,.3]])}
    result=physical_activations(sparse,7,(3,5))
    assert result[2,3]==pytest.approx(.8) and result[0,1]==pytest.approx(.4)
    assert np.count_nonzero(result)==2
    sparse['feature_indices'][0]=[7,7]
    with pytest.raises(ValueError,match='twice'):
        physical_activations(sparse,7,(3,5))


def test_examples_are_class_conditional_maxima_with_fixed_row_ties():
    examples=select_examples(np.array([99,5,5,7,8.]),np.array([1,1,1,0,0]),np.arange(1,5))
    assert [row['panel_row']for row in examples]==[1,4]
    assert [row['molecular_concept_present']for row in examples]==[True,False]
