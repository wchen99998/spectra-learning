import numpy as np

from scripts.analyze_encoder_chemistry_matched_geometry import candidate_banks, retrieve


def test_candidate_calipers_are_inclusive_and_missing_energy_matches_only_missing():
    panel = {'fold': np.array(['train']*5+['test']*2), 'molecule': np.arange(7),
        'peak_valid_mask': np.ones((7, 7)), 'instrument_type': np.array(['Orbitrap']*7),
        'adduct': np.array(['H']*7), 'collision_energy_present': np.array([1,1,1,0,0,1,0]),
        'collision_energy_raw': np.array([20,40,40.1,np.nan,np.nan,30,np.nan]),
        'precursor_mz_raw': np.array([95,105,100,100,106,100,100.])}
    test, banks = candidate_banks(panel)
    np.testing.assert_array_equal(test, [5,6])
    np.testing.assert_array_equal(banks[0], [0,1])
    np.testing.assert_array_equal(banks[1], [3])


def test_retrieval_requires_choice_and_does_not_filter_candidates_by_chemistry():
    features = np.array([[1.,0],[0,1],[1,0],[1,0]])
    panel = {'precursor_mz_raw':np.array([100,101,100,100])}
    labels = {'element_names':np.array([]),'binary':np.eye(4,dtype=int),'morgan':np.eye(4,dtype=int)}
    result = retrieve(features,np.array([2,3]),[np.array([0,1]),np.array([0])],panel,labels)
    assert result['nearest_train_row'].tolist()==[0,-1]
    assert np.isnan(result['morgan_tanimoto'][1])
    labels['morgan'][:]=1;labels['binary'][:]=1
    changed = retrieve(features,np.array([2,3]),[np.array([0,1]),np.array([0])],panel,labels)
    np.testing.assert_array_equal(changed['nearest_train_row'],result['nearest_train_row'])
