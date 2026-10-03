import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts.analyze_encoder_chemistry_sae import (
    CONCEPTS, assign_random_controls, choose_threshold, donor_ordering,
    multi_feature_auroc, nuisance_profile, select_candidates,
)


def test_multifeature_auc_matches_sklearn_for_sparse_tied_activations():
    rng=np.random.default_rng(19)
    x=rng.integers(0,5,(73,11)).astype(float)
    x[:40,:4]=0
    y=rng.integers(0,2,(73,4));y[:,3]=0
    actual=multi_feature_auroc(x,y,block_size=3)
    for c in range(3):
        for f in range(11):
            assert actual[c,f]==pytest.approx(roc_auc_score(y[:,c],x[:,f]),abs=2e-15)
    assert np.isnan(actual[3]).all()


def discovery_fixture():
    rng=np.random.default_rng(17)
    folds=np.array(['train']*80+['val']*60+['test']*20)
    labels=np.empty((160,6),np.uint8)
    for j in range(6):
        for lo,n in ((0,80),(80,60),(140,20)):
            labels[lo:lo+n,j]=rng.permutation(np.repeat([0,1],n//2))
    features=rng.uniform(0,.2,(160,18))
    features[:,:6]+=labels
    features[:,6:12]+=labels
    return features,labels,folds


def test_selection_is_independent_of_test_values_and_labels_and_respects_budget():
    x,y,folds=discovery_fixture()
    first=select_candidates(x,y,list(CONCEPTS),folds)
    altered_x=x.copy();altered_y=y.copy()
    altered_x[folds=='test']=np.nan
    altered_y[folds=='test']=1-altered_y[folds=='test']
    second=select_candidates(altered_x,altered_y,list(CONCEPTS),folds)
    assert first==second
    assert len(first)==12
    assert len({r['feature']for r in first})==12
    assert all(sum(r['concept']==c for r in first)==2 for c in CONCEPTS)
    assert all(r['train_auc']==r['val_auc']==1 for r in first)


def test_no_forced_filler_from_rare_or_unpredictable_concepts():
    x,y,folds=discovery_fixture()
    x[:]=1 # Constant features cannot provide meaningful rankings.
    assert select_candidates(x,y,list(CONCEPTS),folds)==[]
    x,y,folds=discovery_fixture();y[folds=='train',0]=0
    selected=select_candidates(x,y,list(CONCEPTS),folds)
    assert not any(r['concept']==CONCEPTS[0]for r in selected)


def test_thresholds_come_only_from_train_quantiles_and_use_strict_greater():
    train=np.array([0,0,0,1,2,3.])
    validation=np.array([0,1,2,3.]);labels=np.array([0,0,1,1])
    result=choose_threshold(train,validation,labels)
    assert result['selected']['threshold'] in result['candidates_from_train_quantiles']
    assert result['selected']['comparison_operator']=='>'
    assert result['selected']['f1']==1


def test_random_controls_exclude_selected_features_and_ignore_labels():
    rng=np.random.default_rng(4);x=rng.uniform(size=(40,20))
    decoder=rng.normal(size=(20,7));decoder/=np.linalg.norm(decoder,axis=1,keepdims=True)
    records=[{'feature':1},{'feature':3}]
    assign_random_controls(records,x,np.arange(40),decoder)
    repeated=[{'feature':1},{'feature':3}]
    assign_random_controls(repeated,x,np.arange(40),decoder)
    assert records==repeated
    assert all(r['random_control_feature']not in {1,3}for r in records)
    assert all(r['random_control_matching']['feature_decoder_l2']==pytest.approx(1)for r in records)


def test_nuisance_profiles_explicitly_flag_mass_or_peakcount_associations():
    n=30;scores=np.arange(n,dtype=float)
    panel={'exact_mass':np.arange(n)+100.,'precursor_mz_raw':np.arange(n)+101.,
        'peak_valid_mask':np.arange(n+1)[None,:]<=np.arange(n)[:,None],
        'collision_energy_present':np.ones(n),'collision_energy_raw':np.full(n,30.),
        'instrument_type':np.array(['Orbitrap']*n),'adduct':np.array(['[M+H]+']*n)}
    profile,flags=nuisance_profile(scores,np.arange(n),panel)
    assert 'strong_exact_mass_association'in flags
    assert 'strong_processed_peak_count_association'in flags
    assert profile['spearman']['collision_energy']is None
    assert profile['categorical']['instrument_type']['eta_squared']==0


def test_fixed_donor_ordering_reports_half_credit_ties():
    control={'opposite_concept_donor':np.array([[1],[0],[0]])}
    result=donor_ordering(np.array([2.,1.,2.]),np.array([1,0,0]),np.arange(3),control,0)
    assert result['accuracy']==pytest.approx(2.5/3)
    assert result['ties']==1 and result['unique_donors']==2


def test_geometry_selects_neighbors_without_chemical_labels_and_averages_ties():
    from scripts.analyze_encoder_chemistry_sae import retrieval_geometry
    features=np.array([[1,0],[1,0],[0,1],[1,0],[0,1],[0,0]],float)
    panel={'fold':np.array(['train']*3+['test']*3),'molecule':np.array(['a','b','c','d','e','f']),
           'precursor_mz_raw':np.arange(6)+100.}
    labels={'morgan':np.array([[1,0],[0,1],[0,1],[1,0],[0,1],[1,0]],np.uint8),
            'binary':np.array([[1,1,0],[1,0,1],[1,0,1],[1,1,0],[1,0,1],[1,1,0]],np.uint8),
            'element_names':np.array(['N'])}
    result=retrieval_geometry(features,panel,labels)
    np.testing.assert_array_equal(result['nearest_train_row'],[0,2,-1])
    np.testing.assert_array_equal(result['tie_count'],[2,1,0])
    np.testing.assert_allclose(result['morgan_tanimoto'][:2],[.5,1.])
    changed={**labels,'morgan':1-labels['morgan']}
    other=retrieval_geometry(features,panel,changed)
    np.testing.assert_array_equal(other['nearest_train_row'],result['nearest_train_row'])
    assert np.isnan(result['morgan_tanimoto'][2])


def test_discovery_contract_and_separate_confirmation_round_trip(tmp_path):
    import json,hashlib
    from scripts.analyze_encoder_chemistry_sae import discover,confirm
    x,y,folds=discovery_fixture();panel=tmp_path/'panel';panel.mkdir()
    sae=tmp_path/'encoder'/'sae'/'depth19'/'k32'/'seed0';sae.mkdir(parents=True)
    n=len(folds);rows=np.arange(n)
    np.savez(panel/'inputs.npz',rows=rows,fold=folds,molecule=np.array([f'm{i}'for i in rows]),
      exact_mass=np.arange(n,dtype=float)+100,precursor_mz_raw=np.arange(n,dtype=float)+101,
      peak_valid_mask=np.ones((n,5),bool),collision_energy_present=np.ones(n),
      collision_energy_raw=np.full(n,30.),instrument_type=np.array(['Orbitrap']*n),adduct=np.array(['[M+H]+']*n))
    np.savez(panel/'labels.npz',rows=rows,binary=y,binary_names=np.asarray(CONCEPTS),maccs=np.zeros((n,166),np.uint8))
    np.savez(panel/'matched_controls.npz',rows=rows,opposite_concept_donor=np.full((n,6),-1))
    (panel/'manifest.json').write_text('{}')
    np.save(sae/'molecule_max.npy',x.astype(np.float32))
    np.save(sae/'random_dictionary_molecule_max.npy',np.random.default_rng(1).uniform(size=x.shape).astype(np.float32))
    np.save(sae/'dense_channel_molecule_max.npy',x[:,:6].astype(np.float32))
    decoder=np.random.default_rng(3).normal(size=(18,4));decoder/=np.linalg.norm(decoder,axis=1,keepdims=True)
    np.savez(sae/'params.npz',decoder=decoder)
    (sae/'trained_sae.json').write_text(json.dumps({'complete':True,'toy':True,
        'contract':{'encoder':'encoder','depth':19,'k':32,'seed':0},
        'weights_sha256':hashlib.sha256((sae/'params.npz').read_bytes()).hexdigest(),
        'artifact_files':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in sae.glob('*.npy')}}))
    selection=discover(sae,panel)
    assert len(selection['records'])==12
    contract=selection['contract']
    assert contract['encoder']=='encoder'and contract['depth']==19 and contract['k']==32 and contract['seed']==0
    assert contract['selection_splits']==['train','val']and not contract['test_used_for_selection']
    before=(sae/'chemical_features.json').read_bytes()
    confirmation=confirm(sae,panel)
    assert (sae/'chemical_features.json').read_bytes()==before
    assert len(confirmation['families']['sae'])==12
    assert all('test'in record for record in confirmation['families']['sae'])
