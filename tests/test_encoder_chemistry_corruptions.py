import numpy as np
import pytest

from scripts.encoder_chemistry_corruptions import (
    ENCODER_KEYS, corrupt_inputs, match_peak_controls, observed_ce_donors,
)


def panel():
    counts=np.array([5,6,1])
    return {
        'peak_mz':np.tile(np.array([.100,.110,.120,.130,.140,.150,.0],np.float32),(3,1)),
        'peak_intensity':np.tile(np.array([1.,.8,.6,.4,.2,.1,0.],np.float32),(3,1)),
        'peak_valid_mask':np.arange(7)[None,:]<counts[:,None],
        'precursor_mz':np.full(3,.2,np.float32),
        'collision_energy':np.array([.2,1.,.3],np.float32),
        'collision_energy_present':np.ones(3,np.float32),
        'charge':np.ones(3,np.float32),'charge_present':np.ones(3,np.float32),
        'polarity_id':np.ones(3,np.int64),'instrument_family_id':np.ones(3,np.int64),
        'rows':np.array([111,222,333]),
        'computed_inchikey':np.array(['a','b','c']),
        'canonical_smiles':np.array(['CCO','CCN','CCC']),
        'instrument_type':np.array(['Orbitrap']*3),'adduct':np.array(['[M+H]+']*3),
        'collision_energy_raw':np.array([20.,150.,30.]),
        'binary':np.ones((3,22)),
    }


def test_fraction_doses_keep_slots_padding_and_inputs_unchanged():
    data=panel();original={k:v.copy() for k,v in data.items()}
    rows=np.arange(3)
    attenuation,target,meta=corrupt_inputs(data,rows,name='attenuate_top20')
    removal,removed,_=corrupt_inputs(data,rows,name='remove_top20')
    _,random,_=corrupt_inputs(data,rows,name='remove_random20')
    assert set(attenuation)==set(ENCODER_KEYS)
    assert 'binary' not in attenuation and not meta['chemical_labels_sent_to_encoder']
    np.testing.assert_array_equal(target,removed)
    np.testing.assert_array_equal(target.sum(1),[1,1,0])
    np.testing.assert_array_equal(target.sum(1),random.sum(1))
    assert target[0,0] and target[1,0]
    np.testing.assert_array_equal(attenuation['peak_valid_mask'],original['peak_valid_mask'])
    np.testing.assert_array_equal(attenuation['peak_intensity'][target],original['peak_intensity'][target]*.1)
    np.testing.assert_array_equal(attenuation['peak_intensity'][~target],original['peak_intensity'][~target])
    assert removal['peak_valid_mask'].any(1).all()
    np.testing.assert_array_equal(removal['peak_mz'],original['peak_mz'])
    np.testing.assert_array_equal(removal['peak_intensity'],original['peak_intensity'])
    for key,value in data.items():np.testing.assert_array_equal(value,original[key])


def test_random_corruption_is_source_row_stable_across_batch_order_and_dose():
    data=panel()
    _,whole,_=corrupt_inputs(data,np.array([0,1]),name='attenuate_random20',seed=4)
    _,reordered,_=corrupt_inputs(data,np.array([1,0]),name='remove_random20',seed=4)
    _,single,_=corrupt_inputs(data,np.array([1]),name='attenuate_random20',seed=4)
    np.testing.assert_array_equal(whole,reordered[::-1])
    np.testing.assert_array_equal(whole[1],single[0])


def test_intensity_permutation_never_moves_mz_or_padding_or_intensity_multiset():
    data=panel();result,target,meta=corrupt_inputs(data,np.arange(3),name='permute_intensity')
    for i in range(3):
        slots=data['peak_valid_mask'][i]
        np.testing.assert_array_equal(np.sort(result['peak_intensity'][i,slots]),np.sort(data['peak_intensity'][i,slots]))
    np.testing.assert_array_equal(result['peak_mz'],data['peak_mz'])
    np.testing.assert_array_equal(result['peak_intensity'][~data['peak_valid_mask']],data['peak_intensity'][~data['peak_valid_mask']])
    assert not target[2].any()
    assert meta['supported_rows']==[True,True,False]


def test_mz_artificial_doses_preserve_rank_and_bounded_physical_shift():
    data=panel();data['peak_mz'][0,:5]=np.array([20.,20.,20.005,999.995,1000.])/1000
    for name,bound in [('mz_jitter_0p01da',.01),('mz_shift_plus_0p1da',.1)]:
        result,target,meta=corrupt_inputs(data,np.array([0]),name=name)
        x=result['peak_mz'][0,:5]*1000
        assert np.all(np.diff(x)>=0)
        assert x.min()>=20-1e-5 and x.max()<=1000+1e-5
        assert np.max(np.abs(x-data['peak_mz'][0,:5]*1000))<=bound+1e-4
        np.testing.assert_array_equal(result['peak_mz'][0,5:],data['peak_mz'][0,5:])
        np.testing.assert_array_equal(result['precursor_mz'],data['precursor_mz'][[0]])
        assert meta['facts']['artificial_out_of_distribution']
    data['peak_mz'][0,[0,1]]=[.2,.1]
    with pytest.raises(ValueError,match='m/z order'):
        corrupt_inputs(data,np.array([0]),name='mz_jitter_0p01da')


def test_matching_is_complete_disjoint_and_respects_mass_intensity_calipers():
    mz=np.array([100.,110.,115.,130.,200.])
    intensity=np.array([1.,.5,.9,.25,.9])
    valid=np.ones(5,bool);targets=np.array([True,True,False,False,False])
    control,okay=match_peak_controls(mz,intensity,valid,targets,np.random.default_rng(3))
    assert okay
    np.testing.assert_array_equal(control,[False,False,True,True,False])
    invalid_intensity=intensity.copy();invalid_intensity[3]=.1
    control,okay=match_peak_controls(mz,invalid_intensity,valid,targets,np.random.default_rng(3))
    assert not okay and not control.any()


def test_loss_target_and_control_share_support_counts_and_can_be_attenuated_or_removed():
    data=panel();data['peak_mz'][0,:5]=np.array([100.,110.,115.,130.,200.])/1000
    data['peak_intensity'][0,:5]=[1.,.5,.9,.25,.9]
    weak={'rows':data['rows'].copy(),'loss_names':np.array(['H2O']),
          'peak_matches':np.zeros((3,7,1),bool)}
    weak['peak_matches'][0,:2,0]=True
    weak['peak_matches'][1,:6,0]=True # Never permit full-spectrum removal.
    rows=np.arange(3)
    a,targets,ma=corrupt_inputs(data,rows,name='attenuate_loss_H2O',weak_losses=weak)
    b,controls,mb=corrupt_inputs(data,rows,name='attenuate_loss_matched_H2O',weak_losses=weak)
    removed,rm,mr=corrupt_inputs(data,rows,name='remove_loss_H2O',weak_losses=weak)
    assert ma['supported_rows']==mb['supported_rows']==[True,False,False]
    np.testing.assert_array_equal(targets.sum(1),controls.sum(1))
    np.testing.assert_array_equal(targets,rm)
    assert not (targets&controls).any()
    np.testing.assert_array_equal(a['peak_intensity'][targets],data['peak_intensity'][targets]*.1)
    np.testing.assert_array_equal(b['peak_intensity'][controls],data['peak_intensity'][controls]*.1)
    assert removed['peak_valid_mask'].any(1).all()
    weak['rows'][0]+=1
    with pytest.raises(AssertionError,match='rows differ'):
        corrupt_inputs(data,rows,name='attenuate_loss_H2O',weak_losses=weak)


def test_observed_metadata_replacement_excludes_saturation_mismatches_and_wrong_identity():
    data=panel()
    repeats={
        'primary_index':np.array([0,0,0,1]),'rows':np.array([10,20,30,40]),
        'computed_inchikey':np.array(['a','a','wrong','b']),
        'canonical_smiles':np.array(['CCO','CCO','CCO','CCN']),
        'instrument_type':np.array(['Orbitrap','QTOF','Orbitrap','Orbitrap']),
        'adduct':np.array(['[M+H]+']*4),'collision_energy_present':np.ones(4),
        'collision_energy':np.array([.3,.3,.3,1.]),
        'collision_energy_raw':np.array([30.,30.,30.,140.]),
    }
    result,target,meta=corrupt_inputs(data,np.arange(3),name='observed_ce_metadata',repeats=repeats)
    np.testing.assert_allclose(result['collision_energy'],[.3,1.,.3])
    assert not target.any()
    assert meta['supported_rows']==[True,False,False]
    assert meta['facts']['excluded_raw_ce_changed_but_native_ce_equal_counts']==[0,1,0]
    for key in ENCODER_KEYS:
        if key!='collision_energy':np.testing.assert_array_equal(result[key],data[key])
    empty={key:value[:0] for key,value in repeats.items()}
    _,_,meta=corrupt_inputs(data,np.arange(3),name='observed_ce_metadata',repeats=empty)
    assert meta['facts']['repeat_source_rows']==[-1,-1,-1]


def test_invalid_requests_cannot_silently_select_other_doses():
    data=panel()
    with pytest.raises(ValueError,match='Unknown fixed corruption'):
        corrupt_inputs(data,np.array([0]),name='drop_everything')
    with pytest.raises(ValueError,match='outside'):
        corrupt_inputs(data,np.array([3]),name='attenuate_top20')
    with pytest.raises(ValueError,match='integer'):
        corrupt_inputs(data,np.array([0.]),name='attenuate_top20')
    with pytest.raises(ValueError,match='requires measured'):
        corrupt_inputs(data,np.array([0]),name='observed_ce_metadata')
    with pytest.raises(ValueError,match='require aligned'):
        corrupt_inputs(data,np.array([0]),name='attenuate_loss_H2O')
