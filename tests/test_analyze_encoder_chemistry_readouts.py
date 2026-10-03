import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts.analyze_encoder_chemistry_readouts import (
    auc_matrix, balanced_score_losses, fixed_macro, interval,
    matched_margin_metrics, weighted_auroc,
)


def test_weighted_auc_ties_and_duplicated_bootstrap_molecules_match_sklearn():
    scores=np.array([.1,.2,.2,.2,.9,.9,1.])
    labels=np.array([0,0,1,0,1,0,1])
    weights=np.array([[1,2,3,0,2,1,1],[0,1,2,2,1,3,0],[3,0,0,1,0,1,0]])
    result=weighted_auroc(scores,labels,weights)
    for i,w in enumerate(weights[:2]):
        assert result[i]==pytest.approx(roc_auc_score(labels,scores,sample_weight=w),abs=1e-15)
        indices=np.repeat(np.arange(len(labels)),w)
        assert result[i]==pytest.approx(roc_auc_score(labels[indices],scores[indices]),abs=1e-15)
    assert np.isnan(result[2])
    assert weighted_auroc(np.ones(4),np.array([0,1,0,1]))[0]==.5


def test_fractional_weights_and_many_random_ties_match_sklearn():
    rng=np.random.default_rng(46)
    scores=rng.integers(0,5,79).astype(float)
    labels=rng.integers(0,2,79)
    weights=rng.uniform(0,3,(15,79))
    actual=weighted_auroc(scores,labels,weights)
    expected=[roc_auc_score(labels,scores,sample_weight=w)for w in weights]
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-15)


def test_fixed_macro_does_not_silently_drop_rare_undefined_targets():
    values=np.array([[.8,.6,np.nan],[.9,np.nan,.7],[.5,.5,.5]])
    support=np.array([True,True,False])
    macro=fixed_macro(values,support)
    np.testing.assert_allclose(macro[[0,2]],[.7,.5])
    assert np.isnan(macro[1])
    result=interval(macro,.7)
    assert result['valid_bootstrap_replicates']==2
    assert result['undefined_bootstrap_replicates']==1
    assert np.isnan(fixed_macro(values,np.zeros(3,bool))).all()


def test_seed_prediction_ensemble_is_not_seed_average_metric():
    labels=np.array([[0],[0],[1],[1]])
    first=np.array([[0.],[0.],[1.],[1.]])
    second=np.array([[100.],[100.],[0.],[0.]])
    mean_metric=np.mean([auc_matrix(x,labels)[0,0]for x in (first,second)])
    ensemble_metric=auc_matrix((first+second)/2,labels)[0,0]
    assert mean_metric==.5 and ensemble_metric==0


def test_paired_bootstrap_of_identical_predictions_is_exactly_zero():
    labels=np.array([[0,1],[1,0],[0,0],[1,1],[1,1]],np.uint8)
    scores=np.array([[0,0],[1,2],[2,3],[3,4],[4,5.]])
    counts=np.random.default_rng(8).multinomial(5,np.full(5,.2),size=1000)
    boot=auc_matrix(scores,labels,counts)
    contrast=fixed_macro(boot-boot,np.ones(2,bool))
    finite=np.isfinite(contrast)
    assert np.all(contrast[finite]==0)
    assert interval(contrast,0)['ci95']==[0.,0.]


def test_balanced_losses_do_not_weight_rare_class_by_prevalence():
    labels=np.array([[1],[0],[0],[0]])
    logits=np.zeros((4,1))
    losses=balanced_score_losses(logits,labels)
    np.testing.assert_allclose(losses['balanced_bce'],np.log(2))
    np.testing.assert_allclose(losses['balanced_brier'],.25)
    np.testing.assert_allclose(balanced_score_losses(np.array([[2.],[-2.],[-2.],[-2.]]),labels)['balanced_bce'],np.logaddexp(0,-2))


def test_strict_donor_margin_accuracy_handles_reuse_ties_and_panel_indices():
    scores=np.array([[.8],[.2],[.8]])
    labels=np.array([[1],[0],[0]])
    controls={'binary_names':np.array(['fg_amide']),
              'opposite_concept_donor':np.full((8,1),-1,int)}
    controls['opposite_concept_donor'][[3,5,7],0]=[5,3,3]
    result=matched_margin_metrics(scores,labels,['fg_amide'],np.array([3,5,7]),controls,np.array([True]))['per_concept'][0]
    assert result['directed_pairs']==3
    assert result['unique_undirected_pairs']==2
    assert result['ties']==1
    assert result['accuracy']==pytest.approx(2.5/3)
    assert result['balanced_anchor_accuracy']==pytest.approx(.875)
    assert result['unique_donors']==2


def test_bad_scores_and_weights_fail_loudly():
    with pytest.raises(ValueError,match='Nonfinite'):
        weighted_auroc(np.array([0,np.nan]),np.array([0,1]))
    with pytest.raises(ValueError,match='Weights'):
        weighted_auroc(np.array([0,1]),np.array([0,1]),np.array([1,-1]))
