"""Train/validation-only SAE chemistry discovery, followed by separate confirmation.

Molecular token-max activations test concept accessibility, not measured fragment
identity. Candidate-feature search is explicitly separated from held-out test
confirmation. Learned, random and dense dictionaries receive the same selector.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from scripts.analyze_encoder_chemistry_readouts import json_value

CONCEPTS = ("fg_amide", "fg_amine", "fg_hydroxyl", "fg_halide", "fg_nitrile", "fg_sulfonyl")
SEED = 20260930
POLICY = {"concepts": list(CONCEPTS), "train_candidate_features_per_concept": 64,
          "features_per_concept": 2, "minimum_train_and_validation_class_count": 20,
          "minimum_train_activated_molecules": 20,
          "minimum_train_and_validation_auroc": .6,
          "maximum_selected_training_rank_correlation": .95,
          "threshold_training_quantiles": [.5, .75, .9, .95, .99],
          "threshold_selection": "Highest validation F1, then precision, then lowest threshold; strict activation > threshold",
          "selection": "Top64 eligible features by training AUROC for each concept, then validation AUROC with train-AUROC and feature-ID tie-breaks. Fixed concept order, globally unique features; redundancy filtered on train only.",
          "nuisance_flags": "|Spearman|>=0.5 for mass/CE/count or categorical eta-squared>=0.2; descriptive flags, never post-test exclusion"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def multi_feature_auroc(features: np.ndarray, labels: np.ndarray, block_size: int = 256) -> np.ndarray:
    """Target-by-feature exact tie-correct AUROC, without fitting feature scales."""
    if features.ndim != 2 or labels.ndim != 2 or len(features) != len(labels):
        raise ValueError("Features and labels must share molecular rows")
    positive = labels.sum(0).astype(float)
    negative = len(labels) - positive
    denominator = positive * negative
    result = np.full((labels.shape[1], features.shape[1]), np.nan)
    y = labels.astype(np.float64).T
    for start in range(0, features.shape[1], block_size):
        block = np.asarray(features[:, start:start + block_size], np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite dictionary activations")
        ranks = rankdata(block, axis=0, method="average")
        numerator = y @ ranks - (positive * (positive + 1) / 2)[:, None]
        result[:, start:start + block_size] = np.divide(numerator, denominator[:, None],
            out=np.full_like(numerator, np.nan), where=denominator[:, None] > 0)
    return result


def classification(scores: np.ndarray, labels: np.ndarray, threshold: float | None = None) -> dict:
    positive = int(labels.sum())
    result = {"positive": positive, "negative": len(labels) - positive,
              "auroc": None, "average_precision": None}
    if positive and positive < len(labels):
        result.update(auroc=float(roc_auc_score(labels, scores)),
                      average_precision=float(average_precision_score(labels, scores)))
    if threshold is not None:
        predicted = scores > threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", zero_division=0)
        result.update(threshold=float(threshold), comparison_operator=">", precision=float(precision),
                      recall=float(recall), f1=float(f1), predicted_positive=int(predicted.sum()))
    return result


def choose_threshold(train_scores: np.ndarray, val_scores: np.ndarray, val_labels: np.ndarray) -> dict:
    candidates = np.unique(np.quantile(train_scores, POLICY["threshold_training_quantiles"]))
    records = [classification(val_scores, val_labels, float(value)) for value in candidates]
    chosen = sorted(records, key=lambda r: (-r["f1"], -r["precision"], r["threshold"]))[0]
    return {"selected": chosen, "candidates_from_train_quantiles": candidates.tolist()}


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.ptp(x[finite]) == 0 or np.ptp(y[finite]) == 0:
        return None
    return float(spearmanr(x[finite], y[finite]).statistic)


def nuisance_profile(scores: np.ndarray, rows: np.ndarray, panel: dict) -> tuple[dict, list[str]]:
    numeric = {"exact_mass": panel["exact_mass"][rows],
        "precursor_mz": panel["precursor_mz_raw"][rows],
        "processed_peak_count": panel["peak_valid_mask"][rows].sum(1),
        "collision_energy": np.where(panel["collision_energy_present"][rows].astype(bool),
                                     panel["collision_energy_raw"][rows], np.nan)}
    profile = {"spearman": {name: rank_correlation(scores, values) for name, values in numeric.items()},
               "categorical": {}}
    flags = [f"strong_{name}_association" for name, value in profile["spearman"].items()
             if value is not None and abs(value) >= .5]
    for name in ("instrument_type", "adduct"):
        categories = panel[name][rows]
        center = scores.mean()
        total = np.sum((scores - center) ** 2)
        records, between = {}, 0.
        for category in np.unique(categories):
            mask = categories == category
            group_mean = float(scores[mask].mean())
            between += int(mask.sum()) * (group_mean - center) ** 2
            records[str(category)] = {"molecules": int(mask.sum()), "mean": group_mean,
                                       "nonzero_fraction": float(np.mean(scores[mask] > 0))}
        eta = float(between / total) if total > 0 else None
        profile["categorical"][name] = {"eta_squared": eta, "groups": records}
        if eta is not None and eta >= .2:
            flags.append(f"strong_{name}_association")
    return profile, flags


def donor_ordering(scores: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                   controls: dict, concept_index: int) -> dict:
    """Fixed observed donor graph; donor reuse is reported, not treated as IID."""
    position = {int(row): i for i, row in enumerate(rows)}
    donors = controls["opposite_concept_donor"][rows, concept_index]
    anchor = np.flatnonzero(donors >= 0)
    if not len(anchor):
        return {"directed_pairs": 0, "accuracy": None}
    other = np.asarray([position[int(v)] for v in donors[anchor]], int)
    if (labels[anchor] == labels[other]).any():
        raise ValueError("Opposite-concept donor labels agree")
    signed = (scores[anchor] - scores[other]) * (2 * labels[anchor].astype(float) - 1)
    correct = (signed > 0).astype(float) + .5 * (signed == 0)
    pos = labels[anchor].astype(bool)
    return {"directed_pairs": len(anchor), "unique_donors": len(np.unique(other)),
            "positive_anchors": int(pos.sum()), "negative_anchors": int((~pos).sum()),
            "accuracy": float(correct.mean()),
            "balanced_anchor_accuracy": float(.5 * (correct[pos].mean() + correct[~pos].mean())) if pos.any() and (~pos).any() else None,
            "ties": int((signed == 0).sum()),
            "interpretation": "Descriptive strict nuisance-matched ordering; shared donors, not independent molecular counterfactuals"}


def select_candidates(features: np.ndarray, labels: np.ndarray, binary_names: list[str],
                      folds: np.ndarray, *, candidate_ids: np.ndarray | None = None,
                      train_auc: np.ndarray | None = None, val_auc: np.ndarray | None = None) -> list[dict]:
    """Only train and validation rows influence returned feature IDs/thresholds."""
    train, val = np.flatnonzero(folds == "train"), np.flatnonzero(folds == "val")
    ids = np.arange(features.shape[1]) if candidate_ids is None else np.asarray(candidate_ids)
    train_values = np.asarray(features[train][:, ids])
    val_values = np.asarray(features[val][:, ids])
    tr_auc = multi_feature_auroc(train_values, labels[train]) if train_auc is None else train_auc[:, ids]
    va_auc = multi_feature_auroc(val_values, labels[val]) if val_auc is None else val_auc[:, ids]
    # Positive activations define firing for nonnegative SAE/random features.
    # Dense signed-channel baseline instead requires nonconstant train values;
    # it remains eligible irrespective of its arbitrary signed zero location.
    nonconstant = np.ptp(train_values, axis=0) > 0
    firing = np.count_nonzero(train_values > 0, axis=0)
    dense_signed = bool(np.any(train_values < 0))
    eligible_feature = nonconstant & (True if dense_signed else firing >= POLICY["minimum_train_activated_molecules"])
    selected, seen = [], set()
    selected_train_ranks = []
    for concept in CONCEPTS:
        column = binary_names.index(concept)
        tr_y, va_y = labels[train, column], labels[val, column]
        if min(tr_y.sum(),len(tr_y)-tr_y.sum(),va_y.sum(),len(va_y)-va_y.sum()) < POLICY["minimum_train_and_validation_class_count"]:
            continue
        candidate = np.flatnonzero(eligible_feature & (tr_auc[column] >= POLICY["minimum_train_and_validation_auroc"]))
        candidate = sorted(candidate, key=lambda j: (-tr_auc[column,j],int(ids[j])))[:POLICY["train_candidate_features_per_concept"]]
        candidate = sorted(candidate,key=lambda j:(-va_auc[column,j],-tr_auc[column,j],int(ids[j])))
        count = 0
        for local_id in candidate:
            feature = int(ids[local_id])
            if feature in seen or va_auc[column,local_id] < POLICY["minimum_train_and_validation_auroc"]:
                continue
            ranks = rankdata(train_values[:,local_id],method="average")
            if any(abs(float(np.corrcoef(ranks, previous)[0,1])) >= POLICY["maximum_selected_training_rank_correlation"] for previous in selected_train_ranks):
                continue
            threshold = choose_threshold(train_values[:,local_id],val_values[:,local_id],va_y)
            validation = classification(val_values[:,local_id],va_y,threshold["selected"]["threshold"])
            positive = train_values[tr_y.astype(bool),local_id]
            record = {"feature": feature,"concept": concept,"concept_index": column,
                "train_auc": float(tr_auc[column,local_id]),"val_auc": float(va_auc[column,local_id]),
                "val_ap": validation["average_precision"],"validation":validation,
                "threshold":threshold["selected"]["threshold"],"threshold_selection":threshold,
                "positive_activation_quantiles":np.quantile(positive,[.25,.5,.75,.9,.99]).tolist(),
                "positive_activation_quantile_levels":[.25,.5,.75,.9,.99],
                "positive_activation_definition":"Molecular max activations among concept-positive TRAIN molecules, including zeros",
                "train_molecular_nonzero_fraction":float(np.mean(train_values[:,local_id]>0)),
                "validation_molecular_nonzero_fraction":float(np.mean(val_values[:,local_id]>0)),
                "selection_criterion":POLICY["selection"]}
            selected.append(record);seen.add(feature);selected_train_ranks.append(ranks);count+=1
            if count==POLICY["features_per_concept"]:break
    return selected


def assign_random_controls(records: list[dict], features: np.ndarray, train: np.ndarray,
                           decoder: np.ndarray) -> None:
    values=np.asarray(features[train],np.float64)
    firing=np.mean(values>0,axis=0)
    rms=np.sqrt(np.mean(values**2,axis=0))
    norms=np.linalg.norm(decoder,axis=1)
    forbidden={r['feature'] for r in records}
    for record in records:
        feature=record['feature']
        distance=(np.abs(np.log((firing+1/len(train))/(firing[feature]+1/len(train))))
          +np.abs(np.log(np.maximum(rms,1e-12)/max(rms[feature],1e-12)))
          +np.abs(np.log(np.maximum(norms,1e-12)/max(norms[feature],1e-12))))
        candidates=[int(i) for i in np.lexsort((np.arange(len(firing)),distance))
                    if i not in forbidden and firing[i]>0 and norms[i]>0][:16]
        if not candidates:
            raise ValueError('No active random-control dictionary feature')
        control=int(np.random.default_rng(np.random.SeedSequence([SEED,feature])).choice(candidates))
        record['random_control_feature']=control
        record['random_control_matching']={'criterion':'Uniform among16 nearest nonselected active features by log training molecule-firing, activation-RMS and decoder-L2 gaps; no chemical labels used',
          'feature_train_firing':float(firing[feature]),'control_train_firing':float(firing[control]),
          'feature_train_activation_rms':float(rms[feature]),'control_train_activation_rms':float(rms[control]),
          'feature_decoder_l2':float(norms[feature]),'control_decoder_l2':float(norms[control]),
          'distance':float(distance[control]),
          'limitation':'Approximate training-level match. Tokenwise edit norms still need direct L2 matching during causal tests; an inactive control feature can yield no natural ablation effect.'}


def load_panel(path:Path)->tuple[dict,dict,dict]:
    values=[]
    for name in ('inputs','labels','matched_controls'):
        with np.load(path/f'{name}.npz') as f:values.append({k:f[k] for k in f.files})
    panel,labels,controls=values
    np.testing.assert_array_equal(panel['rows'],labels['rows'])
    np.testing.assert_array_equal(panel['rows'],controls['rows'])
    return panel,labels,controls


def discover(sae:Path,panel_path:Path)->dict:
    panel,labels,controls=load_panel(panel_path)
    dictionary=json.loads((sae/'trained_sae.json').read_text())
    identity={'encoder':sae.parents[3].name,'depth':int(sae.parents[1].name.removeprefix('depth')),
              'k':int(sae.parent.name.removeprefix('k')),'seed':int(sae.name.removeprefix('seed'))}
    if not dictionary['complete'] or any(dictionary['contract'][key]!=value for key,value in identity.items()):
        raise ValueError('SAE completion or dictionary identity differs from its directory')
    if sha256(sae/'params.npz')!=dictionary['weights_sha256']:
        raise ValueError('SAE parameter hash differs from final training metadata')
    with np.load(sae/'params.npz') as f:decoder=f['decoder']
    sources={'sae':sae/'molecule_max.npy', 'random_dictionary':sae/'random_dictionary_molecule_max.npy',
             'dense_channel':sae/'dense_channel_molecule_max.npy'}
    for path in sources.values():
        if sha256(path)!=dictionary['artifact_files'][path.name]['sha256']:
            raise ValueError('Exported SAE/baseline activations differ from final training metadata')
    matrices={name:np.load(path,mmap_mode='r')for name,path in sources.items()}
    if any(len(value)!=len(panel['rows'])for value in matrices.values()):
        raise ValueError('Feature molecules differ from panel')
    if matrices['sae'].shape[1]!=len(decoder) or matrices['random_dictionary'].shape!=matrices['sae'].shape:
        raise ValueError('Learned/random dictionaries must have the same feature budget and decoder indexing')
    train=np.flatnonzero(panel['fold']=='train');val=np.flatnonzero(panel['fold']=='val')
    binary_names=labels['binary_names'].tolist();all_labels=np.concatenate([labels['maccs'],labels['binary']],axis=1)
    all_names=[f'maccs_{i:03d}'for i in range(1,167)]+binary_names
    associations={};selections={}
    for name,features in matrices.items():
        print('Train/validation associations',name,features.shape,flush=True)
        train_auc=multi_feature_auroc(np.asarray(features[train]),all_labels[train])
        val_auc=multi_feature_auroc(np.asarray(features[val]),all_labels[val])
        associations[name+'_train_auroc']=train_auc.astype(np.float32)
        associations[name+'_val_auroc']=val_auc.astype(np.float32)
        selections[name]=select_candidates(features,labels['binary'],binary_names,panel['fold'],
            train_auc=train_auc[166:],val_auc=val_auc[166:])
    subset=np.sort(np.random.default_rng(SEED).choice(matrices['sae'].shape[1],matrices['dense_channel'].shape[1],replace=False))
    selections['sae_equal_dense_search_budget']=select_candidates(matrices['sae'],labels['binary'],binary_names,
        panel['fold'],candidate_ids=subset,train_auc=associations['sae_train_auroc'][166:],val_auc=associations['sae_val_auroc'][166:])
    assign_random_controls(selections['sae'],matrices['sae'],train,decoder)
    for baseline,records in selections.items():
        matrix=matrices['sae'] if baseline=='sae_equal_dense_search_budget' else matrices[baseline]
        for record in records:
            feature=record['feature'];column=record['concept_index']
            record['train_nuisance_profile'],flags1=nuisance_profile(matrix[train,feature],train,panel)
            record['validation_nuisance_profile'],flags2=nuisance_profile(matrix[val,feature],val,panel)
            record['nuisance_flags']=sorted(set(flags1+flags2))
            record['validation_matched_donor_ordering']=donor_ordering(matrix[val,feature],labels['binary'][val,column],val,controls,column)
    contract={**identity,
        'params_filename':'params.npz','molecule_max_filename':'molecule_max.npy','metadata_filename':'trained_sae.json',
        'params_sha256':sha256(sae/'params.npz'),'trained_sae_sha256':sha256(sae/'trained_sae.json'),
        'molecule_max_sha256':sha256(sae/'molecule_max.npy'),
        'panel_sha256':{name:sha256(panel_path/name)for name in ('inputs.npz','labels.npz','manifest.json','matched_controls.npz')},
        'molecule_ids_sha256':hashlib.sha256(panel['molecule'].tobytes()).hexdigest(),
        'selection_splits':['train','val'],'test_used_for_selection':False,
        'dictionary_directory':str(sae),'dictionary_metadata':dictionary,
        'source_sha256':sha256(Path(__file__))}
    report={'schema':'sae_chemical_feature_selection_v1','contract':contract,'policy':POLICY,
        'records':selections['sae'],'baseline_selections':{k:v for k,v in selections.items()if k!='sae'},
        'selection_counts':{name:len(records)for name,records in selections.items()},
        'feature_search_budgets':{name:value.shape[1]for name,value in matrices.items()},
        'equal_dense_sae_feature_ids':subset.tolist(),
        'baseline_files':{name:{'path':str(path),'sha256':sha256(path)}for name,path in sources.items()},
        'association_target_names':all_names,
        'selection_bias':'Searching thousands of features inflates train/validation associations. Only separate test confirmation supports out-of-sample claims. Baselines share the selector; equal feature search budget does not imply equal representation capacity.',
        'reconstruction_interpretation':'Full trained_sae.json is attached in contract. Chemical associations do not establish sufficient reconstruction, native model faithfulness, or a causal fragment mechanism. Reconstruction replacement controls and SAE error-preserving interventions remain necessary.',
        'stability_status':'Seed/k stability is not established by train/validation agreement; align decoder/activation features across dictionaries before claiming stability.',
        'peak_interpretation':'Molecule-max feature associations and raw m/z/intensity correlations are not observed atom/fragment annotations.'}
    path=sae/'chemical_features.json'
    serialized=json.dumps(json_value(report),indent=2,allow_nan=False)+'\n'
    if path.exists() and path.read_text()!=serialized:
        raise ValueError('Frozen feature selection already exists with different contents; use a new output experiment directory')
    path.write_text(serialized)
    np.savez_compressed(sae/'chemical_associations_train_val.npz',**associations,target_names=np.asarray(all_names))
    return report


def confirm(sae:Path,panel_path:Path)->dict:
    selection=json.loads((sae/'chemical_features.json').read_text())
    if selection['contract']['test_used_for_selection'] or selection['contract']['selection_splits']!=['train','val']:
        raise ValueError('Feature discovery used invalid selection splits')
    if sha256(sae/'params.npz')!=selection['contract']['params_sha256']:
        raise ValueError('SAE parameters changed after feature selection')
    for filename,checksum in selection['contract']['panel_sha256'].items():
        if sha256(panel_path/filename)!=checksum:raise ValueError('Panel changed after feature selection')
    panel,labels,controls=load_panel(panel_path);rows=np.flatnonzero(panel['fold']=='test')
    y=np.concatenate([labels['maccs'],labels['binary']],axis=1)
    names=selection['association_target_names'];families={'sae':selection['records'],**selection['baseline_selections']}
    report={'selection_sha256':sha256(sae/'chemical_features.json'),'selection_frozen_before_test':True,
        'test_molecules':len(rows),'families':{},'multiplicity':'188 chemical targets and several baseline/feature choices; confirmation is descriptive, without discovery p-values or unadjusted significance claims.'}
    for family,records in families.items():
        source='sae' if family=='sae_equal_dense_search_budget'else family
        metadata=selection['baseline_files'][source];path=Path(metadata['path'])
        if sha256(path)!=metadata['sha256']:raise ValueError('Feature activations changed after discovery')
        features=np.load(path,mmap_mode='r');confirmed=[]
        for record in records:
            feature,column=record['feature'],record['concept_index'];scores=np.asarray(features[rows,feature])
            result={**record,'test':classification(scores,labels['binary'][rows,column],record['threshold']),
               'test_matched_donor_ordering':donor_ordering(scores,labels['binary'][rows,column],rows,controls,column)}
            result['test_nuisance_profile'],result['test_nuisance_flags']=nuisance_profile(scores,rows,panel)
            result['all_test_target_associations']=[{'name':name,**classification(scores,y[rows,i])}for i,name in enumerate(names)]
            confirmed.append(result)
        report['families'][family]=confirmed
    write_json(sae/'chemical_features_test_confirmation.json',report)
    return report


def retrieval_geometry(features:np.ndarray,panel:dict,labels:dict,feature_ids:np.ndarray|None=None)->dict[str,np.ndarray]:
    """Label-free cosine nearest train molecule; labels only score fixed neighbors.

    Latent maxima aggregate peak-level SAE activations, not mean residual tokens.
    No test-tuned distance transform or IDF weighting is applied. Exact cosine
    ties receive the mean chemistry score across all tied nearest candidates.
    """
    train=np.flatnonzero(panel['fold']=='train');test=np.flatnonzero(panel['fold']=='test')
    if set(panel['molecule'][train])&set(panel['molecule'][test]):
        raise ValueError('Retrieval molecule identity crosses train/test')
    x=np.asarray(features[train],np.float32);q=np.asarray(features[test],np.float32)
    if feature_ids is not None:x,q=x[:,feature_ids],q[:,feature_ids]
    norm=np.linalg.norm(x,axis=1);qnorm=np.linalg.norm(q,axis=1)
    good=norm>0;x=x[good]/norm[good,None];bank=train[good]
    if not len(bank):raise ValueError('No nonzero training feature vectors')
    q=np.divide(q,qnorm[:,None],out=np.zeros_like(q),where=qnorm[:,None]>0)
    result={'test_rows':test,'nearest_train_row':np.full(len(test),-1,np.int64),
        'nearest_cosine':np.full(len(test),np.nan),'tie_count':np.zeros(len(test),np.int32),
        'morgan_tanimoto':np.full(len(test),np.nan),'functional_group_jaccard':np.full(len(test),np.nan),
        'precursor_mass_gap_da':np.full(len(test),np.nan),'eligible':qnorm>0}
    fg=labels['binary'][:,len(labels['element_names']):].astype(bool)
    morgan=labels['morgan'].astype(bool)
    for start in range(0,len(test),128):
        similarity=q[start:start+128]@x.T
        for offset,sim in enumerate(similarity):
            position=start+offset;row=test[position]
            if qnorm[position]==0:continue
            best=np.max(sim);neighbors=bank[sim==best]
            result['nearest_train_row'][position]=neighbors[0]
            result['nearest_cosine'][position]=best;result['tie_count'][position]=len(neighbors)
            a=np.logical_and(morgan[neighbors],morgan[row]).sum(1)
            b=np.logical_or(morgan[neighbors],morgan[row]).sum(1)
            result['morgan_tanimoto'][position]=np.mean(np.divide(a,b,out=np.ones(len(a)),where=b>0))
            a=np.logical_and(fg[neighbors],fg[row]).sum(1);b=np.logical_or(fg[neighbors],fg[row]).sum(1)
            result['functional_group_jaccard'][position]=np.mean(np.divide(a,b,out=np.ones(len(a)),where=b>0))
            result['precursor_mass_gap_da'][position]=np.mean(np.abs(panel['precursor_mz_raw'][neighbors]-panel['precursor_mz_raw'][row]))
    return result


def analyze_retrieval(sae:Path,panel_path:Path,bootstrap:int=1000)->dict:
    if bootstrap<1000:raise ValueError('Use at least1000 paired molecule bootstrap resamples')
    selection=json.loads((sae/'chemical_features.json').read_text())
    panel,labels,_=load_panel(panel_path);representations={}
    for name,metadata in selection['baseline_files'].items():
        path=Path(metadata['path'])
        if sha256(path)!=metadata['sha256']:raise ValueError('Representation changed after discovery')
        features=np.load(path,mmap_mode='r')
        representations[name]=retrieval_geometry(features,panel,labels)
        if name=='sae':
            representations['sae_equal_dense_search_budget']=retrieval_geometry(features,panel,labels,
                np.asarray(selection['equal_dense_sae_feature_ids']))
    rng=np.random.default_rng(SEED);n=len(next(iter(representations.values()))['test_rows'])
    counts=rng.multinomial(n,np.full(n,1/n),size=bootstrap)
    report={'selection_sha256':sha256(sae/'chemical_features.json'),'pool':'All fixed4096 training molecules with nonzero feature vectors',
        'metric':'Raw L2-normalized cosine nearest train molecule; exact cosine ties averaged for chemical scores',
        'geometry':'Molecular maximum over token SAE activations; token-aware latent pooling, not raw residual-token mean pooling',
        'target_use':'Morgan and functional-group labels score neighbors only; never determine normalization, distance, features, or candidate choice',
        'jaccard_empty_sets':'Two empty fingerprints/groups receive1; one empty receives0',
        'bootstrap':{'replicates':bootstrap,'seed':SEED,'unit':'one test molecule','paired':True},
        'representations':{},'contrasts':{}}
    metrics=('morgan_tanimoto','functional_group_jaccard','precursor_mass_gap_da')
    for name,rows in representations.items():
        point={}
        for metric in metrics:
            finite=np.isfinite(rows[metric]);weights=counts[:,finite]
            samples=weights@rows[metric][finite]/weights.sum(1)
            point[metric]={'mean':float(np.mean(rows[metric][finite])),
                'ci95':np.quantile(samples,[.025,.975]).tolist(),'eligible_test_molecules':int(finite.sum())}
        report['representations'][name]={'metrics':point,'max_nearest_tie_count':int(rows['tie_count'].max())}
    comparisons=(('sae','random_dictionary'),('sae_equal_dense_search_budget','dense_channel'),('sae','dense_channel'))
    for left,right in comparisons:
        record={}
        for metric in metrics:
            a,b=representations[left][metric],representations[right][metric]
            finite=np.isfinite(a)&np.isfinite(b);weights=counts[:,finite];difference=a[finite]-b[finite]
            samples=weights@difference/weights.sum(1)
            record[metric]={'left_minus_right':float(difference.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
                            'paired_test_molecules':int(finite.sum())}
        report['contrasts'][left+'_minus_'+right]=record
    write_json(sae/'chemical_geometry.json',report)
    np.savez_compressed(sae/'chemical_geometry_rows.npz',**{name+'/'+key:array for name,rows in representations.items()for key,array in rows.items()})
    return report


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sae',type=Path,required=True)
    parser.add_argument('--panel',type=Path,default=Path('artifacts/encoder_chemistry_20260930/panel'))
    parser.add_argument('--mode',choices=('select','confirm','both','retrieval'),default='select')
    args=parser.parse_args()
    if args.mode in ('select','both'):
        result=discover(args.sae,args.panel);print(json.dumps(result['selection_counts']),flush=True)
    if args.mode=='retrieval':
        analyze_retrieval(args.sae,args.panel)
    if args.mode in ('confirm','both'):
        result=confirm(args.sae,args.panel);print('Confirmed test feature counts', {k:len(v)for k,v in result['families'].items()},flush=True)


if __name__=='__main__':main()
