import copy

import numpy as np
import pytest

from scripts.analyze_encoder_chemistry_diffusion import (
    BASELINES, stage_immutable_file, summarize_readouts, validate_sae_contract,
)


def test_three_arm_bootstrap_cancels_identical_predictions_and_excludes_rare_target():
    rng=np.random.default_rng(14)
    labels=np.tile(np.array([[0,1,0,0],[1,0,1,0]]),(24,1))
    labels[:4,3]=1
    logits=rng.integers(-2,3,size=(3,*labels.shape)).astype(float)
    predictions={(arm,27,kind):logits for arm in (*BASELINES,"diffusion_toy") for kind in ("mean","attention")}
    result,_=summarize_readouts(predictions,labels,labels,
        ["maccs_001","fg_amide","elem_N","maccs_002"],"diffusion_toy")
    assert result["groups"]["MACCS"]["names"]==["maccs_001"]
    assert len(result["models"])==6
    assert len(result["contrasts"])==7
    for contrast in result["contrasts"].values():
        for group in contrast["groups"].values():
            for method in group.values():
                assert method["estimate"]==0
                assert method["ci95"]==[0,0]


def test_sae_contract_rejects_mismatched_panel_and_nonexact_cache():
    hashes={"inputs.npz":"input","labels.npz":"labels","manifest.json":"manifest","matched_controls.npz":"controls"}
    metadata={"complete":True,"contract":{"encoder":"diffusion_toy","depth":19,
        "features":8192,"k":32,"seed":0,"panel_sha256":hashes,
        "cache_final_parity":{key:{"finite":True,"bitwise_equal":True} for key in ("native","trace")}}}
    validate_sae_contract(metadata,"diffusion_toy",32,0,hashes)
    wrong=copy.deepcopy(metadata)
    wrong["contract"]["panel_sha256"]["inputs.npz"]="different"
    with pytest.raises(ValueError,match="training/export panel"):
        validate_sae_contract(wrong,"diffusion_toy",32,0,hashes)
    wrong=copy.deepcopy(metadata)
    wrong["contract"]["cache_final_parity"]["native"]["bitwise_equal"]=False
    with pytest.raises(ValueError,match="bitwise-qualified"):
        validate_sae_contract(wrong,"diffusion_toy",32,0,hashes)


def test_staged_analysis_view_cannot_overwrite_existing_data(tmp_path):
    source=tmp_path/"source.json"
    destination=tmp_path/"extension"/"source.json"
    source.write_text('{"checkpoint":831000}')
    stage_immutable_file(source,destination)
    source.write_text('{"checkpoint":832000}')
    with pytest.raises(ValueError,match="differs from source"):
        stage_immutable_file(source,destination)
    assert destination.read_text()=='{"checkpoint":831000}'
