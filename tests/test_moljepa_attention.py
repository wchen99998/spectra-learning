import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from moljepa_attention import ChemicalBatch,RetrievalModel,maxsim,spectrum_tokens

def test_cls_is_always_valid_and_affects_both_reductions():
    p=torch.tensor([[[1.,0.],[99.,99.]]]); c=torch.tensor([[0.,1.]])
    p,m=spectrum_tokens(p,c,torch.tensor([[True,False]]))
    mol=torch.tensor([[[[0.,1.]]]])
    assert m.tolist()==[[True,False,True]]
    torch.testing.assert_close(maxsim(p,m,mol),torch.tensor([[.75]]))

def test_padding_is_invisible_to_self_attention():
    torch.manual_seed(0)
    model=RetrievalModel(width=16,heads=4,dropout=0).eval()
    x=torch.randn(2,5,2048);v=torch.tensor([[True,True,False,False,True],[True,False,True,False,True]])
    altered=x.clone();altered[~v]=1e9
    torch.testing.assert_close(model.spectrum(x,v),model.spectrum(altered,v))
    assert not model.spectrum(x,v)[~v].any()

def test_chemistry_gradients_and_shared_initialization():
    torch.manual_seed(0);plain=RetrievalModel(width=16,heads=4,dropout=0)
    torch.manual_seed(0);extra=RetrievalModel(True,width=16,heads=4,dropout=0)
    for k,v in plain.state_dict().items():torch.testing.assert_close(v,extra.state_dict()[k])
    chem=ChemicalBatch(torch.randn(2,3,49),torch.tensor([[True,True,False],[True,True,True]]),torch.tensor([[0,1,3,4],[1,0,4,3]]),torch.randn(4,8))
    z=torch.randn(2,13,512);valid=torch.ones(2,13,dtype=torch.bool)
    result=extra.molecule(z,valid,chem)
    (result*torch.randn_like(result)).sum().backward()
    for name,p in extra.named_parameters():
        if name.startswith('molecule.'):
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert extra.molecule.memory.neighbor[0].weight.grad.abs().sum()>0
    changed=ChemicalBatch(chem.atoms.clone(),chem.valid,chem.edge_index,chem.bonds)
    changed.atoms[~chem.valid]=1e8
    torch.testing.assert_close(extra.molecule(z,valid,changed),result)

def test_uniform_score_is_candidate_independent_and_permutation_equivariant():
    torch.manual_seed(1)
    p=torch.randn(2,5,8);m=torch.ones(2,5,dtype=torch.bool);z=torch.randn(2,4,13,8)
    scores=maxsim(p,m,z)
    torch.testing.assert_close(scores[:,[2,0]],maxsim(p,m,z[:,[2,0]]))
    torch.testing.assert_close(scores,maxsim(p,m,z[:,:,:,torch.arange(8)]))
    torch.testing.assert_close(scores,maxsim(p,m,z[:,:,torch.randperm(13)]))

def test_retrieval_logits_stay_float32_inside_autocast():
    p=torch.randn(1,3,8);m=torch.randn(1,2,13,8);valid=torch.ones(1,3,dtype=torch.bool)
    expected=maxsim(p,valid,m)
    with torch.autocast('cpu',dtype=torch.bfloat16):actual=maxsim(p,valid,m)
    assert actual.dtype==torch.float32
    torch.testing.assert_close(actual,expected)

def test_mass_transfer_masks_padding_and_cls_and_has_finite_gradients():
    from moljepa_attention import ModalityMassAgreement
    model=ModalityMassAgreement(width=16,map_dim=4)
    chem=ChemicalBatch(torch.randn(1,3,49),torch.tensor([[True,True,False]]),torch.tensor([[0,1],[1,0]]),torch.randn(2,8))
    mol=torch.randn(1,13,16,requires_grad=True)
    weights=model.atom_weights(mol,chem)
    torch.testing.assert_close(weights.sum(-1),torch.ones(1,13))
    assert not weights[:,:,2].any()
    fractions=torch.tensor([[[[.5,1.,-1.,-1.,-1.,-1.],[.25,1.,-1.,-1.,-1.,-1.],[-1.,-1.,-1.,-1.,-1.,-1.]]]])
    bonus=model.bonus(torch.tensor([[.5,.2,0.]]),fractions,chem.valid[:,None],weights[:,None])
    assert bonus.shape==(1,1,3,13)
    assert not bonus[:,:,-1].any()
    changed=fractions.clone();changed[:,:,2]=.5
    torch.testing.assert_close(bonus,model.bonus(torch.tensor([[.5,.2,0.]]),changed,chem.valid[:,None],weights[:,None]))
    bonus.sum().backward()
    for name,param in model.named_parameters():assert param.grad is not None and torch.isfinite(param.grad).all(),name

def test_agreement_disabled_recovers_parent_readouts_exactly():
    torch.manual_seed(3);parent=RetrievalModel(width=16,heads=4,dropout=0).eval()
    extended=RetrievalModel(width=16,heads=4,dropout=0,agreement=True).eval()
    missing,unexpected=extended.load_state_dict(parent.state_dict(),strict=False)
    assert not unexpected and all(k.startswith('mass.') for k in missing)
    tokens=torch.randn(1,4,2048);valid=torch.ones(1,4,dtype=torch.bool)
    z=torch.randn(1,13,512);mv=torch.ones(1,13,dtype=torch.bool)
    a=maxsim(parent.spectrum(tokens,valid),valid,parent.molecule(z,mv)[:,None])
    b=maxsim(extended.spectrum(tokens,valid),valid,extended.molecule(z,mv)[:,None])
    torch.testing.assert_close(a,b,atol=0,rtol=0)
