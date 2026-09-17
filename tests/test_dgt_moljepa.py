import sys
from pathlib import Path
import numpy as np
import torch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'results/dgt_moljepa_atom_latest/runtime')]
from dgt_chemistry import features
from dgt_data import pack_graphs
from dgt_moljepa import MolecularDGT,AtomAgreement,EdgeAttention,structural_features

def test_atom_equivariance_padding_and_context_gradient():
    torch.set_num_threads(1); torch.manual_seed(3)
    graphs=[features('CC(=O)Oc1ccccc1C(=O)O'),features('[Na+].[Cl-]')]
    memory=torch.randn(2,13,512)
    model=MolecularDGT(width=32,layers=2,heads=4,dropout=0).eval()
    batch=pack_graphs(graphs,memory,'cpu'); original=model(batch)
    x,ei,ea,af=graphs[0]; permutation=np.random.default_rng(4).permutation(len(x)); inverse=np.argsort(permutation)
    changed=[(x[permutation],inverse[ei],ea,af[permutation]),graphs[1]]
    permuted=model(pack_graphs(changed,memory,'cpu'))
    torch.testing.assert_close(permuted[0,:len(x)],original[0,permutation],atol=2e-6,rtol=2e-5)
    alone=model(pack_graphs([graphs[1]],memory[1:],'cpu'))
    torch.testing.assert_close(alone[0],original[1,:2],atol=2e-6,rtol=2e-5)
    assert not original[1,2:].count_nonzero()
    model.train(); result=model(batch); (result*torch.randn_like(result)).sum().backward()
    assert torch.isfinite(result).all()
    for block in model.blocks:
        if block.cross:
            assert block.context_gate.grad.abs()>0
            assert block.cross_attention.in_proj_weight.grad.abs().sum()>0
    assert model.bond_input.weight.grad.abs().sum()>0


def test_bond_orientation_and_full_pool_evidence():
    from run_dgt_flare import components
    from flare_candidate import candidate_scores
    torch.manual_seed(2)
    graph=features('CC(=O)Oc1ccccc1')
    x,ei,ea,af=graph
    reversed_graph=(x,ei[:,::-1].copy(),ea.reshape(-1,2,57)[:,::-1].reshape(-1,57).copy(),af)
    model=MolecularDGT(width=32,layers=2,heads=4,dropout=0).eval()
    memory=torch.randn(1,13,512)
    a=model(pack_graphs([graph],memory,'cpu'))
    b=model(pack_graphs([reversed_graph],memory,'cpu'))
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    q=torch.randn(2,9,16); atoms=torch.randn(2,7,5,16)
    pm=torch.ones(2,9,dtype=torch.bool); am=torch.ones(2,7,5,dtype=torch.bool)
    am[:,2,-2:]=False;pm[:,-1]=False
    full=components(q,pm,atoms,am)
    chunks=[components(q,pm,atoms[:,start:start+3],am[:,start:start+3]) for start in range(0,7,3)]
    merged=[torch.cat([c[i] for c in chunks],1) for i in range(2)]
    valid=torch.ones(2,7,dtype=torch.bool)
    torch.testing.assert_close(candidate_scores(*full,pm,valid,'evidence'),candidate_scores(*merged,pm,valid,'evidence'))

def test_mass_cls_and_atom_correspondence():
    module=AtomAgreement()
    af=torch.tensor([[[[.5,1,-1,-1,-1,-1],[.8,1,-1,-1,-1,-1]]]])
    peaks=torch.full((1,128),.5)
    bonus=module(peaks,af)
    torch.testing.assert_close(bonus[0,0,0,0],torch.tensor(.5))
    assert bonus[0,0,0,1]<1e-6
    assert bonus[:,:,127].count_nonzero()==0
    bonus.sum().backward()
    assert torch.isfinite(module.log_sigma.grad)

def test_channel_attention_reference_and_disconnected_distances():
    torch.manual_seed(7); layer=EdgeAttention(8,2,0).eval()
    x=torch.randn(1,3,8); bias=torch.randn(1,3,3,8); values=torch.randn(1,3,3,8)
    valid=torch.tensor([[True,True,False]])
    q,k,v=layer.qkv(x).reshape(1,3,3,2,4).unbind(2)
    reference=torch.zeros_like(x)
    for i in range(2):
        for h in range(2):
            for c in range(4):
                logits=torch.stack([(q[0,i,h]*k[0,j,h]).sum()/2+bias[0,i,j,h*4+c] for j in range(2)])
                reference[0,i,h*4+c]=sum(logits.softmax(0)[j]*(v[0,j,h,c]+values[0,i,j,h*4+c]) for j in range(2))
    reference=layer.output(reference).masked_fill(~valid[...,None],0)
    torch.testing.assert_close(layer(x,valid,bias,values),reference)
    adj=torch.tensor([[[0,1,0],[1,0,0],[0,0,0]]])
    distance,walk=structural_features(adj)
    assert distance[0,0,1]==1 and distance[0,0,2]==7 and distance[0,2,2]==0
    assert walk[0,0,0,1]==1 and walk[0,2].count_nonzero()==0
