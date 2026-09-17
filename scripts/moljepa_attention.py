"""Two-layer token readouts and uniform, bidirectional MaxSim retrieval."""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ChemicalBatch:
    atoms: torch.Tensor
    valid: torch.Tensor
    edge_index: torch.Tensor
    bonds: torch.Tensor

class ReadoutBlock(nn.Module):
    def __init__(self,width:int,heads:int,dropout:float,cross:bool=False):
        super().__init__()
        self.norm1=nn.LayerNorm(width)
        self.self_attention=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
        self.norm2=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*width,width))
        self.drop=nn.Dropout(dropout)
        self.cross=cross
        if cross:
            self.cross_norm=nn.LayerNorm(width)
            self.memory_norm=nn.LayerNorm(width)
            self.cross_attention=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
    def forward(self,x,valid,memory=None,memory_valid=None):
        h=self.norm1(x)
        x=x+self.drop(self.self_attention(h,h,h,key_padding_mask=~valid,need_weights=False)[0])
        if self.cross:
            m=self.memory_norm(memory)
            x=x+self.drop(self.cross_attention(self.cross_norm(x),m,m,key_padding_mask=~memory_valid,need_weights=False)[0])
        x=x+self.drop(self.ffn(self.norm2(x)))
        return x.masked_fill(~valid[...,None],0)

class ChemicalMemory(nn.Module):
    """Original 49 atom / 8 bond features with one-hop, directed neighborhood context."""
    def __init__(self,width:int):
        super().__init__()
        self.atom=nn.Linear(49,width)
        self.neighbor=nn.Sequential(nn.Linear(49+8,width),nn.GELU(),nn.Linear(width,width))
        self.norm=nn.LayerNorm(width)
    def forward(self,batch:ChemicalBatch):
        b,a,_=batch.atoms.shape
        flat=batch.atoms.reshape(b*a,49)
        src,dst=batch.edge_index
        message=self.neighbor(torch.cat((flat[src],batch.bonds),-1))
        summed=torch.zeros(b*a,self.atom.out_features,device=flat.device,dtype=message.dtype).index_add(0,dst,message)
        degree=torch.zeros(b*a,1,device=flat.device,dtype=message.dtype).index_add(0,dst,torch.ones(len(dst),1,device=flat.device,dtype=message.dtype))
        out=self.atom(flat)+summed/degree.clamp_min(1)
        return self.norm(out.reshape(b,a,-1)).masked_fill(~batch.valid[...,None],0)

class TokenReadout(nn.Module):
    def __init__(self,input_dim:int,width:int=512,heads:int=8,dropout:float=.1,chemical:bool=False):
        super().__init__()
        self.input=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,width))
        # Initialize all shared weights before extra pathways so matched seeds are identical.
        self.blocks=nn.ModuleList([ReadoutBlock(width,heads,dropout) for _ in range(2)])
        self.output=nn.LayerNorm(width)
        self.chemical=chemical
        if chemical:
            with torch.random.fork_rng(devices=[]):
                self.memory=ChemicalMemory(width)
                for block in self.blocks:
                    block.cross=True
                    block.cross_norm=nn.LayerNorm(width)
                    block.memory_norm=nn.LayerNorm(width)
                    block.cross_attention=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
    def forward(self,tokens,valid,chemistry:ChemicalBatch|None=None):
        x=self.input(tokens.masked_fill(~valid[...,None],0))
        memory=self.memory(chemistry) if self.chemical else None
        memory_valid=chemistry.valid if self.chemical else None
        for block in self.blocks:x=block(x,valid,memory,memory_valid)
        return F.normalize(self.output(x).float(),dim=-1).masked_fill(~valid[...,None],0)

class RetrievalModel(nn.Module):
    def __init__(self,chemical:bool=False,width:int=512,heads:int=8,dropout:float=.1,agreement:bool=False):
        super().__init__()
        self.spectrum=TokenReadout(2048,width,heads,dropout)
        self.molecule=TokenReadout(512,width,heads,dropout,chemical)
        self.mass=ModalityMassAgreement(width) if agreement else None

def spectrum_tokens(peaks,cls,peak_valid):
    tokens=torch.cat((peaks,cls[:,None]),1)
    valid=torch.cat((peak_valid,torch.ones(len(peaks),1,dtype=torch.bool,device=peaks.device)),1)
    return tokens,valid

def maxsim(peaks,peak_valid,molecules,candidate_valid=None,bonus=None):
    """Molecules are [batch,candidate,13,width]; all 13 modality/CLS tokens are valid."""
    # Retrieval logits must remain float32 even inside an autocast training forward.
    with torch.autocast(peaks.device.type,enabled=False):
        similarity=torch.einsum('bpd,bctd->bcpt',peaks.float(),molecules.float())
    if bonus is not None:similarity=similarity+bonus
    peak_term=(similarity.amax(-1)*peak_valid[:,None]).sum(-1)/peak_valid.sum(-1)[:,None]
    mol_term=similarity.masked_fill(~peak_valid[:,None,:,None],-torch.inf).amax(-2).mean(-1)
    scores=.5*(peak_term+mol_term)
    return scores if candidate_valid is None else scores.masked_fill(~candidate_valid,-torch.inf)

class ModalityMassAgreement(nn.Module):
    """Transfer the original atom mass bonus to modality tokens through attention."""
    def __init__(self,width:int=512,map_dim:int=64):
        super().__init__()
        self.memory=ChemicalMemory(width)
        self.query=nn.Linear(width,map_dim)
        self.key=nn.Linear(width,map_dim)
        self.log_sigma=nn.Parameter(torch.tensor(.03).log())
        self.log_beta=nn.Parameter(torch.tensor(.5).log())
    def atom_weights(self,molecules,chemistry):
        memory=self.memory(chemistry)
        with torch.autocast(molecules.device.type,enabled=False):
            logits=torch.einsum('btd,bad->bta',self.query(molecules.float()),self.key(memory.float()))/self.query.out_features**.5
        return logits.masked_fill(~chemistry.valid[:,None],-torch.inf).softmax(-1)
    def bonus(self,peak_fractions,atom_fractions,atom_valid,weights):
        # Shapes: fp B,P; fractions B,C,A,6; weights B,C,T,A.
        with torch.autocast(peak_fractions.device.type,enabled=False):
            fp=peak_fractions.float()[:,None,:,None]
            af=atom_fractions.float()
            distance=fp.new_full((*af.shape[:2],fp.shape[2],af.shape[2]),1e9)
            for k in range(af.shape[-1]):
                f=af[...,k][:,:,None,:]
                distance=torch.minimum(distance,(fp-f).abs().masked_fill(f<0,1e9))
            atom_bonus=self.log_beta.exp()*torch.exp(-(distance/self.log_sigma.exp()).square())
            atom_bonus=atom_bonus.masked_fill(~atom_valid[:,:,None],0)
            result=torch.einsum('bcpa,bcta->bcpt',atom_bonus,weights.float())
            # CLS has no physical m/z, but still participates in both reductions.
            physical=torch.ones(result.shape[-2],dtype=torch.bool,device=result.device);physical[-1]=False
            return result.masked_fill(~physical[None,None,:,None],0)
