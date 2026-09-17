"""DGT-inspired dual atom/bond attention with gated frozen Mol-JEPA context.

The channel-wise edge-aware attention follows DGT (MIT), commit
2f5a4c3ef2e376d60238f481c36a87eeb5a581e8. This retrieval adaptation uses
LayerNorm, dynamic atom/bond exchange, and keeps all atom outputs.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from moljepa_attention import TokenReadout

@dataclass
class GraphBatch:
    atoms: torch.Tensor
    atom_valid: torch.Tensor
    bonds: torch.Tensor  # B,E,2,57; both orientations of each bond
    bond_valid: torch.Tensor
    endpoints: torch.Tensor
    moljepa: torch.Tensor
    fractions: torch.Tensor

def structural_features(adjacency):
    """Six-step shortest-path bins and sixteen full random-walk kernels."""
    with torch.no_grad(), torch.autocast(adjacency.device.type, enabled=False):
        adjacency = adjacency.float()
        transition = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1)
        walk = transition
        distance = torch.full(adjacency.shape, 7, device=adjacency.device, dtype=torch.long)
        eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=torch.bool)[None]
        distance.masked_fill_(eye, 0)
        powers = []
        for k in range(1, 17):
            powers.append(walk)
            if k <= 6: distance = torch.where((distance == 7) & (walk > 0), k, distance)
            if k < 16: walk = walk @ transition
        return distance, torch.stack(powers, -1)

class EdgeAttention(nn.Module):
    """DGT attention: a head dot product plus a separate bias for every channel."""
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3*width)
        self.output = nn.Linear(width, width)
        self.dropout = dropout

    def forward(self, x, valid, bias, edge_values):
        b,n,d = x.shape; h = self.heads; k = d//h
        q,key,v = self.qkv(x).reshape(b,n,3,h,k).unbind(2)
        base = torch.einsum('bihk,bjhk->bijh', q, key)*(k**-.5)
        logits = base[...,None] + bias.reshape(b,n,n,h,k)
        # A harmless dummy key prevents all-masked softmax for bondless molecules.
        safe = valid.clone(); safe[:,0] |= ~valid.any(-1)
        logits = logits.masked_fill(~safe[:,None,:,None,None], -torch.inf)
        weights = logits.float().softmax(2).to(v.dtype)
        weights = F.dropout(weights, self.dropout, self.training)
        value = torch.einsum('bijhk,bjhk->bihk', weights, v)
        value = value + (weights*edge_values.reshape(b,n,n,h,k)).sum(2)
        return self.output(value.reshape(b,n,d)).masked_fill(~valid[...,None],0)

class StreamBlock(nn.Module):
    def __init__(self,width,heads,dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width); self.norm2 = nn.LayerNorm(width)
        self.attention = EdgeAttention(width,heads,dropout)
        self.ffn = nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*width,width))
        self.drop = nn.Dropout(dropout)
    def forward(self,x,valid,bias,values):
        x = x+self.drop(self.attention(self.norm1(x),valid,bias,values))
        return (x+self.drop(self.ffn(self.norm2(x)))).masked_fill(~valid[...,None],0)

class DualBlock(nn.Module):
    def __init__(self,width,heads,dropout,cross):
        super().__init__()
        self.atom = StreamBlock(width,heads,dropout)
        self.bond = StreamBlock(width,heads,dropout)
        self.atom_exchange = nn.Sequential(nn.LayerNorm(2*width),nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,width))
        self.bond_exchange = nn.Sequential(nn.LayerNorm(2*width),nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,width))
        self.drop = nn.Dropout(dropout)
        self.cross = cross
        if cross:
            self.cross_norm = nn.LayerNorm(width)
            self.cross_attention = nn.MultiheadAttention(width,8,dropout=dropout,batch_first=True)
            self.context_gate = nn.Parameter(torch.tensor(.01))
    def forward(self,a,e,av,ev,ab,ae,eb,ee,incidence,memory):
        a = self.atom(a,av,ab,ae); e = self.bond(e,ev,eb,ee)
        atom_context = incidence@e / incidence.sum(-1,keepdim=True).clamp_min(1)
        bond_context = incidence.transpose(1,2)@a / 2
        a = a+self.drop(self.atom_exchange(torch.cat((a,atom_context),-1)))
        e = e+self.drop(self.bond_exchange(torch.cat((e,bond_context),-1)))
        if self.cross:
            context = self.cross_attention(self.cross_norm(a),memory,memory,need_weights=False)[0]
            a = a+self.context_gate.tanh()*self.drop(context)
        return a.masked_fill(~av[...,None],0),e.masked_fill(~ev[...,None],0)

class MolecularDGT(nn.Module):
    def __init__(self,width=512,layers=8,heads=64,dropout=.2):
        super().__init__()
        self.environments = nn.ModuleList([nn.Embedding(32768,64,padding_idx=0) for _ in range(4)])
        self.atom_input = nn.Linear(127+4*64,width)
        self.bond_input = nn.Linear(57,width)
        self.atom_initial = nn.Linear(2*width,width)
        self.bond_initial = nn.Linear(2*width,width)
        self.atom_distance = nn.Embedding(8,2*width)
        self.bond_distance = nn.Embedding(8,2*width)
        self.atom_walk = nn.Linear(16,2*width,bias=False)
        self.bond_walk = nn.Linear(16,2*width,bias=False)
        self.edge_pair = nn.Linear(57,2*width,bias=False)
        self.node_pair = nn.Linear(width,2*width,bias=False)
        self.memory = nn.Sequential(nn.LayerNorm(512),nn.Linear(512,width),nn.LayerNorm(width))
        self.blocks = nn.ModuleList([DualBlock(width,heads,dropout,cross=(i%2==0)) for i in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width))
        self.width = width

    def forward(self,g: GraphBatch):
        x = torch.cat((g.atoms[...,:127],*[embed(g.atoms[...,127+i].long()) for i,embed in enumerate(self.environments)]),-1)
        a = self.atom_input(x).masked_fill(~g.atom_valid[...,None],0)
        directed = self.bond_input(g.bonds).masked_fill(~g.bond_valid[...,None,None],0)
        e = directed.mean(2)
        b,n,_ = a.shape; count = e.shape[1]
        incidence = a.new_zeros(b,n,count)
        bi = torch.arange(b,device=a.device)[:,None].expand(b,count)
        ei = torch.arange(count,device=a.device)[None].expand(b,count)
        u,v = g.endpoints.unbind(-1)
        incidence[bi,u,ei] = g.bond_valid.to(a.dtype)
        incidence[bi,v,ei] = g.bond_valid.to(a.dtype)
        adjacency = (incidence@incidence.transpose(1,2))>0
        adjacency.diagonal(dim1=-2,dim2=-1).zero_()
        line = (incidence.transpose(1,2)@incidence)>0
        line.diagonal(dim1=-2,dim2=-1).zero_()
        # Oriented bond descriptors contribute to their destination atoms.
        local = a.new_zeros(b,n,self.width)
        flat = local.reshape(b*n,self.width)
        flat.index_add_(0,(bi*n+v).reshape(-1),directed[:,:,0].reshape(-1,self.width))
        flat.index_add_(0,(bi*n+u).reshape(-1),directed[:,:,1].reshape(-1,self.width))
        a = self.atom_initial(torch.cat((a,local),-1))
        e = self.bond_initial(torch.cat((e,incidence.transpose(1,2)@a),-1))
        ad,aw = structural_features(adjacency); ed,ew = structural_features(line)
        atom_pair = self.atom_distance(ad)+self.atom_walk(aw)
        bond_pair = self.bond_distance(ed)+self.bond_walk(ew)
        ep = self.edge_pair(g.bonds)*g.bond_valid[...,None,None]
        atom_pair[bi,u,v] = atom_pair[bi,u,v]+ep[:,:,0]
        atom_pair[bi,v,u] = atom_pair[bi,v,u]+ep[:,:,1]
        # Shared atoms supply chemical context to neighboring bond pairs.
        shared = torch.einsum('bne,bnd,bnf->befd',incidence,self.node_pair(a),incidence)
        bond_pair = bond_pair+shared
        ab,ae = atom_pair.chunk(2,-1); eb,ee = bond_pair.chunk(2,-1)
        memory = self.memory(g.moljepa)
        for block in self.blocks:
            args = (a,e,g.atom_valid,g.bond_valid,ab,ae,eb,ee,incidence,memory)
            a,e = checkpoint(block,*args,use_reentrant=False) if self.training else block(*args)
        return F.normalize(self.output(a).float(),dim=-1).masked_fill(~g.atom_valid[...,None],0)

class AtomAgreement(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(.03).log())
        self.log_beta = nn.Parameter(torch.tensor(.5).log())
    def forward(self,peak_fractions,atom_fractions):
        fp = peak_fractions[:,None,:,None].float()
        distance = fp.new_full((*atom_fractions.shape[:2],fp.shape[2],atom_fractions.shape[2]),1e9)
        for k in range(6):
            af = atom_fractions[...,k][:,:,None,:].float()
            distance = torch.minimum(distance,(fp-af).abs().masked_fill(af<0,1e9))
        bonus = self.log_beta.exp()*torch.exp(-(distance/self.log_sigma.exp()).square())
        physical = torch.arange(bonus.shape[2],device=bonus.device)<127
        return bonus*physical[None,None,:,None]

class Model(nn.Module):
    def __init__(self,width=512,layers=8,heads=64,dropout=.2):
        super().__init__()
        self.spectrum = TokenReadout(2048,width,8,dropout)
        self.molecule = MolecularDGT(width,layers,heads,dropout)
        self.global_head = nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,1024),nn.GELU(),nn.Dropout(dropout),nn.Linear(1024,width))
        self.mass = AtomAgreement()

def compile_graph_blocks(model):
    """Compile the dual blocks without changing module/checkpoint parameter names.

    Independent symbolic dimensions avoid the installed Inductor's CantSplit
    failure when batch size happens to equal the number of atoms or bonds.
    Graph construction stays eager; it contains an unsupported 3.14 context.
    """
    import torch.fx.experimental._config as symbolic_config
    symbolic_config.use_duck_shape = False
    for block in model.molecule.blocks:
        block.compile(dynamic=True)
