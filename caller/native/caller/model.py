"""Independent allele-sequence CNN and molecular-evidence neural caller.

This is a supervised residual model, not a multiplication of independent
posteriors. External-only LD inference is an input feature/initial anchor.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import numpy as np
import torch
from torch import nn

MODEL_VERSION='0.2.0-neural-cohort'
MAX_ALLELES=12
MAX_GENOTYPES=78
PAIR_I=torch.tensor([i for j in range(MAX_ALLELES) for i in range(j+1)])
PAIR_J=torch.tensor([j for j in range(MAX_ALLELES) for i in range(j+1)])

@dataclass
class ModelConfig:
    hidden: int=48
    dropout: float=0.1
    cohort_dropout: float=0.35
    prior_dropout: float=0.1
    prior_floor: float=1e-7
    residual_scale: float=12.0

def mlp(inputs,hidden,outputs,dropout=0.):
    return nn.Sequential(nn.Linear(inputs,hidden),nn.SiLU(),nn.Dropout(dropout),nn.Linear(hidden,outputs))

class NeuralCaller(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or ModelConfig();h=self.config.hidden
        self.bases=nn.Embedding(6,8,padding_idx=0)
        self.sequence=nn.Sequential(nn.Conv1d(8,24,7,padding=3),nn.SiLU(),
                                    nn.Conv1d(24,h,5,padding=2),nn.SiLU())
        # One shared molecular encoder per candidate allele. Molecule order and
        # donor order carry no information about a genotype.
        self.molecule=mlp(10,h,h,self.config.dropout)
        self.molecule_weight=nn.Linear(h,1)
        self.allele=mlp(2*h+8,h,h,self.config.dropout)
        self.cohort=mlp(6,h,h,self.config.dropout)
        self.context_gate=mlp(2*h+3,h,1)
        self.pair_read=mlp(3*h+5,h,1,self.config.dropout)
        self.pair_joint=mlp(6*h+7,2*h,1,self.config.dropout)
        self.register_buffer('pair_i',PAIR_I.clone());self.register_buffer('pair_j',PAIR_J.clone())
        # Begin at the external-only posterior, then learn a discriminative
        # correction. Random untrained weights must not silently produce calls.
        nn.init.zeros_(self.pair_joint[-1].weight);nn.init.zeros_(self.pair_joint[-1].bias)

    def compact_sequence_encoding(self,seq,am):
        """Reuse unchanged CNN positions; retain the full-width pooling order.

        Two convolutions (7 and 5 bases) have radius five. A short localized
        substitution needs only its changed output interval plus that halo.
        Padding-token masks must agree; other alleles use the full encoder.
        """
        active=seq[am]
        if not len(active):return self.bases.weight.new_zeros((0,seq.shape[-1],self.config.hidden))
        if seq.device.type!='cpu' or not bool(am[:,0].all()):
            return self.sequence(self.bases(active).transpose(1,2)).transpose(1,2)
        owners=torch.arange(len(seq),device=seq.device)[:,None].expand_as(am)[am]
        first=torch.cumsum(am.sum(1),0)-am.sum(1)
        reference=seq[:,0][owners];different=active!=reference;l=seq.shape[-1]
        positions=torch.arange(l,device=seq.device)[None,:]
        left=torch.where(different,positions,l).amin(1)
        right=torch.where(different,positions,-1).amax(1)
        patch=((active!=0)==(reference!=0)).all(1)&(left>=10)&(right<l-10)&(right>=left)&(right-left<=24)
        if not patch.any():return self.sequence(self.bases(active).transpose(1,2)).transpose(1,2)
        encoded=self.bases.weight.new_empty((len(active),l,self.config.hidden))
        full=~patch
        encoded[full]=self.sequence(self.bases(active[full]).transpose(1,2)).transpose(1,2)
        patch_ids=torch.nonzero(patch,as_tuple=True)[0]
        encoded[patch_ids]=encoded[first[owners[patch_ids]]]
        spans=right-left+21
        for size in torch.unique(spans[patch_ids]).tolist():
            ids=patch_ids[spans[patch_ids]==size]
            starts=left[ids]-10
            windows=active[ids[:,None],starts[:,None]+torch.arange(size,device=seq.device)[None,:]]
            values=self.sequence(self.bases(windows).transpose(1,2)).transpose(1,2)[:,5:-5]
            targets=starts[:,None]+5+torch.arange(size-10,device=seq.device)[None,:]
            encoded[ids[:,None],targets]=values
        return encoded

    def forward(self,batch,*,return_aux=False,cohort_enabled=True):
        seq=batch['allele_sequences'].long();am=batch['allele_mask'].bool()
        fm=batch['fragment_mask'].bool();ff=batch['fragment_features'].float()
        af=batch['allele_features'].float();depth=batch['depth'].float().reshape(-1,1)
        gm=batch['genotype_mask'].bool()
        if seq.ndim!=3 or seq.shape[1]!=12 or ff.shape[-1]!=20:raise ValueError('Feature schema mismatch')
        if not gm.any(1).all():raise ValueError('At least one valid genotype required per record')
        compact=bool(getattr(self,'compact_inference',False)) and not self.training
        need_read=return_aux or not compact
        if compact and not return_aux:
            # Remove only trailing slots which are masked in every batch row.
            # Base-GP normalization and the final 78-column softmax stay intact.
            active_alleles=max(1,int((am*torch.arange(1,13,device=am.device)).max()))
            active_fragments=max(1,int((fm*torch.arange(1,fm.shape[1]+1,device=fm.device)).max()))
            seq=seq[:,:active_alleles];am=am[:,:active_alleles];af=af[:,:active_alleles]
            fm=fm[:,:active_fragments];ff=ff[:,:active_fragments]
        b,a,l=seq.shape;h=self.config.hidden
        if compact:
            # Masked allele embeddings cannot influence any retained output.
            # Keep training unchanged; validate floating-point effects separately.
            active_seq=seq[am];se=self.bases.weight.new_zeros((b,a,h))
            if len(active_seq):
                encoded=self.compact_sequence_encoding(seq,am)
                sm=(active_seq!=0).unsqueeze(-1)
                se[am]=(encoded*sm).sum(1)/sm.sum(1).clamp_min(1)
        else:
            embed=self.bases(seq).reshape(b*a,l,8).transpose(1,2)
            encoded=self.sequence(embed).transpose(1,2)
            sm=(seq.reshape(b*a,l)!=0).unsqueeze(-1)
            se=(encoded*sm).sum(1)/sm.sum(1).clamp_min(1)
            se=se.reshape(b,a,h)*am.unsqueeze(-1)

        ll=ff[:,:,:a].clamp(-20,0)
        # Correctly ignore padded allele entries in the maximum.
        mx=ll.masked_fill(~am[:,None,:],-1e4).amax(2,keepdim=True)
        meta=ff[:,:,12:20].clamp(-10,10)[:,:,None,:].expand(-1,-1,a,-1)
        molecule_input=torch.cat((ll[:,:,:,None]/20,(ll-mx)[:,:,:,None]/20,meta),-1)
        if compact:
            keep=fm[:,:,None]&am[:,None,:];active_input=molecule_input[keep]
            molecular=ff.new_zeros((b,ff.shape[1],a,h))
            if len(active_input):molecular[keep]=self.molecule(active_input)
        else:molecular=self.molecule(molecule_input)
        scores=self.molecule_weight(molecular).squeeze(-1).masked_fill(~fm[:,:,None],-1e4)
        weights=torch.softmax(scores,dim=1)*fm[:,:,None]
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        re=(molecular*weights[:,:,:,None]).sum(1)
        alleles=self.allele(torch.cat((se,re,af.clamp(-20,20)),dim=-1))*am[:,:,None]
        if need_read:
            read_af=af.clone();read_af[:,:,1]=0;read_af[:,:,5]=0
            read_alleles=self.allele(torch.cat((se,re,read_af.clamp(-20,20)),dim=-1))*am[:,:,None]

        gp=batch['base_gp'].float().clamp_min(0)*gm
        gp=gp/gp.sum(1,keepdim=True).clamp_min(1e-30)
        entropy=-(gp*gp.clamp_min(1e-30).log()).sum(1,keepdim=True)
        entropy=entropy/gm.sum(1,keepdim=True).float().clamp_min(2).log()
        normalized_depth=torch.log1p(depth.clamp_min(0))/np.log(101.)
        context=batch.get('cohort_context',torch.zeros(b,12,6,device=seq.device)).float()[:,:a]
        strength=batch.get('context_strength',torch.zeros(b,device=seq.device)).float().reshape(-1,1)
        active=(strength>0).float()*float(cohort_enabled)
        if self.training and self.config.cohort_dropout:
            active=active*(torch.rand_like(active)>=self.config.cohort_dropout)
        if compact and not return_aux and not active.any():
            # An unavailable cohort is explicitly masked to zero in the trained
            # architecture. Singleton inference needs neither masked MLP.
            ce=torch.zeros_like(alleles);gate=torch.zeros_like(active)
        else:
            context=context*active[:,:,None]
            ce=self.cohort(context.clamp(-20,20))*am[:,:,None]*active[:,:,None]
            global_read=(alleles*am[:,:,None]).sum(1)/am.sum(1,keepdim=True).clamp_min(1)
            global_context=ce.sum(1)/am.sum(1,keepdim=True).clamp_min(1)
            gate=torch.sigmoid(self.context_gate(torch.cat((global_read,global_context,normalized_depth,
                                  entropy,strength.clamp(0,1)*active),dim=1)))*active
            ce=ce*gate[:,:,None]

        width=int((gm*torch.arange(1,79,device=gm.device)).max()) if compact and not return_aux else 78
        i,j=self.pair_i[:width],self.pair_j[:width]
        pair=torch.cat((alleles[:,i]+alleles[:,j],(alleles[:,i]-alleles[:,j]).abs(),
                        alleles[:,i]*alleles[:,j]),dim=-1)
        if need_read:
            read_pair=torch.cat((read_alleles[:,i]+read_alleles[:,j],(read_alleles[:,i]-read_alleles[:,j]).abs(),
                                 read_alleles[:,i]*read_alleles[:,j]),dim=-1)
        cp=torch.cat((ce[:,i]+ce[:,j],(ce[:,i]-ce[:,j]).abs(),ce[:,i]*ce[:,j]),dim=-1)
        pl=batch['read_pl'][:,:width].float()
        read_lp=(-pl*np.log(10.)/10.).clamp(-30,0)/30.
        log_gp=gp.clamp_min(self.config.prior_floor).log()[:,:width]
        is_het=(i!=j).float()[None,:,None].expand(b,-1,-1)
        is_ref=((i==0)&(j==0)).float()[None,:,None].expand(b,-1,-1)
        d=normalized_depth[:,None,:].expand(-1,width,-1)
        en=entropy[:,None,:].expand(-1,width,-1)
        shared=torch.cat((read_lp[:,:,None],d,en,is_het,is_ref),-1)
        if need_read:
            read_shared=torch.cat((read_lp[:,:,None],d,torch.zeros_like(en),is_het,is_ref),-1)
            read_logits=self.pair_read(torch.cat((read_pair,read_shared),-1)).squeeze(-1)
        anchor=log_gp
        if self.training and self.config.prior_dropout:
            dropped=(torch.rand(b,1,device=seq.device)<self.config.prior_dropout)
            anchor=torch.where(dropped,torch.zeros_like(anchor),anchor)
        residual=self.config.residual_scale*torch.tanh(self.pair_joint(torch.cat((pair,cp,shared,
                                    anchor[:,:,None]/16.,gate[:,None,:].expand(-1,width,-1)),-1)).squeeze(-1))
        logits=(anchor+residual).masked_fill(~gm[:,:width],-1e4)
        if need_read:read_logits=read_logits.masked_fill(~gm[:,:width],-1e4)
        if width<78:logits=torch.nn.functional.pad(logits,(0,78-width),value=-1e4)
        if return_aux:return {'logits':logits,'read_logits':read_logits,'cohort_gate':gate.squeeze(1)}
        return logits

def tensors(batch,device):
    keys=['allele_sequences','allele_mask','fragment_features','fragment_mask','base_gp','read_pl',
          'genotype_mask','depth','allele_features','cohort_context','context_strength']
    return {k:torch.as_tensor(batch[k],device=device) for k in keys if k in batch}

def load_model(checkpoint,device='cpu'):
    # Only tensors and primitive metadata are accepted from our saved checkpoint.
    if str(device).startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    if str(device)=='cpu' and 'VARDEPTH_NEURAL_CPU_THREADS' in os.environ:
        threads=int(os.environ['VARDEPTH_NEURAL_CPU_THREADS'])
        if threads<1:raise ValueError('Neural CPU threads must be positive')
        torch.set_num_threads(threads)
    state=torch.load(checkpoint,map_location=device,weights_only=True)
    if state.get('model_version')!=MODEL_VERSION:raise ValueError('Unsupported model version')
    model=NeuralCaller(ModelConfig(**state['config'])).to(device)
    model.load_state_dict(state['model_state']);model.eval()
    model.compact_inference=os.environ.get('VARDEPTH_COMPACT_NEURAL','0')=='1'
    model.checkpoint_metadata={k:v for k,v in state.items() if k not in ['model_state','optimizer_state','rng_state']}
    model.temperature=float(state.get('temperature',1.0))
    return model

@torch.inference_mode()
def predict(model,batch):
    device=next(model.parameters()).device
    x=tensors(batch,device);logits=model(x)/getattr(model,'temperature',1.0)
    gp=torch.softmax(logits,dim=1)*x['genotype_mask']
    return (gp/gp.sum(1,keepdim=True)).cpu().numpy()
