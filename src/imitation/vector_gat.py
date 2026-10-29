"""GAT à bases vectorielles : équilibre exact, gains appris, aucun QP à l’inférence."""
from dataclasses import dataclass,asdict
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VectorConfig:
    hidden_dim:int=48
    init_seed:int=42


class VectorGAT(nn.Module):
    def __init__(self,config=VectorConfig(),maximum=2.):
        super().__init__();self.config=config;self.max_acceleration=float(maximum)
        if config.hidden_dim<1 or not math.isfinite(maximum) or maximum<=0:raise ValueError('Configuration invalide')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.init_seed);h=config.hidden_dim
            self.log_kp=nn.Parameter(torch.zeros(()));self.log_kd=nn.Parameter(torch.zeros(()))
            self.ego=nn.Sequential(nn.Linear(4,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU())
            self.edge=nn.Sequential(nn.Linear(8,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU())
            self.query=nn.Linear(h,h,bias=False);self.key=nn.Linear(h,h,bias=False)
            self.coefficients=nn.Sequential(nn.Linear(2*h,h),nn.SiLU(),nn.Linear(h,3))
            nn.init.zeros_(self.coefficients[-1].weight);nn.init.zeros_(self.coefficients[-1].bias)

    def nominal(self,own):
        return self.log_kp.exp()*(own[...,4:6]-own[...,:2])-self.log_kd.exp()*own[...,2:4]

    def forward(self,own,neighbors,mask):
        if own.shape[-1]!=8 or neighbors.shape[-1]!=4 or neighbors.shape[:-2]!=own.shape[:-1] or mask.shape!=neighbors.shape[:-1] or mask.dtype!=torch.bool:
            raise ValueError('Observations locales incompatibles')
        if not torch.isfinite(own).all() or not torch.isfinite(neighbors[mask]).all():raise ValueError('Observations non finies')
        goal=own[...,4:6]-own[...,:2];velocity=own[...,2:4]
        relative=torch.where(mask[...,None],neighbors,torch.zeros_like(neighbors))
        r,w=relative[...,:2],relative[...,2:];d=torch.linalg.vector_norm(r,dim=-1).clamp_min(1e-6)
        unit=r/d[...,None];away=-unit;tangent=torch.stack((-away[...,1],away[...,0]),dim=-1)
        gnorm=torch.linalg.vector_norm(goal,dim=-1);vnorm=torch.linalg.vector_norm(velocity,dim=-1)
        own_features=torch.stack((gnorm/28.,vnorm,own[...,6]/3.,own[...,7]/64.),dim=-1)
        own_encoding=self.ego(own_features)
        base=self.nominal(own)
        base=base*(self.max_acceleration/torch.linalg.vector_norm(base,dim=-1,keepdim=True).clamp_min(self.max_acceleration))
        gu=(base.unsqueeze(-2)*unit).sum(-1);vu=(velocity.unsqueeze(-2)*unit).sum(-1)
        wu=(w*unit).sum(-1);wt=(w*tangent).sum(-1)
        gt=(base.unsqueeze(-2)*tangent).sum(-1);vt=(velocity.unsqueeze(-2)*tangent).sum(-1)
        features=torch.stack((d/3.,wu,wt,gu/self.max_acceleration,gt/self.max_acceleration,vu,vt,torch.linalg.vector_norm(w,dim=-1)),dim=-1)
        encoding=self.edge(features);query=self.query(own_encoding).unsqueeze(-2)
        scores=(query*self.key(encoding)).sum(-1)/math.sqrt(self.config.hidden_dim)
        scores=scores.masked_fill(~mask,float('-inf'))
        scores=torch.where(mask.any(-1,keepdim=True),scores,torch.zeros_like(scores))
        weights=torch.softmax(scores,dim=-1).masked_fill(~mask,0.)
        ego=own_encoding.unsqueeze(-2).expand_as(encoding)
        coefficients=self.coefficients(torch.cat((ego,encoding),dim=-1))
        activity=torch.tanh(gnorm.unsqueeze(-1)+vnorm.unsqueeze(-1)+torch.linalg.vector_norm(w,dim=-1))
        messages=(coefficients[...,0:1]*away+coefficients[...,1:2]*tangent+coefficients[...,2:3]*w)*activity[...,None]
        degree=mask.sum(-1).to(own.dtype).sqrt().unsqueeze(-1)
        residual=(weights[...,None]*messages).sum(-2)*degree
        raw=base+residual
        return raw*(self.max_acceleration/torch.linalg.vector_norm(raw,dim=-1,keepdim=True).clamp_min(self.max_acceleration))

    def metadata(self):
        return dict(model_type='VectorBasisGAT',schema_version=1,feature_schema_version=2,config=asdict(self.config),
                    max_acceleration=self.max_acceleration,inference='learned_positive_gains_and_attention_vector_messages_no_safety_projection')


def save_vector(path,model):
    with Path(path).open('xb') as stream:torch.save(dict(metadata=model.metadata(),state_dict=model.state_dict()),stream)


def load_vector(path):
    payload=torch.load(path,map_location='cpu',weights_only=True);meta=payload['metadata']
    if meta.get('model_type')!='VectorBasisGAT' or meta.get('schema_version')!=1:raise ValueError('Checkpoint incompatible')
    model=VectorGAT(VectorConfig(**meta['config']),meta['max_acceleration'])
    if model.metadata()!=meta:raise ValueError('Contrat différent')
    model.load_state_dict(payload['state_dict'],strict=True);return model.eval()


def barrier_penalty(prediction,neighbors,mask,safety_distance=.4):
    # Contrainte réciproque utilisée seulement pour apprendre, jamais pour projeter en inférence.
    r=-neighbors[...,:2];w=-neighbors[...,2:];d=torch.linalg.vector_norm(r,dim=-1).clamp_min(.1)
    bound=-.5*(w*w).sum(-1)-3.*(r*w).sum(-1)-2.25*((r*r).sum(-1)-safety_distance**2)
    violation=torch.relu((bound-(r*prediction.unsqueeze(-2)).sum(-1))/d)
    return (violation.square()*mask).sum()/mask.sum().clamp_min(1)
