"""Explicit bounded-future regressors, separate from sealed reference models."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from virtual_imu.utils.seed import seed_everything

class TokenNorm(nn.Module):
    def __init__(self,width):
        super().__init__(); self.norm=nn.LayerNorm(width)
    def forward(self,x):
        return self.norm(x.transpose(1,2)).transpose(1,2)

class CausalConv(nn.Conv1d):
    def forward(self,x):
        pad=self.dilation[0]*(self.kernel_size[0]-1)
        # Explicit taps avoid the slow tiny CUDA dilated-convolution kernel.
        z=F.pad(x,(pad,0)).unfold(-1,pad+1,1)[...,::self.dilation[0]]
        z=z.permute(0,2,1,3).reshape(x.shape[0],x.shape[-1],-1)
        return F.linear(z,self.weight.flatten(1),self.bias).transpose(1,2)

class CausalFront(nn.Module):
    def __init__(self,width=64,depth=3,kernel=5,dilation=1,norm='ln',activation='gelu'):
        super().__init__()
        self.lift=nn.Conv1d(6,width,1)
        blocks=[]
        for _ in range(depth):
            blocks.append(CausalConv(width,width,kernel,dilation=dilation))
            blocks.append(TokenNorm(width) if norm=='ln' else nn.BatchNorm1d(width) if norm=='bn' else nn.Identity())
            blocks.append(nn.GELU() if activation=='gelu' else nn.Identity())
        self.blocks=nn.Sequential(*blocks)
        self.project=nn.Conv1d(width,64,1)
        self.receptive_field=1+depth*(kernel-1)*dilation
    def forward(self,x):
        return self.project(self.blocks(self.lift(x)))

class CausalTCNBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.a=nn.Linear(64,32); self.b=CausalConv(32,32,3,dilation=dilation)
        self.c=nn.Linear(32,64); self.norm=nn.LayerNorm(64)
    def forward(self,x):
        z=F.gelu(self.a(x)); z=F.gelu(self.b(z.transpose(1,2)).transpose(1,2))
        return self.norm(x+self.c(z))

class ContextModel(nn.Module):
    def __init__(self,backend='gru',lookahead=0,width=64,hidden=None,depth=3,kernel=5,dilation=1,norm='ln',activation='gelu'):
        super().__init__()
        if backend not in {'lstm','gru','bigru','transformer','tcn'}: raise ValueError('Unknown backend')
        if lookahead!='full' and (not isinstance(lookahead,int) or lookahead<0): raise ValueError('Invalid lookahead')
        if lookahead=='full' and backend not in {'bigru','transformer'}: raise ValueError('Full future only for noncausal backends')
        self.backend_name,self.lookahead=backend,lookahead
        self.feature_extractor=CausalFront(width,depth,kernel,dilation,norm,activation)
        if backend in {'lstm','gru'}:
            h=hidden or (64 if backend=='lstm' else 72)
            self.backend=(nn.LSTM if backend=='lstm' else nn.GRU)(64,h,batch_first=True)
            self.projection=nn.Linear(h,64)
        elif backend=='bigru':
            h=hidden or 40
            self.forward_gru=nn.GRU(64,h,batch_first=True)
            self.backward_gru=nn.GRU(64,h,batch_first=True)
            self.projection=nn.Linear(2*h,64)
        elif backend=='transformer':
            layer=nn.TransformerEncoderLayer(64,4,128,0.0,activation='gelu',batch_first=True,norm_first=True)
            self.backend=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
            pos=torch.arange(640).float()[:,None]; rates=torch.exp(torch.arange(0,64,2).float()*(-math.log(10000.0)/64))
            pe=torch.zeros(640,64);pe[:,0::2]=torch.sin(pos*rates);pe[:,1::2]=torch.cos(pos*rates)
            self.register_buffer('position',pe[None],persistent=True)
        else:
            self.backend=nn.Sequential(*(CausalTCNBlock(d) for d in [1,2,4,8]))
        self.head=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,6))

    def encode(self,x):
        z=self.feature_extractor(x).transpose(1,2); B,T,C=z.shape
        if self.backend_name in {'gru','lstm'}:
            z=self.projection(self.backend(z)[0])
        elif self.backend_name=='bigru':
            left=self.forward_gru(z)[0]
            if self.lookahead=='full':
                right=self.backward_gru(z.flip(1))[0].flip(1)
            else:
                span=self.lookahead+1
                padded=F.pad(z.transpose(1,2),(0,self.lookahead)).transpose(1,2)
                windows=padded.unfold(1,span,1).permute(0,1,3,2).reshape(B*T,span,C)
                right=self.backward_gru(windows.flip(1))[0][:,-1].reshape(B,T,-1)
            return self.projection(torch.cat([left,right],dim=-1))
        elif self.backend_name=='transformer':
            z=z+self.position[:,:T]
            mask=None if self.lookahead=='full' else torch.ones(T,T,device=z.device,dtype=torch.bool).triu(1)
            z=self.backend(z,mask=mask)
        else: z=self.backend(z)
        if self.lookahead!='full' and self.lookahead:
            ix=(torch.arange(T,device=z.device)+self.lookahead).clamp(max=T-1)
            z=z[:,ix]
        return z

    def forward(self,x):
        return self.head(self.encode(x)).transpose(1,2)


def build_model(job):
    seed_everything(int(job['seed']))
    family=job['family']; opts=job.get('model_options',{})
    if job['stage'] in {'future','mechanism'}:
        model=ContextModel(**opts)
        meta={'implementation':'context_study.models.ContextModel','future_points':opts.get('lookahead',0),
              'frontend_receptive_field':model.feature_extractor.receptive_field,'model_options':opts}
    elif family in {'informer','transformer'}:
        from virtual_imu.models.matched import build_matched_model
        model=build_matched_model({'method':family,'seed':job['seed'],'model':{
            'condition_channels':6,'d_model':64,'nhead':4,'num_layers':2,'dim_feedforward':128,
            'dropout':opts.get('dropout',0.0),'max_length':320,'factor':5,'distil':True}})
        meta={'implementation':'frozen_matched_adapter','model_options':opts,'future_points':'whole_window'}
    else:
        from virtual_imu.replacements.models import build_replacement_model
        cfg=job['config']; mo=dict(cfg['model']);mo.update(opts)
        model,meta=build_replacement_model(family,job['seed'],mo,cfg['replacement_models'])
    meta.update(parameters=sum(p.numel() for p in model.parameters()),family=family)
    return model,meta
