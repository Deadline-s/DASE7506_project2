"""Fit the retained verifier architecture at frozen epochs and seeds."""
import numpy as np
import torch
from torch import nn
from training_data import load_training_split
from verification_features import verification_features
from ranker_inference import answer_readout

EPOCHS=(10,20,40,80)


def load(split):
    d=load_training_split(split)
    d['checks']=verification_features(d['bank']).reshape(len(d['bank']),4,-1)
    return d


def prep(data,config):
    out={}
    for key in ('emb','features','checks'):
        x=data[key].reshape(-1,data[key].shape[-1]);out[key+'_mean']=x.mean(0);out[key+'_scale']=np.maximum(x.std(0),1e-5)
    if 'power' in config:
        scale=out['emb_scale']**config['power']
        z=(data['emb']-out['emb_mean'])/scale
        out['emb_scale']=scale*np.sqrt(np.mean(z*z))
    return out


def inputs(data,p):
    a=[np.clip((data[k]-p[k+'_mean'])/p[k+'_scale'],-8,8) for k in ('emb','features','checks')]
    return torch.from_numpy(np.concatenate(a,-1).astype('float32')),torch.from_numpy(a[-1].astype('float32'))


class Verifier(nn.Module):
    def __init__(self,dim,checks,skip):
        super().__init__();self.main=nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,1))
        self.skip=nn.Linear(checks,1,bias=False) if skip else None
        if self.skip is not None:nn.init.zeros_(self.skip.weight)
    def forward(self,x,c):
        return self.main(x).squeeze(-1)+(self.skip(c).squeeze(-1) if self.skip is not None else 0.)


def fit(train,val,config,seed,epochs):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);p=prep(train,config);xt,ct=inputs(train,p)
    if val is not None:xv,cv=inputs(val,p)
    y=torch.from_numpy(train['y']);valid=torch.from_numpy(train['valid']);pool=torch.from_numpy(train['pool']);unique=torch.from_numpy(train['unique'])
    active=np.arange(len(y)) if config['objective'] in ('binary_brier','binary_log') else np.where(np.any(~train['y'] & train['valid'],axis=1))[0]
    model=Verifier(xt.shape[-1],ct.shape[-1],config['skip']);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    snapshots={};average=None;averages=0
    for epoch in range(1,epochs+1):
        model.train()
        for ids in np.array_split(rng.permutation(active),max(1,int(np.ceil(len(active)/128)))):
            raw=model(xt[ids],ct[ids]);s=torch.bmm(pool[ids],raw[:,:,None]).squeeze(-1)
            objective=config['objective']
            if objective=='answer':loss=nn.functional.cross_entropy(s.masked_fill(~unique[ids],-1e4),y[ids].float().argmax(1))
            elif objective=='brier':
                prob=torch.softmax(s.masked_fill(~unique[ids],-1e4),1)
                target=(y[ids]&unique[ids]).float()
                loss=((prob-target)**2).sum(1).mean()
            elif objective=='binary_brier':
                loss=(((torch.sigmoid(raw)-y[ids].float())**2)*valid[ids]).sum()/valid[ids].sum()
            elif objective=='binary_log':
                loss=(nn.functional.binary_cross_entropy_with_logits(raw,y[ids].float(),reduction='none')*valid[ids]).sum()/valid[ids].sum()
            elif objective=='smooth_log':
                target=(y[ids]&unique[ids]).float()*.95+.05*unique[ids]/unique[ids].sum(1,keepdim=True)
                loss=-(target*torch.log_softmax(s.masked_fill(~unique[ids],-1e4),1)).sum(1).mean()
            elif objective=='margin':
                mask=y[ids,:,None] & ~y[ids,None,:] & valid[ids,None,:]
                delta=raw[:,:,None]-raw[:,None,:]
                loss=((delta[mask]-1)**2).mean()+.01*raw.mean(1).square().mean()
            elif config['objective']=='mass':loss=(torch.logsumexp(raw.masked_fill(~valid[ids],-1e4),1)-torch.logsumexp(raw.masked_fill(~y[ids],-1e4),1)).mean()
            else:
                mask=y[ids,:,None] & ~y[ids,None,:] & valid[ids,None,:]
                loss=nn.functional.softplus(-(raw[:,:,None]-raw[:,None,:]))[mask].mean()
            if config.get('consistency'):
                trace_probability=torch.sigmoid(raw);pooled=torch.bmm(pool[ids],trace_probability[:,:,None]).squeeze(-1)
                loss+=config['consistency']*((trace_probability-pooled).square()*valid[ids]).sum()/valid[ids].sum()
            opt.zero_grad();loss.backward();opt.step()
        if config.get('swa') and epoch>=10:
            averages+=1
            if average is None:average={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:
                for k,v in model.state_dict().items():average[k].add_((v-average[k])/averages)
        if val is not None and epoch in EPOCHS:
            current=None
            if average is not None:
                current={k:v.detach().clone() for k,v in model.state_dict().items()};model.load_state_dict(average)
            model.eval()
            with torch.no_grad():s=model(xv,cv).numpy()
            snapshots[epoch]=answer_readout(s,val['pool'],config.get('readout','logit'))
            if current is not None:model.load_state_dict(current)
    if average is not None:model.load_state_dict(average)
    layers=[(m.weight.detach().numpy().T.copy(),m.bias.detach().numpy().copy()) for m in model.main if isinstance(m,nn.Linear)]
    return dict(prep=p,layers=layers,skip=model.skip.weight.detach().numpy().ravel().copy() if model.skip is not None else None,config=dict(**config,seed=seed,epoch=epochs)),snapshots


def score(data,model):
    x,c=inputs(data,model['prep']);x=x.numpy();c=c.numpy()
    for i,(w,b) in enumerate(model['layers']):
        x=x@w+b
        if i+1<len(model['layers']):x=np.maximum(x,0)
    raw=x.squeeze(-1)
    if model['skip'] is not None:raw+=c@model['skip']
    return answer_readout(raw,data['pool'],model['config'].get('readout','logit'))

