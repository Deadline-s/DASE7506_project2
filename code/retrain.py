"""Reproduce retained neural components from configuration, without old searches."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='4'
import copy
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(4)


def model_spec(bundle):
    if 'components' in bundle:
        return {k:([dict(c, bundle=model_spec(c['bundle'])) if 'bundle' in c else model_spec(c) for c in v] if k=='components' else copy.deepcopy(v)) for k,v in bundle.items()}
    result={k:copy.deepcopy(v) for k,v in bundle.items() if k!='models'}
    result['models']=[{k:copy.deepcopy(v) for k,v in m.items() if k in ('mode','config','context','pool_inputs','activation')} for m in bundle['models']]
    return result


def fit_model(data,spec):
    cfg=spec['config'];legacy='mode' in spec
    torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    e=data['emb'];f=data['features']
    if legacy:
        es=StandardScaler().fit(e.reshape(-1,1024));fs=StandardScaler().fit(f.reshape(-1,f.shape[-1]))
        xe=np.clip(es.transform(e.reshape(-1,1024)),-8,8).reshape(e.shape)
        xf=np.clip(fs.transform(f.reshape(-1,f.shape[-1])),-8,8).reshape(f.shape)
        p=dict(mean=es.mean_.astype('float32'),scale=es.scale_.astype('float32'),feature_mean=fs.mean_.astype('float32'),feature_scale=fs.scale_.astype('float32'))
        widths=[cfg['width'],32];dropout=cfg['dropout'];objective=cfg['loss'];decay=cfg['decay'];bce=0.
    else:
        p=dict(mean=e.reshape(-1,1024).mean(0),scale=np.maximum(e.reshape(-1,1024).std(0),1e-5),
               feature_mean=f.reshape(-1,f.shape[-1]).mean(0),feature_scale=np.maximum(f.reshape(-1,f.shape[-1]).std(0),1e-5))
        xe=np.clip((e-p['mean'])/p['scale'],-8,8);xf=np.clip((f-p['feature_mean'])/p['feature_scale'],-8,8)
        widths=cfg.get('widths',[64,32]);dropout=cfg.get('dropout',.4);objective=cfg.get('objective','answer');decay=.03;bce=cfg.get('bce',.3)
    x=np.concatenate([xe,xf],-1).astype('float32')
    if spec.get('pool_inputs'):x=data['pool']@x
    xt=torch.from_numpy(x);yt=torch.from_numpy(data['y']);valid=torch.from_numpy(data['valid']);pool=torch.from_numpy(data['pool']);unique=torch.from_numpy(data['unique'])
    active=np.arange(len(xt)) if bce else np.where(np.any(~data['y'] & data['valid'],1))[0]
    layers=[];dim=xt.shape[-1]
    for width in widths:layers.extend([nn.Linear(dim,width),nn.ReLU(),nn.Dropout(dropout)]);dim=width
    layers.append(nn.Linear(dim,1));net=nn.Sequential(*layers);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=decay)
    for epoch in range(cfg['epoch']):
        net.train()
        for ids in np.array_split(rng.permutation(active),max(1,int(np.ceil(len(active)/128)))):
            raw=net(xt[ids]).squeeze(-1)
            s=raw if legacy and objective!='answer_ce' else torch.bmm(pool[ids],raw[:,:,None]).squeeze(-1)
            if objective in ('answer','answer_ce'):loss=nn.functional.cross_entropy(s.masked_fill(~unique[ids],-1e4),yt[ids].float().argmax(1))
            elif objective=='mass':loss=(torch.logsumexp(s.masked_fill(~valid[ids],-1e4),1)-torch.logsumexp(s.masked_fill(~yt[ids],-1e4),1)).mean()
            else:
                pairs=yt[ids,:,None] & ~yt[ids,None,:] & valid[ids,None,:]
                losses=nn.functional.softplus(-(s[:,:,None]-s[:,None,:]))
                if legacy:loss=losses[pairs].mean()
                else:
                    count=pairs.sum((1,2));loss=((losses*pairs).sum((1,2))/count.clamp(min=1))[count>0].mean()
            if bce:loss+=bce*(nn.functional.binary_cross_entropy_with_logits(raw,yt[ids].float(),reduction='none')*valid[ids]).sum()/valid[ids].sum()
            opt.zero_grad();loss.backward();opt.step()
    weights=[(m.weight.detach().numpy().T.copy(),m.bias.detach().numpy().copy()) for m in net if isinstance(m,nn.Linear)]
    return dict(**copy.deepcopy(spec),**p,layers=weights)


def fit_bundle(data,spec):
    result={k:copy.deepcopy(v) for k,v in spec.items() if k not in ('components','models')}
    if 'components' in spec:
        result['components']=[dict(c,bundle=fit_bundle(data,c['bundle'])) if 'bundle' in c else fit_bundle(data,c) for c in spec['components']]
    else:result['models']=[fit_model(data,m) for m in spec['models']]
    return result
