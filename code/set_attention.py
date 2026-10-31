"""Independent candidate-set verifier with train-only candidate-count augmentation.

Design motivation: Wenqi-Ge/DASE7506-MP2 at commit
54abbb793ddb6bd2142aa8b5cf56c5a0b0681b88 reports benefits from balancing
answer counts and candidate attention. This implementation uses our existing
features, PyTorch TransformerEncoderLayer, separate recipes and own training.
No peer parameters, predictions or implementation files are loaded.
"""
import copy
import numpy as np
from training_data import subset
from reasoning_features import rich_features
from ranker_inference import answer_matrix


def augmented(data,seed,counts=(1,2,3)):
    rng=np.random.default_rng(seed);origins=[];choices=[];bank=[]
    for i,y in enumerate(data['y']):
        correct=np.flatnonzero(y);wrong=np.flatnonzero(~y&data['valid'][i])
        if not len(wrong):continue
        # Cover minority, tied and majority-correct groups without new solutions.
        for count in counts:
            if count not in (1,2,3):raise ValueError('Augmented correct count must be 1, 2 or 3')
            indices=np.concatenate([rng.choice(correct,count,replace=True),rng.choice(wrong,4-count,replace=True)])
            rng.shuffle(indices)
            bank.append(dict(data['bank'][i],candidates=[copy.deepcopy(data['bank'][i]['candidates'][j]) for j in indices]))
            origins.append(i);choices.append(indices)
    origins=np.asarray(origins);choices=np.asarray(choices)
    if not len(bank):return data
    features,valid,_=rich_features(bank);pool=answer_matrix(bank)
    unique=(np.diagonal(pool,axis1=1,axis2=2)>0)&(np.tril(pool,-1).sum(-1)==0)
    extra=dict(bank=bank,features=features.reshape(-1,4,features.shape[-1]),valid=valid,pool=pool,unique=unique)
    for key in ('emb','checks','y'):extra[key]=data[key][origins[:,None],choices]
    return {key:(data[key]+extra[key] if key=='bank' else np.concatenate([data[key],extra[key]])) for key in extra}


def features(data,mode):
    e=np.asarray(data['emb'],np.float32)
    if mode=='compact':return np.concatenate([e,data['features'][:,:,:27],data['checks']],-1)
    if mode=='no_vote_counts':return np.concatenate([e,data['features'][:,:,:23],data['checks']],-1)
    if mode=='extended':return np.concatenate([e,data['features'],data['checks']],-1)
    if mode=='rich':return np.concatenate([e,e-e.mean(1,keepdims=True),data['features'],data['checks']],-1)
    if mode=='contrast':
        f=np.concatenate([data['features'],data['checks']],-1)
        center=f.mean(1,keepdims=True,dtype=np.float64).astype(np.float32)
        return np.concatenate([e,f,f-center,f-f.max(1,keepdims=True)],-1)
    if mode=='unit_aware':
        from verification_features import verification_features
        checks=verification_features(data['bank'],allow_rhs_units=True).reshape(len(e),4,32)
        return np.concatenate([e,data['features'][:,:,:27],checks],-1)
    raise ValueError('Unknown attention input mode: '+str(mode))


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))


def logits(z,model):
    s=model['state'];cfg=model['config']
    def linear(x,name):return x@s[name+'.weight'].T+s[name+'.bias']
    def norm(x,name):return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)*s[name+'.weight']+s[name+'.bias']
    h=linear(z,'embedding');heads=cfg.get('heads',4);part=h.shape[-1]//heads
    for index in range(cfg['layers']):
        key=f'blocks.{index}.';v=norm(h,key+'norm1')
        q,k,value=np.split(v@s[key+'self_attn.in_proj_weight'].T+s[key+'self_attn.in_proj_bias'],3,-1)
        q,k,value=[a.reshape(len(a),4,heads,part).transpose(0,2,1,3) for a in (q,k,value)]
        scores=q@k.swapaxes(-1,-2)/np.sqrt(part);p=np.exp(scores-scores.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        attended=(p@value).transpose(0,2,1,3).reshape(h.shape);h=h+linear(attended,key+'self_attn.out_proj')
        v=linear(norm(h,key+'norm2'),key+'linear1')
        if cfg.get('activation','relu')=='gelu_tanh':v=.5*v*(1+np.tanh(np.sqrt(2/np.pi)*(v+.044715*v**3)))
        else:v=np.maximum(v,0)
        h=h+linear(v,key+'linear2')
    out=linear(norm(h,'normalization') if cfg.get('final_norm',True) else h,'output').squeeze(-1)
    if cfg.get('skip'):out+=(z@s['direct.weight'].T).squeeze(-1)
    return out


def scores(data,bundle):
    result=np.zeros(data['valid'].shape,dtype=np.float64)
    cached_inputs={}
    for m in bundle['models']:
        if any(v.dtype==np.float16 for v in m['state'].values()):
            m=dict(m,state={k:v.astype(np.float32) for k,v in m['state'].items()})
        mode=m['config']['mode']
        if mode not in cached_inputs:cached_inputs[mode]=features(data,mode)
        x=cached_inputs[mode]
        for start in range(0,len(x),128):
            z=np.clip((x[start:start+128]-m['mean'])/m['scale'],-8,8).astype(np.float32)
            result[start:start+128]+=sigmoid(logits(z,m))
    return result/len(bundle['models'])


def quantize_parameters(bundle):
    """Store native float16 parameters; keep feature normalization in float32."""
    models=[]
    for m in bundle['models']:
        state={k:v.astype(np.float16) for k,v in m['state'].items()}
        if not all(np.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite quantized parameters')
        models.append(dict(m,state=state))
    return dict(models=models)


def average_snapshots(models,weights):
    """Average matching checkpoints from one seed and the same fitting fold."""
    assert len(models)==len(weights) and abs(sum(weights)-1)<1e-10
    first=models[0]
    for m in models[1:]:
        assert np.array_equal(first['mean'],m['mean']) and np.array_equal(first['scale'],m['scale'])
        assert first['state'].keys()==m['state'].keys()
    result=copy.deepcopy(first)
    result['state']={k:sum(np.float32(w)*m['state'][k] for w,m in zip(weights,models)).astype(np.float32) for k in first['state']}
    result['config']=dict(first['config'],epoch=max(m['config']['epoch'] for m in models),
                          averaged_epochs=[m['config']['epoch'] for m in models],averaged_weights=list(weights))
    return result


def fit_recipe(data,recipe,seed):
    epochs=recipe.get('average_epochs')
    if epochs:
        _,_,exports=fit(data,None,recipe['config'],seed,max(epochs),checkpoints=epochs)
        return average_snapshots([exports[e] for e in epochs],recipe['average_weights'])
    return fit(data,None,recipe['config'],seed,recipe['epoch'])[0]


def fit(data,validation,config,seed,epochs,checkpoints=(8,16)):
    import torch
    from torch import nn
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    if config.get('hard_only'):data=subset(data,np.flatnonzero(np.any(~data['y']&data['valid'],1)))
    training=augmented(data,seed+17,config.get('augment_counts',(1,2,3))) if config.get('augment') else data
    x=features(training,config['mode']);mean=x.reshape(-1,x.shape[-1]).mean(0);scale=np.maximum(x.reshape(-1,x.shape[-1]).std(0),1e-5)
    x=torch.from_numpy(np.clip((x-mean)/scale,-8,8).astype(np.float32));y=torch.from_numpy(training['y']);valid=torch.from_numpy(training['valid']);hard=(~y&valid).any(1)
    class Model(nn.Module):
        def __init__(self):
            super().__init__();width=config['width'];self.embedding=nn.Linear(x.shape[-1],width)
            activation=(lambda value:nn.functional.gelu(value,approximate='tanh')) if config.get('activation','relu')=='gelu_tanh' else 'relu'
            self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(width,config.get('heads',4),width*2,dropout=config['dropout'],activation=activation,batch_first=True,norm_first=True) for _ in range(config['layers'])])
            self.normalization=nn.LayerNorm(width) if config.get('final_norm',True) else nn.Identity();self.output=nn.Linear(width,1)
            self.direct=nn.Linear(x.shape[-1],1,bias=False) if config.get('skip') else None
            if self.direct is not None:nn.init.zeros_(self.direct.weight)
        def forward(self,z):
            h=self.embedding(z)
            for block in self.blocks:h=block(h)
            out=self.output(self.normalization(h)).squeeze(-1)
            return out+self.direct(z).squeeze(-1) if self.direct is not None else out
    net=Model();opt=torch.optim.AdamW(net.parameters(),lr=config['lr'],weight_decay=config.get('decay',.01))
    snapshots={};exports={}
    def export(epoch):return dict(mean=mean.copy(),scale=scale.copy(),state={k:v.detach().numpy().copy() for k,v in net.state_dict().items()},config=dict(config,seed=seed,epoch=epoch))
    for epoch in range(1,epochs+1):
        net.train()
        for ids in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/config.get('batch_size',128)))):
            raw=net(x[ids]);bce=nn.functional.binary_cross_entropy_with_logits(raw,y[ids].float(),reduction='none')
            bce=(bce*valid[ids]).sum(1)/valid[ids].sum(1).clamp_min(1)
            mass=torch.logsumexp(raw.masked_fill(~valid[ids],-1e4),1)-torch.logsumexp(raw.masked_fill(~y[ids],-1e4),1)
            local=config.get('binary_weight',1.)*bce+config.get('mass_weight',.4)*mass
            if config.get('rank_weight',0):
                pair=y[ids,:,None]&(~y[ids,None,:])&valid[ids,None,:]
                margin=raw[:,:,None]-raw[:,None,:]
                ranking=(nn.functional.softplus(-margin)*pair).sum((1,2))/pair.sum((1,2)).clamp_min(1)
                local=local+config['rank_weight']*ranking
            loss=(local*(1+config.get('hard_weight',1.)*hard[ids].float())).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        if epoch in checkpoints:
            m=export(epoch);exports[epoch]=m
            if validation is not None:snapshots[epoch]=scores(validation,dict(models=[m]))
    m=export(epochs);net.eval()
    with torch.no_grad():expected=net(x[:24]).numpy()
    actual=logits(x[:24].numpy(),m)
    if not np.allclose(expected,actual,atol=3e-5,rtol=3e-5):raise ValueError('Attention export mismatch')
    return m,snapshots,exports
