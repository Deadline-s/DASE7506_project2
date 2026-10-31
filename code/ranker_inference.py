"""Pure NumPy inference for frozen candidate and answer-level rankers."""
import numpy as np
from common import parse_answer


def network_scores(embeddings, features, model):
    if model['mode'] not in ('rich','answer_loss'):
        raise ValueError('Unsupported frozen ranking architecture')
    e=np.asarray(embeddings,dtype=np.float32).reshape(-1,4,1024)
    e=np.clip((e-model['mean'])/model['scale'],-8,8)
    f=np.clip((features-model['feature_mean'])/model['feature_scale'],-8,8).reshape(len(e),4,-1)
    x=np.concatenate([e,f],axis=-1).reshape(4*len(e),-1)
    for i,(weight,bias) in enumerate(model['layers']):
        x=x@weight+bias
        if i+1<len(model['layers']):x=np.maximum(x,0)
    return x.reshape(-1,4)


def standardized(scores, valid):
    count=np.maximum(1,valid.sum(-1,keepdims=True))
    mean=(scores*valid).sum(-1,keepdims=True)/count
    sd=np.sqrt(((scores-mean)**2*valid).sum(-1,keepdims=True)/count)
    return (scores-mean)/np.maximum(sd,.1)


def ensemble_scores(bank,embeddings,features,valid,bundle):
    results=[]
    for model in bundle['models']:
        raw=network_scores(embeddings,features,model)
        kind=bundle['aggregation']
        if kind=='normalized':raw=standardized(raw,valid)
        elif kind=='probability':
            raw=np.where(valid,raw,-1e9)
            raw=np.exp(raw-raw.max(-1,keepdims=True));raw/=raw.sum(-1,keepdims=True)
        results.append(raw)
    scores=np.mean(results,axis=0);out=scores.copy();mix=bundle['answer_pooling']
    if mix:
        for i,row in enumerate(bank):
            answers=[parse_answer(c['text']) for c in row['candidates']]
            for j,answer in enumerate(answers):
                if answer is not None:
                    same=[k for k,b in enumerate(answers) if b==answer]
                    out[i,j]=(1-mix)*scores[i,j]+mix*np.mean(scores[i,same])
    return out


def composite_scores(bank,embeddings,features,valid,bundle):
    scores=np.zeros(valid.shape,dtype=np.float64)
    for component in bundle['components']:
        raw=ensemble_scores(bank,embeddings,features,valid,component)
        scores+=component['weight']*standardized(raw,valid)
    return scores


def answer_matrix(bank):
    """Average candidates by parsed answer; no reference answers are involved."""
    pool = np.zeros((len(bank), 4, 4), dtype=np.float32)
    for i, row in enumerate(bank):
        answers = [parse_answer(c['text']) for c in row['candidates']]
        for j, answer in enumerate(answers):
            if answer is not None:
                same = np.array([a == answer for a in answers])
                pool[i, j] = same / same.sum()
    return pool


def projected_scores(bank, embeddings, features, valid, bundle):
    """NumPy inference for grouped-CV v4/v5 models, including input pooling."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    features = features.reshape(len(bank), 4, -1)
    pool = answer_matrix(bank)
    scores = []
    for model in bundle['models']:
        if 'projection' in model:
            e = (embeddings-model['mean']) @ model['projection']
        else:
            e = np.clip((embeddings-model['mean'])/model['scale'], -8, 8)
        f = np.clip((features-model['feature_mean'])/model['feature_scale'], -8, 8)
        parts = [e, f]
        if model.get('context'):
            parts += [e-e.mean(1, keepdims=True), f-f.mean(1, keepdims=True)]
        x = np.concatenate(parts, -1).astype(np.float32)
        if model.get('pool_inputs'): x = pool @ x
        for i, (weight, bias) in enumerate(model['layers']):
            x = x @ weight + bias
            if i+1 < len(model['layers']):
                x = np.tanh(x) if model.get('activation') == 'tanh' else np.maximum(x, 0)
        scores.append((pool @ x).squeeze(-1))
    return np.mean(scores, axis=0)


def frozen_scores(bank, embeddings, features, valid, bundle):
    version = bundle.get('version')
    if version == 12:
        from verification_features import verification_features
        from set_attention import scores as attention_scores
        pool=answer_matrix(bank)
        data=dict(bank=bank,emb=np.asarray(embeddings,dtype=np.float32),features=features.reshape(len(bank),4,-1),
                  checks=verification_features(bank).reshape(len(bank),4,32),valid=valid,pool=pool)
        unique=(np.diagonal(pool,axis1=1,axis2=2)>0)&(np.tril(pool,-1).sum(-1)==0)
        result=np.zeros(valid.shape,dtype=np.float64)
        def aggregate(raw):
            mean=(pool@raw[:,:,None]).squeeze(-1)
            amount=bundle.get('answer_max_weight',0.)
            if not 0<=amount<=1:raise ValueError('Invalid answer max weight')
            if amount:
                maximum=np.max(np.where(pool>0,raw[:,None,:],0.),axis=-1)
                return (1-amount)*mean+amount*maximum
            return mean
        for component in bundle['components']:
            if component['kind']=='attention':
                raw=attention_scores(data,component['model'])
                raw=aggregate(raw)
                if bundle.get('normalization')!='after_fusion':raw/=np.maximum((raw*unique).sum(-1,keepdims=True),1e-15)
            elif component['kind']=='existing':
                raw=frozen_scores(bank,embeddings,features,valid,component['model'])
            else:raise ValueError('Unknown attention mixture component')
            result+=component['weight']*raw
        if bundle.get('normalization')=='after_fusion':result/=np.maximum((result*unique).sum(-1,keepdims=True),1e-15)
        return result
    if version == 9:
        from verification_features import verification_features
        from comparison_features import comparison_features
        from text_ranker import text_scores
        pool=answer_matrix(bank)
        checks=verification_features(bank).reshape(len(bank),4,32)
        extra=comparison_features(bank,embeddings)
        base=frozen_scores(bank,embeddings,features,valid,bundle['baseline'])
        lexical=text_scores(bank,embeddings,features,checks,extra,pool,bundle['text'])
        def probabilities(raw,temperature):
            z=np.where(valid,raw/temperature,-1e9).astype(np.float64)
            p=np.exp(z-z.max(-1,keepdims=True))*valid
            unique=(np.diagonal(pool,axis1=1,axis2=2)>0)&(np.tril(pool,-1).sum(-1)==0)
            return p/np.maximum((p*unique).sum(-1,keepdims=True),1e-15)
        return (1-bundle['weight'])*probabilities(base,1)+bundle['weight']*probabilities(lexical,bundle['temperature'])
    if version in (7, 8):
        from verification_features import verification_features, repair_evidence
        checks=verification_features(bank).reshape(len(bank),4,-1)
        pool=answer_matrix(bank)
        score=np.zeros(valid.shape,dtype=np.float64)
        for component in bundle['components']:
            kind=component['kind'];model=component['model']
            if kind=='legacy':raw=frozen_scores(bank,embeddings,features,valid,model)
            elif kind=='kernel':raw=kernel_scores(embeddings,features,pool,model)
            elif kind=='verified':raw=verified_scores(embeddings,features,checks,pool,model)
            else:raise ValueError(f'Unknown component kind: {kind}')
            if version == 8:
                z=np.log(np.maximum(raw,1e-15)) if component['probability'] else raw
                z=np.where(valid,z/component['temperature'],-1e9).astype(np.float64)
                probability=np.exp(z-z.max(-1,keepdims=True))*valid
                unique=(np.diagonal(pool,axis1=1,axis2=2)>0)&(np.tril(pool,-1).sum(-1)==0)
                probability/=np.maximum((probability*unique).sum(-1,keepdims=True),1e-15)
                score+=component['weight']*probability
            else:
                score+=component['weight']*standardized(raw,valid)
        if version == 8:score=np.log(np.maximum(score,1e-15))
        score-=bundle['penalty']*(pool@checks[:,:,10:11]).squeeze(-1)
        if bundle['reward']:score+=bundle['reward']*repair_evidence(bank)
        return score
    if version == 6:
        scores = np.zeros(valid.shape, dtype=np.float64)
        for component in bundle['components']:
            raw = frozen_scores(bank, embeddings, features, valid, component['bundle'])
            scores += component['weight'] * standardized(raw, valid)
        return scores
    if version in (4, 5):
        return projected_scores(bank, embeddings, features, valid, bundle)
    if version == 3:
        return composite_scores(bank, embeddings, features, valid, bundle)
    if version == 2:
        return ensemble_scores(bank, embeddings, features, valid, bundle)
    raise ValueError(f'Unsupported frozen model version: {version}')


def kernel_scores(embeddings,features,pool,model):
    p=model['prep'];e=np.asarray(embeddings,dtype=np.float32)
    if p['centered']:e=e-e.mean(1,keepdims=True)
    e=np.clip((e-p['mean'])/p['scale'],-8,8)@p['projection']
    f=features.reshape(len(e),4,-1)
    f=np.clip((f-p['feature_mean'])/p['feature_scale'],-8,8)
    x=pool@np.concatenate([e,f],-1).astype(np.float32);x=x.reshape(-1,x.shape[-1])
    anchors=model['anchors'].astype(np.float64)
    anorm=(anchors*anchors).sum(-1);out=[]
    # Float64 accumulation matches sklearn's stable float32-distance path.
    for start in range(0,len(x),256):
        query=x[start:start+256].astype(np.float64)
        dist=np.maximum((query*query).sum(-1)[:,None]+anorm[None,:]-2*(query@anchors.T),0).astype(np.float32)
        out.append(np.exp(-model['gamma']*dist)@model['alpha'])
    return np.concatenate(out).reshape(-1,4)


def verified_scores(embeddings,features,checks,pool,bundle):
    arrays=dict(emb=np.asarray(embeddings,dtype=np.float32),features=features.reshape(len(embeddings),4,-1),checks=checks)
    scores=[]
    for model in bundle['models']:
        p=model['prep'];parts=[np.clip((arrays[k]-p[k+'_mean'])/p[k+'_scale'],-8,8) for k in ('emb','features','checks')]
        x=np.concatenate(parts,-1).astype(np.float32)
        for i,(w,b) in enumerate(model['layers']):
            x=x@w+b
            if i+1<len(model['layers']):x=np.maximum(x,0)
        raw=x.squeeze(-1)
        if model['skip'] is not None:raw+=parts[-1]@model['skip']
        scores.append(answer_readout(raw,pool,model['config'].get('readout','logit')))
    return np.mean(scores,axis=0)


def answer_readout(raw,pool,mode='logit'):
    valid=pool.sum(-1)>0
    unique=(np.diagonal(pool,axis1=1,axis2=2)>0)&(np.tril(pool,k=-1).sum(-1)==0)
    if mode=='binary_probability':
        p=1/(1+np.exp(-np.clip(raw,-60,60)))
        p=(pool@p[:,:,None]).squeeze(-1)
        return p/np.maximum((p*unique).sum(-1,keepdims=True),1e-12)
    pooled=(pool@raw[:,:,None]).squeeze(-1)
    if mode=='answer_probability':
        z=np.where(valid,pooled,-1e9);p=np.exp(z-z.max(-1,keepdims=True))*valid
        return p/np.maximum((p*unique).sum(-1,keepdims=True),1e-12)
    return pooled
