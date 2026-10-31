"""Portable lexical comparison ranker: NumPy inference, train-only sklearn fit."""
import re
from collections import Counter
import numpy as np
from comparison_features import comparison_features
from verification_features import verification_features


def normalized_text(text):return re.sub(r'\d+(?:\.\d+)?',' NUM ',text.lower()).lower()


def tokens(text,mode):
    text=normalized_text(text)
    if mode=='word':
        words=re.findall(r'(?u)\b\w\w+\b',text)
        return words+[' '.join(words[i:i+2]) for i in range(len(words)-1)]
    result=[]
    for word in re.sub(r'\s\s+',' ',text).split():
        word=' '+word+' '
        for n in range(3,6):
            result.extend(word[i:i+n] for i in range(max(1,len(word)-n+1)))
            if len(word)<=n:break
    return result


def dense_inputs(embeddings,features,checks,extra):
    return np.concatenate([embeddings,features.reshape(len(embeddings),4,-1),checks,extra],-1).reshape(len(embeddings)*4,-1)


def text_scores(bank,embeddings,features,checks,extra,pool,model):
    x=dense_inputs(embeddings,features,checks,extra)
    scores=np.clip((x-model['mean'])/model['std'],-8,8)@model['dense_coef']
    for i,c in enumerate(c for row in bank for c in row['candidates']):
        values=[];weights=[]
        for token,count in Counter(tokens(c['text'],model['mode'])).items():
            item=model['vocabulary'].get(token)
            if item is not None:
                idf,weight=item;values.append((1+np.log(count))*idf);weights.append(weight)
        if values:
            values=np.asarray(values,dtype=np.float32);values/=max(np.linalg.norm(values),1e-12)
            scores[i]+=model.get('text_scale',20)*float(values@np.asarray(weights))
    return (pool@scores.reshape(-1,4,1)).squeeze(-1)


def fit_text(data,config):
    from scipy.sparse import hstack,csr_matrix,vstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    checks=verification_features(data['bank']).reshape(-1,4,32);extra=comparison_features(data['bank'],data['emb'])
    dense=dense_inputs(data['emb'],data['features'],checks,extra);mean=dense.mean(0);std=np.maximum(dense.std(0),1e-5)
    mode=config['mode'];vec=TfidfVectorizer(analyzer=mode,ngram_range=(1,2) if mode=='word' else (3,5),min_df=5,max_features=15000,sublinear_tf=True,dtype=np.float32)
    texts=[normalized_text(c['text']) for r in data['bank'] for c in r['candidates']];sparse=vec.fit_transform(texts)
    x=(sparse*config.get('text_scale',20)).tocsr() if config.get('text_only') else hstack([csr_matrix(np.clip((dense-mean)/std,-8,8)),sparse*config.get("text_scale",20)]).tocsr()
    pos=[];neg=[]
    for i,(y,valid) in enumerate(zip(data['y'],data['valid'])):
        for j in range(4):
            for k in range(4):
                if y[j] and not y[k] and valid[k]:pos.append(4*i+j);neg.append(4*i+k)
    pairs=x[pos]-x[neg];xx=vstack([pairs,-pairs]);yy=np.r_[np.ones(len(pos)),np.zeros(len(pos))]
    if config.get('loss')=='logistic':
        from sklearn.linear_model import LogisticRegression
        m=LogisticRegression(C=config['C'],fit_intercept=False,solver='liblinear',dual=True,max_iter=3000,tol=1e-5,random_state=7506).fit(xx,yy)
    else:m=LinearSVC(C=config['C'],fit_intercept=False,dual=True,max_iter=3000,tol=1e-3,random_state=7506).fit(xx,yy)
    coef=m.coef_.ravel();n=dense.shape[-1]
    dense_coef=np.zeros(n) if config.get('text_only') else coef[:n].copy()
    offset=0 if config.get('text_only') else n
    result=dict(mode=mode,text_scale=config.get("text_scale",20),mean=mean,std=std,dense_coef=dense_coef,vocabulary={word:(float(vec.idf_[i]),float(coef[offset+i])) for word,i in vec.vocabulary_.items()},config=config)
    portable=text_scores(data['bank'],data['emb'],data['features'],checks,extra,data['pool'],result)
    expected=(data['pool']@m.decision_function(x).reshape(-1,4,1)).squeeze(-1)
    if not np.allclose(portable,expected,atol=2e-6):raise ValueError('Portable text export mismatch')
    return result
