"""Question/candidate alignment and permutation-invariant comparison features."""
import re
from fractions import Fraction
import numpy as np
from common import parse_answer
from number_features import tokenize
from verification_features import arithmetic_steps

GROUPS=(r'left|remain|rest|unused',r'total|altogether|combined|sum',r'each|per|every|rate',r'more|increase|gain|additional',r'less|fewer|decrease|lose|lost',r'percent|percentage|discount|%',r'ratio|fraction|half|quarter|third',r'average|mean',r'difference|apart|than',r'time|hour|minute|second',r'distance|mile|kilometer|speed',r'cost|price|dollar|profit|pay|paid|spent',r'area|length|width|square',r'before|after|then|now|initial|original',r'at least|at most|minimum|maximum|round|whole',r'not|except|excluding|without',r'double|twice|triple|times',r'equal|same|equally')
STOP=set('a an the of to in on at and or is are was were be been it its this that for with from by as how many much what does do did will would can he she they his her their has have had'.split())
def words(s):return set(re.findall(r'[a-z]+',s.lower()))-STOP

def comparison_features(bank,embeddings):
    output=[]
    for row,e in zip(bank,embeddings):
        q=row['question'].lower();qn,_=tokenize(q);qw=words(q);target=words(re.split(r'[.!?]',q.rstrip(' ?'))[-1]);qflags=np.array([bool(re.search(p,q)) for p in GROUPS],dtype=float)
        answers=[parse_answer(c['text']) for c in row['candidates']];av=[float(Fraction(a)) if a is not None else 0. for a in answers]
        traces=[c['text'].lower() for c in row['candidates']];tw=[words(t) for t in traces];steps=[arithmetic_steps(t) for t in traces];local=[]
        for j,t in enumerate(traces):
            tail=t[-300:];w=tw[j];tf=np.array([bool(re.search(p,t)) for p in GROUPS],dtype=float);lf=np.array([bool(re.search(p,tail)) for p in GROUPS],dtype=float)
            f=list(qflags)+list(qflags*tf)+list(qflags*(1-tf))+list(qflags*lf)
            f += [len(qw&w)/max(1,len(qw)),len(target&w)/max(1,len(target)),len(target&words(tail))/max(1,len(target)),len(w-qw)/max(1,len(w))]
            vals=[s['right'] for s in steps[j]];leaves=[v for s in steps[j] for v in s['leaves']]
            f += [sum(any(abs(x-v)<1e-5*max(1,abs(x)) for v in leaves) for x in qn)/max(1,len(qn)),sum(any(abs(x-v)<1e-5*max(1,abs(x)) for v in vals) for x in qn)/max(1,len(qn))]
            for pat in (r'\+',r'(?<=\d)\s*-',r'\*|\\times|multiply|product',r'/|\\frac|divide|quotient'):
                n=len(re.findall(pat,t));f += [np.log1p(n),qflags[0]*np.log1p(n),qflags[1]*np.log1p(n),qflags[2]*np.log1p(n)]
            nums=np.array(qn or [0]);a=av[j];scale=max(1,float(np.max(np.abs(nums))))
            f += [np.sign(a)*np.log1p(abs(a)/scale),np.log1p(abs(a-float(nums.sum()))/scale),np.log1p(min(abs(a-v) for v in nums)/scale),a<0,a!=int(a)]
            others=[k for k in range(4) if k!=j];same=[k for k in others if answers[k]==answers[j]];diff=[k for k in others if answers[k]!=answers[j]]
            for ids in (others,same,diff):
                sim=[len(w&tw[k])/max(1,len(w|tw[k])) for k in ids];shared=[len(set(round(v,5) for v in vals)&set(round(s['right'],5) for s in steps[k]))/max(1,len(vals)) for k in ids]
                f += [np.mean(sim) if sim else 0,max(sim,default=0),np.mean(shared) if shared else 0]
            f += [sum(a<v for v in av)/4,sum(a==v for v in av)/4,np.log1p(abs(a-np.median(av))/max(1,abs(np.median(av))))]
            local.append(f)
        local=np.asarray(local,dtype=np.float32);norm=e/np.maximum(np.linalg.norm(e,axis=-1,keepdims=True),1e-8);similar=norm@norm.T
        relative=local-local.mean(0,keepdims=True)
        output.append(np.concatenate([local,relative,np.sort(similar,axis=-1)[:,:3]],axis=-1))
    return np.asarray(output,dtype=np.float32)
