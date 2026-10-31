"""Check portable inference invariants and optionally exact training reproduction."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='4'
import argparse,copy,hashlib,json,pickle
from pathlib import Path
import numpy as np
from training_data import load_training_split,subset
from ranker_inference import frozen_scores
from submission import predict

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reproduced',type=Path);parser.add_argument('--groups',type=int,default=40)
    args=parser.parse_args();model=pickle.loads(args.model.read_bytes())
    full=load_training_split('validation');d=subset(full,np.arange(min(args.groups,len(full['bank']))))
    def score(data):return frozen_scores(data['bank'],data['emb'],data['features'],data['valid'],model)
    raw=score(d);assert np.isfinite(raw).all()
    batch=np.concatenate([score(subset(d,np.arange(i,min(i+7,len(d['bank']))))) for i in range(0,len(d['bank']),7)])
    assert np.allclose(raw,batch,atol=2e-5,rtol=2e-5)
    changed=copy.deepcopy(d)
    for i,row in enumerate(changed['bank']):row['id']='unseen-'+str(i)
    assert np.allclose(raw,score(changed),atol=2e-5,rtol=2e-5)
    perm=np.array([2,0,3,1]);changed=copy.deepcopy(d)
    for row in changed['bank']:row['candidates']=[row['candidates'][j] for j in perm]
    for key in ('emb','features','valid'):changed[key]=changed[key][:,perm]
    assert np.allclose(score(changed),raw[:,perm],atol=2e-5,rtol=2e-5)
    shuffled=np.random.default_rng(7506).permutation(len(raw))
    assert np.allclose(score(subset(d,shuffled)),raw[shuffled],atol=2e-5,rtol=2e-5)
    assert predict([],{},args.model.parent)==[]
    count=0
    def compare(a,b):
        nonlocal count
        if isinstance(a,np.ndarray):assert np.array_equal(a,b);count+=1
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a:compare(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        else:assert a==b
    if args.reproduced:compare(model,pickle.loads(args.reproduced.read_bytes()))
    result=dict(model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),validation_groups=len(d['bank']),finite_scores=True,batch_independence=True,candidate_permutation=True,question_permutation=True,unseen_ids=True,empty_input=True,reproduced_arrays=count if args.reproduced else None)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
