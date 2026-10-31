"""Validation-only fixed-weight ablations and correct-candidate-count diagnostics."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='4'
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np
from training_data import load_training_split,accuracy
from ranker_inference import frozen_scores

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();m=pickle.loads(args.model.read_bytes());assert m['version']==12
    d=load_training_split('validation')
    def evaluate(model):
        p=frozen_scores(d['bank'],d['emb'],d['features'],d['valid'],model)
        choice=np.where(d['valid'],p,-1e9).argmax(1);correct=d['y'][np.arange(len(choice)),choice]
        return dict(**accuracy(d,p),subgroups=[dict(correct_candidates=k,n=int((d['y'].sum(1)==k).sum()),correct=int(correct[d['y'].sum(1)==k].sum())) for k in (1,2,3,4)])
    result=dict(model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),protocol='Validation only; remove one frozen branch and renormalize remaining fixed weights without fitting or reoptimizing. No test data.',full=evaluate(m),ablations=[])
    for i,component in enumerate(m['components']):
        name=component['kind']
        if name=='attention':name=component['model']['models'][0]['config']['name']
        kept=[c for j,c in enumerate(m['components']) if j!=i];total=sum(c['weight'] for c in kept)
        reduced=dict(m,components=[dict(c,weight=c['weight']/total) for c in kept])
        result['ablations'].append(dict(removed=name,**evaluate(reduced)))
    args.output.write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
