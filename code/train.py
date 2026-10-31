"""Reproduce the selected verifier from supplied training data and frozen config.

Training does not read validation or test references. Use --output to avoid
replacing the released checkpoint while verifying a local training run.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='4'
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from training_data import load_training_split
from retrain import fit_bundle
from kernel_training import prepare, inputs, solve
from verified_training import fit as fit_verified
from verification_features import verification_features

ROOT=Path(__file__).resolve().parent.parent


def train(config, data=None):
    torch.set_num_threads(4)
    if data is None:data=load_training_split('train')
    if config['version']==12:
        from sklearn.model_selection import StratifiedKFold
        from training_data import subset
        from set_attention import fit_recipe
        data['checks']=verification_features(data['bank']).reshape(len(data['bank']),4,32)
        components=[]
        for component in config['components']:
            if component['kind']=='existing':model=train(component['recipe'],data)
            elif component['kind']=='attention':
                recipe=component['recipe'];models=[]
                if recipe.get('fit_scope')=='full_train':
                    models=[fit_recipe(data,recipe,seed) for seed in recipe['seeds']]
                else:
                    folds=list(StratifiedKFold(recipe.get('fold_count',len(recipe['seeds'])),shuffle=True,random_state=recipe['fold_seed']).split(data['y'],data['y'].sum(1)))
                    fold_indices=recipe.get('fold_indices',list(range(len(recipe['seeds']))))
                    if len(fold_indices)!=len(recipe['seeds']):raise ValueError('Fold/seed count mismatch')
                    for seed,index in zip(recipe['seeds'],fold_indices):models.append(fit_recipe(subset(data,folds[index][0]),recipe,seed))
                model=dict(models=models)
                if recipe.get('parameter_dtype')=='float16':
                    from set_attention import quantize_parameters
                    model=quantize_parameters(model)
            else:raise ValueError('Unknown v12 component')
            components.append(dict(kind=component['kind'],weight=component['weight'],model=model))
        bundle=dict(version=12,components=components)
        if 'normalization' in config:bundle['normalization']=config['normalization']
        if 'answer_max_weight' in config:bundle['answer_max_weight']=config['answer_max_weight']
        return bundle
    if config['version']==9:
        from text_ranker import fit_text
        return dict(version=9,weight=config['weight'],temperature=config['temperature'],baseline=train(config['baseline'],data),text=fit_text(data,config['text']))
    bundle=dict(version=config['version'],penalty=config['penalty'],reward=config['reward'],components=[])
    for component in config['components']:
        kind=component['kind'];cfg=component['configuration']
        if kind=='legacy':
            model=fit_bundle(data,cfg)
        elif kind=='kernel':
            prep=prepare(data,cfg['centered']);x=inputs(data,prep)
            hard=np.any(~data['y'] & data['valid'],1);mask=(data['unique'] & hard[:,None]).ravel()
            x=x[mask];y=data['y'].ravel()[mask].astype('float32');groups=np.repeat(np.arange(len(data['y'])),4)[mask]
            dist=euclidean_distances(x,squared=True).astype('float32');gamma=cfg['gamma']/float(dist.mean())
            alpha,status=solve(np.exp(-gamma*dist),y,groups,cfg['ridge'])
            if status!=0:raise RuntimeError(f'Kernel solver did not converge: {status}')
            model=dict(version='kernel_ranker',prep=prep,anchors=x,alpha=alpha,gamma=gamma,selection=cfg)
        elif kind=='verified':
            data['checks']=verification_features(data['bank']).reshape(len(data['bank']),4,-1)
            models=[fit_verified(data,None,cfg['config'],seed,cfg['epoch'])[0] for seed in cfg['seeds']]
            model=dict(version='verified_ranker',models=models)
        else:raise ValueError(f'Unsupported component: {kind}')
        bundle['components'].append(dict(**{k:v for k,v in component.items() if k!='configuration'},model=model))
        print('Trained',kind,flush=True)
    return bundle


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'code/artifacts/model.pkl')
    parser.add_argument('--config',type=Path,default=ROOT/'code/training_config.json')
    args=parser.parse_args();start=time.perf_counter()
    config=json.loads(args.config.read_text());model=train(config)
    payload=pickle.dumps(model,protocol=5)
    if len(payload)>16*1024**2:raise ValueError('Inference assets exceed 16 MiB')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(payload)
    report=dict(config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),
                training_seconds=time.perf_counter()-start,training_split='train',threads=4,
                validation_references_read=False,test_references_read=False)
    report_path=ROOT/'results/training.json' if args.output.resolve()==(ROOT/'code/artifacts/model.pkl').resolve() else args.output.with_suffix('.training.json')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
