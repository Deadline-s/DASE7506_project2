"""Measure fresh official prediction and scorer processes, including startup."""
import os
import argparse
import sys
import json
import hashlib
import platform
import subprocess
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=root/'artifacts')
    parser.add_argument('--output-dir',type=Path,default=root.parent/'results')
    args=parser.parse_args()
    checkpoint=args.checkpoint.resolve();out=args.output_dir.resolve()
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name] = '4'
    assets = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in checkpoint.iterdir() if p.is_file()}
    assert sum(a['bytes'] for a in assets.values()) <= 16*1024**2
    out.mkdir(parents=True, exist_ok=True)
    # Record assets before scoring; complete the same resource record below.
    (out/'resources.json').write_text(json.dumps(dict(assets=assets,status='evaluation_started'),indent=2)+'\n')
    timings = {}
    for split in ('validation', 'test'):
        start = time.perf_counter()
        subprocess.run([sys.executable,'predict.py','--method','submission.py','--checkpoint',str(checkpoint),'--split',split,'--output',str(out/f'{split}-predictions.json')],cwd=root,env=env,check=True)
        subprocess.run([sys.executable,'score_predictions.py','--split',split,'--predictions',str(out/f'{split}-predictions.json'),'--output',str(out/f'{split}-score.json')],cwd=root,env=env,check=True)
        timings[split] = time.perf_counter()-start
    info = dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(), python=sys.version, threads=4, fresh_prediction_plus_scoring_seconds=timings, assets=assets, reference_machine_verified=False)
    (out/'resources.json').write_text(json.dumps(info,indent=2)+'\n')

if __name__ == '__main__': main()
