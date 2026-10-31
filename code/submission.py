"""CPU candidate verifier. Never reads references or uses question IDs."""
import pickle
from pathlib import Path
from fractions import Fraction
import re
import numpy as np
from common import parse_answer


def features(bank):
    rows, masks, votes = [], [], []
    for group in bank:
        texts = [c['text'] for c in group['candidates']]
        answers = [parse_answer(t) for t in texts]
        counts = [sum(a == b for b in answers) if a is not None else 0 for a in answers]
        lengths = [len(t) for t in texts]
        for text, answer, count in zip(texts, answers, counts):
            val = float(Fraction(answer)) if answer is not None else 0.
            low = text.lower()
            row = [answer is not None, np.log1p(abs(val)), val < 0, val == int(val),
                   np.log1p(len(text)), len(text)/max(1, np.mean(lengths)),
                   np.log1p(len(group['question'])), len(text.split())/max(1,len(group['question'].split()))]
            row += [np.log1p(low.count(s)) for s in ['\n', '=', '####', '\\boxed', '**', '$', '\\frac', 'step', 'therefore', 'however', 'wait', 'approximately', 'remaining', 'total', '%']]
            row += [count, len(set(a for a in answers if a is not None)), max(counts), count == max(counts)]
            rows.append(row)
        masks.append([a is not None for a in answers]); votes.append(counts)
    return np.asarray(rows, dtype=np.float32), np.asarray(masks), np.asarray(votes)


def choose(scores, valid):
    return np.where(valid, np.asarray(scores).reshape(-1, 4), -1e9).argmax(axis=1)


def predict(bank, arrays, checkpoint):
    """Frozen CPU inference; no reference files, IDs, fitting or test-specific logic."""
    if not bank:
        return []
    if checkpoint is None:
        checkpoint = Path(__file__).resolve().parent/'artifacts'
    with (Path(checkpoint)/'model.pkl').open('rb') as f:
        bundle = pickle.load(f)
    if bundle.get('version') not in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        return legacy_predict(bank, arrays, bundle)
    from reasoning_features import rich_features
    from ranker_inference import frozen_scores
    x, valid, _ = rich_features(bank)
    scores = frozen_scores(bank, arrays['embeddings'], x, valid, bundle)
    return choose(scores, valid).astype(int).tolist()


def legacy_predict(bank, arrays, bundle):
    """Reproduce the disclosed v1 baseline with its separate saved checkpoint."""
    x, valid, _ = features(bank)
    emb = np.asarray(arrays['embeddings']).reshape(-1, 1024)
    linear = bundle['linear'].decision_function(emb)
    mode = bundle['mode']
    if mode == 'linear':
        scores = linear
    else:
        if mode == 'style':
            x = x[:, :23]
        if mode == 'fusion':
            x = np.column_stack([x, linear])
        scores = bundle['tree'].predict_proba(x)[:, 1]
    return choose(scores, valid).astype(int).tolist()
