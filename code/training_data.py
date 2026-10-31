"""Shared train/development loading. Test labels are explicitly disallowed."""
import numpy as np
from common import DATA, load_inputs, parse_answer, read_jsonl
from reasoning_features import rich_features


def load_training_split(split):
    if split not in ('train', 'validation'):
        raise ValueError('Training code only accepts train or validation')
    bank, arrays = load_inputs(split)
    features, valid, _ = rich_features(bank)
    refs = read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert [r['id'] for r in refs] == [r['id'] for r in bank]
    answers = [[parse_answer(c['text']) for c in r['candidates']] for r in bank]
    y = np.array([[a is not None and a == parse_answer(r['answer']) for a in ans] for ans, r in zip(answers, refs)])
    pool = np.array([[[a is not None and a == b for b in ans] for a in ans] for ans in answers], dtype=np.float32)
    pool /= np.maximum(1, pool.sum(-1, keepdims=True))
    unique = np.array([[a is not None and a not in ans[:j] for j,a in enumerate(ans)] for ans in answers])
    return dict(bank=bank, emb=arrays['embeddings'], features=features.reshape(len(bank),4,-1), valid=valid, y=y, pool=pool, unique=unique)


def subset(data, indices):
    return {k:([v[i] for i in indices] if k == 'bank' else v[indices]) for k,v in data.items()}


def accuracy(data, scores):
    choice = np.where(data['valid'], scores, -1e9).argmax(1)
    correct = data['y'][np.arange(len(choice)), choice]
    hard = np.any(~data['y'] & data['valid'], axis=1)
    return dict(correct=int(correct.sum()), n=len(correct), accuracy=float(100*correct.mean()),
                hard_correct=int(correct[hard].sum()), hard_n=int(hard.sum()), hard_accuracy=float(100*correct[hard].mean()))
