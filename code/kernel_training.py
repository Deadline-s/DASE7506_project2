"""Training-only kernel preprocessing and centered ridge solve."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.decomposition import PCA


def prepare(data, centered):
    e = data['emb']
    if centered: e = e-e.mean(1, keepdims=True)
    mean = e.reshape(-1,1024).mean(0)
    scale = np.maximum(e.reshape(-1,1024).std(0), 1e-5)
    z = np.clip((e-mean)/scale, -8, 8).reshape(-1,1024)
    pca = PCA(n_components=256, svd_solver='randomized', random_state=7506).fit(z)
    f = data['features'].reshape(-1,data['features'].shape[-1])
    return dict(centered=centered, mean=mean, scale=scale, projection=pca.components_.T.astype('float32'),
                feature_mean=f.mean(0), feature_scale=np.maximum(f.std(0),1e-5))


def inputs(data, prep):
    e = data['emb']
    if prep['centered']: e = e-e.mean(1,keepdims=True)
    e = np.clip((e-prep['mean'])/prep['scale'],-8,8) @ prep['projection']
    f = np.clip((data['features']-prep['feature_mean'])/prep['feature_scale'],-8,8)
    x = np.concatenate([e, f],-1).astype('float32')
    # One representation per answer removes repeated-answer weighting.
    return (data['pool'] @ x).reshape(-1,x.shape[-1])


def solve(kernel, y, group, ridge):
    counts = np.bincount(group)
    def center(x):
        return x - (np.bincount(group, weights=x, minlength=len(counts))/np.maximum(counts,1))[group]
    b = center(y).astype('float64')
    def matvec(x):
        return center(kernel @ center(x).astype('float32')).astype('float64') + ridge*x
    op = LinearOperator(kernel.shape, matvec=matvec, dtype=np.float64)
    alpha, status = cg(op, b, rtol=1e-4, atol=1e-6, maxiter=200)
    return center(alpha).astype('float32'), int(status)

