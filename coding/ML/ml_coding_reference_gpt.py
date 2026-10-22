"""Waymo ML coding practice: all 15 problems, PyTorch first, NumPy backup.

Use *_torch with torch tensors and *_numpy with NumPy arrays. Shared helpers
have the same suffix as their caller. Importing/running this file only defines
functions; it does not train models or execute examples automatically.

Assumptions: CPU inputs, compatible floating feature/parameter dtypes, integer
class labels (torch.long for CE), and boolean masks. No device movement, input
dtype conversions, or assertions. Use float64 for finite-difference checks.
Floating inputs are essential: integer feature arithmetic can overflow and
integer arrays cannot safely receive floating gradient updates.

Shape notation: N/M=examples, D=features, C=classes, B=batch, T=time,
H=hidden width or image height (by context), K/k=neighbors/clusters/kernel.
() denotes a scalar tensor/array. Prefer training-only preprocessing and
validation-only hyperparameter/threshold selection.

Dependencies: numpy, torch. This is an interview reference, with the same
algorithms as the practice PDF; broad input validation is intentionally omitted.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn.functional as F



# ============================================================================
# Q01. Stable softmax and cross entropy
# ============================================================================

# PyTorch (primary)
# logits:(N,C), y:(N,) integer class IDs; N>0, finite logits.
# Return mean cross entropy, probabilities:(N,C), logits gradient:(N,C).
def softmax_cross_entropy_torch(logits, y):
    n, c = logits.shape
    # Gotcha: Subtract the row maximum before exp to avoid overflow.
    shifted = logits - logits.amax(dim=1, keepdim=True)  # (N,C); row max: (N,1)
    logp = shifted - shifted.exp().sum(1, keepdim=True).log()  # (N,C); sum: (N,1)
    p = logp.exp()
    rows = torch.arange(n)
    # Gotcha: Paired row/label indexing selects one target-class log probability per example.
    loss = -logp[rows, y].mean()  # (); selected logp: (N,)
    grad = p.clone()
    grad[rows, y] -= 1
    # Gotcha: Mean loss requires dividing its gradient by N exactly once.
    return loss, p, grad / n

# NumPy (backup)
# logits:(N,C), y:(N,) integer class IDs; N>0, finite logits.
# Return mean cross entropy, probabilities:(N,C), logits gradient:(N,C).
def softmax_cross_entropy_numpy(logits, y):
    z = np.asarray(logits)
    n, c = z.shape
    y = np.asarray(y)
    # Gotcha: Subtract the row maximum before exp to avoid overflow.
    shifted = z - z.max(axis=1, keepdims=True)  # (N,C); row max: (N,1)
    logp = shifted - np.log(
        np.exp(shifted).sum(axis=1, keepdims=True))  # (N,C); sum: (N,1)
    p = np.exp(logp)
    # Gotcha: Paired row/label indexing selects one target-class log probability per example.
    loss = -logp[np.arange(n), y].mean()  # (); selected logp: (N,)
    grad = p.copy()
    grad[np.arange(n), y] -= 1
    # Gotcha: Mean loss requires dividing its gradient by N exactly once.
    grad /= n
    return loss, p, grad


# ============================================================================
# Q02. Pairwise distances and vectorized KNN
# ============================================================================

# PyTorch (primary)
# x:(N,D), y:(M,D) floating features.
# Return all squared Euclidean distances:(N,M), without an (N,M,D) temporary.
def squared_distances_torch(x, y):
    d = ((x*x).sum(1)[:, None] + (y*y).sum(1)[None, :]
         - 2 * x @ y.T)  # (N,M)
    # Gotcha: Dot-product distances can be slightly negative from roundoff; clamp to zero.
    return d.clamp_min(0)

# train_x:(M,D), train_y:(M,), query_x:(N,D); 1<=k<=M.
# Return predictions:(N,); distance ties use train order, vote ties use smaller label.
def knn_predict_torch(train_x, train_y, query_x, k):
    m = len(train_x)
    labels = train_y
    c = int(labels.max().item()) + 1
    d = squared_distances_torch(query_x, train_x)  # (N,M)
    # Gotcha: Stable sorting resolves equal distances by the original training index.
    idx = torch.argsort(d, dim=1, stable=True)[:, :k]  # (N,k)
    neighbors = labels[idx]  # (N,k)
    votes = torch.zeros((len(query_x), c), dtype=torch.long)
    # Gotcha: Repeated class IDs must accumulate; each query row remains separate.
    votes.scatter_add_(1, neighbors, torch.ones_like(neighbors))  # (N,C); accumulate k votes per row
    # Gotcha: argmax chooses the first maximum, so vote ties favor the smaller class ID.
    return votes.argmax(dim=1)  # (N,)

# NumPy (backup)
# x:(N,D), y:(M,D) floating features.
# Return all squared Euclidean distances:(N,M), without an (N,M,D) temporary.
def squared_distances_numpy(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    d = (np.sum(x*x, axis=1)[:, None]
         + np.sum(y*y, axis=1)[None, :] - 2*x@y.T)
    # Gotcha: Large nearly equal vectors can suffer cancellation; direct blocked differences are safer.
    return np.maximum(d, 0.0)

# train_x:(M,D), train_y:(M,), query_x:(N,D); 1<=k<=M.
# Return predictions:(N,); distance ties use train order, vote ties use smaller label.
def knn_predict_numpy(train_x, train_y, query_x, k):
    m = len(train_x)
    labels = np.asarray(train_y)
    c = int(labels.max()) + 1
    d = squared_distances_numpy(query_x, train_x)  # (N,M)
    # Gotcha: Stable sorting resolves equal distances by the original training index.
    idx = np.argsort(d, axis=1, kind='stable')[:, :k]  # (N,k)
    neighbors = labels[idx]  # (N,k)
    votes = np.zeros((len(query_x), c), dtype=int)
    rows = np.repeat(np.arange(len(query_x)), k)  # (N*k,)
    # Gotcha: Repeated advanced indices need add.at; votes[rows, labels] += 1 can lose repeats.
    np.add.at(votes, (rows, neighbors.ravel()), 1)  # (N,C); accumulate k votes per row
    # Gotcha: argmax chooses the first maximum, so vote ties favor the smaller class ID.
    return votes.argmax(axis=1)  # (N,)


# ============================================================================
# Q03. Broadcasting and masked tensor operations
# ============================================================================

# PyTorch (primary)
# x:(N,D); normalize each feature using population mean/std.
# Return (N,D); constant features become zero. Fit statistics on training data only.
def standardize_torch(x, eps=1e-8):
    mu = x.mean(dim=0, keepdim=True)  # (1,D)
    # Gotcha: Use population std (correction=0 / NumPy default), not sample std.
    sd = x.std(dim=0, correction=0, keepdim=True)  # (1,D)
    return (x-mu) / sd.clamp_min(eps)

# h:(B,T,D), valid:(B,T) boolean; average valid timesteps.
# Return (B,D); sequences with no valid timesteps return zeros.
def masked_pool_torch(h, valid):
    # Gotcha: Use where rather than mask multiplication: NaN padding times zero is still NaN.
    h0 = torch.where(valid[..., None], h, 0.)
    # Gotcha: Clamp the count to avoid zero division for all-invalid sequences.
    count = valid.sum(dim=1, keepdim=True).clamp_min(1)  # (B,1)
    return h0.sum(dim=1) / count  # (B,D); time sum: (B,D)

# q:(B,L,D), k:(B,S,D); compare every query/key pair.
# Return batched dot products:(B,L,S).
def pairwise_dot_torch(q, k):
    return q @ k.transpose(-1, -2)  # (B,L,S); transposed k: (B,D,S)

# scores:(B,L,S), valid boolean and broadcastable to scores; True means allowed.
# Softmax over S; allowed scores finite; all-masked rows return zeros.
def masked_softmax_torch(scores, valid):
    valid = torch.broadcast_to(valid, scores.shape)  # (B,L,S)
    z = scores.masked_fill(~valid, -torch.inf)
    has = valid.any(dim=-1, keepdim=True)  # (B,L,1)
    # Gotcha: Replace the all-masked -inf maximum so -inf - (-inf) never occurs.
    peak = torch.where(has, z.amax(dim=-1, keepdim=True), 0.)  # (B,L,1)
    e = (z-peak).exp()
    total = e.sum(dim=-1, keepdim=True)  # (B,L,1)
    # Gotcha: A zero denominator represents an all-masked row; return zeros there.
    return e / torch.where(total > 0, total, 1.)

# NumPy (backup)
# x:(N,D); normalize each feature using population mean/std.
# Return (N,D); constant features become zero. Fit statistics on training data only.
def standardize_numpy(x, eps=1e-8):
    mu = x.mean(axis=0, keepdims=True)  # (1,D)
    # Gotcha: Use population std (correction=0 / NumPy default), not sample std.
    sd = x.std(axis=0, keepdims=True)  # (1,D)
    return (x-mu) / np.maximum(sd, eps)

# h:(B,T,D), valid:(B,T) boolean; average valid timesteps.
# Return (B,D); sequences with no valid timesteps return zeros.
def masked_pool_numpy(h, valid):
    # Gotcha: Use where rather than mask multiplication: NaN padding times zero is still NaN.
    h0 = np.where(valid[..., None], h, 0.0)
    # Gotcha: Clamp the count to avoid zero division for all-invalid sequences.
    count = valid.sum(axis=1, keepdims=True)  # (B,1)
    return h0.sum(axis=1) / np.maximum(count, 1)  # (B,D); time sum: (B,D)

# q:(B,L,D), k:(B,S,D); compare every query/key pair.
# Return batched dot products:(B,L,S).
def pairwise_dot_numpy(q, k):
    return q @ k.swapaxes(-1, -2)  # (B,L,S); transposed k: (B,D,S)

# scores:(B,L,S), valid boolean and broadcastable to scores; True means allowed.
# Softmax over S; allowed scores finite; all-masked rows return zeros.
def masked_softmax_numpy(scores, valid):
    # valid is broadcastable to scores; valid scores finite.
    z = np.where(valid, scores, -np.inf)
    has = np.any(valid, axis=-1, keepdims=True)  # (B,L,1)
    # Gotcha: Replace the all-masked -inf maximum so -inf - (-inf) never occurs.
    peak = np.where(has, z.max(axis=-1, keepdims=True), 0)  # (B,L,1)
    e = np.exp(z-peak)
    total = e.sum(axis=-1, keepdims=True)  # (B,L,1)
    # Gotcha: A zero denominator represents an all-masked row; return zeros there.
    return e / np.where(total > 0, total, 1)


# ============================================================================
# Q04. K Means clustering
# ============================================================================

# PyTorch (primary)
# x:(N,D), 1<=k<=N, max_iter>=1; cluster using squared Euclidean distance.
# Return centers:(k,D), labels:(N,); empty clusters retain their previous centers.
@torch.no_grad()
def kmeans_torch(x, k, max_iter=100, tol=1e-6, seed=0):
    n, d = x.shape
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=gen)[:k]  # (k,)
    centers = x[idx].clone()  # (k,D)
    for _ in range(max_iter):
        # Gotcha: Assign by nearest center; the final reassignment must use the returned centers.
        labels = squared_distances_torch(x, centers).argmin(dim=1)  # (N,); distances: (N,k)
        counts = torch.bincount(labels, minlength=k)  # (k,)
        sums = x.new_zeros((k, d))
        # Gotcha: Each label routes a whole feature row; duplicate labels sum into one cluster.
        sums.index_add_(0, labels, x)  # (k,D); sum rows by cluster
        used = counts > 0
        # Gotcha: Empty clusters have no mean; retain their old centers (reseeding is an alternative).
        new = centers.clone()
        # Gotcha: Filter out zero counts; (U,1) denominators divide each of U rows across D features.
        new[used] = sums[used] / counts[used].unsqueeze(dim=1)
        shift = torch.linalg.vector_norm(new-centers, dim=1).max()  # (); row norms: (k,)
        centers = new
        if shift.item() <= tol: break
    # Gotcha: Assign by nearest center; the final reassignment must use the returned centers.
    labels = squared_distances_torch(x, centers).argmin(dim=1)  # (N,); distances: (N,k)
    return centers, labels

# NumPy (backup)
# x:(N,D), 1<=k<=N, max_iter>=1; cluster using squared Euclidean distance.
# Return centers:(k,D), labels:(N,); empty clusters retain their previous centers.
def kmeans_numpy(x, k, max_iter=100, tol=1e-6, seed=0):
    x = np.asarray(x)
    n, d = x.shape
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, k, replace=False)].copy()  # (k,D)
    for _ in range(max_iter):
        # Gotcha: Assign by nearest center; the final reassignment must use the returned centers.
        labels = squared_distances_numpy(x, centers).argmin(1)  # (N,); distances: (N,k)
        counts = np.bincount(labels, minlength=k)  # (k,)
        sums = np.zeros((k, d))
        # Gotcha: Accumulate every row assigned to a cluster, including repeated labels.
        np.add.at(sums, labels, x)  # (k,D); sum rows by cluster
        # Gotcha: Empty clusters have no mean; retain their old centers (reseeding is an alternative).
        new = centers.copy()
        used = counts > 0
        # Gotcha: Filter out zero counts; (U,1) denominators divide each of U rows across D features.
        new[used] = sums[used] / counts[used, None]
        shift = np.linalg.norm(new-centers, axis=1).max()  # (); row norms: (k,)
        centers = new
        if shift <= tol:
            break
    # Gotcha: Assign by nearest center; the final reassignment must use the returned centers.
    labels = squared_distances_numpy(x, centers).argmin(1)  # (N,); distances: (N,k)
    return centers, labels


# ============================================================================
# Q05. Logistic regression with a custom objective
# ============================================================================

# PyTorch (primary)
# z: arbitrary floating shape; return sigmoid probabilities of the same shape.
def sigmoid_torch(z):
    return torch.sigmoid_torch(z)

# x:(N,D), binary y:(N,), w:(D,), b:scalar; alpha>0, lam>=0.
# Return weighted BCE + L2 loss, dw:(D,), db:scalar; bias is not regularized.
def logistic_loss_grad_torch(x, y, w, b, alpha=1., lam=0.):
    n = len(x)
    z = x @ w + b  # (N,)
    weight = torch.where(y == 1, alpha, 1.)
    # Gotcha: Normalize weighted BCE by N, not sum(weights); L2 is 0.5*lam*||w||^2.
    loss = (weight*(torch.logaddexp(torch.zeros_like(z), z)
                    - y*z)).mean() + 0.5*lam*(w*w).sum()  # (); mean over N
    r = weight*(sigmoid_torch(z)-y)
    return loss, x.T @ r/n + lam*w, r.mean()  # (), (D,), ()

# x:(N,D), binary y:(N,); full-batch gradient descent with positive weight and L2.
# Return w:(D,), b:scalar; predict with sigmoid(x @ w + b).
@torch.no_grad()
def fit_logistic_torch(x, y, steps=500, lr=0.1, alpha=1., lam=0.):
    w = x.new_zeros(x.shape[1]); b = x.new_zeros(())
    for _ in range(steps):
        loss, dw, db = logistic_loss_grad_torch(x,y,w,b,alpha,lam)  # (), (D,), ()
        # Gotcha: Too-large learning rates can diverge; feature scaling helps conditioning.
        w -= lr*dw; b -= lr*db
    return w, b
# Probabilities: sigmoid_torch(x @ w + b)

# NumPy (backup)
# z: arbitrary floating shape; return sigmoid probabilities of the same shape.
def sigmoid_numpy(z):
    z = np.asarray(z)
    return np.exp(-np.logaddexp(0.0, -z))

# x:(N,D), binary y:(N,), w:(D,), b:scalar; alpha>0, lam>=0.
# Return weighted BCE + L2 loss, dw:(D,), db:scalar; bias is not regularized.
def logistic_loss_grad_numpy(x, y, w, b, alpha=1., lam=0.):
    n = len(x)
    z = x @ w + b  # (N,)
    weight = np.where(y == 1, alpha, 1.)
    # Gotcha: Normalize weighted BCE by N, not sum(weights); L2 is 0.5*lam*||w||^2.
    loss = np.mean(weight*(np.logaddexp(0., z)-y*z))
    loss += 0.5*lam*np.dot(w, w)
    r = weight*(sigmoid_numpy(z)-y)
    return loss, x.T @ r/n + lam*w, r.mean()  # (), (D,), ()

# x:(N,D), binary y:(N,); full-batch gradient descent with positive weight and L2.
# Return w:(D,), b:scalar; predict with sigmoid(x @ w + b).
def fit_logistic_numpy(x, y, steps=500, lr=0.1,
                 alpha=1., lam=0.):
    w = np.zeros(x.shape[1]); b = 0.
    for _ in range(steps):
        loss, dw, db = logistic_loss_grad_numpy(
            x, y, w, b, alpha, lam)
        # Gotcha: Too-large learning rates can diverge; feature scaling helps conditioning.
        w -= lr*dw; b -= lr*db
    return w, b
# Probability: sigmoid_numpy(X @ w + b)
# Class: probability >= an explicitly selected threshold


# ============================================================================
# Q06. Linear regression and gradient descent
# ============================================================================

# PyTorch (primary)
# x:(N,D), y:(N,), w:(D,), b:scalar; use half-MSE = mean(residual**2)/2.
# Return scalar loss, dw:(D,), db:scalar.
def linear_loss_grad_torch(x, y, w, b):
    r = x @ w + b - y  # (N,)
    return 0.5*(r*r).mean(), x.T @ r/len(x), r.mean()  # (), (D,), ()

# x:(N,D), y:(N,); fit linear regression with full-batch gradient descent.
# Return w:(D,), b:scalar.
@torch.no_grad()
def fit_linear_torch(x, y, steps=1000, lr=0.05):
    w = x.new_zeros(x.shape[1]); b = x.new_zeros(())
    for _ in range(steps):
        loss, dw, db = linear_loss_grad_torch(x, y, w, b)
        w -= lr*dw; b -= lr*db
    return w, b

# x:(N,D), y:(N,); solve least squares with an appended bias column.
# Return w:(D,), b:scalar; use a solver rather than an explicit inverse.
def least_squares_torch(x, y):
    design = torch.cat([x, x.new_ones((len(x), 1))], dim=1)  # (N,D+1)
    # CPU gelsd supports rank-deficient inputs.
    # Gotcha: CPU gelsd handles rank-deficient designs; avoid inverse(X.T @ X).
    theta = torch.linalg.lstsq(design, y, driver='gelsd').solution  # (D+1,)
    return theta[:-1], theta[-1]

# NumPy (backup)
# x:(N,D), y:(N,), w:(D,), b:scalar; use half-MSE = mean(residual**2)/2.
# Return scalar loss, dw:(D,), db:scalar.
def linear_loss_grad_numpy(x, y, w, b):
    r = x @ w + b  # (N,)
    r = r-y
    return 0.5*np.mean(r*r), x.T @ r/len(x), r.mean()  # (), (D,), ()

# x:(N,D), y:(N,); fit linear regression with full-batch gradient descent.
# Return w:(D,), b:scalar.
def fit_linear_numpy(x, y, steps=1000, lr=0.05):
    w = np.zeros(x.shape[1]); b = 0.
    for _ in range(steps):
        loss, dw, db = linear_loss_grad_numpy(x, y, w, b)
        w -= lr*dw; b -= lr*db
    return w, b

# x:(N,D), y:(N,); solve least squares with an appended bias column.
# Return w:(D,), b:scalar; use a solver rather than an explicit inverse.
def least_squares_numpy(x, y):
    design = np.column_stack([x, np.ones(len(x))])  # (N,D+1)
    # Gotcha: lstsq handles rank deficiency without explicitly forming a matrix inverse.
    theta, *_ = np.linalg.lstsq(design, y, rcond=None)  # (D+1,)
    return theta[:-1], theta[-1]


# ============================================================================
# Q07. Convolution and max pooling
# ============================================================================

# PyTorch (primary)
# x:(B,Cin,H,W), weight:(Cout,Cin,Kh,Kw), bias:(Cout,); stride>0, padding>=0.
# Return (B,Cout,Ho,Wo); deep-learning convolution is unflipped cross-correlation.
def conv2d_torch(x, weight, bias, stride=1, padding=0):
    kh, kw = weight.shape[-2:]
    xp = F.pad(x, (padding, padding, padding, padding))  # (B,Cin,H+2p,W+2p)
    # Gotcha: unfold appends window axes; the second unfold still targets original width axis 3.
    win = xp.unfold(2, kh, stride).unfold(3, kw, stride)  # (B,Cin,Ho,Wo,Kh,Kw)
    # win: B,Cin,Ho,Wo,Kh,Kw; no conv2d_torch helper.
    # Gotcha: Contract Cin/Kh/Kw; do not flip the kernel for deep-learning cross-correlation.
    out = torch.einsum('bchwij,ocij->bohw', win, weight)  # (B,Cout,Ho,Wo)
    return out + bias[None, :, None, None]

# x:(B,C,H,W); square kernel, positive stride, no padding.
# Return (B,C,Ho,Wo); reduce each spatial window by its maximum.
def max_pool2d_torch(x, kernel=2, stride=2):
    win = x.unfold(2, kernel, stride).unfold(3, kernel, stride)  # (B,C,Ho,Wo,K,K)
    return win.amax(dim=(-2, -1))  # (B,C,Ho,Wo)

# NumPy (backup)

# x:(B,Cin,H,W), weight:(Cout,Cin,Kh,Kw), bias:(Cout,); stride>0, padding>=0.
# Return (B,Cout,Ho,Wo); deep-learning convolution is unflipped cross-correlation.
def conv2d_numpy(x, weight, bias, stride=1, padding=0):
    kh, kw = weight.shape[-2:]
    xp = np.pad(x, ((0,0),(0,0),(padding,padding),
                    (padding,padding)))
    win = sliding_window_view(xp, (kh, kw), axis=(2,3))
    win = win[:, :, ::stride, ::stride, :, :]  # (B,Cin,Ho,Wo,Kh,Kw)
    # win: B,Cin,Ho,Wo,Kh,Kw
    # Gotcha: Contract Cin/Kh/Kw; do not flip the kernel for deep-learning cross-correlation.
    out = np.einsum('bchwij,ocij->bohw', win, weight)  # (B,Cout,Ho,Wo)
    return out + np.asarray(bias)[None, :, None, None]

# x:(B,C,H,W); square kernel, positive stride, no padding.
# Return (B,C,Ho,Wo); reduce each spatial window by its maximum.
def max_pool2d_numpy(x, kernel=2, stride=2):
    win = sliding_window_view(
        x, (kernel, kernel), axis=(2,3))
    return win[:, :, ::stride, ::stride].max(axis=(-2,-1))  # (B,C,Ho,Wo)


# ============================================================================
# Q08. Two layer neural network and backpropagation
# ============================================================================

# PyTorch (primary)
# x:(N,D), y:(N,), w1:(D,H), b1:(H,), w2:(H,C), b2:(C,).
# Return mean CE and analytic (dw1,db1,dw2,db2); ReLU hidden layer, no autograd.
def mlp_loss_grad_torch(x, y, w1, b1, w2, b2):
    a1 = x @ w1 + b1  # (N,H)
    h = a1.clamp_min(0)
    logits = h @ w2 + b2  # (N,C)
    loss, p, dz = softmax_cross_entropy_torch(logits, y)  # (), (N,C), (N,C)
    dw2 = h.T @ dz  # (H,C)
    db2 = dz.sum(dim=0)  # (C,)
    # Gotcha: CE already averaged over N; do not divide again. ReLU derivative at zero is defined as 0.
    da1 = (dz @ w2.T) * (a1 > 0)  # (N,H)
    dw1 = x.T @ da1  # (D,H)
    db1 = da1.sum(dim=0)  # (H,)
    return loss, (dw1, db1, dw2, db2)
# Analytic gradients: no autograd required.
# Apply updates under torch.no_grad().
# ReLU initialization: torch.randn(D,H) * (2/D)**0.5

# NumPy (backup)
# x:(N,D), y:(N,), w1:(D,H), b1:(H,), w2:(H,C), b2:(C,).
# Return mean CE and analytic (dw1,db1,dw2,db2); ReLU hidden layer, no autograd.
def mlp_loss_grad_numpy(x, y, w1, b1, w2, b2):
    a1 = x @ w1 + b1  # (N,H)
    h = np.maximum(a1, 0)
    logits = h @ w2 + b2  # (N,C)
    loss, p, dz = softmax_cross_entropy_numpy(logits, y)  # (), (N,C), (N,C)
    dw2 = h.T @ dz  # (H,C)
    db2 = dz.sum(axis=0)  # (C,)
    # Gotcha: CE already averaged over N; do not divide again. ReLU derivative at zero is defined as 0.
    da1 = (dz @ w2.T) * (a1 > 0)  # (N,H)
    dw1 = x.T @ da1  # (D,H)
    db1 = da1.sum(axis=0)  # (H,)
    return loss, (dw1, db1, dw2, db2)

# One step: param -= lr * corresponding_grad
# Example initialization for the ReLU layer:
# rng = np.random.default_rng(0)
# w1 = rng.normal(size=(D,H)) * np.sqrt(2/D)
# b1 = np.zeros(H)


# ============================================================================
# Q09. PCA using SVD
# ============================================================================

# PyTorch (primary)
# x:(N,D), N>=2, 1<=k<=min(N,D); centered PCA via reduced SVD.
# Return mean:(D,), components:(k,D), sample explained variance:(k,).
def fit_pca_torch(x, k):
    n, d = x.shape
    mean = x.mean(dim=0)  # (D,)
    # Gotcha: SVD signs are arbitrary, and repeated singular values can rotate the basis.
    _, s, vh = torch.linalg.svd(x-mean, full_matrices=False)  # U:(N,R), s:(R,), Vh:(R,D)
    # Gotcha: Sample variance uses N-1. Zero-variance components need a cutoff before whitening.
    return mean, vh[:k], s[:k].square() / (n-1)  # (D,), (k,D), (k,); R=min(N,D)

# x:(N,D), mean:(D,), components:(k,D) fitted on training data.
# Return projected features:(N,k).
def transform_pca_torch(x, mean, components):
    return (x-mean) @ components.T  # (N,k)

# NumPy (backup)
# x:(N,D), N>=2, 1<=k<=min(N,D); centered PCA via reduced SVD.
# Return mean:(D,), components:(k,D), sample explained variance:(k,).
def fit_pca_numpy(x, k):
    x = np.asarray(x)
    n, d = x.shape
    mean = x.mean(axis=0)  # (D,)
    # Gotcha: SVD signs are arbitrary, and repeated singular values can rotate the basis.
    _, s, vt = np.linalg.svd(x-mean, full_matrices=False)  # U:(N,R), s:(R,), Vt:(R,D)
    # Gotcha: Sample variance uses N-1. Zero-variance components need a cutoff before whitening.
    return mean, vt[:k], s[:k]**2/(n-1)  # (D,), (k,D), (k,); R=min(N,D)

# x:(N,D), mean:(D,), components:(k,D) fitted on training data.
# Return projected features:(N,k).
def transform_pca_numpy(x, mean, components):
    return (x-mean) @ components.T  # (N,k)

# Reconstruction: projected @ components + mean
# Whiten: projected / sqrt(max(variance, eps))
# Fit on train data and reuse mean/components on test.


# ============================================================================
# Q10. Decision tree with Gini impurity
# ============================================================================

# PyTorch (primary)
# y:(N,) nonnegative integer class IDs, c:number of classes.
# Return Gini impurity; empty labels have impurity zero.
def gini_torch(y, c):
    if len(y) == 0: return 0.
    p = torch.bincount(y, minlength=c)/len(y)  # (C,)
    return (1-(p*p).sum()).item()

# x:(N,D), y:(N,) class IDs; nonempty finite data, depth>=0, min_leaf>=1.
# Return a recursive classification tree minimizing size-weighted child Gini.
@torch.no_grad()
def fit_tree_torch(x, y, depth=3, min_leaf=1, c=None):
    if c is None: c = int(y.max().item())+1
    pred = int(torch.bincount(y, minlength=c).argmax().item())
    node = {'pred': pred}
    if depth == 0 or y.unique().numel() == 1: return node
    best = gini_torch(y, c); split = None
    for j in range(x.shape[1]):
        vals = x[:,j].unique(sorted=True)  # (U,); U distinct feature values
        for lo, hi in zip(vals[:-1], vals[1:]):
            # Gotcha: Only thresholds between distinct feature values can create different partitions.
            t = (lo/2 + hi/2).item()
            left = x[:,j] <= t
            nl = int(left.sum().item()); nr = len(y)-nl
            # Gotcha: Both child nodes must satisfy min_leaf.
            if min(nl, nr) < min_leaf: continue
            score = (nl*gini_torch(y[left],c)
                     + nr*gini_torch(y[~left],c))/len(y)
            # Gotcha: Strict improvement preserves the first encountered split when scores tie.
            if score < best:
                best = score; split = (j,t,left)
    if split is None: return node
    j,t,left = split
    node.update(j=j, t=t,
        left=fit_tree_torch(x[left],y[left],depth-1,min_leaf,c),
        right=fit_tree_torch(x[~left],y[~left],depth-1,min_leaf,c))
    return node

# node: fitted tree dictionary, row:(D,) feature vector.
# Return a Python integer class prediction for one observation.
def predict_tree_torch(node, row):
    while 'j' in node:
        side = 'left' if row[node['j']].item()<=node['t'] else 'right'
        node = node[side]
    return node['pred']
# Python branching makes this CPU-oriented, not a GPU speedup.

# NumPy (backup)
# y:(N,) nonnegative integer class IDs, c:number of classes.
# Return Gini impurity; empty labels have impurity zero.
def gini_numpy(y, c):
    if len(y) == 0: return 0.
    p = np.bincount(y, minlength=c) / len(y)  # (C,)
    return 1. - p @ p

# x:(N,D), y:(N,) class IDs; nonempty finite data, depth>=0, min_leaf>=1.
# Return a recursive classification tree minimizing size-weighted child Gini.
def fit_tree_numpy(x, y, depth=3, min_leaf=1, c=None):
    if c is None: c = int(y.max())+1
    pred = int(np.bincount(y, minlength=c).argmax())
    node = {'pred': pred}
    if depth == 0 or np.unique(y).size == 1: return node
    best = gini_numpy(y, c); split = None
    for j in range(x.shape[1]):
        vals = np.unique(x[:,j])  # (U,); U distinct feature values
        for lo, hi in zip(vals[:-1], vals[1:]):
            # Gotcha: Only thresholds between distinct feature values can create different partitions.
            t = lo/2 + hi/2
            left = x[:,j] <= t
            nl = int(left.sum()); nr = len(y)-nl
            # Gotcha: Both child nodes must satisfy min_leaf.
            if min(nl, nr) < min_leaf: continue
            score = (nl*gini_numpy(y[left],c)
                     + nr*gini_numpy(y[~left],c))/len(y)
            # Gotcha: Strict improvement preserves the first encountered split when scores tie.
            if score < best:
                best = score; split = (j,t,left)
    if split is None: return node
    j,t,left = split
    node.update(j=j, t=t,
        left=fit_tree_numpy(x[left],y[left],depth-1,min_leaf,c),
        right=fit_tree_numpy(x[~left],y[~left],depth-1,min_leaf,c))
    return node

# node: fitted tree dictionary, row:(D,) feature vector.
# Return a Python integer class prediction for one observation.
def predict_tree_numpy(node, row):
    while 'j' in node:
        node = node['left' if row[node['j']]<=node['t']
                    else 'right']
    return node['pred']


# ============================================================================
# Q11. Reservoir and stratified sampling
# ============================================================================

# PyTorch (primary)
# stream: arbitrary items of unknown total length, k>=0.
# Return up to k uniformly sampled items using O(k) memory.
def reservoir_sample_torch(stream, k, seed=0):
    if k == 0: return []
    gen = torch.Generator().manual_seed(seed)
    out = []
    for i, item in enumerate(stream):
        if i < k: out.append(item)
        else:
            # Gotcha: Draw from 0 through i inclusive; each new item enters with probability k/(i+1).
            j = int(torch.randint(i+1, (), generator=gen).item())
            if j < k: out[j] = item
    return out
# y:(N,) class labels, q>=0; sample exactly q examples per observed class.
# Return shuffled dataset indices:(C*q,), without replacement; equal class quotas.
def stratified_indices_torch(y, q, seed=0):
    gen = torch.Generator().manual_seed(seed)
    parts = []
    for label in y.unique(sorted=True):
        # Gotcha: nonzero/flatnonzero returns original dataset positions for this class.
        idx = (y == label).nonzero(as_tuple=True)[0]  # (n_class,)
        if len(idx) < q: raise ValueError('class too small')
        # Gotcha: These are local positions inside idx, not original dataset indices.
        perm = torch.randperm(len(idx), generator=gen)  # (n_class,)
        # Gotcha: Map the random local positions back to dataset indices; no duplicates within a class.
        parts.append(idx[perm[:q]])  # (q,) appended per class
    selected = (torch.cat(parts) if parts else
                torch.empty(0, dtype=torch.long))
    # Gotcha: Shuffle again to avoid returning examples grouped by class.
    perm = torch.randperm(len(selected), generator=gen)  # (C*q,)
    return selected[perm]  # (C*q,)
# CPU reservoir RNG; seeds need not match NumPy draws.

# NumPy (backup)
# stream: arbitrary items of unknown total length, k>=0.
# Return up to k uniformly sampled items using O(k) memory.
def reservoir_sample_numpy(stream, k, seed=0):
    if k == 0: return []
    rng = np.random.default_rng(seed)
    out = []
    for i, item in enumerate(stream):
        if i < k: out.append(item)
        else:
            # Gotcha: Draw from 0 through i inclusive; each new item enters with probability k/(i+1).
            j = int(rng.integers(i+1))
            if j < k: out[j] = item
    return out
# y:(N,) class labels, q>=0; sample exactly q examples per observed class.
# Return shuffled dataset indices:(C*q,), without replacement; equal class quotas.
def stratified_indices_numpy(y, q, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for label in np.unique(y):
        # Gotcha: nonzero/flatnonzero returns original dataset positions for this class.
        idx = np.flatnonzero(y == label)  # (n_class,)
        if len(idx) < q: raise ValueError('class too small')
        # Gotcha: Sample original dataset indices directly, without replacement.
        parts.append(rng.choice(idx, q, replace=False))
    selected = np.concatenate(parts) if parts else np.array([],int)
    # Gotcha: The final shuffle mixes the per-class selections, not the class quotas.
    rng.shuffle(selected)
    return selected


# ============================================================================
# Q12. SGD momentum and Adam updates
# ============================================================================

# PyTorch (primary)
# theta, grad, v: same parameter shape; v initialized to zeros.
# Return updated theta and v; convention is v=beta*v+grad.
@torch.no_grad()
def momentum_step_torch(theta, grad, v, lr=0.1, beta=0.9):
    v = beta*v + grad
    return theta-lr*v, v

# theta, grad, m, v: same shape; m/v initially zero, t initially 0.
# Return updated theta, m, v, t; t counts optimizer updates.
@torch.no_grad()
def adam_step_torch(theta, grad, m, v, t, lr=1e-3,
              beta1=0.9, beta2=0.999, eps=1e-8):
    # Gotcha: Increment only on an optimizer step, not on each gradient-accumulation minibatch.
    t += 1
    m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*grad.square()
    # Gotcha: Bias correction compensates for zero-initialized first/second moments.
    mh = m/(1-beta1**t); vh = v/(1-beta2**t)
    # Gotcha: Epsilon goes outside sqrt; copy the functional result into parameters under no_grad.
    updated = theta-lr*mh/(vh.sqrt()+eps)
    return updated, m, v, t
# Functional API returns new tensors. For an nn.Parameter:
# with torch.no_grad(): parameter.copy_(updated)
# Keep separate m, v and t state for each parameter.

# NumPy (backup)
# theta, grad, v: same parameter shape; v initialized to zeros.
# Return updated theta and v; convention is v=beta*v+grad.
def momentum_step_numpy(theta, grad, v, lr=0.1, beta=0.9):
    v = beta*v + grad
    return theta-lr*v, v

# theta, grad, m, v: same shape; m/v initially zero, t initially 0.
# Return updated theta, m, v, t; t counts optimizer updates.
def adam_step_numpy(theta, grad, m, v, t, lr=1e-3,
              beta1=0.9, beta2=0.999, eps=1e-8):
    # Gotcha: Increment only on an optimizer step, not on each gradient-accumulation minibatch.
    t += 1
    m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*grad*grad
    # Gotcha: Bias correction compensates for zero-initialized first/second moments.
    mh = m/(1-beta1**t)
    vh = v/(1-beta2**t)
    # Gotcha: Epsilon is outside sqrt; AdamW adds separate decay using the old theta.
    theta = theta-lr*mh/(np.sqrt(vh)+eps)
    return theta, m, v, t

# Adam initialization for each parameter:
# m = np.zeros_like(theta); v = np.zeros_like(theta); t = 0
# AdamW: use old theta for an extra -lr*wd*theta term.


# ============================================================================
# Q13. PyTorch training step and weighted masked loss
# ============================================================================

# PyTorch (primary)

# model(x):(B,T,C), targets/valid:(B,T), weights:(C,) nonnegative.
# Return weighted mean CE as a Python float; skip optimizer update if no effective weight.
def train_step_torch(model, optimizer, x, targets, valid, weights):
    model.train()
    # Gotcha: Clear gradients each step; otherwise backward accumulates them.
    optimizer.zero_grad(set_to_none=True)
    if not bool(valid.any()): return None
    logits = model(x)  # (B,T,C)
    # Gotcha: Filter padding before indexing weights or CE: invalid targets may contain -1.
    z = logits[valid]  # (M,C)
    y = targets[valid]  # (M,)
    w = weights[y]  # (M,)
    denom = w.sum()  # ()
    # Gotcha: Skip zero effective weight; model forward may already have changed BatchNorm statistics.
    if denom.item() == 0: return None
    # Gotcha: Use unreduced CE and divide by total selected weight, not valid-token count.
    loss = (F.cross_entropy(z, y, reduction='none')*w).sum()  # (); per-token CE: (M,)
    loss = loss/denom
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach().item()

# NumPy (backup)
# logits:(B,T,C), targets/valid:(B,T), weights:(C,) nonnegative.
# Return weighted mean CE and logits gradient; None + zero gradient if no effective weight.
def weighted_masked_ce_numpy(logits, targets, valid, weights):
    # Loss and logits gradient; network backprop is separate.
    grad = np.zeros_like(logits)
    if not np.any(valid): return None, grad
    # Gotcha: Filter padding before indexing weights or CE: invalid targets may contain -1.
    z = logits[valid]; y = targets[valid]  # z:(M,C), y:(M,)
    w = weights[y]; denom = w.sum()  # w:(M,), denom:()
    # Gotcha: Zero effective weight means skip the loss and parameter update.
    if denom == 0: return None, grad
    shifted = z-z.max(axis=1, keepdims=True)
    logp = shifted-np.log(np.exp(shifted).sum(1, keepdims=True))
    rows = np.arange(len(y))
    loss = -(w*logp[rows,y]).sum()/denom  # (); indexed logp: (M,)
    dz = np.exp(logp)
    dz[rows,y] -= 1
    # Gotcha: Scatter valid-token gradients back into (B,T,C); padding gradients remain zero.
    grad[valid] = dz * (w/denom)[:,None]
    return loss, grad
# NumPy has no generic model/optimizer autograd API.
# Backprop grad through a manually implemented model (Q8),
# then apply its parameter updates (Q12); skip if loss is None.


# ============================================================================
# Q14. Sequence processing with an RNN and LSTM cell
# ============================================================================

# PyTorch (primary)
# x:(B,T,D), lengths:(B,), wx:(D,H), wh:(H,H), b:(H,); 0<=lengths<=T.
# Return final hidden states:(B,H); trailing padding is ignored.
def rnn_final_torch(x, lengths, wx, wh, b):
    batch, steps, _ = x.shape
    h = x.new_zeros((batch, wh.shape[0]))
    for t in range(steps):
        live = (t < lengths)[:, None]
        # Gotcha: Zero invalid inputs before matmul so NaN padding cannot contaminate computations.
        xt = torch.where(live, x[:,t], 0.)
        candidate = torch.tanh(xt @ wx + h @ wh + b)  # (B,H)
        # Gotcha: Inactive sequences retain their prior state; length-zero sequences stay zero.
        h = torch.where(live, candidate, h)
    return h

# x:(B,D), h/c:(B,H), wx:(D,4H), wh:(H,4H), b:(4H,).
# Return h_new/c_new:(B,H); gate order is input, forget, candidate, output.
def lstm_cell_torch(x, h, c, wx, wh, b):
    gates = x @ wx + h @ wh + b  # (B,4H)
    # Gotcha: Gate order must match how wx/wh/b are packed: input, forget, candidate, output.
    i, f, g, o = gates.chunk(4, dim=-1)  # four tensors, each (B,H)
    c_new = f.sigmoid_torch()*c + i.sigmoid_torch()*g.tanh()
    h_new = o.sigmoid_torch()*c_new.tanh()
    return h_new, c_new

# NumPy (backup)
# x:(B,T,D), lengths:(B,), wx:(D,H), wh:(H,H), b:(H,); 0<=lengths<=T.
# Return final hidden states:(B,H); trailing padding is ignored.
def rnn_final_numpy(x, lengths, wx, wh, b):
    batch, steps, _ = x.shape
    h = np.zeros((batch, wh.shape[0]), dtype=x.dtype)
    for t in range(steps):
        # Zero padding before matmul, including NaN padding.
        live = (t < lengths)[:, None]
        # Gotcha: Zero invalid inputs before matmul so NaN padding cannot contaminate computations.
        xt = np.where(live, x[:,t], 0.)
        candidate = np.tanh(xt @ wx + h @ wh + b)  # (B,H)
        # Gotcha: Inactive sequences retain their prior state; length-zero sequences stay zero.
        h = np.where(live, candidate, h)
    return h

# x:(B,D), h/c:(B,H), wx:(D,4H), wh:(H,4H), b:(4H,).
# Return h_new/c_new:(B,H); gate order is input, forget, candidate, output.
def lstm_cell_numpy(x, h, c, wx, wh, b):
    # wx:(D,4H), wh:(H,4H), b:(4H,)
    gates = x @ wx + h @ wh + b  # (B,4H)
    # Gotcha: Gate order must match how wx/wh/b are packed: input, forget, candidate, output.
    i, f, g, o = np.split(gates, 4, axis=-1)  # four tensors, each (B,H)
    c_new = sigmoid_numpy(f)*c + sigmoid_numpy(i)*np.tanh(g)
    h_new = sigmoid_numpy(o)*np.tanh(c_new)
    return h_new, c_new


# ============================================================================
# Q15. Formulate and code a rare road actor classifier
# ============================================================================

# PyTorch (primary)
# y:(N,) binary labels, prob:(N,); positive prediction means prob>=threshold.
# Return confusion counts, precision and recall; undefined rates are NaN.
def binary_metrics_torch(y, prob, threshold=0.5):
    positive = y == 1; pred = prob >= threshold
    tp = int((pred & positive).sum().item())  # Python scalar; sum reduces N
    fp = int((pred & ~positive).sum().item())  # Python scalar
    fn = int((~pred & positive).sum().item())  # Python scalar
    tn = int((~pred & ~positive).sum().item())  # Python scalar
    # Gotcha: No predicted positives means undefined precision; report NaN rather than a perfect score.
    precision = tp/(tp+fp) if tp+fp else float('nan')
    # Gotcha: No actual positives means undefined recall.
    recall = tp/(tp+fn) if tp+fn else float('nan')
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall)

# y/prob:(N,), thresholds: iterable of scalar cutoffs.
# Return (threshold, metrics) pairs; select thresholds on validation data only.
def threshold_sweep_torch(y, prob, thresholds):
    return [(float(t), binary_metrics_torch(y, prob, float(t)))
            for t in thresholds]
# Fit with Q5 using training-only standardization statistics.
# Choose threshold on validation; report final test metrics once.

# NumPy (backup)
# y:(N,) binary labels, prob:(N,); positive prediction means prob>=threshold.
# Return confusion counts, precision and recall; undefined rates are NaN.
def binary_metrics_numpy(y, prob, threshold=0.5):
    y = np.asarray(y) == 1
    pred = prob >= threshold
    tp = int(np.sum(pred & y))  # Python scalar; sum reduces N
    fp = int(np.sum(pred & ~y))  # Python scalar
    fn = int(np.sum(~pred & y))  # Python scalar
    tn = int(np.sum(~pred & ~y))  # Python scalar
    # Gotcha: No predicted positives means undefined precision; report NaN rather than a perfect score.
    precision = tp/(tp+fp) if tp+fp else float('nan')
    # Gotcha: No actual positives means undefined recall.
    recall = tp/(tp+fn) if tp+fn else float('nan')
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall)

# y/prob:(N,), thresholds: iterable of scalar cutoffs.
# Return (threshold, metrics) pairs; select thresholds on validation data only.
def threshold_sweep_numpy(y, prob, thresholds):
    return [(float(t), binary_metrics_numpy(y, prob, t))
            for t in thresholds]
# Undefined metrics are NaN rather than fabricated perfect scores.
# Pick a threshold using VALIDATION labels only.
# Evaluate the chosen threshold once on the held-out test set.
