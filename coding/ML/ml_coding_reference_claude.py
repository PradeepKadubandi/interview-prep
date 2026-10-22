"""
ML / Tensor / RL coding reference.

Combined from:
  - tesla_ml_tensor_rl_solutions.md  (Part 1: fundamentals, attention, RL, sampling, classic ML)
  - waymo_ml_tensor_solutions.md     (Part 2: AV-flavored NumPy / broadcasting problems)

Conventions: shapes are annotated inline as [B, T, D]. NumPy versions are the reference;
PyTorch versions (suffix _t) are what you'd write in production.
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # NumPy-only sections still work without torch
    torch = None
    F = None


# =============================================================================
# PART 1 — TESLA SET
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 Tensor / NumPy fundamentals
# -----------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Numerically stable softmax over `axis`. x: any shape."""
    # Subtract max so largest exponent is 0 (no overflow); exact since constant cancels.
    # keepdims=True on BOTH max and sum, so broadcasting works on any axis.
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_t(x, dim=-1):
    """Torch stable softmax. Built-in: F.softmax(x, dim=-1)."""
    # tensor.max(dim=...) returns (values, indices) namedtuple -> need .values
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def log_softmax(x, axis=-1):
    """Log-softmax via log-sum-exp. Never compute log(softmax(x)) (tiny -> -inf)."""
    m = np.max(x, axis=axis, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def cross_entropy(logits, labels):
    """Mean cross-entropy. logits [N,C], labels [N] int class ids -> scalar."""
    logp = log_softmax(logits, axis=-1)
    n = logits.shape[0]
    # Advanced indexing picks each row's true-class log-prob -> [N]
    return -np.mean(logp[np.arange(n), labels])


def log_softmax_t(x, dim=-1):
    """Torch log-softmax via log-sum-exp."""
    m = x.max(dim=dim, keepdim=True).values
    shifted = x - m
    return shifted - torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))


def cross_entropy_t(logits, labels):
    """Torch CE. logits [N,C] RAW logits, labels [N] LongTensor."""
    # F.cross_entropy takes RAW logits; passing softmax outputs = double-softmax bug.
    return F.cross_entropy(logits, labels)


def conv2d(x, kernel, stride=1, pad=0):
    """Single-channel 2D conv (cross-correlation, no kernel flip). x [H,W], kernel [kh,kw].
    Output size: (H - kh + 2*pad) // stride + 1."""
    if pad:
        x = np.pad(x, pad, mode="constant")
    H, W = x.shape
    kh, kw = kernel.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    # Speed-up follow-up: im2col (unfold patches -> one matmul); torch F.unfold does this.
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            out[i, j] = np.sum(x[r:r + kh, c:c + kw] * kernel)
    return out


def conv2d_t(x, kernel, stride=1, pad=0):
    """Torch conv for x [H,W], kernel [kh,kw] -> [out_h, out_w]."""
    # F.conv2d requires 4-D input [N, C_in, H, W] — the reshape is the gotcha.
    x4 = x.reshape(1, 1, *x.shape)
    # Weight layout is [C_out, C_in, kh, kw].
    k4 = kernel.reshape(1, 1, *kernel.shape)
    out = F.conv2d(x4, k4, stride=stride, padding=pad)
    return out[0, 0]


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """LayerNorm forward over last (feature) axis, per sample. x [..., D]."""
    mu = x.mean(axis=-1, keepdims=True)
    # Biased (population) variance — matches F.layer_norm; unbiased would diverge.
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    if gamma is not None:
        xhat = xhat * gamma
    if beta is not None:
        xhat = xhat + beta
    return xhat


def layer_norm_t(x, gamma=None, beta=None, eps=1e-5):
    """Torch LayerNorm. Module form: nn.LayerNorm(d_model)."""
    # normalized_shape = trailing dims to normalize over.
    return F.layer_norm(x, (x.shape[-1],), weight=gamma, bias=beta, eps=eps)


def iou(box_a, box_b):
    """IoU of two boxes, each [x1,y1,x2,y2] -> float."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    # Clamp at 0: non-overlapping boxes otherwise give spurious positive area.
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thresh):
    """Greedy NMS. boxes [N,4] xyxy, scores [N] -> list of kept indices (score order)."""
    idxs = np.argsort(scores)[::-1].tolist()        # highest score first
    keep = []
    while idxs:
        cur = idxs.pop(0)
        keep.append(cur)
        idxs = [i for i in idxs if iou(boxes[cur], boxes[i]) <= iou_thresh]
    return keep


def box_iou_t(boxes_a, boxes_b):
    """Vectorized IoU. [N,4], [M,4] xyxy -> [N,M]. Prod: torchvision.ops.box_iou."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    # [:, None] vs [None, :] broadcasts to [N,M,2] — narrate this.
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    # Clamp negatives to 0 (no overlap).
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


def nms_t(boxes, scores, iou_thresh):
    """Torch greedy NMS. Prod: torchvision.ops.nms(boxes, scores, iou_thresh)."""
    idxs = scores.argsort(descending=True).tolist()
    keep = []
    while idxs:
        cur = idxs.pop(0)
        keep.append(cur)
        if not idxs:
            break
        ious = box_iou_t(boxes[cur:cur + 1], boxes[idxs])[0]
        idxs = [idxs[i] for i in range(len(idxs)) if ious[i] <= iou_thresh]
    return keep


# -----------------------------------------------------------------------------
# 1.2 Attention / Transformer
# -----------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """softmax(QK^T / sqrt(d_k)) V. Q,K [T,d_k], V [T,d_v], mask [T,T] bool -> [T,d_v]."""
    d_k = Q.shape[-1]
    # /sqrt(d_k) keeps dot products from growing with dim and saturating softmax.
    scores = Q @ K.T / np.sqrt(d_k)                 # [T,T]
    if mask is not None:
        # Mask BEFORE softmax. -1e9 overflows in fp16; prefer -inf / finfo.min there.
        scores = np.where(mask, scores, -1e9)
    attn = softmax(scores, axis=-1)
    return attn @ V


def sdpa_t(Q, K, V, mask=None):
    """Torch SDPA, works batched [..., T, d]. Built-in: F.scaled_dot_product_attention."""
    d_k = Q.shape[-1]
    # transpose(-2,-1), not .T, so it generalizes to batched [B,H,T,d].
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)     # [..., T, T]
    if mask is not None:
        # mask True = keep; ~mask fills disallowed positions with -inf.
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ V


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads):
    """MHA. X [T,d_model], W* [d_model,d_model] -> [T,d_model]. d_head = d_model // n_heads."""
    T, d_model = X.shape
    d_head = d_model // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv

    def split(M):                                   # [T,d_model] -> [n_heads,T,d_head]
        # reshape THEN transpose: reshape splits last dim into (heads, d_head) per token.
        return M.reshape(T, n_heads, d_head).transpose(1, 0, 2)

    Qh, Kh, Vh = split(Q), split(K), split(V)
    outs = [scaled_dot_product_attention(Qh[h], Kh[h], Vh[h]) for h in range(n_heads)]
    return np.concatenate(outs, axis=-1) @ Wo


def mha_t(X, Wq, Wk, Wv, Wo, n_heads):
    """Torch MHA, heads folded into batch dim (no python loop). X [T,d_model].
    Module form: nn.MultiheadAttention(d_model, n_heads, batch_first=True)."""
    T, d_model = X.shape
    d_head = d_model // n_heads

    def split(M):                                   # [T,d_model] -> [h,T,d_head]
        return (X @ M).reshape(T, n_heads, d_head).transpose(0, 1)

    Q, K, V = split(Wq), split(Wk), split(Wv)
    out = sdpa_t(Q, K, V)                           # [h,T,d_head], one batched matmul
    # transpose back to [T,h,d_head] before merging heads, else heads interleave wrongly.
    concat = out.transpose(0, 1).reshape(T, d_model)
    return concat @ Wo


def positional_encoding(T, d):
    """Sinusoidal PE -> [T,d]. Even dims sin, odd dims cos; no learned params."""
    pos = np.arange(T)[:, None]
    i = np.arange(d)[None, :]
    # 2*(i//2): paired dims (2k, 2k+1) share the same frequency.
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def positional_encoding_t(T, d):
    """Torch sinusoidal PE -> [T,d]."""
    pos = torch.arange(T)[:, None]
    i = torch.arange(d)[None, :]
    angle = pos / torch.pow(torch.tensor(10000.0), (2 * (i // 2)) / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


# -----------------------------------------------------------------------------
# 1.3 RL
# -----------------------------------------------------------------------------

def importance_sampling_estimate(rewards, pi_probs, b_probs, normalize=True):
    """Off-policy value estimate, weights w = pi(a|s)/b(a|s). All inputs [N].
    Ordinary IS: unbiased, high variance. Self-normalized: slightly biased, lower variance."""
    # b must cover pi's support (no /0); in log-space use exp(log_pi - log_b).
    w = pi_probs / b_probs
    if normalize:
        return np.sum(w * rewards) / np.sum(w)
    return np.mean(w * rewards)


def importance_sampling_t(rewards, pi, b, normalize=True):
    """Torch IS estimate (ordinary or self-normalized)."""
    w = pi / b
    return (w * rewards).sum() / w.sum() if normalize else (w * rewards).mean()


def per_decision_is(rewards, pi_probs, b_probs, gamma):
    """Per-decision IS. All inputs [B,T]. Weight r_{t+1} by prefix product rho_{0:t}
    (not full-trajectory product) -> same expectation, lower variance. Returns scalar."""
    log_ratios = np.log(pi_probs) - np.log(b_probs)        # [B,T]
    # cumsum along T (axis=1) WITHIN a trajectory — swapping axes is the classic bug.
    cum_ratios = np.exp(np.cumsum(log_ratios, axis=1))     # [B,T] prefix product
    discounts = gamma ** np.arange(rewards.shape[1])       # [T]
    weighted = discounts[None, :] * cum_ratios * rewards   # [B,T]
    # Sum over time per trajectory, THEN average over batch.
    return weighted.sum(axis=1).mean()


def per_decision_is_t(rewards, pi_probs, b_probs, gamma):
    """Torch per-decision IS. All [B,T]. Ragged trajs: zero padded log-ratios and rewards."""
    log_ratios = torch.log(pi_probs) - torch.log(b_probs)
    cum_ratios = torch.exp(torch.cumsum(log_ratios, dim=1))
    T = rewards.shape[1]
    discounts = gamma ** torch.arange(T)
    weighted = discounts[None, :] * cum_ratios * rewards
    return weighted.sum(dim=1).mean()


class ReplayBuffer:
    """Uniform replay: O(1) circular append, uniform sample. Valid entries always [0, size)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = [None] * capacity
        self.pos = 0                                # write head, NOT a read boundary
        self.size = 0

    def add(self, transition):
        self.buf[self.pos] = transition             # overwrite oldest
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        # Sample over self.size, not capacity (buffer isn't full early on).
        idxs = rng.integers(0, self.size, size=batch_size)
        return [self.buf[i] for i in idxs]


class TorchReplayBuffer:
    """Pre-allocated struct-of-arrays replay (infra-grade). states [cap, state_dim]."""

    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states = torch.zeros(capacity, state_dim)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self.next_states = torch.zeros(capacity, state_dim)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        self.pos = 0
        self.size = 0

    def add(self, s, a, r, s2, done):
        i = self.pos
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = s2
        self.dones[i] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Storing full (s,a,r,s',done) per slot avoids frame-stack straddling-pos bugs.
        idx = torch.randint(0, self.size, (batch_size,))
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])


class SumTree:
    """Prioritized replay. Sample ∝ priority in O(log n), update in O(log n).
    tree: internal nodes [1..cap-1], leaves [cap..2cap-1]; root tree[1] = total.
    PER: priority = (|TD|+eps)^alpha, IS weights (1/(N*P(i)))^beta."""

    def __init__(self, capacity):
        self.capacity = capacity
        # Keep on CPU even in torch: descent is sequential/branchy, GPU won't help.
        self.tree = np.zeros(2 * capacity)
        self.data = [None] * capacity
        self.pos = 0
        self.size = 0

    def add(self, priority, item):
        idx = self.pos + self.capacity              # leaf index for this slot
        self.data[self.pos] = item
        self.update(idx, priority)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        # Propagate the DELTA (not the new value) up to the root — commonly gotten wrong.
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        idx //= 2
        while idx >= 1:
            self.tree[idx] += change
            idx //= 2

    def total(self):
        return self.tree[1]

    def sample(self, value):                        # value ~ U[0, total)
        idx = 1
        while idx < self.capacity:                  # descend to a leaf
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                # Going right: subtract the left subtree's mass.
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return data_idx, self.tree[idx], self.data[data_idx]


def discounted_returns(rewards, gamma):
    """Return-to-go G_t = r_t + gamma * G_{t+1}. rewards [T] -> [T]."""
    G = np.zeros(len(rewards))
    running = 0.0
    # Always accumulate BACKWARD in time.
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def gae(rewards, values, gamma, lam, last_value=0.0):
    """GAE advantages. rewards, values [T] -> [T]. lam=0: 1-step TD; lam=1: MC advantage.
    Sanity: lam=1 with zero values == discounted_returns."""
    T = len(rewards)
    adv = np.zeros(T)
    acc = 0.0
    # last_value bootstraps beyond the rollout; pass 0 if the episode terminated.
    next_v = last_value
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_v - values[t]    # TD residual
        acc = delta + gamma * lam * acc
        adv[t] = acc
        next_v = values[t]
    return adv


def discounted_returns_t(rewards, gamma):
    """Torch return-to-go. rewards [T] -> [T]."""
    G = torch.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def gae_t(rewards, values, gamma, lam, last_value=0.0):
    """Torch GAE. rewards, values [T] -> [T]. Recurrence is inherently sequential."""
    T = len(rewards)
    adv = torch.zeros(T)
    acc = 0.0
    next_v = last_value
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_v - values[t]
        acc = delta + gamma * lam * acc
        adv[t] = acc
        next_v = values[t]
    return adv


def epsilon_greedy(q_values, epsilon, rng):
    """q_values [A] -> action int. Explore w.p. epsilon."""
    if rng.random() < epsilon:
        return rng.integers(0, len(q_values))
    return int(np.argmax(q_values))


def q_learning_update(Q, s, a, r, s_next, alpha, gamma, done):
    """Tabular Q-learning, Q [S,A]. Off-policy: target uses max over next actions."""
    # On terminal, drop the bootstrap or value leaks across episode boundaries.
    target = r + (0 if done else gamma * np.max(Q[s_next]))
    Q[s, a] += alpha * (target - Q[s, a])
    return Q


def sample_categorical(probs, rng):
    """Inverse-CDF sampling. probs [K] (sums to 1) -> index."""
    cdf = np.cumsum(probs)
    return int(np.searchsorted(cdf, rng.random()))


def epsilon_greedy_t(q_values, epsilon):
    """Torch epsilon-greedy. q_values [A]."""
    if torch.rand(1).item() < epsilon:
        return torch.randint(len(q_values), (1,)).item()
    return int(q_values.argmax())


def q_update_t(Q, s, a, r, s_next, alpha, gamma, done):
    """Torch tabular Q-learning update, Q [S,A]."""
    target = r + (0 if done else gamma * Q[s_next].max())
    Q[s, a] += alpha * (target - Q[s, a])
    return Q


def sample_categorical_t(probs):
    """Torch categorical sample. Use Categorical(logits=...) to skip manual softmax."""
    return torch.distributions.Categorical(probs).sample().item()
    # or: torch.multinomial(probs, 1).item()


# -----------------------------------------------------------------------------
# 1.4 Sampling & streaming
# -----------------------------------------------------------------------------

def reservoir_sample(stream, k, rng):
    """Uniform k-sample from a stream of unknown length; one pass, O(k) memory.
    Proof: item i kept w.p. k/(i+1), survives later evictions -> k/n overall."""
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # Upper bound i+1 (inclusive of current index) — off-by-one breaks uniformity.
            j = rng.integers(0, i + 1)
            if j < k:
                reservoir[j] = item
    return reservoir
    # torch: rng.integers(0, i+1) -> torch.randint(0, i+1, (1,)).item()


def weighted_sample_without_replacement(items, weights, k, rng):
    """Efraimidis–Spirakis: key = u^(1/w), take top-k. weights [N] positive."""
    weights = np.asarray(weights, dtype=float)
    keys = rng.random(len(items)) ** (1.0 / weights)   # higher weight -> larger key
    top = np.argsort(keys)[::-1][:k]
    return [items[i] for i in top]


def weighted_sample_wor_t(weights, k):
    """Torch built-in; weights [N] need not be normalized -> sampled indices [k]."""
    return torch.multinomial(weights, k, replacement=False)


class Welford:
    """One-pass, stable online mean/variance (vs naive E[x^2]-E[x]^2 cancellation).
    Per-feature: make mean/M2 arrays of shape [num_features]; same recurrence."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        # Second factor uses the UPDATED mean — the crux of Welford.
        self.M2 += delta * (x - self.mean)

    def variance(self):
        # Population variance; use M2/(n-1) for sample variance.
        return self.M2 / self.n if self.n else 0.0


# -----------------------------------------------------------------------------
# 1.5 Classic ML from scratch
# -----------------------------------------------------------------------------

def kmeans(X, k, iters, rng):
    """Lloyd's k-means. X [N,D] -> centroids [k,D], labels [N]."""
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        # Broadcast to [N,k,D] (memory!); see pairwise_sq_dist for the matmul trick.
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)   # [N,k]
        labels = dists.argmin(axis=1)
        # Empty cluster: keep old centroid, else mean of empty -> nan.
        new = np.array([X[labels == c].mean(axis=0) if np.any(labels == c)
                        else centroids[c] for c in range(k)])
        if np.allclose(new, centroids):
            break
        centroids = new
    return centroids, labels


def kmeans_t(X, k, iters):
    """Torch k-means using torch.cdist. X [N,D]."""
    centroids = X[torch.randperm(len(X))[:k]]
    for _ in range(iters):
        d = torch.cdist(X, centroids)               # [N,k]
        labels = d.argmin(dim=1)
        new = torch.stack([X[labels == c].mean(0) if (labels == c).any() else centroids[c]
                           for c in range(k)])
        if torch.allclose(new, centroids):
            break
        centroids = new
    return centroids, labels


def logistic_regression_step(X, y, w, b, lr):
    """One GD step. X [N,D], y [N] in {0,1}, w [D], b scalar. BCE grad wrt logits = (p - y)."""
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    grad_w = X.T @ (p - y) / len(y)
    grad_b = np.mean(p - y)
    return w - lr * grad_w, b - lr * grad_b


def logreg_step_autograd(X, y, w, b, lr):
    """Torch autograd step. Real loop: opt.zero_grad(); loss.backward(); opt.step()."""
    w = w.clone().detach().requires_grad_(True)
    b = b.clone().detach().requires_grad_(True)
    # For stability prefer F.binary_cross_entropy_with_logits(X @ w + b, y).
    p = torch.sigmoid(X @ w + b)
    loss = F.binary_cross_entropy(p, y)
    loss.backward()
    # no_grad so the manual update isn't tracked by autograd.
    with torch.no_grad():
        return w - lr * w.grad, b - lr * b.grad


def precision_recall_f1(y_true, y_pred):
    """Binary P/R/F1. y_true, y_pred [N] in {0,1}."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    # Guard all three denominators.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def prf1_t(y_true, y_pred):
    """Torch binary P/R/F1."""
    # .float() before dividing, else integer tensors.
    tp = ((y_pred == 1) & (y_true == 1)).sum().float()
    fp = ((y_pred == 1) & (y_true == 0)).sum().float()
    fn = ((y_pred == 0) & (y_true == 1)).sum().float()
    p = tp / (tp + fp) if tp + fp > 0 else torch.tensor(0.)
    r = tp / (tp + fn) if tp + fn > 0 else torch.tensor(0.)
    f = 2 * p * r / (p + r) if p + r > 0 else torch.tensor(0.)
    return p.item(), r.item(), f.item()


# =============================================================================
# PART 2 — WAYMO SET (NumPy / broadcasting heavy)
# =============================================================================

# -----------------------------------------------------------------------------
# Tier A
# -----------------------------------------------------------------------------

# A1. Pairwise distances + kNN

def pairwise_sq_dist(A, B):
    """A [N,D], B [M,D] -> [N,M] squared distances via ||a||^2 + ||b||^2 - 2a.b (one matmul,
    avoids [N,M,D] intermediate)."""
    a2 = (A * A).sum(axis=1)[:, None]               # [N,1]
    b2 = (B * B).sum(axis=1)[None, :]               # [1,M]
    # Clamp at 0: cancellation can give -1e-15, then sqrt -> nan.
    return np.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0)


def knn_predict(X_train, y_train, X_test, k, num_classes):
    """kNN classify. X_train [N,D], y_train [N] int, X_test [M,D] -> [M] predicted class."""
    # Memory is [M,N]; chunk over test points for large N.
    d2 = pairwise_sq_dist(X_test, X_train)                  # [M,N]
    # argpartition is O(N) per row vs argsort O(N log N); result is UNORDERED.
    nn_idx = np.argpartition(d2, k - 1, axis=1)[:, :k]      # [M,k]
    votes = y_train[nn_idx]                                 # [M,k]
    counts = np.zeros((X_test.shape[0], num_classes), dtype=int)
    rows = np.arange(X_test.shape[0])[:, None]              # [M,1] broadcasts with [M,k]
    # np.add.at is unbuffered; counts[rows, votes] += 1 silently drops duplicate votes.
    np.add.at(counts, (rows, votes), 1)
    return counts.argmax(axis=1)                            # ties -> lowest class id


# A2. BatchNorm forward + backward

class BatchNorm:
    """BatchNorm, channel axis = 1. Works for [N,C] and [N,C,H,W].
    Reduce over every axis except C. Fold into conv for inference: W'=gW/s, b'=g(b-mu)/s+beta."""

    def __init__(self, C, eps=1e-5, momentum=0.1):
        self.gamma, self.beta = np.ones(C), np.zeros(C)
        self.running_mean, self.running_var = np.zeros(C), np.ones(C)
        self.eps = eps
        # PyTorch-style: weight on NEW value. Keras/TF is the opposite (0.99 on old).
        self.momentum = momentum

    def _shape(self, x):
        """Broadcast shape for per-channel params: (1, C, 1, 1, ...)."""
        return (1, -1) + (1,) * (x.ndim - 2)

    def forward(self, x, training=True):
        axes = (0,) + tuple(range(2, x.ndim))       # everything except C
        s = self._shape(x)
        # #1 miss: eval path must use running stats, not batch stats.
        if training:
            mu = x.mean(axis=axes)                  # [C]
            # Biased variance for normalization...
            var = x.var(axis=axes)
            m = x.size // x.shape[1]                # elements per channel
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            # ...but running_var stores the UNBIASED estimate (m/(m-1)), PyTorch convention.
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * m / (m - 1)
        else:
            mu, var = self.running_mean, self.running_var
        inv_std = 1.0 / np.sqrt(var + self.eps)     # [C]
        x_hat = (x - mu.reshape(s)) * inv_std.reshape(s)
        self.cache = (x_hat, inv_std, axes, s)
        return self.gamma.reshape(s) * x_hat + self.beta.reshape(s)

    def backward(self, dy):
        """Training-mode backward. dy same shape as x -> (dx, dgamma [C], dbeta [C])."""
        x_hat, inv_std, axes, s = self.cache
        m = dy.size // dy.shape[1]
        dgamma = (dy * x_hat).sum(axis=axes)
        dbeta = dy.sum(axis=axes)
        dx_hat = dy * self.gamma.reshape(s)
        # Three paths through x (direct, via mean, via var) summed into one formula:
        # dx = (1/m) * inv_std * (m*dx_hat - sum(dx_hat) - x_hat*sum(dx_hat*x_hat))
        dx = (inv_std.reshape(s) / m) * (
            m * dx_hat
            - dx_hat.sum(axis=axes).reshape(s)
            - x_hat * (dx_hat * x_hat).sum(axis=axes).reshape(s))
        return dx, dgamma, dbeta


# A3. Sample uniformly inside bounding boxes

def sample_in_aabb(lo, hi, n, rng):
    """Axis-aligned boxes. lo, hi [M,D] -> samples [M,n,D]."""
    u = rng.random((lo.shape[0], n, lo.shape[1]))
    return lo[:, None, :] + u * (hi - lo)[:, None, :]


def sample_in_rotated_bev_box(boxes, n, rng):
    """Rotated BEV boxes [M,5] = (cx,cy,l,w,yaw) -> [M,n,2].
    Sample in local frame then rotate+translate (rotation preserves uniformity, det=1)."""
    cx, cy, l, w, yaw = boxes.T                     # each [M]
    # Don't sample the AABB hull + reject: wasteful for thin boxes near 45°, random runtime.
    local = (rng.random((boxes.shape[0], n, 2)) - 0.5) * np.stack([l, w], -1)[:, None, :]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]   # [M,1]
    x = c * local[..., 0] - s * local[..., 1] + cx[:, None]
    y = s * local[..., 0] + c * local[..., 1] + cy[:, None]
    return np.stack([x, y], axis=-1)


def sample_across_boxes_by_area(boxes, n, rng):
    """n points uniform over the UNION of disjoint rotated boxes [M,5] -> ([n,2], box_idx [n])."""
    area = boxes[:, 2] * boxes[:, 3]
    # Pick box ∝ area (not uniformly) or small boxes get over-dense.
    # Overlapping boxes: area-weighting over-counts overlap -> reject over union or state bias.
    which = rng.choice(len(boxes), size=n, p=area / area.sum())
    local = (rng.random((n, 2)) - 0.5) * boxes[which, 2:4]
    c, s = np.cos(boxes[which, 4]), np.sin(boxes[which, 4])
    x = c * local[:, 0] - s * local[:, 1] + boxes[which, 0]
    y = s * local[:, 0] + c * local[:, 1] + boxes[which, 1]
    return np.stack([x, y], -1), which


# A4. Points in rotated 3D boxes

def points_in_boxes(points, boxes):
    """Lidar labeling. points [N,3], boxes [M,7]=(cx,cy,cz,l,w,h,yaw) -> [M,N] bool."""
    # Memory [M,N,3]; for 100k+ pts chunk over boxes or prefilter by circumscribed radius.
    d = points[None, :, :] - boxes[:, None, :3]     # [M,N,3]
    c = np.cos(boxes[:, 6])[:, None]                # [M,1]
    s = np.sin(boxes[:, 6])[:, None]
    # Rotate by -yaw (world -> box frame); sign error is easy, check with a point ahead.
    lx = c * d[..., 0] + s * d[..., 1]
    ly = -s * d[..., 0] + c * d[..., 1]
    # Inclusive <= on the boundary is a convention — state it. boxes[:, 3:4] keeps [M,1].
    return ((np.abs(lx) <= boxes[:, 3:4] / 2) &
            (np.abs(ly) <= boxes[:, 4:5] / 2) &
            (np.abs(d[..., 2]) <= boxes[:, 5:6] / 2))


# A5. Rigid transforms and agent-centric frames

def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi). Mean of angles: atan2(mean sin, mean cos)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def se3(R, t):
    """Build homogeneous transform. R [...,3,3], t [...,3] -> [...,4,4]. Name as T_dst_src."""
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def invert_se3(T):
    """Closed-form inverse: (R^T, -R^T t). Exact, O(1), keeps R orthonormal. T [...,4,4]."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    # swapaxes(-1,-2), not .T, so batched [...,3,3] works.
    Rt = np.swapaxes(R, -1, -2)
    return se3(Rt, -(Rt @ t[..., None])[..., 0])


def transform_points(T, P):
    """Apply SE(3). T [...,4,4], P [...,N,3] row vectors -> [...,N,3]."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    # Row-vector convention: p' = p R^T + t (NOT R @ P). State convention first.
    return P @ np.swapaxes(R, -1, -2) + t[..., None, :]


def to_agent_frame(traj, heading, t_ref):
    """Normalize each agent into its own frame at step t_ref.
    traj [A,T,2], heading [A,T] -> (local traj [A,T,2], relative heading [A,T])."""
    origin = traj[:, t_ref, :]                      # [A,2]
    yaw = heading[:, t_ref]                         # [A]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    d = traj - origin[:, None, :]                   # [A,T,2]
    # Rotate by -yaw so agent heading maps to +x.
    x = c * d[..., 0] + s * d[..., 1]
    y = -s * d[..., 0] + c * d[..., 1]
    # Always wrap heading differences.
    return np.stack([x, y], -1), wrap_angle(heading - yaw[:, None])


# A6. Trajectory tokenization + next-step prediction (MotionLM-style)

def quantize(d, dmax, nb):
    """Continuous delta in [-dmax, dmax] -> bin index in [0, nb-1], per axis."""
    # Outliers clip silently — pick dmax from data percentiles and count clipped values.
    return np.clip(np.round((d + dmax) / (2 * dmax) * (nb - 1)), 0, nb - 1).astype(int)


def dequantize(q, dmax, nb):
    """Bin index -> bin-center delta."""
    return q / (nb - 1) * 2 * dmax - dmax


def tokenize_open_loop(traj, dmax, nb):
    """traj [A,T,3] -> tokens [A,T-1]. Tokenize per-step DELTA (not absolute position).
    Open loop: rounding errors accumulate on reconstruction (random-walk drift)."""
    q = quantize(np.diff(traj, axis=1), dmax, nb)   # [A,T-1,3]
    # Joint vocab nb^3 — factorize per axis or drop z for ground vehicles.
    return np.ravel_multi_index((q[..., 0], q[..., 1], q[..., 2]), (nb, nb, nb))


def tokenize_closed_loop(traj, dmax, nb):
    """traj [A,T,3] -> tokens [A,T-1]. Quantize relative to RECONSTRUCTED position so error
    stays <= half a bin forever (error feedback, like delta modulation)."""
    A, T, _ = traj.shape
    tokens = np.zeros((A, T - 1), dtype=int)
    # .copy(): otherwise recon aliases traj and in-place updates would corrupt it.
    recon = traj[:, 0].copy()                       # [A,3]
    for t in range(1, T):                           # loop over time, vectorized over agents
        # Delta vs recon (not vs traj[:, t-1]) — this is the whole closed-loop trick.
        q = quantize(traj[:, t] - recon, dmax, nb)  # [A,3]
        tokens[:, t - 1] = np.ravel_multi_index(tuple(q.T), (nb, nb, nb))
        recon = recon + dequantize(q, dmax, nb)
    return tokens


def detokenize(tokens, start, dmax, nb):
    """tokens [A,T-1], start [A,3] -> reconstructed traj [A,T,3]."""
    q = np.stack(np.unravel_index(tokens, (nb, nb, nb)), axis=-1)   # [A,T-1,3]
    steps = dequantize(q, dmax, nb)
    return np.concatenate([start[:, None], start[:, None] + np.cumsum(steps, axis=1)], axis=1)


def fit_bigram(tokens, vocab, alpha=1.0):
    """Baseline next-token model P(next | current), Laplace smoothed. tokens [A,T-1] -> [V,V].
    Always compare against a trivial 'repeat last token' baseline."""
    counts = np.full((vocab, vocab), alpha)
    # add.at, not fancy +=, so repeated (cur, next) pairs all count.
    np.add.at(counts, (tokens[:, :-1].ravel(), tokens[:, 1:].ravel()), 1)
    return counts / counts.sum(axis=1, keepdims=True)


# A7. minADE / minFDE / miss rate with validity masks

def min_ade_fde(pred, gt, valid, miss_thresh=2.0):
    """Multi-modal metrics. pred [B,K,T,2], gt [B,T,2], valid [B,T] bool
    -> (mean minADE, mean minFDE, miss rate). minADE and minFDE may pick different modes."""
    B, K, T, _ = pred.shape
    err = np.linalg.norm(pred - gt[:, None], axis=-1)            # [B,K,T]
    v = valid[:, None, :].astype(float)                          # [B,1,T]
    n_valid = valid.sum(axis=1)                                  # [B]
    # Average only over valid steps; max(...,1) guards agents with no valid steps.
    ade = (err * v).sum(-1) / np.maximum(n_valid, 1)[:, None]    # [B,K]
    # FDE uses LAST VALID step, not T-1: argmax on reversed mask finds it.
    last = T - 1 - np.argmax(valid[:, ::-1], axis=1)             # [B]
    # Index arrays [B,1], [1,K], [B,1] broadcast to [B,K].
    fde = err[np.arange(B)[:, None], np.arange(K)[None, :], last[:, None]]
    # Drop agents with zero valid future steps.
    has = n_valid > 0
    min_ade, min_fde = ade.min(1)[has], fde.min(1)[has]
    return min_ade.mean(), min_fde.mean(), (min_fde > miss_thresh).mean()


# A8. Masked ops on padded sequences

def lengths_to_mask(lengths, T):
    """lengths [B] -> mask [B,T] bool (True = valid)."""
    return np.arange(T)[None, :] < lengths[:, None]


def masked_mean(x, mask):
    """x [B,T,D], mask [B,T] -> [B,D]. Empty rows -> 0."""
    m = mask[..., None].astype(x.dtype)
    # max(count, 1) avoids 0/0 on fully-masked rows.
    return (x * m).sum(1) / np.maximum(m.sum(1), 1)


def masked_max(x, mask):
    """PointNet/VectorNet polyline max-pool. x [B,T,D], mask [B,T] -> [B,D]."""
    out = np.where(mask[..., None], x, -np.inf).max(axis=1)      # [B,D]
    # Empty polyline would give -inf; replace with 0.
    return np.where(mask.any(1)[:, None], out, 0.0)


def masked_softmax(logits, mask, axis=-1):
    """Softmax over valid entries only. logits, mask same shape. All-masked rows -> zeros."""
    # Use boolean where with -inf; -1e9 overflows in fp16 (max ~65504) — else finfo.min.
    z = np.where(mask, logits, -np.inf)
    zmax = z.max(axis=axis, keepdims=True)
    # All-masked row: zmax=-inf and exp(-inf - -inf)=nan — replace with 0.
    zmax = np.where(np.isfinite(zmax), zmax, 0.0)
    e = np.exp(z - zmax) * mask
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-30)


# A9. MLP backprop from scratch + gradient check

def mlp_init(D, H, C, rng):
    """2-layer MLP params (He init). D in, H hidden, C classes."""
    return {"W1": rng.normal(0, np.sqrt(2 / D), (D, H)), "b1": np.zeros(H),
            "W2": rng.normal(0, np.sqrt(2 / H), (H, C)), "b2": np.zeros(C)}


def mlp_loss_and_grads(p, X, y):
    """Linear-ReLU-Linear-softmax-CE. X [N,D], y [N] int -> (loss, grads dict)."""
    N = X.shape[0]
    z1 = X @ p["W1"] + p["b1"]                      # [N,H]
    h = np.maximum(z1, 0)
    logits = h @ p["W2"] + p["b2"]                  # [N,C]
    sh = logits - logits.max(1, keepdims=True)
    logp = sh - np.log(np.exp(sh).sum(1, keepdims=True))
    loss = -logp[np.arange(N), y].mean()
    # dlogits = softmax - onehot ...
    dlogits = np.exp(logp)
    dlogits[np.arange(N), y] -= 1
    # ... divided by N exactly ONCE (double-dividing is a classic bug grad-check catches).
    dlogits /= N                                    # [N,C]
    g = {"W2": h.T @ dlogits, "b2": dlogits.sum(0)}
    dh = dlogits @ p["W2"].T
    # Mask on z1 > 0 (pre-activation), not h > 0 — differs for leaky variants.
    dz1 = dh * (z1 > 0)
    g["W1"] = X.T @ dz1
    g["b1"] = dz1.sum(0)
    return loss, g


def grad_check(f, p, eps=1e-5):
    """Central-difference check; f(p) -> (loss, grads). Returns max relative error.
    Use float64. <1e-6 good, >1e-3 bug. ReLU kinks can cause spurious failures."""
    worst = 0.0
    _, g = f(p)
    for k, v in p.items():
        num = np.zeros_like(v)
        it = np.nditer(v, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = v[i]
            # Perturb in place (v is a view into p), then RESTORE.
            v[i] = old + eps
            lp, _ = f(p)
            v[i] = old - eps
            lm, _ = f(p)
            v[i] = old
            num[i] = (lp - lm) / (2 * eps)
        rel = np.abs(num - g[k]) / np.maximum(np.abs(num) + np.abs(g[k]), 1e-12)
        worst = max(worst, rel.max())
    return worst


# -----------------------------------------------------------------------------
# Tier B
# -----------------------------------------------------------------------------

# B1. Voxelization and BEV scatter (PointPillars / VoxelNet front end)

def voxelize(points, feats, pc_range, voxel_size):
    """points [N,3], feats [N,F], pc_range (x0,y0,z0,x1,y1,z1), voxel_size [3]
    -> coords [V,3], mean feats [V,F], counts [V]."""
    lo, hi = np.asarray(pc_range[:3]), np.asarray(pc_range[3:])
    vs = np.asarray(voxel_size)
    # Half-open (< hi) so floor never yields index == grid size.
    keep = np.all((points >= lo) & (points < hi), axis=1)
    pts, f = points[keep], feats[keep]
    grid = np.round((hi - lo) / vs).astype(int)                      # [3]
    ijk = np.floor((pts - lo) / vs).astype(int)                      # [N',3]
    # Float-rounding guard: (x-lo)/vs can land exactly on grid edge.
    ijk = np.minimum(ijk, grid - 1)
    # Linearize 3D idx so np.unique can group points per voxel.
    lin = np.ravel_multi_index(ijk.T, grid)                          # [N']
    uniq, inv, cnt = np.unique(lin, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), f.shape[1]))
    # Scatter-add; sums[inv] += f would count duplicates once (buffered).
    np.add.at(sums, inv, f)
    coords = np.stack(np.unravel_index(uniq, grid), axis=1)          # [V,3]
    return coords, sums / cnt[:, None], cnt


def bev_max_height(points, pc_range, cell):
    """Dense BEV max-z map. points [N,3] -> [X,Y] (empty cells = -inf)."""
    lo, hi = np.asarray(pc_range[:2]), np.asarray(pc_range[3:5])
    keep = np.all((points[:, :2] >= lo) & (points[:, :2] < hi), axis=1)
    p = points[keep]
    gx, gy = np.round((hi - lo) / cell).astype(int)
    ij = np.minimum(np.floor((p[:, :2] - lo) / cell).astype(int), [gx - 1, gy - 1])
    bev = np.full((gx, gy), -np.inf)
    # Scatter-max; fancy-index assignment would keep an arbitrary duplicate, not the max.
    np.maximum.at(bev, (ij[:, 0], ij[:, 1]), p[:, 2])
    return bev


# B2. Detection Average Precision

def box_iou(a, b):
    """Vectorized NumPy IoU. a [P,4], b [G,4] xyxy -> [P,G]."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=-1)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def match_detections(pred, scores, gt, iou_thr):
    """One image. pred [P,4], scores [P], gt [G,4] -> (tp flags [P] in score order, sorted scores)."""
    order = np.argsort(-scores, kind="stable")
    tp = np.zeros(len(pred))
    if len(gt):
        iou_m = box_iou(pred[order], gt)
        taken = np.zeros(len(gt), dtype=bool)
        for i in range(len(order)):                 # greedy: highest score claims best free GT
            # Each GT claimable once; duplicates become FPs (why NMS matters).
            cand = np.where(taken, -1.0, iou_m[i])
            j = cand.argmax()
            if cand[j] >= iou_thr:
                tp[i] = 1
                taken[j] = True
    return tp, scores[order]


def average_precision(images, iou_thr=0.5):
    """images: list of (pred [P,4], scores [P], gt [G,4]) -> AP (all-point interpolation).
    Match PER IMAGE, rank GLOBALLY. COCO: 101-pt, IoU 0.5:0.95; Waymo adds APH."""
    tps, scs, n_gt = [], [], 0
    for pred, scores, gt in images:
        tp, s = match_detections(pred, scores, gt, iou_thr)
        tps.append(tp)
        scs.append(s)
        n_gt += len(gt)
    tp = np.concatenate(tps)
    s = np.concatenate(scs)
    # Re-rank across images by score before cumulating.
    order = np.argsort(-s, kind="stable")
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    # Precision envelope: running max from the right makes it monotone non-increasing.
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


# B3. Bilinear sampling

def bilinear_sample(feat, xy):
    """Sample feature map at continuous pixel coords. feat [H,W,C], xy [N,2] -> [N,C].
    Zero padding outside the map (cf. grid_sample align_corners / padding_mode)."""
    H, W, _ = feat.shape
    # x is COLUMN, y is ROW — say it out loud.
    x, y = xy[:, 0], xy[:, 1]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = (x - x0)[:, None], (y - y0)[:, None]

    def at(yy, xx):
        ok = ((xx >= 0) & (xx < W) & (yy >= 0) & (yy < H))[:, None]
        # Clip for safe indexing, then zero out-of-bounds via `ok`.
        return feat[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)] * ok

    return (at(y0, x0) * (1 - wx) * (1 - wy) + at(y0, x1) * wx * (1 - wy) +
            at(y1, x0) * (1 - wx) * wy + at(y1, x1) * wx * wy)


# B4. Top-k modes + gather

def topk_modes(traj, conf, k):
    """traj [B,M,T,2], conf [B,M] -> (top-k traj [B,k,T,2], conf [B,k]) sorted desc.
    NumPy analogue of torch.topk + torch.gather. O(M + k log k)."""
    # argpartition result is unordered — sort only the k survivors afterwards.
    part = np.argpartition(-conf, k - 1, axis=1)[:, :k]  # [B,k]
    part_conf = np.take_along_axis(conf, part, axis=1)
    order = np.argsort(-part_conf, axis=1)
    idx = np.take_along_axis(part, order, axis=1)        # [B,k]
    # Index shape [B,k,1,1] broadcasts over T and xy in take_along_axis.
    return (np.take_along_axis(traj, idx[:, :, None, None], axis=1),
            np.take_along_axis(conf, idx, axis=1))


# B5. Rotated BEV IoU (stretch)

def box_corners(b):
    """b = (cx,cy,l,w,yaw) -> corners [4,2], CCW."""
    cx, cy, l, w, yaw = b
    local = np.array([[l, w], [-l, w], [-l, -w], [l, -w]]) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    # Row-vector rotation: local @ R^T, with R^T = [[c, s], [-s, c]].
    return local @ np.array([[c, s], [-s, c]]) + [cx, cy]


def poly_area(P):
    """Shoelace area of polygon P [K,2]."""
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_poly(subject, clipper):
    """Sutherland–Hodgman: clip subject [K,2] by convex CCW clipper [L,2] -> polygon."""
    def inside(p, a, b):
        # Cross product >= 0 => left of edge a->b (inside for CCW clipper).
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p, q, a, b):
        d1, d2 = q - p, b - a
        t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / (d1[0] * d2[1] - d1[1] * d2[0])
        return p + t * d1

    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        # No overlap left — stop early.
        if not inp:
            break
        for j in range(len(inp)):
            # inp[j-1] with j=0 wraps to last vertex (closing edge).
            p, q = inp[j - 1], inp[j]
            if inside(q, a, b):
                if not inside(p, a, b):
                    out.append(intersect(p, q, a, b))
                out.append(q)
            elif inside(p, a, b):
                out.append(intersect(p, q, a, b))
    return np.array(out)


def rotated_iou(b1, b2):
    """Rotated BEV IoU of two (cx,cy,l,w,yaw) boxes. Production uses a CUDA kernel."""
    inter_poly = clip_poly(box_corners(b1), box_corners(b2))
    # Degenerate (<3 vertices) intersection has zero area.
    inter = poly_area(inter_poly) if len(inter_poly) >= 3 else 0.0
    return inter / (b1[2] * b1[3] + b2[2] * b2[3] - inter)


# B6. Constant-velocity Kalman Filter, batched over tracks

def cv_matrices(dt, q, r):
    """CV model, state [x,y,vx,vy], measure [x,y]. Returns F [4,4], H [2,4], Q [4,4], R [2,2]."""
    F_ = np.eye(4)
    F_[0, 2] = F_[1, 3] = dt
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1
    # Process noise from white acceleration: Q = q * G G^T.
    G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
    return F_, H, q * G @ G.T, r * np.eye(2)


def kf_predict(x, P, F_, Q):
    """x [N,4], P [N,4,4] -> predicted (x, P). [4,4] @ [N,4,4] broadcasts."""
    # Row-vector states: x @ F^T (not F @ x).
    return x @ F_.T, F_ @ P @ F_.T + Q


def kf_update(x, P, z, H, R):
    """x [N,4], P [N,4,4], z [N,2] -> updated (x, P)."""
    y = z - x @ H.T                                 # innovation [N,2]
    S = H @ P @ H.T + R                             # [N,2,2]
    K = P @ H.T @ np.linalg.inv(S)                  # [N,4,2]
    # Batched mat-vec via einsum.
    x = x + np.einsum("nij,nj->ni", K, y)
    I_KH = np.eye(4) - K @ H
    # Joseph form keeps P symmetric PSD under float error (vs (I-KH)P).
    P = I_KH @ P @ np.swapaxes(I_KH, -1, -2) + K @ R @ np.swapaxes(K, -1, -2)
    return x, P


# Quick hits — stable BCE and focal loss

def bce_with_logits(z, y):
    """Elementwise stable BCE from logits z, labels y in {0,1}."""
    # max(z,0) - z*y + log1p(exp(-|z|)) never overflows (vs log(sigmoid(z))).
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def focal_loss(z, y, alpha=0.25, gamma=2.0):
    """Focal loss (RetinaNet/CenterPoint): down-weights easy negatives. gamma=0, alpha=.5 -> BCE/2."""
    p = 1 / (1 + np.exp(-z))
    p_t = y * p + (1 - y) * (1 - p)
    a_t = y * alpha + (1 - y) * (1 - alpha)
    return a_t * (1 - p_t) ** gamma * bce_with_logits(z, y)
