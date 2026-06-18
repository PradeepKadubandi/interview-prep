# Tesla AP ML Infra — ML / Tensor / RL Solution Reference
 
Each problem shows a **NumPy reference** (clear, dependency-light — good for explaining the mechanism) and a **PyTorch** version (what you'd actually write on an ML-infra team; verified against the numpy output). All code is run-verified: deterministic ops checked against scipy/numpy, samplers against statistics.
 
This is your differentiator set. The *why* (numerical stability, estimator variance, sim-to-real) is the senior signal — say it out loud. Annotate tensor shapes inline (`# [B, T, d]`). Where a built-in exists (`F.scaled_dot_product_attention`, `nn.MultiheadAttention`, `torch.multinomial`, `torchvision.ops.nms`), know it *and* be able to implement it from scratch — interviewers ask for the latter, production uses the former.
 
---
 
## Tensor / NumPy fundamentals
 
### Numerically stable softmax
Subtract the per-row max before `exp` so the largest exponent is 0 (no overflow). Mathematically identical (the constant cancels).
 
**NumPy (reference)**
```python
import numpy as np
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # stability shift
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)
```
**PyTorch**
```python
import torch
import torch.nn.functional as F
 
def softmax_t(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values   # .values: max returns (vals, idxs)
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)
 
# built-in: F.softmax(x, dim=-1)
```
Gotchas: `keepdims/keepdim=True` on both max and sum so broadcasting works on any axis. In torch, `tensor.max(dim=...)` returns a `(values, indices)` namedtuple — use `.values`. "Why subtract the max?" → overflow prevention, and it's exact.
 
### log-softmax & cross-entropy
Never `log(softmax(x))` (log of a tiny number → -inf). Use log-sum-exp.
 
**NumPy (reference)**
```python
def log_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
 
def cross_entropy(logits, labels):            # logits [N,C], labels [N] int
    logp = log_softmax(logits, axis=-1)
    n = logits.shape[0]
    return -np.mean(logp[np.arange(n), labels])
```
**PyTorch**
```python
def log_softmax_t(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    shifted = x - m
    return shifted - torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
 
def cross_entropy_t(logits, labels):          # labels: LongTensor of class idxs
    return F.cross_entropy(logits, labels)    # fuses log_softmax + NLL, numerically stable
```
Gotchas: `F.cross_entropy` takes **raw logits**, not probabilities — passing softmax outputs is a classic bug (double softmax). The numpy advanced-index `logp[np.arange(n), labels]` picks each row's true-class log-prob; torch's `F.nll_loss` does this internally.
 
### 2D convolution from scratch
ML "convolution" is cross-correlation (no kernel flip).
 
**NumPy (reference)**
```python
def conv2d(x, kernel, stride=1, pad=0):        # x [H,W], kernel [kh,kw]
    if pad:
        x = np.pad(x, pad, mode="constant")
    H, W = x.shape
    kh, kw = kernel.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            out[i, j] = np.sum(x[r:r+kh, c:c+kw] * kernel)
    return out
```
**PyTorch**
```python
def conv2d_t(x, kernel, stride=1, pad=0):      # x [H,W], kernel [kh,kw]
    x4 = x.reshape(1, 1, *x.shape)             # F.conv2d needs [N, C_in, H, W]
    k4 = kernel.reshape(1, 1, *kernel.shape)   # weight [C_out, C_in, kh, kw]
    out = F.conv2d(x4, k4, stride=stride, padding=pad)   # cross-correlation
    return out[0, 0]
```
Gotchas: derive output size `(H − kh + 2·pad)//stride + 1`. `F.conv2d` requires 4-D `[N,C,H,W]` input and `[C_out,C_in,kh,kw]` weights — the reshape is the gotcha. If asked to speed up the numpy loop: `im2col` (unfold patches → one matmul) — `torch.nn.functional.unfold` is exactly this, and it's how convs hit hardware efficiency (ties to your inference-optimization background).
 
### LayerNorm forward
Normalize over the feature axis per-sample (vs BatchNorm over the batch).
 
**NumPy (reference)**
```python
def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    if gamma is not None: xhat = xhat * gamma
    if beta is not None:  xhat = xhat + beta
    return xhat
```
**PyTorch**
```python
def layer_norm_t(x, gamma=None, beta=None, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), weight=gamma, bias=beta, eps=eps)
# module form: nn.LayerNorm(d_model)
```
Gotchas: `F.layer_norm` uses **biased** (population) variance — matches the numpy `.var()` default; if you hand-roll with `unbiased=True` they'll diverge. `normalized_shape` is the trailing dims to normalize over. LN vs BN: LN is batch-size-independent (why transformers use it).
 
### IoU + Non-Max Suppression (NMS)
AV/perception bread-and-butter.
 
**NumPy (reference)**
```python
def iou(box_a, box_b):                          # box = [x1,y1,x2,y2]
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)   # clamp: no negative overlap
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0
 
def nms(boxes, scores, iou_thresh):             # -> indices to keep
    idxs = np.argsort(scores)[::-1].tolist()    # highest score first
    keep = []
    while idxs:
        cur = idxs.pop(0)
        keep.append(cur)
        idxs = [i for i in idxs if iou(boxes[cur], boxes[i]) <= iou_thresh]
    return keep
```
**PyTorch** (vectorized, batched IoU)
```python
def box_iou_t(boxes_a, boxes_b):                # [N,4],[M,4] -> [N,M]
    area_a = (boxes_a[:,2]-boxes_a[:,0]) * (boxes_a[:,3]-boxes_a[:,1])
    area_b = (boxes_b[:,2]-boxes_b[:,0]) * (boxes_b[:,3]-boxes_b[:,1])
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])    # [N,M,2]
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)                 # clamp negatives to 0
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)
 
def nms_t(boxes, scores, iou_thresh):
    idxs = scores.argsort(descending=True).tolist()
    keep = []
    while idxs:
        cur = idxs.pop(0); keep.append(cur)
        if not idxs: break
        ious = box_iou_t(boxes[cur:cur+1], boxes[idxs])[0]
        idxs = [idxs[i] for i in range(len(idxs)) if ious[i] <= iou_thresh]
    return keep
 
# production: torchvision.ops.nms(boxes, scores, iou_thresh) and ops.box_iou(a, b)
```
Gotchas: the `clamp(min=0)` / `max(0, …)` on width-height is essential — non-overlapping boxes otherwise give spurious area. The `[:, None]` vs `[None, :]` broadcasting to `[N,M,2]` is the torch idiom worth narrating. Mention `torchvision.ops.nms` exists but be ready to hand-roll.
 
---
 
## Attention / Transformer
 
### Scaled dot-product attention
`Attention(Q,K,V) = softmax(QKᵀ/√d_k) · V`.
 
**NumPy (reference)**
```python
def scaled_dot_product_attention(Q, K, V, mask=None):   # Q,K:[T,d_k] V:[T,d_v]
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)             # [T,T]
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    attn = softmax(scores, axis=-1)
    return attn @ V                             # [T,d_v]
```
**PyTorch**
```python
def sdpa_t(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)      # [..., T, T]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ V
 
# built-in (fused, flash-attention kernels under the hood):
# F.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=True)
```
Gotchas: the `/√d_k` keeps dot products from growing with dimension and saturating softmax (vanishing gradients) — be ready to explain *why* the scale is there. Use `transpose(-2,-1)` not `.T` so it works on batched `[B,H,T,d]`. `masked_fill(~mask, -inf)` for causal masking *before* softmax. The built-in `F.scaled_dot_product_attention` dispatches to FlashAttention — good to name on an infra team.
 
### Multi-head attention
Project to Q/K/V, split into heads, attend per head, concat, output-project.
 
**NumPy (reference)**
```python
def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads):   # X:[T,d_model]
    T, d_model = X.shape
    d_head = d_model // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    def split(M):                               # [T,d_model] -> [n_heads,T,d_head]
        return M.reshape(T, n_heads, d_head).transpose(1, 0, 2)
    Qh, Kh, Vh = split(Q), split(K), split(V)
    outs = [scaled_dot_product_attention(Qh[h], Kh[h], Vh[h]) for h in range(n_heads)]
    return np.concatenate(outs, axis=-1) @ Wo
```
**PyTorch** (no per-head python loop — fold heads into a batch dim)
```python
def mha_t(X, Wq, Wk, Wv, Wo, n_heads):          # X:[T,d_model]
    T, d_model = X.shape
    d_head = d_model // n_heads
    def split(M):                               # [T,d_model] -> [h,T,d_head]
        return (X @ M).reshape(T, n_heads, d_head).transpose(0, 1)
    Q, K, V = split(Wq), split(Wk), split(Wv)
    out = sdpa_t(Q, K, V)                        # batched over head dim -> [h,T,d_head]
    concat = out.transpose(0, 1).reshape(T, d_model)
    return concat @ Wo
 
# module form: nn.MultiheadAttention(d_model, n_heads, batch_first=True)
```
Gotchas: the reshape→transpose to form heads is exactly what they probe — narrate dims. The torch version attends to all heads in one batched matmul (no python loop) — that parallelism is the point. Tie-in: head count vs accelerator core count (your 56-heads-on-64-cores co-design) is a strong hardware-aware aside.
 
### Sinusoidal positional encoding
**NumPy (reference)**
```python
def positional_encoding(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(angle[:, 0::2])        # even dims sin
    pe[:, 1::2] = np.cos(angle[:, 1::2])        # odd dims cos
    return pe
```
**PyTorch**
```python
def positional_encoding_t(T, d):
    pos = torch.arange(T)[:, None]
    i = torch.arange(d)[None, :]
    angle = pos / torch.pow(torch.tensor(10000.0), (2 * (i // 2)) / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe
```
Gotchas: even indices sin, odd cos; bounded in [-1,1]; gives relative-position info with no learned params.
 
---
 
## RL (highest priority for this role)
 
### Importance sampling for off-policy estimation
`w = π(a|s) / b(a|s)`.
 
**NumPy (reference)**
```python
def importance_sampling_estimate(rewards, pi_probs, b_probs, normalize=True):
    w = pi_probs / b_probs
    if normalize:                               # self-normalized IS
        return np.sum(w * rewards) / np.sum(w)
    return np.mean(w * rewards)                 # ordinary IS
```
**PyTorch**
```python
def importance_sampling_t(rewards, pi, b, normalize=True):
    w = pi / b
    return (w * rewards).sum() / w.sum() if normalize else (w * rewards).mean()
```
Gotchas (articulate this one — it's RL-core and appeared in a documented Tesla ML screen):
- **Ordinary IS**: unbiased, high variance (weights explode when `b` is small).
- **Self-normalized IS** (÷ Σw): slightly biased, much lower variance, usually preferred.
- Variance levers: weight clipping, per-decision IS, and `b` must cover the support of `π` (no ÷0). In log-space, compute `exp(logπ − logb)` for stability.
### Per-decision IS (trajectory form)
Over trajectories `[B, T]`, weight each reward `r_{t+1}` only by the prefix product `ρ_{0:t}` (ratios up to that step), not the full-trajectory product — same expectation, lower variance. The prefix product is a `cumsum` of log-ratios along the time axis.
 
**NumPy (reference)**
```python
def per_decision_is(rewards, pi_probs, b_probs, gamma):   # all [B, T]
    log_ratios = np.log(pi_probs) - np.log(b_probs)        # [B, T]  per-step log ρ_t
    cum_ratios = np.exp(np.cumsum(log_ratios, axis=1))     # [B, T]  prefix product ρ_{0:t} along T
    discounts  = gamma ** np.arange(rewards.shape[1])      # [T]
    weighted   = discounts[None, :] * cum_ratios * rewards # [B, T]  γ^t · ρ_{0:t} · r_{t+1}
    return weighted.sum(axis=1).mean()                     # scalar  (per-traj return, then avg over B)
```
**PyTorch**
```python
def per_decision_is_t(rewards, pi_probs, b_probs, gamma):  # all [B, T]
    log_ratios = torch.log(pi_probs) - torch.log(b_probs)
    cum_ratios = torch.exp(torch.cumsum(log_ratios, dim=1))    # [B, T]  ρ_{0:t}
    T = rewards.shape[1]
    discounts = gamma ** torch.arange(T)
    weighted = discounts[None, :] * cum_ratios * rewards        # [B, T]
    return weighted.sum(dim=1).mean()
```
Gotchas: accumulate along `T` (time) *within* a trajectory, then average over `B` — swapping the axes is the classic bug. Self-normalized variant: normalize per timestep across the batch, `(disc * (cum*r).sum(0) / cum.sum(0)).sum()`. For ragged trajectories, mask padded steps (set their log-ratio to 0, zero their rewards). Equivalent backward recursion: `G = ρ_t·(r_t + γG)`.
 
### Replay buffer (uniform)
O(1) circular append + uniform sampling.
 
**NumPy (reference)**
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = [None] * capacity
        self.pos = 0
        self.size = 0
    def add(self, transition):
        self.buf[self.pos] = transition         # overwrite oldest
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size, rng):
        idxs = rng.integers(0, self.size, size=batch_size)
        return [self.buf[i] for i in idxs]
```
**PyTorch** (pre-allocated tensor storage — what you'd use in practice)
```python
class TorchReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states     = torch.zeros(capacity, state_dim)
        self.actions    = torch.zeros(capacity, dtype=torch.long)
        self.rewards    = torch.zeros(capacity)
        self.next_states = torch.zeros(capacity, state_dim)
        self.dones      = torch.zeros(capacity, dtype=torch.bool)
        self.pos = 0; self.size = 0
    def add(self, s, a, r, s2, done):
        i = self.pos
        self.states[i] = s; self.actions[i] = a; self.rewards[i] = r
        self.next_states[i] = s2; self.dones[i] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])
```
Gotchas: sample over `self.size`, not `capacity` (buffer isn't full early). Pre-allocating contiguous tensors (struct-of-arrays) and indexing with a `LongTensor` is the infra-grade pattern — no python list, batched gather, pinned-memory/GPU-friendly.
 
Indexing invariant: valid entries are always exactly `[0, size)`. `pos` is the *write head* (where the next overwrite lands), **not** a boundary on what's readable — the buffer is never both partially-full *and* wrapped, so no gap ever forms inside `[0, size)`. Absolute indexing is therefore correct; relative-to-`pos` is only needed for recency-biased sampling. The real boundary trap is in **memory-optimized buffers** that store each observation once and reconstruct a transition from adjacent slots `(buf[i], buf[i+1])` (e.g. DQN frame stacks): there, sampling `i` at/near the write head pairs a state with a `next_state` from overwritten or cross-episode data — exclude indices straddling `pos`, and never bootstrap across a `done`. Storing the full `(s, a, r, s', done)` tuple per slot (as above) avoids this entirely since each entry is self-contained.
 
### Prioritized Experience Replay via Sum-Tree — *the crossover problem*
Sample proportional to priority in **O(log n)**; update priority in **O(log n)**. Pure index arithmetic, so numpy and torch differ only in the array type — shown once with a note.
 
```python
import numpy as np   # (use torch.zeros for the tree to keep everything on-device; logic identical)
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)      # internal nodes [1..cap-1], leaves [cap..2cap-1]
        self.data = [None] * capacity
        self.pos = 0; self.size = 0
    def add(self, priority, item):
        idx = self.pos + self.capacity          # leaf for this slot
        self.data[self.pos] = item
        self.update(idx, priority)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        idx //= 2
        while idx >= 1:                         # propagate delta to root
            self.tree[idx] += change
            idx //= 2
    def total(self):
        return self.tree[1]                     # root = sum of priorities
    def sample(self, value):                    # value in [0, total)
        idx = 1
        while idx < self.capacity:              # descend to a leaf
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]        # go right, subtract left mass
                idx = left + 1
        data_idx = idx - self.capacity
        return data_idx, self.tree[idx], self.data[data_idx]
```
Gotchas: sampling = draw uniform `value ∈ [0,total)`, walk down comparing against the left subtree's sum → lands on a leaf with prob ∝ priority. The "propagate the delta up to the root" in `update` is what people get wrong. Full PER picture: priority = `(|TD-error| + ε)^α`, plus IS weights `(1/(N·P(i)))^β` to correct the induced sampling bias (loops back to IS above). Torch note: `self.tree = torch.zeros(2*capacity)` works unchanged; keep it on CPU since the descent is sequential/branchy (GPU won't help a single-sample tree walk).
 
### Discounted returns & GAE
**NumPy (reference)**
```python
def discounted_returns(rewards, gamma):
    G = np.zeros(len(rewards)); running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G
 
def gae(rewards, values, gamma, lam, last_value=0.0):
    T = len(rewards); adv = np.zeros(T); acc = 0.0; next_v = last_value
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_v - values[t]    # TD residual δ_t
        acc = delta + gamma * lam * acc
        adv[t] = acc; next_v = values[t]
    return adv
```
**PyTorch**
```python
def discounted_returns_t(rewards, gamma):
    G = torch.zeros(len(rewards)); running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G
 
def gae_t(rewards, values, gamma, lam, last_value=0.0):
    T = len(rewards); adv = torch.zeros(T); acc = 0.0; next_v = last_value
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_v - values[t]
        acc = delta + gamma * lam * acc
        adv[t] = acc; next_v = values[t]
    return adv
```
Gotchas: always accumulate **backward**. GAE's λ trades bias/variance: λ=0 → one-step TD (low variance, biased), λ=1 → Monte-Carlo advantage (unbiased, high variance); at λ=1 with a zero baseline it reduces to discounted return-to-go (sanity check). The recurrence is inherently sequential — that's fine; vectorizing it isn't the point.
 
### Epsilon-greedy, Q-learning update, categorical sampling
**NumPy (reference)**
```python
def epsilon_greedy(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, len(q_values))   # explore
    return int(np.argmax(q_values))             # exploit
 
def q_learning_update(Q, s, a, r, s_next, alpha, gamma, done):
    target = r + (0 if done else gamma * np.max(Q[s_next]))   # bootstrap
    Q[s, a] += alpha * (target - Q[s, a])
    return Q
 
def sample_categorical(probs, rng):             # inverse-CDF
    cdf = np.cumsum(probs)
    return int(np.searchsorted(cdf, rng.random()))
```
**PyTorch**
```python
def epsilon_greedy_t(q_values, epsilon):
    if torch.rand(1).item() < epsilon:
        return torch.randint(len(q_values), (1,)).item()
    return int(q_values.argmax())
 
def q_update_t(Q, s, a, r, s_next, alpha, gamma, done):
    target = r + (0 if done else gamma * Q[s_next].max())
    Q[s, a] += alpha * (target - Q[s, a])
    return Q
 
def sample_categorical_t(probs):
    return torch.distributions.Categorical(probs).sample().item()
    # or: torch.multinomial(probs, 1).item()
```
Gotchas: Q-learning is **off-policy** — target uses `max` over next actions regardless of the action taken. On terminal (`done`) drop the bootstrap, or value leaks across episode boundaries (very common bug). `torch.distributions.Categorical` takes either `probs` or `logits` (use `logits=` to skip a manual softmax and stay stable).
 
---
 
## Sampling & streaming (data-infra flavored — this team owns data management)
 
### Reservoir sampling
Uniform k-sample from a stream of unknown length, one pass, O(k) memory. Pure control-flow — identical in numpy/torch; only the RNG call differs.
```python
def reservoir_sample(stream, k, rng):           # rng = np.random.default_rng()
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.integers(0, i + 1)          # keep new item w.p. k/(i+1)
            if j < k:
                reservoir[j] = item
    return reservoir
# torch: swap rng.integers(0, i+1) -> torch.randint(0, i+1, (1,)).item()
```
Gotchas: the survival proof is the likely follow-up — element i is picked w.p. k/(i+1) and survives all later evictions; telescoping gives k/n.
 
### Weighted sampling without replacement
**NumPy (reference)** — Efraimidis–Spirakis: key = `u^(1/w)`, take top-k.
```python
def weighted_sample_without_replacement(items, weights, k, rng):
    weights = np.asarray(weights, dtype=float)
    keys = rng.random(len(items)) ** (1.0 / weights)   # higher weight -> larger key
    top = np.argsort(keys)[::-1][:k]
    return [items[i] for i in top]
```
**PyTorch** — there's a clean built-in.
```python
def weighted_sample_wor_t(weights, k):          # weights need not be normalized
    idx = torch.multinomial(weights, k, replacement=False)
    return idx                                  # the sampled indices
```
Gotchas: `torch.multinomial(w, k, replacement=False)` is the idiomatic torch path (handles unnormalized weights). For *with*-replacement O(1) draws after O(n) setup, mention the alias method. The numpy ES-trick is worth knowing for the "no library" version.
 
### Welford's online mean/variance
One-pass, numerically stable (vs naive `E[x²]−E[x]²` which catastrophically cancels). Scalar accumulators — same in both; torch only matters if you keep per-feature running stats as tensors.
```python
class Welford:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)      # uses the UPDATED mean
    def variance(self):
        return self.M2 / self.n if self.n else 0.0    # population; n-1 for sample
# torch (per-feature): make mean/M2 tensors of shape [num_features]; same recurrence elementwise
```
Gotchas: the second `delta` deliberately uses the post-update mean — the crux. This is exactly how you'd compute dataset-normalization stats over a streaming pipeline; `nn.BatchNorm`'s running stats use the same online idea.
 
---
 
## Classic ML from scratch (lower probability, cheap to prep)
 
### K-means
**NumPy (reference)**
```python
def kmeans(X, k, iters, rng):
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)   # [N,k]
        labels = dists.argmin(axis=1)
        new = np.array([X[labels == c].mean(axis=0) if np.any(labels == c)
                        else centroids[c] for c in range(k)])
        if np.allclose(new, centroids): break
        centroids = new
    return centroids, labels
```
**PyTorch** (`torch.cdist` for pairwise distances)
```python
def kmeans_t(X, k, iters):
    centroids = X[torch.randperm(len(X))[:k]]
    for _ in range(iters):
        d = torch.cdist(X, centroids)           # [N,k] pairwise distances
        labels = d.argmin(dim=1)
        new = torch.stack([X[labels == c].mean(0) if (labels == c).any() else centroids[c]
                           for c in range(k)])
        if torch.allclose(new, centroids): break
        centroids = new
    return centroids, labels
```
Gotchas: numpy `X[:, None] - centroids[None]` broadcasts to `[N,k,dims]`; torch has `cdist` so you skip the manual broadcast. Handle empty clusters (keep old centroid) or you hit `nan`.
 
### One gradient step (logistic regression)
**NumPy (reference)** — manual gradient; BCE gradient w.r.t. logits is just `(p − y)`.
```python
def logistic_regression_step(X, y, w, b, lr):
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    grad_w = X.T @ (p - y) / len(y)             # clean gradient: (p - y)
    grad_b = np.mean(p - y)
    return w - lr * grad_w, b - lr * grad_b
```
**PyTorch** (autograd — the idiomatic torch way; let the engine compute gradients)
```python
def logreg_step_autograd(X, y, w, b, lr):
    w = w.clone().detach().requires_grad_(True)
    b = b.clone().detach().requires_grad_(True)
    p = torch.sigmoid(X @ w + b)
    loss = F.binary_cross_entropy(p, y)
    loss.backward()                             # fills w.grad, b.grad
    with torch.no_grad():
        return w - lr * w.grad, b - lr * b.grad
# in real code: optimizer = torch.optim.SGD([w, b], lr); loss.backward(); optimizer.step()
```
Gotchas: the BCE-on-logits gradient collapsing to `(p − y)` is worth stating (why logistic-regression gradients are so simple). In torch, show you know the real loop — `optimizer.zero_grad(); loss.backward(); optimizer.step()` — and that `with torch.no_grad()` is needed for the manual update to avoid tracking. For stability use `F.binary_cross_entropy_with_logits` (fuses sigmoid + BCE) instead of `sigmoid` then `binary_cross_entropy`.
 
### Precision / Recall / F1
**NumPy (reference)**
```python
def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return precision, recall, f1
```
**PyTorch**
```python
def prf1_t(y_true, y_pred):
    tp = ((y_pred==1) & (y_true==1)).sum().float()
    fp = ((y_pred==1) & (y_true==0)).sum().float()
    fn = ((y_pred==0) & (y_true==1)).sum().float()
    p = tp/(tp+fp) if tp+fp > 0 else torch.tensor(0.)
    r = tp/(tp+fn) if tp+fn > 0 else torch.tensor(0.)
    f = 2*p*r/(p+r) if p+r > 0 else torch.tensor(0.)
    return p.item(), r.item(), f.item()
```
Gotchas: guard all three denominators. Explain the precision/recall trade-off and when F1 (harmonic mean) beats accuracy (imbalanced classes).
 
---
 
## Future explorations — to deepen (no code yet)
 
Two RL variance topics flagged for a dedicated study pass:
 
- **GAE — Generalized Advantage Estimation.** Code already in the doc (`gae()`); the gap is conceptual. It's an advantage-estimate variance lever (*not* an IS lever — it operates on-policy, e.g. PPO/A2C). The `λ` knob is a bias/variance dial over TD residuals: `λ=0` → one-step TD (low variance, biased), `λ=1` → Monte-Carlo advantage `G_t − V(s_t)` (unbiased, high variance). Essentially TD(λ) applied to the advantage. To understand: why the residuals telescope at `λ=1`, and how `λ` interacts with the value baseline.
- **V-trace (IMPALA).** Off-policy actor-critic target with *two* clipped IS weights doing different jobs: `ρ̄` clips the TD-residual weight and sets the **fixed point** (which policy's value you converge to → bias), while `c̄` clips the **trace product** `Π c̄_i` and bounds backward-credit **variance**; constraint `ρ̄ ≥ c̄`, defaults `1`. It's the off-policy generalization of GAE's λ-trace — where IS clipping and λ-traces become the same machinery. Lineage to study together: Retrace(λ), Tree-Backup, TD(λ). This is the principled fix for the unbounded prefix-product variance in per-decision IS above.
---
 
## Things to say out loud (the senior signal)
 
- **Stability first:** subtract-max in softmax, log-sum-exp, `eps` in norms, `*_with_logits` fused losses. Naming these unprompted reads as production maturity.
- **Bias/variance framing:** ordinary vs self-normalized IS; GAE's λ; MC vs TD targets. RL-infra interviewers listen for this.
- **Built-in *and* from-scratch:** know `F.scaled_dot_product_attention`, `nn.MultiheadAttention`, `torch.multinomial`, `torchvision.ops.nms` — but be able to hand-roll each. They ask for the hand-roll; production uses the built-in.
- **Shape discipline:** annotate every tensor's dims; state the conv output-size formula; use `transpose(-2,-1)` so code generalizes to batched/multi-head.
- **Hardware bridge:** conv → im2col/`unfold` → matmul efficiency; attention → FlashAttention kernels; head-count vs core layout. Your Amazon kernel-fusion and accelerator co-design experience is exactly this team's "bridge hardware and ML" mandate.
- **Vectorize on request:** have the clear loop ready, then the batched/`cdist`/`multinomial` version — mirrors the real infra job.