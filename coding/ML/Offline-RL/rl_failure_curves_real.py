"""
Developed by Claude: Project link: https://claude.ai/project/019c7d87-6129-729d-9e64-771130861a30
Thread link: https://claude.ai/chat/0d90e810-e310-401f-970e-359e6eaf6c76

Real failure-signature figures (numpy + matplotlib only, no GPU, no RL libraries).
Reproduces the two panels shown in chat:
  (1) Q-value explosion from OOD bootstrapping  -- a real fitted-Q toy computation
  (2) AWR weight collapse vs temperature        -- exact math, no training
 
These are deliberately TOY/EXACT computations chosen to isolate each mechanism.
They prove the failure mode is real and reproducible; they are NOT claimed to match
any specific real dataset's curve. For real-data curves, see the GPU scaffold.
 
Run:  python rl_failure_curves_real.py
"""
import numpy as np
import matplotlib.pyplot as plt
 
# =============================================================================
# PANEL 1 -- Q-value explosion from OOD bootstrapping (real fitted-Q on a toy MDP)
# =============================================================================
# Mechanism isolated: a critic that bootstraps the max over OOD actions inflates
# its own targets, climbing past the true value and past the R_max/(1-gamma)
# ceiling. The in-sample target (next action drawn from the dataset) stays calibrated.
 
rng = np.random.default_rng(0)
GAMMA, R_MAX = 0.9, 0.5
CEILING = R_MAX / (1 - GAMMA)                       # = 5.0 (analytic upper bound)
 
def reward(s):  return 0.5 * (1.0 - s**2)            # bounded in [0, 0.5]
def step(s, a): return np.clip(s + 0.3*a + 0.02*rng.standard_normal(s.shape), -1, 1)
def behavior_action(s): return np.clip(0.2*rng.standard_normal(s.shape), -1, 1)  # NARROW -> large|a| is OOD
 
# ---- fixed offline dataset of trajectories ----
N_TRAJ, H = 400, 30
S, A, R, S2, A2 = [], [], [], [], []
for _ in range(N_TRAJ):
    s = rng.uniform(-1, 1, size=(1,))
    for _ in range(H):
        a = behavior_action(s); s2 = step(s, a); a2 = behavior_action(s2)
        S.append(s); A.append(a); R.append(reward(s)); S2.append(s2); A2.append(a2)
        s = s2
S, A = np.array(S), np.array(A); R = np.array(R).ravel(); S2, A2 = np.array(S2), np.array(A2)
X = np.concatenate([S, A], axis=1)
 
# ---- true achievable value via Monte Carlo (reference) ----
def mc_value(policy_fn, n=2000, h=200):
    s = rng.uniform(-1, 1, size=(n, 1)); G = np.zeros(n); disc = 1.0
    for _ in range(h):
        G += disc * reward(s).ravel(); s = step(s, policy_fn(s)); disc *= GAMMA
    return G.mean()
V_TRUE = mc_value(lambda s: np.zeros_like(s))        # a=0 is the in-support optimum
 
# ---- minimal numpy MLP critic (2 hidden, ReLU) ----
def init():
    h = 64
    return [rng.standard_normal((2, h))*0.3, np.zeros(h),
            rng.standard_normal((h, h))*0.3, np.zeros(h),
            rng.standard_normal((h, 1))*0.3, np.zeros(1)]
def fwd(P, x):
    W1, b1, W2, b2, W3, b3 = P
    z1 = x@W1 + b1; h1 = np.maximum(z1, 0)
    z2 = h1@W2 + b2; h2 = np.maximum(z2, 0)
    return (h2@W3 + b3).ravel(), (x, z1, h1, z2, h2)
def grad_step(P, x, y, lr=2e-3):
    W1, b1, W2, b2, W3, b3 = P
    yp, (x, z1, h1, z2, h2) = fwd(P, x)
    d = (2.0/len(y))*(yp - y)[:, None]
    gW3 = h2.T@d; gb3 = d.sum(0)
    dz2 = (d@W3.T)*(z2 > 0); gW2 = h1.T@dz2; gb2 = dz2.sum(0)
    dz1 = (dz2@W2.T)*(z1 > 0); gW1 = x.T@dz1; gb1 = dz1.sum(0)
    for p, g in zip(P, [gW1, gb1, gW2, gb2, gW3, gb3]): p -= lr*g
    return P
 
def fit(target_kind, iters=60, inner=40, K=16):
    P = init(); Pt = [p.copy() for p in P]
    eval_state = rng.uniform(-1, 1, size=(512, 1)); grid = np.linspace(-1, 1, 21); curve = []
    for _ in range(iters):
        if target_kind == "insample":
            qn, _ = fwd(Pt, np.concatenate([S2, A2], axis=1))
        else:  # OOD-max: max over K random actions across the FULL range
            qs = [fwd(Pt, np.concatenate([S2, rng.uniform(-1, 1, size=A2.shape)], axis=1))[0]
                  for _ in range(K)]
            qn = np.max(np.stack(qs, 0), 0)
        y = R + GAMMA*qn
        for _ in range(inner): P = grad_step(P, X, y)
        Pt = [p.copy() for p in P]
        qbest = np.full(len(eval_state), -1e9)
        for av in grid:
            q, _ = fwd(P, np.concatenate([eval_state, np.full_like(eval_state, av)], axis=1))
            qbest = np.maximum(qbest, q)
        curve.append(qbest.mean())
    return np.array(curve)
 
c_in, c_ood = fit("insample"), fit("oodmax")
print(f"ceiling={CEILING}  V_true={V_TRUE:.2f}  in-sample_final={c_in[-1]:.2f}  ood_final={c_ood[-1]:.2f}")
 
# =============================================================================
# PANEL 2 -- AWR weight collapse vs temperature (exact, no training)
# =============================================================================
# Weights w = exp(A/lambda) over a fixed advantage batch. Effective sample size
# ESS = (sum w)^2 / sum(w^2) measures how many samples actually contribute.
# Low lambda -> one-hot (ESS -> 1 sample); high lambda -> uniform (ESS -> full batch).
B = 256
adv = rng.standard_normal(B); adv = (adv - adv.mean())/adv.std()
lams = np.logspace(-1.0, 1.6, 60)
ess = []
for lam in lams:
    a = adv/lam; a -= a.max()                        # stabilize exp; doesn't change normalized weights
    w = np.exp(a); ess.append((w.sum()**2)/(w**2).sum()/B)
ess = np.array(ess)
assert np.all(np.diff(ess) > -1e-9), "ESS must increase monotonically with temperature"
 
# =============================================================================
# PLOT
# =============================================================================
OOD, INS, CEIL, GRID = "#c0392b", "#2c6e9b", "#b07d2b", "#dddddd"
plt.rcParams.update({"font.size": 11, "figure.dpi": 140, "savefig.dpi": 140})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
 
it = np.arange(1, len(c_in)+1)
ax1.axhspan(CEILING, max(c_ood.max(), CEILING)*1.06, color=OOD, alpha=0.06)
ax1.axhline(CEILING, color=CEIL, ls="--", lw=1.4)
ax1.axhline(V_TRUE, color="#3a3a3a", ls=":", lw=1.4)
ax1.plot(it, c_ood, color=OOD, lw=2.4, label="OOD-max bootstrap target")
ax1.plot(it, c_in,  color=INS, lw=2.4, label="in-sample target")
ax1.text(1.0, CEILING+0.22, r"$R_{\max}/(1-\gamma)$ ceiling = 5.0", color=CEIL, fontsize=9.5)
ax1.text(1.0, V_TRUE-0.62, f"true achievable value (MC) ≈ {V_TRUE:.2f}", color="#3a3a3a", fontsize=9.5)
ax1.text(len(it)*0.50, c_ood.max()*0.80, "extrapolation region\n(above ceiling = impossible)",
         color=OOD, fontsize=9)
ax1.set_xlabel("fitted-Q iteration"); ax1.set_ylabel("mean greedy  Q(s, ·)")
ax1.set_title("Q-value explosion from OOD bootstrapping", fontsize=12, loc="left", weight="bold")
ax1.set_ylim(0, max(c_ood.max(), CEILING)*1.08); ax1.grid(True, color=GRID, lw=0.6)
ax1.legend(frameon=False, loc="center", bbox_to_anchor=(0.62, 0.42), fontsize=10)
 
ax2.axhspan(0, 0.08, color=OOD, alpha=0.06); ax2.axhspan(0.85, 1.0, color=INS, alpha=0.06)
ax2.plot(lams, ess, color="#1c1c1c", lw=2.4); ax2.set_xscale("log")
ax2.set_xlabel("temperature  λ  (log scale)")
ax2.set_ylabel("effective sample size  (fraction of batch)")
ax2.set_title("AWR weight collapse vs temperature", fontsize=12, loc="left", weight="bold")
ax2.set_ylim(0, 1.02); ax2.grid(True, color=GRID, lw=0.6)
ax2.text(lams[2], 0.16, "λ too small →\nweights ≈ one-hot\n(1-sample BC, high variance)",
         color=OOD, fontsize=9, va="top")
ax2.text(lams[-12], 0.93, "λ too large → weights ≈ uniform (plain BC)",
         color=INS, fontsize=9, ha="right", va="top")
ax2.text(1.0, 0.5, "healthy band", color="#3a3a3a", fontsize=9.5, ha="center")
 
fig.tight_layout()
fig.savefig("failure_curves_real.png", bbox_inches="tight", facecolor="white")
print("saved failure_curves_real.png")