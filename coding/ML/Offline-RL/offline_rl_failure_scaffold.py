"""
Developed by Claude: Project link: https://claude.ai/project/019c7d87-6129-729d-9e64-771130861a30
Thread link: https://claude.ai/chat/0d90e810-e310-401f-970e-359e6eaf6c76

Offline-RL failure-curve scaffold  (PyTorch, real Minari data, single-GPU/P5000).
============================================================================
Purpose: produce the "what the curve looks like when it breaks" figures on REAL
data -- the ones that can't be faithfully faked without training:
    * Q-value explosion past R_max/(1-gamma) and past dataset MC returns
    * actor loss going strongly negative
    * action-saturation fraction climbing toward 1.0
    * critic loss low while policy is bad (self-consistency != correctness)
 
Design: a correct, minimal TD3+BC instrumented with every diagnostic the
reference doc names, plus a MODE switch that deliberately induces each failure.
TD3+BC is the right vehicle: small enough to be obviously correct, and three of
its knobs (BC term, alpha, state normalization) map directly onto the failures.
 
CQL undershoot and full IQL/AWAC temperature sweeps: see the notes at the bottom.
Hand-rolled CQL is genuinely error-prone (logsumexp importance correction,
Lagrangian) -- for that one, instrument a vetted implementation instead of trusting
this scaffold; a library-agnostic Q-vs-return logger is provided for exactly that.
 
------------------------------------------------------------------------------
Setup (P5000 = 16GB Pascal, fp32 is fine; no bf16 needed):
    pip install torch minari[all] gymnasium mujoco numpy matplotlib
    # first run downloads the dataset (~hundreds of MB)
 
Verify-before-trusting checklist (these are the stale/fragile spots):
    * DATASET name + version: `minari list remote` then set DATASET below.
      Naming is `mujoco/<env>/<quality>-v0` (e.g. mujoco/halfcheetah/medium-v0).
    * episode field is `.rewards` (plural) in current Minari; older snippets show
      `.reward`. If you get an AttributeError, flip it in load_minari().
    * MuJoCo install is the usual pain point; `recover_environment()` needs it only
      for closed-loop eval, not for training on the static data.
------------------------------------------------------------------------------
Run:
    python offline_rl_failure_scaffold.py --mode no_bc       # -> Q explosion + saturation
    python offline_rl_failure_scaffold.py --mode alpha_huge  # -> divergence, actor loss very negative
    python offline_rl_failure_scaffold.py --mode no_norm     # -> imbalance / instability
    python offline_rl_failure_scaffold.py --mode healthy     # -> the calibrated baseline
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
# ---- MODE switch: each mode flips exactly the knob that induces one failure ----
MODES = {
    "healthy":    dict(alpha=2.5,   use_bc=True,  normalize=True),   # calibrated baseline
    "no_bc":      dict(alpha=2.5,   use_bc=False, normalize=True),   # pure offline TD3 -> OOD explosion
    "alpha_huge": dict(alpha=1e4,   use_bc=True,  normalize=True),   # RL term dominates -> divergence
    "no_norm":    dict(alpha=2.5,   use_bc=True,  normalize=False),  # broken state norm -> imbalance
}
DATASET = "mujoco/halfcheetah/medium-v0"   # <-- verify name/version with `minari list remote`
 
# =============================================================================
# Data
# =============================================================================
def load_minari(name):
    import minari
    ds = minari.load_dataset(name, download=True)
    obs, act, rew, nobs, term = [], [], [], [], []
    for ep in ds.iterate_episodes():
        o = ep.observations[:-1]; no = ep.observations[1:]
        obs.append(o); nobs.append(no); act.append(ep.actions)
        rew.append(ep.rewards)                      # <-- `.rewards` (plural) in current Minari
        d = np.zeros(len(ep.rewards), dtype=np.float32)
        d[-1] = float(getattr(ep, "terminations", np.zeros_like(d))[-1]) if hasattr(ep, "terminations") else 0.0
        term.append(d)
    to = lambda xs: torch.as_tensor(np.concatenate(xs), dtype=torch.float32)
    return dict(obs=to(obs), act=to(act), rew=to(rew).reshape(-1, 1),
                nobs=to(nobs), done=to(term).reshape(-1, 1))
 
def mc_returns(rew, done, gamma):
    """Per-state discounted return along trajectories -> the empirical 'achievable' reference."""
    rew = rew.squeeze(-1).numpy(); done = done.squeeze(-1).numpy()
    G = np.zeros_like(rew); run = 0.0
    for i in range(len(rew)-1, -1, -1):
        run = rew[i] + gamma*run*(1.0 - done[i]); G[i] = run
    return float(np.mean(G)), float(np.max(G))
 
# =============================================================================
# Networks
# =============================================================================
def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
 
class Actor(nn.Module):
    def __init__(self, s, a): super().__init__(); self.net = mlp(s, a)
    def forward(self, x): return torch.tanh(self.net(x))        # actions in [-1, 1]
 
class Critic(nn.Module):
    def __init__(self, s, a):
        super().__init__(); self.q1 = mlp(s+a, 1); self.q2 = mlp(s+a, 1)
    def forward(self, s, a):
        x = torch.cat([s, a], -1); return self.q1(x), self.q2(x)
 
# =============================================================================
# TD3+BC training with full diagnostic logging
# =============================================================================
def train(mode, steps=100_000, batch=256, gamma=0.99, tau=0.005,
          policy_noise=0.2, noise_clip=0.5, policy_freq=2, log_every=1000):
    cfg = MODES[mode]
    D = load_minari(DATASET)
    for k in D: D[k] = D[k].to(DEVICE)
    S, Adim = D["obs"].shape[1], D["act"].shape[1]
    N = D["obs"].shape[0]
 
    # state normalization (TD3+BC's second change; mode 'no_norm' disables it)
    if cfg["normalize"]:
        mu, sd = D["obs"].mean(0, keepdim=True), D["obs"].std(0, keepdim=True) + 1e-3
    else:
        mu, sd = torch.zeros(1, S, device=DEVICE), torch.ones(1, S, device=DEVICE)
    norm = lambda x: (x - mu) / sd
 
    g_mc, g_max = mc_returns(D["rew"].cpu(), D["done"].cpu(), gamma)
    r_max = float(D["rew"].max().cpu()); ceiling = r_max / (1 - gamma)
 
    actor, actor_t = Actor(S, Adim).to(DEVICE), Actor(S, Adim).to(DEVICE)
    critic, critic_t = Critic(S, Adim).to(DEVICE), Critic(S, Adim).to(DEVICE)
    actor_t.load_state_dict(actor.state_dict()); critic_t.load_state_dict(critic.state_dict())
    a_opt = torch.optim.Adam(actor.parameters(), 3e-4)
    c_opt = torch.optim.Adam(critic.parameters(), 3e-4)
 
    eval_idx = torch.randint(0, N, (2048,), device=DEVICE)   # fixed eval batch for diagnostics
    log = {k: [] for k in ["step", "meanQ", "criticL", "actorL", "sat", "ceiling", "mc", "maxret"]}
 
    for t in range(1, steps+1):
        idx = torch.randint(0, N, (batch,), device=DEVICE)
        s, a, r = norm(D["obs"][idx]), D["act"][idx], D["rew"][idx]
        s2, done = norm(D["nobs"][idx]), D["done"][idx]
 
        # --- critic update ---
        with torch.no_grad():
            noise = (torch.randn_like(a)*policy_noise).clamp(-noise_clip, noise_clip)
            a2 = (actor_t(s2) + noise).clamp(-1, 1)
            q1t, q2t = critic_t(s2, a2)
            y = r + gamma*(1 - done)*torch.min(q1t, q2t)
        q1, q2 = critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        c_opt.zero_grad(); critic_loss.backward(); c_opt.step()
 
        actor_loss_val = float("nan")
        # --- delayed actor update ---
        if t % policy_freq == 0:
            pi = actor(s); q1pi, _ = critic(s, pi)
            if cfg["use_bc"]:
                lam = cfg["alpha"] / q1pi.abs().mean().detach()     # <-- detached scale
                actor_loss = -lam*q1pi.mean() + F.mse_loss(pi, a)
            else:
                actor_loss = -q1pi.mean()                           # pure TD3 (no support constraint)
            a_opt.zero_grad(); actor_loss.backward(); a_opt.step()
            actor_loss_val = float(actor_loss.detach().cpu())
            with torch.no_grad():
                for p, pt in zip(critic.parameters(), critic_t.parameters()):
                    pt.mul_(1-tau).add_(tau*p)
                for p, pt in zip(actor.parameters(), actor_t.parameters()):
                    pt.mul_(1-tau).add_(tau*p)
 
        # --- diagnostics ---
        if t % log_every == 0:
            with torch.no_grad():
                se = norm(D["obs"][eval_idx]); pe = actor(se)
                qe, _ = critic(se, pe)
                sat = float((pe.abs() >= 0.999).float().mean().cpu())   # action-saturation fraction
            for k, v in zip(log, [t, float(qe.mean().cpu()), float(critic_loss.detach().cpu()),
                                  actor_loss_val, sat, ceiling, g_mc, g_max]):
                log[k].append(v)
            print(f"[{mode}] step {t:6d}  meanQ {qe.mean():8.2f}  ceiling {ceiling:7.1f} "
                  f" MCret {g_mc:7.2f}  satfrac {sat:4.2f}  actorL {actor_loss_val:8.2f}")
    return log
 
# =============================================================================
# Plot
# =============================================================================
def plot(log, mode):
    st = log["step"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4)); OOD, CAL = "#c0392b", "#2c6e9b"
    a = ax[0]
    a.axhline(log["ceiling"][0], color="#b07d2b", ls="--", lw=1.4, label="R_max/(1-γ) ceiling")
    a.axhline(log["mc"][0], color="#3a3a3a", ls=":", lw=1.4, label="dataset MC return")
    a.plot(st, log["meanQ"], color=OOD, lw=2.2, label="mean predicted Q")
    a.set_title("Q vs achievable ceiling", loc="left", weight="bold"); a.set_xlabel("step"); a.legend(frameon=False)
    ax[1].plot(st, log["actorL"], color=OOD, lw=2.2)
    ax[1].axhline(0, color="#999", lw=0.8); ax[1].set_title("actor loss", loc="left", weight="bold"); ax[1].set_xlabel("step")
    ax[2].plot(st, log["sat"], color=OOD, lw=2.2)
    ax[2].set_ylim(0, 1.02); ax[2].set_title("action-saturation fraction", loc="left", weight="bold"); ax[2].set_xlabel("step")
    for x in ax: x.grid(True, color="#ddd", lw=0.6)
    fig.suptitle(f"TD3+BC failure signatures — mode='{mode}' (real {DATASET})", x=0.012, ha="left", color="#666")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"failure_{mode}.png", bbox_inches="tight", facecolor="white")
    print(f"saved failure_{mode}.png")
 
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", default="no_bc", choices=list(MODES))
    p.add_argument("--steps", type=int, default=100_000); args = p.parse_args()
    plot(train(args.mode, steps=args.steps), args.mode)
 
# =============================================================================
# NOTES for the remaining curves
# =============================================================================
# CQL undershoot (Q sits BELOW empirical returns by design; over-pessimism crushes it):
#   Do NOT hand-roll CQL here -- the logsumexp importance correction and the Lagrangian
#   alpha are easy to get subtly wrong. Train CQL with a vetted single-file impl, e.g.
#   CORL (corl-team/CORL: cql.py) or d3rlpy (CalQL/CQL), then reuse this logger on the
#   trained critic. The undershoot is visible as the gap between these two lines:
#
#   def log_q_vs_return(critic, obs, act, rew, done, gamma, norm):
#       with torch.no_grad():
#           q1, _ = critic(norm(obs), act)            # predicted Q on dataset (s,a)
#       mc, _ = mc_returns(rew, done, gamma)          # empirical discounted return
#       return float(q1.mean()), mc                   # healthy CQL: q1.mean() < mc
#
# IQL tau extremes / AWAC temperature sweep:
#   Swap the update rule, keep this harness's logging. AWAC weight ESS needs no
#   training -- it's the exact-math panel in rl_failure_curves_real.py; reuse that
#   directly with advantages computed from your trained critic if you want it on real data.
#
# Reliability caveats to expect on first run:
#   * 'no_bc'/'alpha_huge' should diverge within ~50-100k steps on medium data; if Q
#     merely overestimates without exploding, lower policy_noise or raise alpha.
#   * normalize the dataset reward (common in TD3+BC: r -> r/1000 or per-dataset scaling)
#     if the healthy baseline's Q scale looks off; reward scale shifts the ceiling check.