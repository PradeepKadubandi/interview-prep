"""
Developed by Claude: Project link: https://claude.ai/project/019c7d87-6129-729d-9e64-771130861a30
Thread link: https://claude.ai/chat/0d90e810-e310-401f-970e-359e6eaf6c76

Offline-dataset & rollout diagnostics — the plots behind the prep doc's
stitchability checks (§5) and the "measure data support" debugging step (§8).

Operates on a generic dataset contract so it works with Minari, robomimic, or D4RL:
    D = {
        "obs":  np.ndarray [N, S],   # all states, flattened across episodes
        "act":  np.ndarray [N, A],   # actions (assumed normalized to [-1, 1])
        "rew":  np.ndarray [N],
        "done": np.ndarray [N],      # 1 at each episode's last transition, else 0
    }
A rollout is just {"obs": [T, S], "act": [T, A]} — e.g. states a BC/learned policy
actually visited at eval time, for the covariate-shift overlay.

Deps: numpy, matplotlib, scikit-learn.  Run `python offline_data_viz.py` for the
synthetic demo (writes offline_data_diagnostics.png).

Loader adapters (uncomment the one you use):
    # Minari:
    #   import minari; ds = minari.load_dataset("mujoco/halfcheetah/medium-v0", download=True)
    #   obs, act, rew, done = [], [], [], []
    #   for ep in ds.iterate_episodes():
    #       obs.append(ep.observations[:-1]); act.append(ep.actions); rew.append(ep.rewards)
    #       d = np.zeros(len(ep.rewards)); d[-1] = 1; done.append(d)
    #   D = {k: np.concatenate(v) for k, v in zip(["obs","act","rew","done"],[obs,act,rew,done])}
    # robomimic (HDF5): iterate demos in data["data"], read obs (flatten dict), actions, rewards,
    #   set done[-1]=1 per demo. See pointers at the bottom of this file.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# ----------------------------------------------------------------------------- core
def episode_slices(done_b):                                  # -> list of (start, end)
    ends = np.where(done_b > 0.5)[0]
    slices, start = [], 0
    for e in ends:
        slices.append((start, int(e) + 1)); start = int(e) + 1
    if start < len(done_b):                                   # trailing unterminated episode
        slices.append((start, len(done_b)))
    return slices

def returns_to_go(rew_b, done_b, gamma):                      # [N] discounted return from each step
    rtg_b = np.zeros_like(rew_b, dtype=np.float64); run = 0.0
    for i in range(len(rew_b) - 1, -1, -1):
        run = rew_b[i] + gamma * run * (1.0 - done_b[i]); rtg_b[i] = run
    return rtg_b


# ----------------------------------------------------------------------------- panel
def diagnostic_panel(D, rollout=None, gamma=0.99, k=15, n_sample=2000, seed=0,
                     out="offline_data_diagnostics.png"):
    rng = np.random.default_rng(seed)
    obs_ns, act_na = D["obs"], D["act"]
    N, A = obs_ns.shape[0], act_na.shape[1]
    rtg_n   = returns_to_go(D["rew"], D["done"], gamma)       # [N]
    slices  = episode_slices(D["done"])
    ep_ret  = np.array([rtg_n[s] for s, _ in slices])         # [n_episodes] total returns

    pca = PCA(n_components=2).fit(obs_ns)                      # 2D projection of state space
    xy_n2 = pca.transform(obs_ns)                             # [N, 2]

    # stitchability signature: variance of return-to-go among each state's k nearest neighbors
    nn_k: NearestNeighbors = NearestNeighbors(n_neighbors=k + 1).fit(obs_ns)
    samp = rng.choice(N, size=min(n_sample, N), replace=False)
    _, nbr = nn_k.kneighbors(obs_ns[samp])                    # [M, k+1]
    rtg_var_m = rtg_n[nbr[:, 1:]].var(axis=1)                 # exclude self-neighbor: [M]

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2)); GRID = "#dddddd"
    plt.rcParams.update({"font.size": 10})

    # (0,0) state coverage, colored by return-to-go, with optional rollout overlay
    a = ax[0, 0]
    sc = a.scatter(xy_n2[:, 0], xy_n2[:, 1], c=rtg_n, s=4, alpha=0.35, cmap="viridis")
    fig.colorbar(sc, ax=a, label="return-to-go", fraction=0.046)
    if rollout is not None:
        rxy = pca.transform(rollout["obs"])                  # [T, 2]
        a.plot(rxy[:, 0], rxy[:, 1], color="#c0392b", lw=1.6, marker="o", ms=2.5,
               label="rollout path"); a.scatter(rxy[0, 0], rxy[0, 1], color="k", s=40, zorder=5, label="start")
        a.legend(frameon=False, fontsize=8, loc="best")
    a.set_title("State coverage (PCA), colored by return\noverlap + quality-spread = stitchable",
                loc="left", weight="bold"); a.set_xlabel("PC1"); a.set_ylabel("PC2")

    # (0,1) stitchability signature: return variance among neighbors
    a = ax[0, 1]
    a.hist(rtg_var_m, bins=50, color="#2c6e9b", alpha=0.85)
    a.axvline(np.median(rtg_var_m), color="#c0392b", ls="--", lw=1.4,
              label=f"median {np.median(rtg_var_m):.2f}")
    a.set_title("Return-variance among k-NN states\nmass at high variance = recombinable value",
                loc="left", weight="bold"); a.set_xlabel(f"var of return-to-go over {k} neighbors")
    a.set_ylabel("count"); a.legend(frameon=False, fontsize=8)

    # (0,2) episode return distribution (quality spread)
    a = ax[0, 2]
    a.hist(ep_ret, bins=40, color="#6a8f3a", alpha=0.85)
    a.set_title(f"Episode return distribution\n{len(ep_ret)} episodes; spread → mixed quality",
                loc="left", weight="bold"); a.set_xlabel("episode return"); a.set_ylabel("count")

    # (1,0) rollout support distance over steps (covariate-shift / first-divergence)
    a = ax[1, 0]
    if rollout is not None:
        nn1 = NearestNeighbors(n_neighbors=1).fit(obs_ns)
        dist_t, _ = nn1.kneighbors(rollout["obs"])           # [T, 1]
        dist_t = dist_t.ravel()
        ref = np.percentile(NearestNeighbors(n_neighbors=2).fit(obs_ns)
                            .kneighbors(obs_ns[rng.choice(N, 1000)])[0][:, 1], 95)
        a.plot(dist_t, color="#c0392b", lw=2.0)
        a.axhline(ref, color="#888", ls="--", lw=1.2, label="95th pct in-data NN dist")
        first = np.argmax(dist_t > ref) if np.any(dist_t > ref) else None
        if first is not None:
            a.axvline(first, color="k", ls=":", lw=1.4, label=f"first divergence @ t={first}")
        a.legend(frameon=False, fontsize=8)
        a.set_title("Rollout distance to dataset support\nspike = policy left the data",
                    loc="left", weight="bold"); a.set_xlabel("rollout step"); a.set_ylabel("dist to nearest dataset state")
    else:
        a.text(0.5, 0.5, "pass a rollout to see\nsupport-distance diagnostic",
               ha="center", va="center"); a.set_axis_off()

    # (1,1) action distributions per dim (multimodality, narrowness, saturation)
    a = ax[1, 1]
    for j in range(A):
        a.hist(act_na[:, j], bins=60, histtype="step", lw=1.3, label=f"a[{j}]")
    a.axvline(-1, color="#888", ls="--", lw=1); a.axvline(1, color="#888", ls="--", lw=1)
    a.set_title("Action distributions per dim\nbimodal → multimodality; piled at ±1 → saturation",
                loc="left", weight="bold"); a.set_xlabel("action value")
    if A <= 8: a.legend(frameon=False, fontsize=7, ncol=2)

    # (1,2) action saturation fraction per dim (the diagnostic the doc names)
    a = ax[1, 2]
    sat_a = (np.abs(act_na) >= 0.999).mean(axis=0)           # [A]
    a.bar(np.arange(A), sat_a, color="#b07d2b")
    a.set_ylim(0, max(0.05, sat_a.max() * 1.2))
    a.set_title("Action-saturation fraction per dim\nhigh = behavior pinned at bounds",
                loc="left", weight="bold"); a.set_xlabel("action dim"); a.set_ylabel("frac |a|≥0.999")

    for row in ax:
        for a in row:
            if a.has_data(): a.grid(True, color=GRID, lw=0.5)
    fig.suptitle("Offline dataset D — diagnostics (real computation on the provided D)",
                 x=0.012, ha="left", color="#666")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")
    return dict(median_rtg_var=float(np.median(rtg_var_m)),
                episode_return_spread=(float(ep_ret.min()), float(ep_ret.max())),
                saturation_per_dim=sat_a)


# ----------------------------------------------------------------------------- synthetic demo
def _synthetic_dataset(rng):
    """Two trajectory families that OVERLAP in a mid-region (stitchable), embedded in R^8."""
    S, A = 8, 3
    W = rng.standard_normal((2, S)) * 0.7                     # 2D latent -> 8D obs embedding
    goal = np.array([2.0, 0.0])
    obs, act, rew, done = [], [], [], []
    def emit(latent_t, quality):
        for t, p in enumerate(latent_t):
            o = p @ W + 0.05 * rng.standard_normal(S)
            a = np.clip((goal - p) * quality + 0.1 * rng.standard_normal(2), -1, 1)
            a = np.concatenate([a, [0.3 * rng.standard_normal()]])  # 3rd action dim = noise
            obs.append(o); act.append(a); rew.append(-np.linalg.norm(p - goal))
            done.append(1.0 if t == len(latent_t) - 1 else 0.0)
    for _ in range(120):                                     # GOOD trajectories: start -> goal through mid
        start = np.array([-2.0, 0.0]) + 0.3 * rng.standard_normal(2)
        latent = np.linspace(start, goal, 25) + 0.1 * rng.standard_normal((25, 2))
        emit(latent, quality=0.9)
    for _ in range(120):                                     # MEDIOCRE: wander through the same mid-region
        start = np.array([-1.0, 1.5]) + 0.3 * rng.standard_normal(2)
        mid = np.array([0.0, 0.0]); end = np.array([0.5, -1.5])
        latent = np.vstack([np.linspace(start, mid, 12), np.linspace(mid, end, 13)]) + 0.15 * rng.standard_normal((25, 2))
        emit(latent, quality=0.3)
    D = {k: np.array(v, dtype=np.float64) for k, v in
         zip(["obs", "act", "rew", "done"], [obs, act, rew, done])}
    D["obs"] = np.array(obs); D["act"] = np.array(act)
    # a rollout that starts in support then DRIFTS off into an unvisited region
    p = np.array([-2.0, 0.0]); path = []
    for t in range(30):
        path.append(p @ W + 0.05 * rng.standard_normal(S))
        p = p + (np.array([0.15, 0.0]) if t < 12 else np.array([0.1, 0.45]))  # veers off after step 12
    return D, {"obs": np.array(path), "act": np.zeros((30, 3))}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D, rollout = _synthetic_dataset(rng)
    stats = diagnostic_panel(D, rollout=rollout, gamma=0.99)
    print("median k-NN return variance:", round(stats["median_rtg_var"], 3),
          "| episode return spread:", tuple(round(x, 1) for x in stats["episode_return_spread"]))