# Behavior & Motion Prediction — Quick Reference
 
Companion to `seminal_cv_models_reference.md`. Kept separate because the input is already-abstracted (agent tracks, map geometry, traffic state) rather than raw pixels or points, and because the taxonomy that matters is scene encoding and output parameterization, not CNN vs. transformer.
 
---
 
## The axes that actually discriminate
 
| Axis | Options |
|---|---|
| **Scene encoding** | Rasterized BEV image · Vectorized polylines · Graph (lane connectivity) · Query-centric |
| **Output form** | Single trajectory · Trajectory set / anchors · Goal-then-path · Latent generative · Occupancy grid · Discrete tokens |
| **Agent coupling** | Marginal (per-agent independently) · Joint / scene-level (consistent futures) |
| **Coordinate frame** | Agent-centric (re-encode per agent) · Scene-centric · Query-centric (encode once, reuse) |
| **Supervision** | Imitation · CVAE / GAN · Diffusion · Autoregressive token prediction |
 
**Legend — Arch:** `RNN` · `CNN` · `GNN` · `TF` (transformer) · `Diff` (diffusion)
 
---
 
## 1. Pedestrian / crowd origins (pre-AV lineage)
 
| Year | Work | Arch | Scene encoding | Output | Note |
|---|---|---|---|---|---|
| 2016 | Social LSTM | RNN | Neighbor pooling | Single | Social pooling layer |
| 2018 | Social GAN | RNN + GAN | Neighbor pooling | Generative | Multimodality via adversarial training |
| 2019 | STGAT / Social-BiGAT | GNN + RNN | Graph attention | Generative | Attention over interactions |
| 2020 | Trajectron++ | GNN + CVAE | Graph + map | Latent generative | Dynamics-aware, heterogeneous agents |
 
---
 
## 2. Rasterized CNN era (AV, ~2018–2020)
 
Render the scene as a multi-channel BEV image, run a CNN over it.
 
| Year | Work | Arch | Scene encoding | Output | Note |
|---|---|---|---|---|---|
| 2018 | Fast and Furious (FaF) | CNN | LiDAR BEV | Short-horizon | Detection + tracking + prediction in one pass |
| 2018 | IntentNet | CNN | LiDAR + map raster | Intent + trajectory | Joint detection and high-level intent |
| 2019 | ChauffeurNet | CNN | Raster BEV | Single (planner) | Imitation + synthesized perturbations |
| 2019 | Rules of the Road | CNN | Raster BEV | Distribution | Entity-centric raster encoding |
| 2019 | MultiPath | CNN | Raster BEV | Anchor set + offsets | Fixed anchors tame multimodality |
| 2019 | PRECOG | CNN + flow | Raster + LiDAR | Joint generative | Conditions on ego plan |
| 2020 | CoverNet | CNN | Raster BEV | Trajectory-set classification | Regression → classification reframing |
 
---
 
## 3. Vectorized / graph era (~2020–2021)
 
Drop the raster. Encode lanes and agent histories as polylines and nodes — orders of magnitude cheaper and lossless on geometry.
 
| Year | Work | Arch | Scene encoding | Output | Note |
|---|---|---|---|---|---|
| 2020 | VectorNet | GNN | Vectorized polylines | Single / set | The raster-to-vector pivot |
| 2020 | LaneGCN | GNN | Lane graph | Set | Actor–lane, lane–lane message passing |
| 2020 | TNT | GNN | Vectorized | Goal → path → score | Target-driven decomposition |
| 2021 | DenseTNT | GNN | Vectorized | Dense goal set | Anchor-free goal prediction |
| 2021 | mmTransformer | TF | Vectorized | Proposal set | Region-based multimodality |
| 2022 | HiVT | TF | Vectorized, hierarchical | Set | Translation-invariant local frames |
 
---
 
## 4. Transformer era (~2021–present)
 
| Year | Work | Arch | Scene encoding | Output | Note |
|---|---|---|---|---|---|
| 2021 | Scene Transformer | TF | Vectorized, scene-centric | **Joint** | Marginal and joint from one model |
| 2022 | MultiPath++ | TF + MLP | Vectorized | Anchor set (learned) | Context fusion without rasters |
| 2022 | Wayformer | TF | Vectorized | Set | Systematic study of fusion strategies |
| 2022 | MTR (Motion Transformer) | TF | Vectorized | Motion query pairs | Global intent + local refinement; WOMD winner |
| 2023 | MTR++ / MTR-A | TF | Vectorized | Joint | Multi-agent extension |
| 2023 | QCNet | TF | **Query-centric** | Set | Encode scene once, reuse across agents/time |
| 2023 | MotionLM | TF | Vectorized | Discrete tokens | Prediction as language modeling |
| 2023 | MotionDiffuser | Diff | Vectorized | Joint generative | Diffusion over joint futures, controllable |
| 2024 | Trajeglish | TF | Discrete tokens | Joint autoregressive | Traffic behavior as a token sequence |
 
---
 
## 5. Occupancy / non-vector output
 
Predicts *where space will be occupied*, sidestepping the detect-then-track-then-predict cascade and its failure modes.
 
| Year | Work | Arch | Inputs | Output | Note |
|---|---|---|---|---|---|
| 2020 | P3 | CNN | LiDAR | Semantic occupancy | Differentiable, planner-coupled |
| 2021 | MP3 | CNN | LiDAR | Online map + occupancy flow | Mapless end-to-end |
| 2021 | FIERY | CNN + TF | Multi-cam | BEV instance futures | Probabilistic future instance seg |
| 2022 | Occupancy Flow Fields | CNN | Vector + raster | Occupancy + flow | Waymo benchmark formulation |
| 2023 | Occ3D / OccNet | TF | Multi-cam | 3D occupancy | Voxel-level scene forecasting |
 
---
 
## 6. End-to-end (perception → prediction → planning)
 
Where the field is converging: prediction stops being a standalone module.
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2022 | ST-P3 | CNN + TF | Multi-cam | Interpretable spatial-temporal E2E |
| 2023 | UniAD | TF | Multi-cam | Query-unified full stack; planning-oriented |
| 2023 | VAD | TF | Multi-cam | Fully vectorized scene, fast |
| 2024 | SparseDrive / GenAD | TF | Multi-cam | Sparse-query and generative E2E variants |
 
---
 
## 7. Closed-loop simulation & controllable generation
 
| Year | Work | Arch | Note |
|---|---|---|---|
| 2021 | TrafficSim | GNN + CVAE | Learned multi-agent traffic simulation |
| 2023 | CTG | Diff | Diffusion with rule-based controllability (STL) |
| 2024 | Trajeglish | TF | Token-level agent simulation for closed-loop eval |
 
---
 
## Benchmarks & metrics
 
| Benchmark | Notes |
|---|---|
| Argoverse 1 / 2 Motion Forecasting | Single focal agent; 2s→3s and 5s→6s horizons |
| nuScenes Prediction | 6s horizon; multi-modal |
| Waymo Open Motion Dataset (WOMD) | 8s; marginal **and** joint (interaction) tracks |
| INTERACTION | Dense negotiation scenarios, roundabouts/merges |
| nuPlan / CARLA | Closed-loop planning, not open-loop forecasting |
 
**Metrics.** `minADE_k` / `minFDE_k` (best-of-k displacement error), `Miss Rate`, `brier-minFDE` (penalizes overconfidence), `mAP` for joint prediction, and `Overlap` / drivable-area compliance for physical plausibility. Note the shared weakness: best-of-k rewards diverse guessing and none of these measure calibration well — which is a large part of why the field is drifting toward closed-loop evaluation.
 
---
 
## The one-line history
 
Raster + CNN → vector + graph → transformer with learned anchors → query-centric encoding and discrete-token / diffusion generative models, with the output moving from *one trajectory per agent* toward *jointly consistent scene futures*, and the module boundary with planning dissolving.
 
*Dates are the commonly cited publication year (conference or arXiv, whichever the work is usually referenced by).*
 