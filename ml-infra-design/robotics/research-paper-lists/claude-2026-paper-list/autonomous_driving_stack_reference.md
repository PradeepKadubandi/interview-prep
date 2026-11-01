# Autonomous Driving — Decomposed Stack Reference
 
Scan-first reference. No method explanations; name + sub-axis values only.
Currency: literature through ~mid-2026. See **Confidence notes** at the end.
 
**Scope.** The *modular* stack — perception, localization, prediction, planning, control, and the
newer language-reasoning layer. End-to-end (E2E) sensor-to-trajectory models are **out of scope**
here except as boundary markers where a modular line of work terminates in one.
 
**Where this sits:**
- `seminal_cv_models_reference.md` — task-generic Computer Vision (CV): 2D detection, segmentation,
  depth, tracking, and the base 3D detectors. This doc picks up the *driving-specific* cuts
  (Bird's-Eye-View (BEV) view transforms, occupancy, online mapping) that generic CV doesn't have.
- `behavior_prediction_reference.md` — **prediction owns its own doc**; not duplicated here (§4 is a
  pointer only).
- `robotics_foundation_models_reference.md` §4–5 — world models and embodied reasoning; the
  driving-domain reasoning table here (§7) is the AV-specific cut of that family.
- `waymo_hm_call_prep.md` — Waymo Foundation Model / World Model / simulation realism specifics.
---
 
## The areas
 
| # | Area | Consumes | Emits | Why it's a separate axis |
|---|---|---|---|---|
| 1 | **Perception** | raw sensors | 3D boxes, occupancy, map elements, per-point labels | The only stage whose input is unabstracted; sub-axis is *how 2D pixels become 3D* |
| 2 | **Localization & mapping** | sensors + prior map | ego pose (SE(3)), map | Distinct objective (geometry against a prior), distinct failure modes (drift, aliasing) |
| 3 | **Prediction** | abstracted tracks + map | agent futures | Owns a separate doc — see `behavior_prediction_reference.md` |
| 4 | **Planning / decision** | perception + prediction + map | ego trajectory or maneuver | The only stage whose output is *chosen* rather than *estimated*; interaction and closed-loop eval dominate |
| 5 | **Control** | reference trajectory | actuator commands | Model-based, largely pre-deep-learning; still the tracking-error owner |
| 6 | **Reasoning (VLM/LLM)** | scene + language | plans, explanations, labels, decisions | Cross-cuts 1–4; latency budget forces a dual-rate architecture |
 
Cross-cutting concerns — module **interfaces**, uncertainty propagation, and metric alignment —
get their own section (§9). That section is the interview-relevant part; the tables are recall aids.
 
---
 
## Legend
 
**Sensors** — `C` monocular camera · `MC` multi-camera rig · `L` LiDAR · `R` radar · `SD/HD` map prior · `IMU`/`GNSS`
**Arch** — `CNN` · `TF` transformer · `GNN` · `Diff` diffusion · `Point` point/sparse-conv · `Hybrid`
**View transform** (camera → 3D) — `Lift` forward projection with a predicted depth distribution ·
`BEVQ` BEV queries with backward projection (deformable attention into images) · `3DQ` sparse 3D
object queries, no BEV grid · `PosEmb` implicit, via 3D position encoding · `IPM` flat-ground
homography · `FB` forward *and* backward
**Temporal** — `—` single frame · `Stack` warp-and-concatenate a fixed window · `Stream` recurrent
propagation of features/queries · `Long` long-horizon (16+ frames) temporal stereo
 
---
 
# 1. Perception
 
## 1a. Camera 3D detection / BEV representation
 
The one axis that matters and does not appear in generic CV: **how image features are moved into 3D**.
Everything else (backbone, head) is inherited from the 2D literature.
 
| Year | Work | Org | Sensors | View transform | Temporal | Note |
|---|---|---|---|---|---|---|
| 2019 | OFT / Pseudo-LiDAR | Cambridge / Cornell | C | `IPM`/depth-to-points | — | Pre-BEV ancestors; depth → fake point cloud |
| 2020 | **LSS (Lift-Splat-Shoot)** | NVIDIA | MC | **`Lift`** | — | Emits a depth *distribution* per pixel — the ancestor of every forward-projection method |
| 2021 | **DETR3D** | MIT/CMU | MC | **`3DQ`** | — | 3D reference points project down into 2D features; no dense BEV |
| 2021 | CaDDN | Toronto | C | `Lift` | — | Categorical depth distribution, explicitly supervised |
| 2022 | **BEVFormer** | Shanghai AI Lab | MC | **`BEVQ`** | `Stack` | Spatial cross-attention + temporal self-attention; the backward-projection reference |
| 2022 | BEVDet / BEVDet4D | PhiGent | MC | `Lift` | — / `Stack` | Productionizable LSS detector; 4D adds temporal |
| 2022 | **BEVDepth** | Megvii | MC (+L train) | `Lift` | `Stack` | LiDAR-supervised depth — the "depth is the bottleneck" result |
| 2022 | **PETR / PETRv2** | Megvii | MC | **`PosEmb`** | `Stack` | 3D position encoding into image features; no explicit projection op |
| 2022 | PolarFormer | — | MC | `BEVQ` (polar) | `Stack` | Polar rather than Cartesian BEV grid |
| 2023 | **SOLOFusion** | — | MC | `Lift` | **`Long`** | Long-horizon temporal stereo; showed history length > single-frame resolution |
| 2023 | BEVFormer v2 | Shanghai AI Lab | MC | `BEVQ` | `Stack` | Perspective-view auxiliary supervision to adapt modern backbones |
| 2023 | **StreamPETR** | Megvii | MC | `PosEmb` | **`Stream`** | Object-centric query propagation — recurrence at the object level, not the feature level |
| 2023 | SparseBEV | NJU | MC | `3DQ` | `Stream` | Adaptive spatio-temporal sampling, fully sparse |
| 2024 | Far3D | — | MC | `3DQ` | `Stream` | Long-range (150 m+) via 2D-proposal-seeded 3D queries |
| 2024 | RayDN / QAF2D | — | MC | `3DQ` | `Stream` | Depth-aware negatives / 2D-conditioned queries — the "false positives along the ray" fix |
 
**Sub-axis worth stating out loud:** dense BEV grid (`Lift`, `BEVQ`) buys a shared spatial substrate
that mapping, occupancy, and planning can all read; sparse queries (`3DQ`, `PosEmb`) buy compute that
scales with objects rather than area, which is what makes long range affordable. That is the real
trade, not accuracy.
 
## 1b. LiDAR & multi-sensor 3D detection
 
Base entries (VoxelNet → PointPillars → CenterPoint → BEVFusion) live in
`seminal_cv_models_reference.md` §2b. This table carries the *later* thread: sparsity and fusion stage.
 
**Fusion stage** — `—` none · `Prop` proposal/frustum-level · `BEV` unified BEV feature space ·
`Query` query/token-level · `Out` late, output-level
**Encoding** — `Vox` voxel · `Pil` pillar · `Pt` raw point · `Rng` range image · `SpTF` sparse transformer
 
| Year | Work | Sensors | Encoding | Fusion stage | Note |
|---|---|---|---|---|---|
| 2022 | **TransFusion** | MC + L | Vox | `Query` | LiDAR queries attend to image features; robust to calibration error |
| 2022 | **BEVFusion** (MIT / PKU) | MC + L | Vox/Pil | **`BEV`** | Independent modality streams into one BEV space; degrades gracefully if one drops |
| 2022 | FSD | L | Pt + Vox | — | Fully sparse — removes the dense BEV bottleneck for long range |
| 2023 | **CMT** (Cross-Modal Transformer) | MC + L | Vox | `Query` (`PosEmb`) | No explicit view transform at all; positional encoding does the alignment |
| 2023 | DSVT | L | `SpTF` | — | Dynamic sparse voxel transformer, deployment-friendly |
| 2023 | VoxelNeXt / FSDv2 | L | Vox sparse | — | Fully sparse prediction, no dense head |
| 2023 | SparseFusion | MC + L | Vox | `Query` | Sparse candidates from both modalities, fused as instances |
| 2024 | DAL ("Detecting As Labeling") | MC + L | Pil | `BEV` | Argues the training-pipeline recipe, not the fusion op, carries the gain |
| 2024 | SAFDNet | L | Vox sparse | — | Adaptive feature diffusion; sparse detector at long range |
 
**Radar & adverse conditions** (relevant to any radar-fusion role):
 
| Year | Work | Sensors | Fusion stage | Note |
|---|---|---|---|---|
| 2020 | RadarNet | L + R | `BEV` + `Out` | Early-BEV plus attention-based late velocity refinement |
| 2021 | CenterFusion | C + R | `Prop` | Frustum association of radar returns to image detections |
| 2023 | **CRN** (Camera Radar Net) | MC + R | `BEV` | Radar-assisted view transform — radar supplies the depth camera lacks |
| 2024 | RCBEVDet / RadarDistill | MC + R | `BEV` / distill | LiDAR-teacher distillation into a radar branch |
| 2024 | DPFT | C + R (4D) | `BEV` | Dual-perspective fusion for 4D imaging radar |
 
Datasets that define this sub-field: nuScenes (radar included), **K-Radar** (4D radar, adverse
weather), **View-of-Delft**, RADIATE, Seeing Through Fog (STF/DENSE), aiMotive.
 
## 1c. 3D semantic occupancy
 
Output is a voxel grid of (occupied, class), including a **free-space** and often an **unknown/unobserved**
label — which is what makes it usable by a planner directly, unlike boxes.
 
**Representation** — `Vox` dense voxel · `TPV` tri-perspective planes · `Sparse` sparse queries ·
`Gauss` 3D Gaussians · `Field` implicit/NeRF field
**Supervision** — `Dense` mesh/multi-sweep-derived dense voxel labels · `Sweep` raw LiDAR sweeps ·
`Render` self-supervised volume rendering (no 3D labels) · `Vis` visibility-masked
 
| Year | Work | Sensors | Repr | Supervision | Note |
|---|---|---|---|---|---|
| 2019 | SemanticKITTI (SSC task) | L | Vox | Dense | The benchmark that created semantic scene completion outdoors |
| 2022 | **MonoScene** | C | Vox | Dense | First monocular outdoor SSC; line-of-sight feature projection |
| 2023 | **TPVFormer** | MC | **TPV** | Sweep | Three orthogonal planes instead of a cubic grid — the memory fix |
| 2023 | SurroundOcc | MC | Vox | Dense | Multi-frame mesh-based dense label generation pipeline |
| 2023 | OccFormer / VoxFormer | MC / C | Vox | Dense | Dual-path transformer / depth-seeded sparse queries then densify |
| 2023 | **Occ3D**, OpenOccupancy | MC | Vox | Dense + `Vis` | The benchmarks; visibility masking is the metric subtlety |
| 2023 | **FB-OCC** | MC | Vox | Dense | **`FB`** forward *and* backward view transform; CVPR'23 challenge winner |
| 2024 | SelfOcc / OccNeRF | MC | Field | **Render** | Self-supervised — no 3D occupancy labels at all |
| 2024 | SparseOcc | MC | Sparse | Dense | Fully sparse; drops the dense grid to cut cost |
| 2024 | GaussianFormer | MC | **Gauss** | Dense | 3D Gaussians as the scene primitive; object-centric compute |
| 2024 | **OccWorld** / Cam4DOcc | MC | Vox | Dense | Occupancy *forecasting* — the bridge to world models |
 
## 1d. Online vectorized HD mapping & topology
 
Motivation is economic, not scientific: HD maps cost money to build and go stale. This line asks
whether the map can be inferred per-frame instead.
 
**Primitive** — `Raster` segmentation then post-hoc vectorization · `Poly` ordered polyline points ·
`Pivot` variable-count pivot points · `Bezier` parametric curves
**Decode** — `AR` autoregressive · `Par` parallel set prediction
 
| Year | Work | Sensors | Primitive | Decode | Temporal | Note |
|---|---|---|---|---|---|---|
| 2021 | **HDMapNet** | MC (+L) | `Raster` | — | — | Framed the task; heuristic vectorization was the weak link |
| 2022 | **VectorMapNet** | MC (+L) | `Poly` | `AR` | — | First end-to-end vectorized output |
| 2022 | **MapTR** / 2023 MapTRv2 | MC (+L) | `Poly` | **`Par`** | — | Permutation-equivalent point sets — removes the ordering ambiguity; the standard baseline |
| 2023 | PivotNet / BeMapNet | MC | `Pivot` / `Bezier` | `Par` | — | Variable-density and parametric alternatives to fixed-N points |
| 2024 | **StreamMapNet** | MC | `Poly` | `Par` | **`Stream`** | Memory buffer across frames; also exposed the train/val geographic-overlap flaw in the standard splits |
| 2024 | MapTracker | MC | `Poly` | `Par` | `Stream` | Mapping as tracking — consistency across time is the objective |
| 2024 | SMERF / P-MapNet | MC + **SD** | `Poly` | `Par` | — | Standard-Definition (SD) map prior (e.g. OpenStreetMap) as cheap context |
| 2024 | MGMap / HIMap / SQD-MapNet | MC | `Poly` | `Par` | —/`Stream` | Mask guidance, hybrid point-element queries, stream query denoising |
 
**Lane & topology** (the part planners actually consume — connectivity, not just geometry):
 
| Year | Work | Output | Note |
|---|---|---|---|
| 2022 | PersFormer + OpenLane | 3D lanes | Benchmark that moved lane detection off the image plane |
| 2023 | **OpenLane-V2** | lane + topology + traffic element | The benchmark defining "which lane goes where, under which light" |
| 2023 | TopoNet | lane graph | Scene-graph reasoning over lane and traffic-element relations |
| 2024 | LaneSegNet / TopoMLP | lane segment + topology | Segment-level primitive; strong simple topology baseline |
 
## 1e. LiDAR semantic & panoptic segmentation
 
| Year | Work | Representation | Note |
|---|---|---|---|
| 2018 | SqueezeSeg (v1–v3) | Range image | Real-time projection-based baseline |
| 2019 | RangeNet++ | Range image | + GPU-accelerated k-NN post-processing for projection artifacts |
| 2019 | MinkowskiNet | Sparse voxel | Generalized sparse convolution |
| 2020 | PolarNet / SPVNAS | Polar BEV / point-voxel | Polar partitioning matches LiDAR density; PV-NAS for efficiency |
| 2021 | **Cylinder3D** | Cylindrical voxel | Asymmetric sparse conv; the long-standing strong baseline |
| 2021 | Panoptic-PolarNet / 4D-PLS | Polar / 4D | Panoptic and *temporally consistent* (4D) panoptic |
| 2022 | 2DPASS | Point + image distill | Camera knowledge distilled in at training, camera-free at inference |
| 2023 | SphereFormer | Radial window TF | Fixes the sparsity-at-distance problem directly |
| 2023 | OpenScene / Seal | Open-vocabulary | CLIP-aligned 3D features; zero-shot 3D scene understanding |
| 2024 | **Point Transformer V3** | Serialized point | Simplicity-over-mechanism scaling result |
 
## 1f. Robustness, corner cases, and open-set
 
| Year | Work | Kind | Note |
|---|---|---|---|
| 2019 | Fishyscapes / Lost & Found | Benchmark | Anomaly segmentation on road surfaces |
| 2021 | SegmentMeIfYouCan | Benchmark | Anomaly + obstacle track, the standard OOD (Out-Of-Distribution) suite |
| 2021 | CODA | Benchmark | Corner-case object detection in real driving data |
| 2022 | LiDAR Snowfall / Fog simulation | Data | Physics-based adverse-weather augmentation for LiDAR |
| 2023 | **Robo3D** | Benchmark | Corruption robustness for LiDAR detection/segmentation |
| 2023 | nuScenes-C / Robo-BEV | Benchmark | Sensor corruption and missing-modality robustness for BEV models |
 
**One-line history (perception).** Image-plane detection → pseudo-LiDAR and flat-ground homography
→ *learned* view transforms splitting into forward (`Lift`) and backward (`BEVQ`) camps → sparse
queries and streaming temporal memory → outputs shifting from boxes toward **dense, class-agnostic,
free-space-aware** representations (occupancy, online map elements) because that is what a planner
can actually consume.
 
---
 
# 2. Localization & mapping (SLAM)
 
Kept separate from perception because the objective is geometric consistency against a prior, and
the failure modes are different in kind: **drift** (accumulating, silent) and **aliasing**
(confident and wrong, e.g. repeated parking structures).
 
**Front-end** — `Feat` sparse feature matching · `Direct` photometric · `ICP` scan matching ·
`Learned` learned correspondence
**Back-end** — `Filter` EKF/UKF · `BA` bundle adjustment · `PGO` pose-graph optimization ·
`Factor` factor graph (iSAM2)
 
| Year | Work | Sensors | Front-end | Back-end | Note |
|---|---|---|---|---|---|
| 2007 | MonoSLAM / PTAM | C | Feat | Filter / BA | Filter-vs-BA split; BA won |
| 2010 | Levinson & Thrun | L + GNSS | ICP | Filter | Probabilistic reflectance-map localization — the AV production pattern |
| 2014 | **LOAM** | L | Edge/planar feats | Scan-to-map | The LiDAR odometry reference; LeGO-LOAM (2018) adds ground segmentation |
| 2014 | LSD-SLAM / 2016 DSO | C | **Direct** | PGO / BA | Photometric, no features |
| 2015 | **ORB-SLAM** → 2017 v2 → 2020 v3 | C/stereo/RGB-D + IMU | Feat | BA + PGO | Multi-map, visual-inertial; the canonical citation |
| 2016 | Cartographer | L (2D/3D) | Scan match | PGO | Google; submap + loop closure |
| 2018 | **VINS-Mono** | C + IMU | Feat | Sliding-window BA | The VIO (Visual-Inertial Odometry) reference |
| 2020 | **LIO-SAM** | L + IMU (+GNSS) | ICP | **Factor** | Tightly-coupled factor-graph LiDAR-inertial |
| 2021 | FAST-LIO2 | L + IMU | Direct point-to-map | Iterated EKF | ikd-Tree; very high rate, no feature extraction |
| 2021 | **DROID-SLAM** | C/stereo | **Learned** | Dense BA (differentiable) | Learned recurrent flow inside a BA loop |
| 2022 | NICE-SLAM / iMAP | RGB-D | Learned | Implicit field | Neural-field map representation |
| 2023 | KISS-ICP | L | ICP | — | "Point-to-point ICP, done right" — the strong simple baseline |
| 2024 | SplaTAM / MonoGS | RGB-D / C | Photometric | 3D Gaussian map | Gaussian-splat map as the SLAM back-end |
 
**Place recognition / relocalization:** NetVLAD (2016), **Scan Context** (2018), PointNetVLAD (2018),
MinkLoc3D (2021), **OrienterNet** (2023, localize against OpenStreetMap), BEV-Locator (2022).
 
---
 
# 3. Prediction — pointer only
 
Covered in full by **`behavior_prediction_reference.md`** (rasterized CNN era → vectorized/graph →
transformer with learned anchors → query-centric and generative/token models; marginal vs. joint;
occupancy-flow outputs; the open-loop metric critique).
 
The only thing worth restating at the interface: prediction's output form determines what the
planner can do with it. A set of `k` marginal trajectories with probabilities forces the planner into
either most-likely-mode following or expensive enumeration; **joint/scene-level** futures are what
make contingency and game-theoretic planning well-posed. That is the single most load-bearing
interface decision in the modular stack.
 
---
 
# 4. Planning & decision making
 
**Paradigm** — `Rule` finite-state/behavior tree · `Search` sampling or graph search ·
`Opt` optimization/MPC · `POMDP` belief-space · `Game` game-theoretic · `IL` imitation ·
`RL` reinforcement learning · `Hybrid` learned proposals + rule-based scoring
**Interaction** — `Static` other agents as obstacles · `PredCond` conditioned on
independently-predicted futures · `Reactive` others modeled as responding to ego ·
`Joint` joint/equilibrium solution
**Eval** — `OL` open-loop (log replay, displacement error) · `CL-NR` closed-loop, non-reactive agents ·
`CL-R` closed-loop, reactive agents · `Real` on-vehicle
 
## 4a. Classical & optimization-based
 
| Year | Work | Paradigm | Interaction | Note |
|---|---|---|---|---|
| 2008 | **DARPA Urban Challenge systems** (Boss, Junior, Talos) | `Rule` + `Search` | Static | The origin of the layered mission/behavior/motion decomposition still in use |
| 2008 | **Hybrid A*** | `Search` | Static | Continuous-state A\* with a kinematic model; the parking/unstructured workhorse |
| 2010 | **Frenet-frame optimal trajectories** (Werling) | `Opt` | Static | Road-aligned lateral/longitudinal decoupling; jerk-optimal quintic primitives |
| 2011 | RRT\* / 2009 CL-RRT | `Search` | Static | Asymptotic optimality; closed-loop RRT propagates through the controller |
| 2005/2011 | State lattice planners | `Search` | Static | Precomputed kinematically-feasible motion primitives |
| 2015 | **MPDM** (multipolicy decision making) → EUDM (2020), MARC (2023) | `POMDP`-lite | `Reactive` | Forward-simulate a small set of *policies* rather than trajectories |
| 2015 | Intention-aware POMDP / DESPOT | `POMDP` | `Reactive` | Belief over other agents' intent; the principled but expensive branch |
| 2016 | **Sadigh et al.** — planning that leverages effects on human actions | `Game` | `Joint` | Ego actions as information-gathering and influence; Stackelberg framing |
| 2018 | **Apollo EM Planner** | `Opt` | PredCond | Production reference: iterative path/speed decomposition in Frenet frame |
| 2019 | iLQGames / hierarchical game-theoretic planning (Fisac) | `Game` | `Joint` | Tractable dynamic-game solutions in real time |
| 2013/2022 | Contingency planning (Hardy & Campbell; RCP) | `Opt` | `PredCond` | One committed short-horizon plan, branching futures — the honest answer to multimodal prediction |
 
## 4b. Learned planners
 
| Year | Work | Org | Paradigm | Interaction | Eval | Note |
|---|---|---|---|---|---|---|
| 2019 | **ChauffeurNet** | Waymo | `IL` | PredCond | `CL-NR` | Synthesized perturbations + auxiliary losses; named the covariate-shift problem for AV planning |
| 2021 | **SafetyNet** | Woven/Lyft L5 | `IL` + `Rule` | PredCond | `Real` | ML planner with a rule-based fallback layer — the deployable pattern |
| 2021 | UrbanDriver | Lyft L5 | `IL` (policy gradient in sim) | `Reactive` | `CL-R` | Closed-loop training on real demonstrations |
| 2021 | **Roach** | ETH | `RL` (privileged teacher) | `Reactive` | `CL-R` | RL coach with BEV ground truth, distilled into a sensor policy |
| 2023 | **Imitation Is Not Enough** (BC-SAC) | Waymo | `IL` + `RL` | `Reactive` | `CL-R` | Adding an RL safety objective to BC fixes long-tail failures BC can't |
| 2023 | **PDM-Closed** ("Parting with Misconceptions…") | Tübingen | `Hybrid`/`Rule` | `Reactive` | `CL-R` | A tuned rule-based planner beat all learned entries on nuPlan — the field's most-cited humbling result |
| 2023 | PlanTF ("Rethinking Imitation-based Planners") | — | `IL` | PredCond | `CL-NR` | Isolates which design choices actually cause the compounding error |
| 2023 | **GameFormer** | HKUST | `IL` + `Game` | `Joint` | `CL-R` | Level-k hierarchical game reasoning inside a transformer |
| 2023 | DTPP | — | `IL` + tree search | `Reactive` | `CL-R` | Differentiable tree policy planning; jointly trained predictor and cost |
| 2024 | **PLUTO** | — | `IL` | `Reactive` | `CL-R` | Contrastive imitation + auxiliary losses; pushes IL close to rule-based on nuPlan |
| 2024 | **Hydra-MDP** | NVIDIA | `IL` + distill | PredCond | `CL-NR` | Multi-target distillation from rule-based *and* human teachers; NAVSIM winner |
| 2024 | Think2Drive | — | `RL` (model-based) | `Reactive` | `CL-R` | World-model RL as the first to clear CARLA v2 |
| 2024 | Diffusion-ES / STR (State Transformer) | — | `IL` + search / `IL` | `Reactive` | `CL-R` | Evolutionary search in diffusion latent space; GPT-style scaling of planning |
| 2025 | Diffusion Planner / CarPlanner | Tsinghua / — | `Diff` / `RL` | `Reactive` | `CL-R` | Diffusion over joint plans; consistent-auto-regressive RL planner |
 
## 4c. Safety & verification layer
 
The part that determines whether a learned planner is deployable at all.
 
| Year | Work | Kind | Note |
|---|---|---|---|
| 2017 | **RSS** (Responsibility-Sensitive Safety) | Formal rule set | Mobileye; interpretable "who is at fault" longitudinal/lateral constraints |
| 2019 | **Safety Force Field** | Formal | NVIDIA; claimed-set collision avoidance |
| 2017 | **Control Barrier Functions** (Ames et al.) | Constraint | Forward-invariance certificate; composes with a quadratic program |
| 2005/2017 | Hamilton-Jacobi reachability; FaSTrack | Reachability | Worst-case safe sets; tracking-error bound for planner/controller mismatch |
| 2019 | Rulebooks (Censi et al.) | Priority structure | Pre-ordered rule hierarchy for resolving unavoidable violations |
| 2021+ | Safety filters / shielding, MPC backup policies | Runtime | Learned nominal + verified fallback — the SafetyNet pattern generalized |
 
## 4d. Benchmarks (planning-specific)
 
| Benchmark | Regime | Note |
|---|---|---|
| **nuPlan** (2021) | `CL-R` / `CL-NR` / `OL` | 1200 h real data; first large-scale *closed-loop* planning benchmark |
| **CARLA Leaderboard v1/v2** (2019/2023) | `CL-R` | Simulated sensors; v2 adds hard scenarios |
| **Bench2Drive** (2024) | `CL-R` | 44 scenarios × towns, per-ability breakdown in CARLA |
| **NAVSIM** (2024) | pseudo-`CL` | Non-reactive sim with closed-loop-aligned metrics (PDM score) — cheap proxy |
| **Waymax** (2023) | `CL-R` | JAX-accelerated data-driven sim; log-divergence and infeasibility metrics |
| **interPlan** (2024) | `CL-R` | Edge cases layered onto nuPlan |
| **WOSAC** (2023) | sim-agent realism | Evaluates agent realism, not ego planning |
 
**The metric point to have ready:** open-loop displacement error against a logged human trajectory is
a weak predictor of closed-loop performance — a planner that copies inertia scores well and drives
badly. This is the same failure structure as best-of-`k` in prediction, and it is why nuPlan,
Bench2Drive, and NAVSIM exist.
 
---
 
# 5. Control
 
Mostly pre-deep-learning and stable; know the selection criteria, not papers.
 
| Method | Model | Constraints | Preview | Typical use |
|---|---|---|---|---|
| Pure Pursuit (1992) | Kinematic, geometric | No | Lookahead point | Low-speed lateral tracking; parking, AGVs |
| **Stanley** (2006) | Kinematic + cross-track | No | Current error only | DARPA Grand Challenge; front-axle referenced |
| LQR | Linearized dynamic bicycle | No | Infinite-horizon, no constraints | Lateral control on structured roads |
| **MPC** | Kinematic or dynamic bicycle | **Yes** (explicit) | Finite horizon | The AV default; handles actuator and comfort limits |
| MPCC (2015) | Dynamic, contouring cost | Yes | Finite horizon | Racing / aggressive maneuvers; path progress in the objective |
| iLQR / DDP | Nonlinear | Soft only | Finite horizon | Fast nonlinear trajectory optimization; blurs into planning |
| **MPPI** (2016) | Sampling, any dynamics | Via cost | Finite horizon | Non-differentiable dynamics / off-road; GPU-parallel |
| Tube MPC / L1 adaptive | Nominal + bounded uncertainty | Yes, robustly | Finite horizon | When model error must be certified, not just tuned |
 
**The interface fact worth stating:** planner and controller share a model. A planner that emits a
trajectory the vehicle model cannot track converts a planning success into a tracking error, and the
usual fix — feasibility checks in the planner's cost — means the "modular boundary" is already leaky
in production systems.
 
---
 
# 6. Reasoning — Vision-Language Models in driving
 
**Role** — `QA` scene understanding / benchmark · `Meta` high-level decision or maneuver selection ·
`Planner` emits a trajectory · `Labeler` offline data engine / auto-annotation ·
`Critic` evaluation, verification, triage
**Runs at** — `Off` offline only · `Slow` low-rate in a dual-system architecture · `On` real-time onboard
 
| Year | Work | Org | Role | Output | Runs at | Note |
|---|---|---|---|---|---|---|
| 2019 | Talk2Car | — | `QA` | Referred object | Off | The grounding-in-driving benchmark ancestor |
| 2023 | NuScenes-QA / Reason2Drive | — | `QA` | Text answer | Off | Chain-of-reasoning benchmarks for perception→prediction→reasoning |
| 2023 | **DriveLM** | Shanghai AI Lab | `QA` → `Planner` | Graph-VQA + waypoints | Off | Graph-structured QA over the whole stack; the reference benchmark |
| 2023 | Talk2BEV | IIIT | `QA` | Language over BEV map | Off | Language interface on top of a BEV representation |
| 2023 | GPT-Driver / LanguageMPC / DiLu | — | `Meta`/`Planner` | Waypoints / MPC params / decisions | Off | LLM-as-planner probes; memory + reflection loops |
| 2023 | DriveGPT4 | HKU | `Planner` | Control + explanation | Off | Interpretable control signal prediction |
| 2024 | **Agent-Driver** | — | `Meta` | Tool calls + plan | Off | LLM as an agent orchestrating perception tools and memory |
| 2024 | **LMDrive** | Shanghai AI Lab | `Planner` | Closed-loop control | `Slow` | Language-conditioned closed-loop driving in CARLA |
| 2024 | **DriveVLM / DriveVLM-Dual** | Li Auto/Tsinghua | `Meta` + `Planner` | Coarse plan refined by a fast planner | **`Slow`** | The dual-system pattern, on-vehicle; slow VLM proposes, fast module executes |
| 2024 | OmniDrive / Senna | NVIDIA / Huawei | `Meta` | 3D-grounded reasoning / maneuver | `Slow` | VLM decides *what*, E2E model decides *how* |
| 2024 | **EMMA** | Waymo | `Planner` | Trajectory + objects, as text | Off (research) | Gemini-based; multimodal, unified as language — the strongest "why decompose at all" datapoint |
| 2024 | WOMD-Reasoning | Waymo | `Labeler`/`QA` | Interaction descriptions | Off | Language annotations over motion data |
| 2025 | AlphaDrive | — | `Meta` | Planning decisions | Off | GRPO-style RL post-training applied to a driving VLM |
| 2025 | DriveBench / DriveMLLM | — | `Critic` | Benchmark | Off | Probes whether VLM answers are *grounded* or plausible-sounding |
| 2025 | **Cosmos-Reason1/2** | NVIDIA | `Meta` | Physical common sense, action tokens | Off/`Slow` | Cross-domain (also in `robotics_foundation_models_reference.md` §5) |
| 2025 | Waymo Foundation Model — Driving VLM | Waymo | `Meta` + `Critic` | Semantic reasoning; teacher for distillation | `Slow` + Off | "Think fast / think slow": sensor-fusion encoder + Gemini-derived VLM; teacher→student distillation |
 
**Where language actually pays today** — ranked by how defensible the claim is:
1. **Offline** — auto-labeling, long-tail mining, scenario description and retrieval, failure triage.
   Latency-free, and the value is measurable in data-engine throughput.
2. **Evaluation and critique** — judging realism, flagging anomalous behavior, explaining a
   disengagement. This is where Waymo's stack places it.
3. **Slow-loop meta-decisions** — 1–3 Hz maneuver selection feeding a fast conventional planner.
4. **Onboard closed-loop planner** — still a research claim; latency and grounding are unsolved.
---
 
# 7. Substrate — datasets & simulators
 
| Name | Year | What it anchors |
|---|---|---|
| KITTI | 2012 | The original detection/odometry/stereo benchmark |
| **nuScenes** | 2019 | Full sensor suite incl. radar; detection, prediction, and the *planning* lineage |
| **Waymo Open Dataset** (+ Motion, Sim Agents) | 2019–23 | Scale; marginal *and* joint prediction; sim-agent realism |
| **Argoverse 1 / 2** | 2019/2021 | Motion forecasting; HD map + lane graph focus |
| SemanticKITTI / Occ3D / OpenLane-V2 | 2019/2023/2023 | Segmentation & occupancy / lane topology |
| **CARLA** | 2017 | The open-source sensor simulator; leaderboard closed-loop eval |
| **nuPlan** | 2021 | Closed-loop planning on real logs |
| **Waymax** | 2023 | Fast data-driven agent simulation (JAX) |
| MetaDrive | 2021 | Lightweight procedural scenario generation for RL |
| GAIA-1/2, Vista, DriveDreamer, **Waymo World Model** | 2023–26 | Generative driving world models — sim, counterfactuals, data generation |
 
---
 
# 8. Cross-cutting — the modular interfaces
 
This is the section that matters in an interview; the tables above are recall.
 
### 8a. What the module boundaries actually cost
 
- **Information bottleneck.** Each interface discards what the schema can't express. A 3D box drops
  articulation, pose intent, gaze, and hesitation; a `k`-mode trajectory set drops the tail. Downstream
  modules cannot recover what upstream serialized away — this is the strongest technical argument
  for end-to-end, and it is a real one.
- **Cascading error and objective mismatch.** Detection is trained on mean Average Precision (mAP),
  which weights a distant parked car and a merging truck identically. **PKL** (Planning KL-divergence,
  nuScenes 2020) is the counter-proposal: score perception by how much its errors perturb a planner.
  Worth naming — it's the cleanest statement of "your module metric is not your system metric."
- **Uncertainty propagation.** Most production stacks pass a point estimate plus a score, and the
  score is a classifier confidence, not a calibrated probability. Doing this properly means passing
  distributions across every boundary and having the planner consume them — which is expensive and
  rarely done end to end.
- **Latency stack-up.** Each module adds queue and compute latency; the planner acts on a world that
  is 100–300 ms stale. Compensation (predict-forward to the actuation timestamp) is a system-design
  decision, not a modeling one.
### 8b. What modularity buys (the case that survives)
 
- **Attributable failure analysis.** A disengagement can be assigned to a module and regression-tested
  against a unit benchmark. In a monolith, the equivalent question is answered with saliency maps.
- **Independent validation and certification.** Safety arguments are constructed per-component;
  formal layers (RSS, CBFs, reachability) attach to interpretable interfaces and have nothing to
  attach to in a pixel-to-steering network.
- **Data efficiency per module.** Perception trains on cheap, abundant, *supervisable* labels.
  Planning trains on scarce expert behavior. Fusing them means the whole system inherits the scarcer
  supervision signal.
- **Team scaling.** Independent module ownership is an organizational fact, not a technical one — but
  it is often the actual reason a stack is modular.
### 8c. The convergence — where the boundary is dissolving
 
The honest current state is neither pole:
 
| Pattern | Example | What it preserves |
|---|---|---|
| **Query passing** (differentiable interfaces) | UniAD, VAD | Module identity and auxiliary supervision, but gradients flow end to end |
| **Dual-system / slow-fast** | DriveVLM-Dual, Waymo "think fast/think slow" | A conventional real-time path with a reasoning layer above it |
| **Learned proposal + verified selection** | PDM-Closed, Hydra-MDP, SafetyNet | A learned generator with a rule-based or formal scorer holding the safety argument |
| **Perception output shift** | occupancy, online mapping | Denser, class-agnostic representations that lose less at the boundary |
 
### 8d. Senior phrasing
 
> "The interesting question isn't modular versus end-to-end — it's *where you put the differentiable
> boundary and where you keep a verifiable one*. Every deployed stack I'd trust today still has a
> non-learned safety layer, because that's the only part you can write an argument about. What has
> actually changed is that the interfaces above that layer got richer — queries instead of boxes,
> occupancy instead of object lists — so the information loss at each boundary shrank without giving
> up attributable failure analysis."
 
> "The result I'd point to on the planning side is PDM-Closed: a well-tuned rule-based planner won the
> nuPlan closed-loop challenge outright. The lesson wasn't that learning doesn't work — it was that
> open-loop imitation metrics were measuring the wrong thing, and the field had been optimizing a
> proxy. That's the same structural failure as best-of-`k` displacement error in prediction."
 
### 8e. Questions worth asking a team building one of these
 
- Which interfaces are differentiable, and which are deliberately not? What's the argument for each?
- How is perception evaluated — mAP, or something planner-aware?
- Where does uncertainty actually get consumed rather than thresholded away?
- What's the closed-loop evaluation loop, and how far do you trust simulated agents in it?
- What does the safety layer look like, and can a learned planner override it?
---
 
# 9. Confidence notes
 
- **High confidence** — everything pre-2024. Names, years, and the stated sub-axis values are
  well-established canon.
- **Medium confidence** — 2024–2025 entries. Names and contributions are right; exact publication
  years and venues may be off by one (arXiv vs. conference), and rankings ("winner of X") should be
  re-verified before being asserted in an interview.
- **Lower confidence** — 2025–2026 entries, particularly in §6 (reasoning), where the field is moving
  fastest and where a claim about what is *deployed* versus *published* dates quickly. Treat the
  Waymo Foundation Model row as sourced from Waymo's own blog posts (see `waymo_hm_call_prep.md`),
  not from a peer-reviewed paper.
- Dates are the commonly cited publication year (conference or arXiv, whichever the work is usually
  referenced by).
# 10. TODO / evolving
 
- [ ] Add a **sensor calibration & synchronization** sub-section under §1 (extrinsics degradation,
      online calibration, time-sync failure modes) — it's a common systems probe and currently absent.
- [ ] Add per-metric definitions for the planning benchmarks (nuPlan closed-loop score components,
      PDM score composition) — currently only named.
- [ ] Decide whether **occupancy forecasting** (OccWorld, Cam4DOcc) belongs here or in
      `behavior_prediction_reference.md` §5 — it currently appears in both framings.
- [ ] Cross-link §8 against the sim-to-real argument in `sim_to_real_reference.md` — the
      "verified by a simulator is not verified" probe applies directly to closed-loop planning eval.
- [ ] Expand §5 (control) only if a role warrants it; currently sized as a recognition aid.
 