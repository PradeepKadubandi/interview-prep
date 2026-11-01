# Robotics Foundation Models — Seminal Works Reference
 
Scan-first reference. No method explanations; name + sub-axis values only.
Currency: literature through ~July 2026.
 
---
 
## The axes (proposed)
 
Your two axes (VLA, World Model) are the right center of gravity, but the field has split the
world-model side in two, and there are three adjacent families that a "robotics foundation model"
question will reach for. Six families, organized by **what the pretrained backbone is** and
**what the model emits**:
 
| # | Family | Backbone origin | Emits | Why it's a separate axis |
|---|---|---|---|---|
| 1 | **Pre-foundation policy ancestors** | scratch | actions | Established chunking, diffusion heads, tokenized actions — every family inherits |
| 2 | **VLAs** | VLM (image-text) | actions | The dominant recipe |
| 3 | **WAMs** (World-Action Models) | video / world model | video + actions | The 2026 second bet — *not* a VLA subtype, different pretraining prior |
| 4 | **World models** (prediction-only) | video / latent dynamics | future states | Never emits motor commands; used for sim, planning, eval, data gen |
| 5 | **Embodied reasoning models** | VLM/LLM | plans, points, trajectories, subgoals | The "System 2" layer; sold separately (Gemini ER, Cosmos-Reason) |
| 6 | **Representation backbones** | self-supervised vision | features | The frozen substrate the other five sit on |
 
Plus a **substrate** section (data, benchmarks, sim) — not models, but the seminal works there are
cited as often as the models.
 
**Sub-axes differ by family**, so columns differ per table. That's deliberate: forcing one column
set across all six flattens the distinctions that matter.
 
---
 
## Legend
 
**Action representation** — `Disc` discrete/tokenized bins · `Flow` flow matching · `Diff` diffusion ·
`VQ` vector-quantized codebook · `Reg` direct regression · `Latent` latent action/plan
**Action space** — `EE` end-effector SE(3) · `Jnt` joint · `Δ` relative/delta · `Abs` absolute · `WB` whole-body
**Inputs** — `RGB` camera · `MC` multi-cam · `Txt` language · `Prop` proprioception · `D` depth ·
`PC` point cloud · `Tac` tactile · `Vid` video history
**Arch** — `Mono` monolithic · `MoT` mixture-of-transformers · `Dual` fast/slow two-system · `Hier` hierarchical
 
---
 
## 1. Pre-foundation policy ancestors
 
| Year | Work | Org | Action repr | Action space | Inputs | Note |
|---|---|---|---|---|---|---|
| 2018 | World Models (Ha & Schmidhuber) | Google Brain | Reg | — | RGB | The original VAE+RNN dream-policy |
| 2019 | Play-LMP / Latent Plans | Google | Latent | Jnt | RGB, Prop | Prior/posterior latent plan — reappears in 2026 |
| 2021 | BC-Z | Google | Reg | EE Δ | RGB, Txt | Language-conditioned BC at scale |
| 2022 | RT-1 | Google | Disc | EE Δ + base | RGB, Txt | First tokenized-action scale anchor |
| 2023 | ACT / ALOHA | Stanford | Reg (CVAE) | Jnt Abs | MC, Prop | Action chunking + temporal ensembling |
| 2023 | Diffusion Policy | Columbia | Diff | EE/Jnt pos | RGB, Prop | Diffusion action head; position > velocity |
| 2023 | RT-X / Open X-Embodiment | 21-lab consortium | Disc | EE Δ | RGB, Txt | Cross-embodiment training is viable |
| 2024 | Octo | Berkeley | Diff | EE Δ | MC, Txt | Open generalist policy, OXE-trained |
| 2024 | UMI | Stanford | Diff | EE Δ SE(3) | RGB | Handheld data collection → policy |
| 2024 | VQ-BeT | KAIST/NYU | VQ | EE | RGB | The vector-quantized third path |
 
---
 
## 2. VLAs — VLM-backbone policies
 
| Year | Model | Org | Backbone | Action repr | Action space | Inputs | Arch | Open |
|---|---|---|---|---|---|---|---|---|
| 2023 | **RT-2** | Google DM | PaLI-X / PaLM-E | Disc | EE Δ | RGB, Txt | Mono | ✗ |
| 2024 | **OpenVLA** | Stanford/Berkeley | Llama-2 + SigLIP+DINOv2 | Disc | EE Δ | RGB, Txt | Mono | ✓ |
| 2024 | **RDT-1B** | Tsinghua | SigLIP + T5-XXL | Diff | Unified EE | MC, Txt, Prop | Mono | ✓ |
| 2024 | **π0** | Physical Intelligence | PaliGemma-3B + 300M expert | Flow | Jnt / EE Δ | MC, Txt, Prop | MoT | ✓ (openpi) |
| 2025 | **Helix** | Figure AI | internet-pretrained VLM | Reg (200Hz S1) | WB upper body | MC, Txt, Prop | Dual | ✗ |
| 2025 | **Gemini Robotics / -ER** | Google DM | Gemini 2.0 | Disc | EE | MC, Txt | Dual | ✗ (ER via API) |
| 2025 | **π0.5** | Physical Intelligence | PaliGemma + KI | Flow | Jnt / EE Δ | MC, Txt, Prop | MoT | ✓ |
| 2025 | **GR00T N1 / N1.5** | NVIDIA | Eagle-2 VLM (1.34B) | Diff (DiT) | per-embodiment | MC, Txt, Prop | Dual | ✓ |
| 2025 | **OpenVLA-OFT** | Stanford | OpenVLA | Disc + parallel decode | EE Δ | MC, Txt, Prop | Mono | ✓ |
| 2025 | **π-FAST** | Physical Intelligence | PaliGemma | Disc (FAST) | EE Δ | MC, Txt, Prop | Mono | ✓ |
| 2025 | **SmolVLA** | Hugging Face | SmolVLM | Flow | Jnt | RGB, Txt, Prop | MoT | ✓ |
| 2025 | **π0.6** | Physical Intelligence | π0.5 + RL | Flow | Jnt / EE Δ | MC, Txt, Prop | MoT | ✗ |
| 2025 | **GR00T N1.7** | NVIDIA | Cosmos-Reason2-2B | Diff (32-layer DiT) | per-embodiment | MC, Txt, Prop | Dual | ✓ |
| 2026 | **Being-H0.5** | BeingBeyond | InternVL3.5 | Flow | cross-embodiment | MC, Txt, Prop | MoT | ✓ |
| 2026 | **Xiaomi-Robotics-0** | Xiaomi | VLM + co-train | Disc + Flow | Jnt / EE | MC, Txt, Prop | MoT | ✓ |
| 2026 | **π0.7** | Physical Intelligence | π-family + BAGEL subgoals | Flow | Jnt / EE Δ | MC, Txt, Prop, subgoal img | Hier | ✗ |
| 2026 | **MolmoAct2** | AI2 | Molmo2-ER | Disc (OpenFAST) | EE Δ | MC, Txt | Mono | ✓ |
| 2026 | **VLA Foundry** | TRI | Foundry-LLM→VLM (1–2B) | Flow | Jnt / EE | MC, Txt, Prop | MoT | ✓ |
| 2026 | **GEN-1** | Generalist AI | robotics-first (from scratch) | — | EE Δ | MC, Txt, Prop | — | ✗ |
| 2026 | **Gemini Robotics 2** | Google DM | Gemini | — | **WB** (legs+torso+arms+fingers) | MC, Txt, Prop | Dual (VLA + ER 2) | ✗ (ER 2 via API) |
 
**Cross-cutting VLA sub-axis — action tokenizers/interfaces:**
FAST (2025, DCT-based) · BEAST (2025, B-spline) · LAPA (2025, latent actions from unlabeled video) ·
OpenFAST (2026) · Genie-style latent action codes.
 
**Cross-cutting VLA sub-axis — modality extensions:**
`3D`: SpatialVLA, PointVLA, 3D-VLA, 4D-VLA · `Tactile`: ForceVLA, Tactile-VLA, VLA-Touch ·
`Reasoning`: ECoT, CoT-VLA, DreamVLA · `Efficiency`: RoboMamba (SSM), CogACT/HiMoE-VLA (MoE) ·
`Cross-embodiment`: X-VLA, UniVLA, XR-1, CrossFormer.
 
---
 
## 3. WAMs — World-Action Models (video-backbone policies)
 
Sub-axes follow the now-standard three: **paradigm** (what it predicts), **action integration**
(how actions enter), **architecture**.
 
| Year | Model | Org | Paradigm | Action integration | Video backbone | Arch |
|---|---|---|---|---|---|---|
| 2023 | **UniPi** | MIT/Google | Inverse dynamics | Action tokens | CNN video diffusion (1.7B, scratch) | Hier |
| 2024 | **GR-1** | ByteDance | Joint prediction | Action tokens | GPT-2-style (scratch) | Mono |
| 2024 | **GR-2** | ByteDance | Joint prediction | Action tokens | video-pretrained transformer | Mono |
| 2024 | **GENIMA** | Dyson RL Lab | Inverse dynamics | **Action-as-image** | Stable Diffusion + ControlNet | Hier |
| 2025 | **VPP** (Video Prediction Policy) | Tsinghua | Inverse dynamics | Action tokens | Stable Video Diffusion | Hier |
| 2025 | **Seer** | — | Inverse dynamics | Action tokens | transformer over vis/act tokens | Mono |
| 2025 | **UWM** (Unified World Model) | — | Joint prediction | Action tokens | joint video-action diffusion | Mono |
| 2025 | **UVA** (Unified Video Action) | MIT | Joint prediction | Action tokens | video diffusion | Mono |
| 2025 | **mimic-video** | Mimic Robotics | Inverse dynamics | Action tokens | Cosmos | Hier |
| 2025 | **Motus** | — | Latent world-action | Latent actions | unified latent WM | MoT |
| 2026 | **DreamZero** | NVIDIA | Joint prediction | Action tokens | Wan 2.1-I2V-14B | Mono DiT |
| 2026 | **LingBot-VA** | Ant Group | Inverse dynamics | Action tokens | Wan 2.2-5B (16k hr pretrain) | MoT |
| 2026 | **Cosmos Policy** | NVIDIA | Joint prediction | **Action-as-image** (latent frames) | Cosmos | Mono DiT |
| 2026 | **Fast-WAM** | — | **Representation-only** | Action tokens | Wan 5.5B | MoT |
| 2026 | **Being-H0.7** | BeingBeyond | Joint (latent) | Latent plans (prior/posterior) | V-JEPA 2.1 + MoT | MoT |
| 2026 | **DVA** (Direct Video Action) | Rhoda AI | Inverse dynamics | Action tokens | causal video model | Hier |
| 2026 | **DiT4DiT** | — | Inverse dynamics | Action tokens | video DiT | Hier |
| 2026 | **BagelVLA** | — | Hybrid VLA+WAM | Interleaved V-L-A | BAGEL | MoT |
| 2026 | **Cortex 2.0** | Sereact | Foresight-as-planning | Scored latent rollouts | latent world model | Hier |
| 2026 | **VLA-JEPA** | — | Latent prediction | Action tokens | V-JEPA | MoT |
 
---
 
## 4. World models — prediction only (no motor commands)
 
| Year | Work | Org | Output | Conditioning | Domain |
|---|---|---|---|---|---|
| 2018 | World Models | Google Brain | latent rollout | action | toy control |
| 2019 | PlaNet | Google | latent | action | MBRL |
| 2020–23 | Dreamer v1/v2/v3 | Google DM | latent | action | MBRL, general |
| 2020 | MuZero | DeepMind | latent value/reward | action | discrete planning |
| 2023 | TD-MPC2 | UCSD | latent | action | continuous control |
| 2024 | **Genie** | Google DM | 2D frames | **learned latent actions** | web video |
| 2024 | Genie 2 | Google DM | 3D-consistent frames | latent action | general |
| 2025 | **Cosmos** (Predict/Transfer/Reason) | NVIDIA | video + latent | text, action, layout | physical AI |
| 2025 | **Wan / Wan 2.2** | Alibaba | video (latent DiT) | text, image | general (the WAM backbone) |
| 2025 | **V-JEPA 2** | Meta | **latent** (no pixels) | action | video understanding + planning |
| 2025 | Veo 3 / 3.1 | Google DM | video | text, image | general |
| 2025 | Genie 3 | Google DM | interactive 720p@24fps | action, text | general-purpose WM |
| 2026 | **Waymo World Model** | Waymo | multi-sensor (cam + lidar) | driving action, scene layout, language | autonomous driving |
| 2026 | **DreamDojo** | NVIDIA | video | continuous latent actions from egocentric video | generalist robot WM |
| 2026 | V-JEPA 2.1 | Meta | dense latent features | — | video SSL |
| 2026 | Summer-22B | — | video | text | video FM reference scale |
 
Adjacent driving-domain world models worth knowing by name: GAIA-1/2 (Wayve), Vista, DriveDreamer.
 
---
 
## 5. Embodied reasoning / high-level planners
 
Emit plans, points, trajectories, or subgoal images — **not** motor commands.
 
| Year | Work | Org | Output form | Inputs |
|---|---|---|---|---|
| 2022 | SayCan | Google | skill selection (affordance × LLM score) | Txt, skill values |
| 2022 | Inner Monologue | Google | replanning text loop | RGB, Txt, feedback |
| 2022 | Code as Policies | Google | executable Python | RGB, Txt |
| 2023 | PaLM-E | Google | multimodal plan text | MC, Txt, state |
| 2023 | VoxPoser | Stanford | 3D value maps | RGB-D, Txt |
| 2024 | RoboPoint | UW/NVIDIA | 2D affordance points | RGB, Txt |
| 2024 | ECoT | Berkeley | embodied chain-of-thought traces | RGB, Txt |
| 2025 | Molmo / MolmoAct | AI2 | pointing + action reasoning | RGB, Txt |
| 2025 | Cosmos-Reason1 | NVIDIA | physical-commonsense reasoning | Vid, Txt |
| 2025 | Gemini Robotics-ER 1.5 | Google DM | points, trajectories, plans | MC, Txt |
| 2026 | Cosmos-Reason2 | NVIDIA | action tokens for task decomposition | Vid, Txt |
| 2026 | **Gemini Robotics-ER 2** | Google DM | spatial plans, multi-robot coordination | MC, Txt |
 
---
 
## 6. Representation backbones (the frozen substrate)
 
| Year | Work | Org | Objective | Modality | Robotics role |
|---|---|---|---|---|---|
| 2021 | CLIP | OpenAI | contrastive image-text | RGB+Txt | semantic features |
| 2022 | R3M | Stanford | time-contrastive + language on Ego4D | RGB | first robot-purposed encoder |
| 2022 | MVP | Berkeley | MAE on ego video | RGB | masked pretraining for control |
| 2023 | Voltron | Stanford | language-driven visual rep | RGB+Txt | — |
| 2023 | VC-1 | Meta | MAE on Ego4D+ | RGB | CortexBench reference |
| 2023 | DINOv2 | Meta | self-distillation | RGB | geometric/spatial features |
| 2023 | SigLIP | Google | sigmoid image-text | RGB+Txt | the default VLA vision encoder |
| 2023 | SAM / SAM 2 | Meta | promptable segmentation | RGB, Vid | open-set masks |
| 2024 | Theia | Boston Dynamics AI | multi-teacher distillation | RGB | compact robot encoder |
| 2025 | V-JEPA 2 | Meta | latent video prediction | Vid | planning-capable features |
 
---
 
## 7. Substrate — data, benchmarks, simulators
 
**Datasets**
 
| Year | Dataset | Scale / note |
|---|---|---|
| 2022 | Ego4D | 3.6k hr egocentric human video |
| 2023 | BridgeData V2 | 60k trajectories, low-cost arms |
| 2023 | **Open X-Embodiment** | 22 embodiments, 60 datasets — the aggregation moment |
| 2024 | **DROID** | 50k+ demos, Franka, in-the-wild |
| 2024 | AgiBot World | large-scale Chinese cross-embodiment |
| 2025 | EgoDex | egocentric + hand pose |
| 2026 | Action100M | large-scale video action dataset |
| 2026 | RoVid-X | embodied video-generation training set |
 
**Benchmarks**
 
| Year | Benchmark | What it isolates |
|---|---|---|
| 2020 | Meta-World / RLBench | multi-task manipulation |
| 2022 | CALVIN | long-horizon language-conditioned |
| 2023 | LIBERO | lifelong / knowledge transfer |
| 2024 | SimplerEnv | sim-real correlated eval |
| 2024 | RoboCasa | large-scale household sim |
| 2025 | RoboTwin 2.0 | bimanual, domain randomization |
| 2025 | **RoboArena** | distributed **real-world** open-ended eval (the elo leaderboard) |
| 2025 | RoboChallenge | large-scale real-robot eval |
| 2026 | RoboLab | high-fidelity generalist-policy analysis |
| 2026 | MolmoSpaces | anti-benchmaxxing nav + manipulation |
| 2026 | ManipArena | reasoning-oriented generalist manipulation |
 
**Simulators:** MuJoCo · Isaac Sim / Isaac Lab · ManiSkill · Genesis · NVIDIA Newton · Waymax (driving)
 
---
 
## The one-line state of play (July 2026)
 
Two competing pretraining bets — **VLM backbone (VLA)** vs **video backbone (WAM)** — with a third
"robotics-first from scratch" path (GEN-1) blocked for most teams by data access. Consensus
architecture across both is **Mixture-of-Transformers with a gradient-isolated action expert**.
The likely resolution is hybrids: π0.7's world-model subgoals inside a VLA stack, Being-H0.7's
latent prior/posterior bridge, Motus/BagelVLA-style unified towers.
 
---
 
## Confidence notes
 
- **Rows through 2025** are well-established and safe to cite from memory.
- **2026 rows** are compiled from the NVIDIA/Reuss WAM survey (June 2026), the NTU *World Model for
  Robot Learning* survey (2605.00080), and the WAM survey (2605.12090). Names, orgs, and axis values
  are reliable; exact parameter counts and dates for the newest entries are not independently verified here.
- **Gemini Robotics 2** released July 30, 2026 — whole-body humanoid under one policy; ER 2 public
  via Gemini API, VLA and On-Device 2 gated behind Trusted Tester. Too new to have settled details.
- Blank cells mean "not reliably known," not "not applicable."
 