# Seminal Deep Learning Works in Computer Vision — Quick Reference
 
**Legend — Arch:** `CNN` · `TF` (transformer) · `Hybrid` (CNN backbone + transformer head, or mixed) · `Point` (point-based MLP/sparse-conv) · `Diff` (diffusion)
**Legend — Input:** RGB = monocular camera · Multi-cam = calibrated camera rig · D = depth/RGB-D · PC = point cloud (LiDAR) · Txt = language
 
**On output structure.** Each task section opens with what the networks actually *emit*, which is often not the task output. The useful frame is **emitted tensor → post-processing → task output**. A task feels clean when that middle step is short: classification has nothing there, detection has NMS, tracking has an entire association algorithm, grasping sometimes has a sampling-and-optimization loop. Much of the progress below is the middle column shrinking — DETR deletes NMS, Mask2Former deletes panoptic merge rules, MOTR deletes association.
 
---
 
## 1. Image Classification (backbones)
 
**Output:** logits over a fixed set of K classes, softmax, argmax. No post-processing. The one break in the pattern is CLIP-style models, which emit an *embedding* rather than logits — no fixed head, no fixed label set, and classification happens as similarity against text embeddings computed at inference. That is what "open vocabulary" means mechanically.
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2012 | AlexNet | CNN | RGB | Started the era |
| 2014 | VGG | CNN | RGB | Depth via 3×3 stacks |
| 2015 | GoogLeNet / Inception | CNN | RGB | Multi-branch efficiency |
| 2015 | ResNet | CNN | RGB | Residual connections |
| 2017 | DenseNet | CNN | RGB | Dense feature reuse |
| 2017 | MobileNet (v1–v3) | CNN | RGB | Edge/mobile baseline |
| 2018 | SENet | CNN | RGB | Channel attention |
| 2019 | EfficientNet | CNN | RGB | Compound scaling |
| 2020 | ViT | TF | RGB | Patches as tokens |
| 2021 | CLIP | TF | RGB + Txt | Open-vocabulary pretraining |
| 2021 | DeiT | TF | RGB | ViT without JFT-scale data |
| 2021 | Swin Transformer | TF | RGB | Hierarchical, windowed attn |
| 2021 | MAE | TF | RGB | Masked-image self-supervision |
| 2021 | DINO (v1) | TF | RGB | Self-distillation features |
| 2022 | ConvNeXt | CNN | RGB | CNN modernized to ViT parity |
| 2023 | DINOv2 | TF | RGB | General-purpose frozen features |
 
---
 
## 2. Object Detection — 2D
 
**Output:** a variable-length set of (class, box, score) — reached from three different emitted tensors.
 
- *Dense anchor-based* (Faster R-CNN, SSD, RetinaNet): per-anchor class logits + 4 box deltas over ~100k anchors → NMS.
- *Dense anchor-free* (FCOS, CenterNet): per-location class + distances to box edges, or a center heatmap + size + sub-pixel offset → peak extraction replaces NMS.
- *Set prediction* (DETR family): N fixed queries (100–900), each emitting a class distribution containing an explicit "no object" token, plus a box. Bipartite matching at train time, nothing at inference.

| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2014 | R-CNN | CNN | RGB | Region proposals + CNN |
| 2015 | Fast / Faster R-CNN | CNN | RGB | RoI pooling; learned RPN |
| 2016 | YOLO (v1) | CNN | RGB | Single-stage, real-time |
| 2016 | SSD | CNN | RGB | Multi-scale single-stage |
| 2017 | FPN | CNN | RGB | Feature pyramid neck |
| 2017 | RetinaNet (Focal Loss) | CNN | RGB | Class-imbalance fix |
| 2017 | Mask R-CNN | CNN | RGB | Instance seg via extra head |
| 2019 | FCOS / CenterNet | CNN | RGB | Anchor-free |
| 2020 | DETR | Hybrid | RGB | Set prediction, no NMS |
| 2021 | Deformable DETR | Hybrid | RGB | Sparse attn, fast convergence |
| 2022 | DINO-DETR | Hybrid | RGB | DETR family reaches SOTA |
| 2023 | Grounding DINO | Hybrid | RGB + Txt | Open-vocabulary detection |
 
## 2b. Object Detection — 3D / Multi-Sensor
 
**Output:** the space changes outright — 7-DoF boxes (x, y, z, l, w, h, yaw), often 9-DoF with velocity so tracking downstream gets a motion prior for free. Yaw is predicted as bin + residual or as (sin, cos), never as a raw angle, because angle regression is discontinuous at the wrap-around.
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2017 | MV3D / AVOD | CNN | Multi-cam + PC | Early camera–LiDAR fusion |
| 2017 | PointNet / PointNet++ | Point | PC | Raw point-cloud learning |
| 2018 | VoxelNet | CNN | PC | End-to-end voxel encoding |
| 2018 | SECOND | CNN | PC | Sparse 3D convolution |
| 2018 | Frustum PointNet | Point | RGB + PC | 2D proposal → 3D frustum |
| 2019 | PointPillars | CNN | PC | Pillar encoding, fast |
| 2019 | PointRCNN | Point | PC | Two-stage point-based |
| 2020 | LSS (Lift-Splat-Shoot) | CNN | Multi-cam | Camera → BEV lifting |
| 2021 | CenterPoint | CNN | PC | Anchor-free 3D + tracking |
| 2022 | BEVFormer | Hybrid | Multi-cam | Spatiotemporal BEV queries |
| 2022 | PETR | Hybrid | Multi-cam | 3D position-aware queries |
| 2022 | TransFusion | Hybrid | Multi-cam + PC | Attention-based fusion |
| 2022 | BEVFusion | Hybrid | Multi-cam + PC | Unified BEV fusion space |
 
---
 
## 3. Semantic Segmentation
 
**Output:** H×W×K logits, per-pixel argmax — true for the FCN → DeepLab → SegFormer lineage. Two breaks: *mask classification* (MaskFormer/Mask2Former) emits N binary masks plus N class labels, with the per-pixel map derived by combining them; and SAM emits class-agnostic masks plus a predicted IoU score, three per prompt to represent ambiguity, with no semantics at all.
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2015 | FCN | CNN | RGB | Fully convolutional dense pred |
| 2015 | U-Net | CNN | RGB | Encoder–decoder + skips |
| 2015 | SegNet | CNN | RGB | Pooling-index upsampling |
| 2016–18 | DeepLab v1–v3+ | CNN | RGB | Atrous conv, ASPP, CRF |
| 2017 | PSPNet | CNN | RGB | Pyramid pooling context |
| 2019 | HRNet | CNN | RGB | High-res throughout |
| 2021 | SETR | TF | RGB | ViT as seg encoder |
| 2021 | Segmenter | TF | RGB | Mask-transformer decoder |
| 2021 | SegFormer | TF | RGB | Hierarchical TF + MLP head |
| 2023 | SAM | TF | RGB + prompt | Promptable, class-agnostic |
| 2024 | SAM 2 | TF | RGB video + prompt | Memory-based video tracking |
 
**Point-cloud semantic segmentation**
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2017 | SqueezeSeg | CNN | PC (range image) | Projection-based |
| 2019 | RangeNet++ | CNN | PC (range image) | Real-time LiDAR seg |
| 2019 | MinkowskiNet | CNN (sparse) | PC | 4D sparse conv |
| 2021 | Cylinder3D | CNN (sparse) | PC | Cylindrical partition |
 
---
 
## 4. Panoptic Segmentation
 
**Output:** the best-defined task output of the six — per-pixel (semantic class, instance id), with instance id null for stuff — because PQ forced the definition before architectures settled. The emitted tensors still diverge sharply:
 
- *Top-down* (Panoptic FPN, UPSNet): instance masks + semantic map, merged by explicit conflict-resolution rules.
- *Bottom-up* (Panoptic-DeepLab): semantic map + instance-center heatmap + per-pixel offset-to-center; grouping is nearest-center assignment.
- *Mask transformer* (MaX-DeepLab, Mask2Former): N (mask, class) pairs; per-pixel labels are an argmax over masks.
That third row is why mask classification unified semantic, instance, and panoptic under one model — one output structure covers all three, and the task only changes what you do with it. Video panoptic adds a temporally consistent instance id, which is a tracking output living inside a segmentation task.
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2019 | Panoptic Segmentation (Kirillov) | — | RGB | Task + PQ metric |
| 2019 | Panoptic FPN | CNN | RGB | Shared-backbone baseline |
| 2019 | UPSNet | CNN | RGB | Parameter-free panoptic head |
| 2020 | Panoptic-DeepLab | CNN | RGB | Bottom-up, box-free |
| 2020 | Axial-DeepLab | TF | RGB | Axial self-attention |
| 2021 | MaX-DeepLab | Hybrid | RGB | First end-to-end mask TF |
| 2021 | MaskFormer | Hybrid | RGB | Per-mask classification |
| 2021 | K-Net | Hybrid | RGB | Dynamic kernels, unified |
| 2022 | Mask2Former | Hybrid | RGB | Masked attn; universal seg |
| 2022 | Panoptic SegFormer | Hybrid | RGB | DETR-style panoptic |
| 2023 | OneFormer | Hybrid | RGB + Txt | One model, three seg tasks |
 
**Video / LiDAR panoptic**
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2020 | VPSNet | CNN | RGB video | Defines VPQ |
| 2021 | ViP-DeepLab | CNN | RGB video | Depth-aware video panoptic |
| 2021 | Panoptic-PolarNet | CNN | PC | LiDAR panoptic |
| 2022 | TubeFormer-DeepLab | TF | RGB video | Tube-level mask prediction |
| 2023 | TarViS | Hybrid | RGB video | Unified video seg targets |
| 2023 | Video-kMaX | Hybrid | RGB video | Near/long-term tracking split |
 
---
 
## 5. Tracking
 
**Output:** a set of tracklets — (track_id, box) over time. But nearly every model emits *per-frame* quantities and assigns identity by a separate mechanism, while the metrics (MOTA, IDF1, HOTA) score the whole sequence. That mismatch between what is supervised and what is measured is the defining awkwardness of the task.
 
The real discriminator is **where identity lives**:
 
| Approach | Emitted per frame | Where ID comes from |
|---|---|---|
| SORT / ByteTrack / OC-SORT | Boxes only | External Kalman filter + Hungarian on IoU |
| DeepSORT / JDE / FairMOT | Boxes + ReID embedding | Cosine distance in embedding space |
| CenterTrack | Boxes + 2D displacement per center | Greedy match using the predicted offset |
| MOTR / TrackFormer | Box per query | Implicit — the query slot *is* the identity |
 
The last row is the structural payoff of the DETR lineage: no association signal is emitted because there is nothing to associate. Two further output variants below — VOS/mask tracking holds state in an explicit memory bank rather than queries, and point tracking adds a channel nothing else has: a per-frame **visibility/occlusion flag** alongside (x, y). Predicting where a point is while it is hidden is a distinct capability, and it is the part that matters for manipulation policies built on point tracks.
 
**2D multi-object tracking**
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2016 | SORT | Detector + Kalman | RGB | Motion-only association baseline |
| 2017 | DeepSORT | CNN (ReID) | RGB | Appearance embedding for association |
| 2019 | Tracktor | CNN | RGB | Detector regression head as tracker |
| 2020 | JDE | CNN | RGB | Joint detection + embedding |
| 2020 | CenterTrack | CNN | RGB | Point tracking via frame-to-frame offsets |
| 2021 | FairMOT | CNN | RGB | Anchor-free, resolves det/ReID conflict |
| 2022 | ByteTrack | Detector + assoc | RGB | Associate low-confidence boxes too |
| 2022 | TrackFormer / MOTR | TF | RGB | Track queries, end-to-end, no heuristics |
| 2022 | BoT-SORT | Detector + assoc | RGB | Camera-motion-compensated SORT |
| 2023 | OC-SORT | Detector + filter | RGB | Observation-centric occlusion recovery |
 
**Mask / video object segmentation tracking**
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2019 | STM | CNN + memory | RGB video | Space-time memory read |
| 2022 | XMem | CNN + memory | RGB video | Long-video hierarchical memory |
| 2024 | Cutie | Hybrid | RGB video | Object-level memory reading |
| 2024 | SAM 2 | TF | RGB video + prompt | Promptable, unified seg + tracking |
 
**3D / multi-sensor tracking**
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2020 | AB3DMOT | Detector + Kalman | PC | 3D baseline + standard metrics |
| 2021 | CenterPoint (track head) | CNN | PC | Velocity-based greedy association |
| 2021 | EagerMOT | Fusion + filter | Multi-cam + PC | Two-stage camera–LiDAR association |
| 2021 | SimpleTrack | Detector + filter | PC | Ablates what actually matters |
| 2022 | MUTR3D | TF | Multi-cam | 3D track queries, end-to-end |
| 2023 | PF-Track | TF | Multi-cam | Past-and-future query propagation |
 
**Dense / point correspondence** (adjacent, increasingly used for manipulation policies)
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2015 | FlowNet | CNN | RGB pair | First learned optical flow |
| 2018 | PWC-Net | CNN | RGB pair | Pyramid, warping, cost volume |
| 2020 | RAFT | CNN (recurrent) | RGB pair | Iterative all-pairs correlation |
| 2022 | PIPs / TAP-Vid | CNN | RGB video | Any-point tracking task + benchmark |
| 2023 | TAPIR | Hybrid | RGB video | Matching + iterative refinement |
| 2023 | CoTracker | TF | RGB video | Joint tracking of correlated points |
 
---
 
## 6. Grasp Planning
 
**Output:** the most heterogeneous of the six, fragmenting along an axis that exists nowhere else — **generator vs. evaluator**. Dex-Net's GQ-CNN takes a crop already aligned to a *candidate* grasp and emits a single scalar (success probability); it generates nothing, and sampling happens outside the network via CEM. QT-Opt has the same shape with a Q-value. 6-DOF GraspNet splits the roles explicitly: a VAE generates SE(3) poses, a separate evaluator scores and gradient-refines them. GG-CNN is the opposite — dense per-pixel (quality, angle, width) maps in one forward pass, argmax.
 
The second axis is the gripper, which defines the output space and has no canonical form:
 
| Parameterization | DoF | Examples |
|---|---|---|
| Planar grasp rectangle (x, y, θ, w) | 4 | Lenz, GG-CNN |
| Suction: contact point + surface normal | 5 | Dex-Net 3.0/4.0 |
| 6-DoF pose + gripper width | 7 | 6-DOF GraspNet, VGN, AnyGrasp |
| Contact-anchored (approach + baseline + width, per point) | 6, constrained | Contact-GraspNet |
| Wrist pose + hand joint configuration | 6 + ~20 | DexGraspNet, UniDexGrasp |
 
Contact-GraspNet's parameterization is the clever one: anchoring at observed points reduces the 6-DoF search to something dense over the point cloud instead of sampled. And some rows below emit no grasp at all — VGN emits a voxel grid where each voxel carries (quality, quaternion, width); Transporter Networks emits a pick affordance map plus a spatial correlation map over placements, i.e. SE(2) *actions*. Once you are in action space the boundary with manipulation policy learning stops being crisp.
 
This fragmentation is why grasping has no PQ or mAP equivalent: success is measured by physical execution, with no agreed intermediate representation to score against.
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2015 | Lenz, Lee & Saxena | CNN | RGB-D | First deep grasp detection |
| 2016 | Levine et al. (hand-eye) | CNN | RGB | Large-scale real-robot grasping |
| 2017 | Dex-Net 2.0 (GQ-CNN) | CNN | D | Synthetic data + grasp quality |
| 2018 | GG-CNN | CNN | D | Pixelwise, closed-loop, real-time |
| 2018 | QT-Opt | CNN | RGB | Off-policy RL grasping |
| 2019 | Dex-Net 4.0 | CNN | D | Ambidextrous (suction + jaw) |
| 2019 | 6-DOF GraspNet | Point | PC | VAE over 6-DoF grasps |
| 2020 | VGN | CNN (3D) | TSDF from D | Volumetric, clutter, one shot |
| 2020 | GraspNet-1Billion | Point | RGB-D / PC | Benchmark + baseline |
| 2020 | Transporter Networks | CNN | RGB-D (top-down) | Pick-and-place as spatial matching |
| 2021 | Contact-GraspNet | Point | D / PC | Grasps from contact points |
| 2021 | CLIPort | Hybrid | RGB-D + Txt | Semantics + spatial precision |
| 2023 | AnyGrasp | Point | PC | Dense grasps, dynamic scenes |
| 2023 | GraspNeRF / neural-field grasping | Hybrid | Multi-cam | Transparent/specular objects |
| 2023 | DexGraspNet / UniDexGrasp | Point | PC | Dexterous multi-finger hands |
 
---
 
## 7. Depth Prediction
 
**Output:** per-pixel scalar — but the scalar means different things, and the differences are load-bearing. Metric depth (meters), relative depth (ordering only), affine-invariant inverse depth (MiDaS: scale *and* shift unknown, which is exactly what permits mixed-dataset training), disparity, ordinal bins (DORN), adaptive per-image bins (AdaBins). LSS is the sharpest case: it emits a full categorical *distribution* over depth bins per pixel rather than a point estimate, and that distribution is what makes the lift-to-BEV operation work — the output structure is not incidental to the method, it *is* the method.
 
**Monocular — supervised**
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2014 | Eigen et al. | CNN | RGB | First deep monodepth |
| 2016 | FCRN (Laina) | CNN | RGB | Fully conv residual up-projection |
| 2018 | DORN | CNN | RGB | Ordinal regression framing |
| 2019 | BTS | CNN | RGB | Local planar guidance |
| 2021 | AdaBins | Hybrid | RGB | Adaptive depth bins |
| 2021 | DPT | TF | RGB | ViT for dense prediction |
 
**Monocular — self-supervised**
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2017 | Monodepth (Godard) | CNN | Stereo pairs (train) | Left-right consistency |
| 2017 | SfMLearner | CNN | RGB video | Joint depth + ego-motion |
| 2019 | Monodepth2 | CNN | RGB video | Min-reproj + auto-masking |
 
**Zero-shot / foundation depth**
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2020 | MiDaS | CNN → Hybrid | RGB | Mixed-dataset relative depth |
| 2023 | ZoeDepth | Hybrid | RGB | Relative → metric |
| 2023 | Metric3D (v1/v2) | TF | RGB | Cross-camera metric depth |
| 2024 | Depth Anything (v1/v2) | TF | RGB | Large-scale pseudo-labeling |
| 2024 | Marigold | Diff | RGB | Diffusion prior repurposed |
| 2024 | UniDepth | TF | RGB | Camera-intrinsics-aware metric |
| 2024 | Depth Pro | TF | RGB | Sharp, fast, metric, zero-shot |
 
**Stereo & depth completion**
 
| Year | Work | Arch | Inputs | Note |
|---|---|---|---|---|
| 2017 | GC-Net | CNN (3D) | Stereo | Learned cost volume |
| 2018 | PSMNet | CNN (3D) | Stereo | Pyramid + stacked hourglass |
| 2018 | Sparse-to-Dense | CNN | RGB + sparse PC | Depth completion baseline |
| 2020 | NLSPN | CNN | RGB + sparse PC | Non-local spatial propagation |
| 2021 | RAFT-Stereo | CNN (recurrent) | Stereo | Iterative refinement |
 
---
 
## Cross-cutting trend
 
Reading the tables top-to-bottom, the same three transitions repeat in every task: **CNN → hybrid (CNN backbone + transformer decoder) → pure transformer**, **task-specific heads → unified mask/set prediction** (Mask2Former, OneFormer, DETR family), and **closed-vocabulary supervised training → foundation models with open vocabulary or zero-shot transfer** (CLIP, SAM, DINOv2, Depth Anything).
 
Tracking shows a fourth transition specific to it: **hand-designed association → learned association → no association at all**, as identity moves from a Hungarian-matching step into a persistent transformer query. The DETR-family lineage is what makes that possible, which is why tracking sits in this document rather than in a separate one.
 
Grasp planning is the outlier — point-based architectures still dominate there, since geometry rather than semantics carries most of the signal.
 
**Scope note.** Every task here maps raw sensor observations to structured output. Behavior/trajectory prediction takes already-abstracted input (agent tracks + map context) and needs a different column schema and metric vocabulary, so it lives in a companion document: `behavior_prediction_reference.md`.
 
*Dates are the commonly cited publication year (conference or arXiv, whichever the work is usually referenced by).*
 