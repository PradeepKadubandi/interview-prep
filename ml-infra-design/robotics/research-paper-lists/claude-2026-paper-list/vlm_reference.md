# Vision-Language Models — Seminal Works Reference
 
Scan-first reference. No method explanations; name + sub-axis values only.
Currency: literature through ~July 2026.
 
**Where this sits:** picks up where `seminal_cv_models_reference.md` ends (§1 here *is* that doc's
classification backbones, re-cut by pretraining objective) and hands off to
`robotics_foundation_models_reference.md` (§8 here is the VLA backbone layer).
 
---
 
## The axes (proposed)
 
The obvious axis — "big VLM that answers questions about images" — collapses six genuinely
different design problems into one row. Organizing by **what the model emits** separates them
cleanly, and the emitted object is what determines the sub-axes that matter:
 
| # | Family | Emits | Why it's a separate axis |
|---|---|---|---|
| 1 | **Encoders & contrastive aligners** | embeddings | No LLM; the substrate every other family sits on |
| 2 | **Generative VLMs** | text | The main line; sub-axes are connector, resolution, fusion depth |
| 3 | **Grounded / region-level** | boxes, points, masks | Output is spatial, not linguistic — different heads, different data |
| 4 | **Unified understanding + generation** | text **and** pixels | Two objectives in one backbone; the tokenizer question is the whole design |
| 5 | **Video VLMs** | text over temporal input | Token budget and positional encoding are the binding constraints, not semantics |
| 6 | **Document / GUI / agentic** | structured text, UI actions | Resolution and OCR fidelity dominate; benchmarks barely overlap with §2 |
| 7 | **Spatial / embodied** | points, trajectories, affordances | The bridge to VLA — 3D and physical reasoning, not scene description |
 
Efficiency (§2b) and reasoning/thinking (§2c) are **cross-cutting sub-axes**, not families —
they appear as variants inside several of the above.
 
---
 
## Legend
 
**Connector** — `X-attn` gated cross-attention · `Q-Former` querying transformer · `MLP` linear/MLP
projector · `Resampler` perceiver resampler · `PixShuf` pixel shuffle · `Expert` per-layer visual
expert · `None` early fusion, no separate encoder
**Resolution** — `Fix` fixed square · `Tile` AnyRes/dynamic tiling · `Native` native variable
resolution (NaViT-style) · `Raw` raw patches, no encoder
**Fusion** — `Late` encoder → connector → LLM · `Deep` fusion inside LLM layers · `Early` single
weight set trained multimodally from the start
**Open** — `W` weights · `WD` weights + data · `API` hosted only
 
---
 
## 1. Encoders & contrastive aligners
 
The vision tower. Choice here propagates into every downstream family.
 
| Year | Work | Org | Objective | Note |
|---|---|---|---|---|
| 2021 | **CLIP** | OpenAI | InfoNCE image-text | The alignment moment |
| 2021 | ALIGN | Google | InfoNCE, noisy web alt-text | Scale over curation |
| 2021 | LiT | Google | locked image tower, tuned text | Freeze-the-vision finding |
| 2022 | OpenCLIP / LAION | LAION | InfoNCE | Reproducible CLIP scaling laws |
| 2022 | CoCa | Google | contrastive **+** captioning | Both objectives in one tower |
| 2022 | FLAVA | Meta | unimodal + multimodal joint | Early "one model" attempt |
| 2022 | BLIP | Salesforce | ITC + ITM + captioning, CapFilt | Bootstrapped caption quality |
| 2023 | **SigLIP** | Google | **sigmoid** pairwise loss | No global negatives; the VLA/VLM default |
| 2023 | EVA-CLIP | BAAI | MIM-init + InfoNCE | Scaled to 18B |
| 2023 | DFN | Apple | learned data filtering | Data curation as the lever |
| 2023 | MetaCLIP | Meta | transparent CLIP data recipe | Reproduces WIT curation |
| 2023 | **DINOv2** | Meta | self-distillation, no text | Geometric/spatial features |
| 2023 | InternViT-6B | Shanghai AI Lab | progressive align to LLM | Largest open vision tower |
| 2024 | AIMv2 | Apple | autoregressive multimodal | AR pretraining for encoders |
| 2024 | AM-RADIO | NVIDIA | agglomerative multi-teacher distill | CLIP+DINO+SAM into one |
| 2025 | **SigLIP 2** | Google | + captioning, self-distill, dense | Current default encoder |
| 2025 | Perception Encoder | Meta | contrastive at scale, layer-selected | Best features aren't the last layer |
| 2025 | DINOv3 | Meta | self-supervised, gram anchoring | Dense-feature SOTA |
 
---
 
## 2. Generative VLMs (encoder + LLM → text)
 
| Year | Model | Org | Vision encoder | Connector | LLM | Res | Fusion | Open |
|---|---|---|---|---|---|---|---|---|
| 2022 | **Flamingo** | DeepMind | NFNet | Resampler + X-attn | Chinchilla (frozen) | Fix | Deep | ✗ |
| 2023 | **BLIP-2** | Salesforce | ViT-g (frozen) | **Q-Former** | OPT / Flan-T5 (frozen) | Fix | Late | W |
| 2023 | MiniGPT-4 | KAUST | BLIP-2 ViT | Linear | Vicuna | Fix | Late | W |
| 2023 | InstructBLIP | Salesforce | ViT-g | Q-Former (instr-aware) | Vicuna / Flan-T5 | Fix | Late | W |
| 2023 | **LLaVA** | UW-Madison/MSR | CLIP ViT-L/14 | **Linear** | Vicuna | Fix | Late | WD |
| 2023 | LLaVA-1.5 | UW-Madison | CLIP ViT-L/14-336 | MLP | Vicuna | Fix | Late | WD |
| 2023 | Qwen-VL | Alibaba | ViT-bigG | X-attn (256 queries) | Qwen-7B | Fix | Late | W |
| 2023 | CogVLM | Zhipu | EVA2-CLIP-E | **Expert** per layer | Vicuna | Fix | **Deep** | W |
| 2023 | Fuyu-8B | Adept | **none** | — | Persimmon | **Raw** | Early | W |
| 2023 | GPT-4V | OpenAI | — | — | GPT-4 | — | — | API |
| 2024 | LLaVA-NeXT (1.6) | UW-Madison | CLIP ViT-L | MLP | Vicuna/Mistral/Yi | **Tile** | Late | WD |
| 2024 | Idefics2 / 3 | Hugging Face | SigLIP-SO400M | Resampler / pool | Mistral / Llama 3 | Tile | Late | WD |
| 2024 | InternVL 1.5 / 2 | Shanghai AI Lab | InternViT-6B | **PixShuf** + MLP | InternLM2 | Tile | Late | W |
| 2024 | **Qwen2-VL** | Alibaba | ViT (retrained) | MLP | Qwen2 | **Native** + M-RoPE | Late | W |
| 2024 | **PaliGemma** | Google | SigLIP-So400m | Linear | Gemma-2B | Fix | Late (prefix-LM) | W |
| 2024 | Cambrian-1 | NYU | multi-encoder | Spatial Vision Aggregator | Llama/Vicuna | Tile | Late | WD |
| 2024 | NVLM | NVIDIA | InternViT | decoder / X-attn / hybrid | Qwen2 | Tile | both | W |
| 2024 | Pixtral 12B | Mistral | ViT from scratch, RoPE-2D | MLP | Mistral Nemo | Native | Late | W |
| 2024 | **Molmo** | AI2 | CLIP ViT | MLP | Qwen2 / OLMo | Tile | Late | **WD** |
| 2024 | DeepSeek-VL2 | DeepSeek | SigLIP | MLP | DeepSeek-MoE | Tile | Late | W |
| 2024 | Llama 3.2 Vision | Meta | ViT | X-attn adapters | Llama 3 (frozen) | Tile | Deep | W |
| 2024 | GPT-4o | OpenAI | — | — | — | — | **Early** | API |
| 2024 | Gemini 1.5 | Google DM | — | — | — | — | Early | API |
| 2025 | **Qwen2.5-VL** | Alibaba | ViT + window attn | MLP | Qwen2.5 | Native | Late | W |
| 2025 | InternVL3 | Shanghai AI Lab | InternViT-V2.5 | PixShuf | Qwen2.5 | Tile + V2PE | native pretrain | W |
| 2025 | Phi-4-multimodal | Microsoft | SigLIP | LoRA mixture-of-adapters | Phi-4 | Tile | Late | W |
| 2025 | Kimi-VL-A3B | Moonshot | MoonViT | MLP | MoE 16B/3B active | Native | Late | W |
| 2025 | Llama 4 (Scout/Maverick) | Meta | — | — | MoE | — | **Early** | W |
| 2025 | InternVL3.5 (+Flash) | Shanghai AI Lab | InternViT | PixShuf | Qwen3 / GPT-OSS | Tile | Late | W |
| 2025 | GLM-4.5V / 4.6V | Zhipu | — | — | MoE 106–108B | Native | Late | W (MIT) |
| 2025 | MiniCPM-V 4.5 | OpenBMB | SigLIP | Resampler | Qwen | Tile | Late | W |
| 2025 | **Qwen3-VL** | Alibaba | ViT | MLP | Qwen3 MoE (235B-A22B / 30B-A3B) | Native, 256K–1M ctx | Late | W |
| 2026 | **Molmo2** | AI2 | — | — | Olmo / Qwen | Tile | Late | **WD** |
| 2026 | **Qwen3.5** | Alibaba | **none (early fusion)** | — | 2B → 397B-A17B | Native, 256K–1M | **Early** | W |
| 2026 | Gemini 3 | Google DM | — | — | — | — | Early | API |
 
**The trend to name:** the bolt-on encoder era is closing. Fuyu (2023) was the early outlier;
GPT-4o and Llama 4 went native; Qwen3.5 (Feb 2026) is the first major family where *every* model is
natively multimodal via early fusion with **no separate `-VL` variant**. Late-fusion
encoder+projector is still the dominant open recipe, but it is now the legacy branch.
 
### 2b. Efficient / small VLMs (cross-cutting)
 
| Year | Model | Org | Size | Angle |
|---|---|---|---|---|
| 2024 | MobileVLM v2 | Meituan | 1.7–7B | Mobile-targeted projector |
| 2024 | Florence-2 | Microsoft | 0.23–0.77B | Unified seq2seq, tiny |
| 2024 | Moondream | Moondream | ~2B | Edge/embedded |
| 2025 | SmolVLM | Hugging Face | 256M–2.2B | Consumer hardware |
| 2025 | PaliGemma 2 | Google | 3–28B | Transfer-oriented base |
| 2025 | InternVL3.5-Flash | Shanghai AI Lab | 1–40B | Visual-token compression |
| 2026 | Qwen3.5-2B / 4B | Alibaba | 2–4B | Native multimodal at small scale |
 
Levers: visual-token pruning/merging, pixel shuffle, MoE sparsity, resolution budgeting, distillation.
 
### 2c. Reasoning / thinking VLMs (cross-cutting)
 
| Year | Model | Org | Note |
|---|---|---|---|
| 2024 | QVQ-72B | Alibaba | First open visual-reasoning preview |
| 2025 | Kimi-VL-A3B-Thinking | Moonshot | Long-CoT at 3B active |
| 2025 | InternVL3.5 (Cascade RL) | Shanghai AI Lab | RL post-training for VLMs |
| 2025 | Qwen3-VL-Thinking | Alibaba | Thinking/Instruct split per size |
| 2025 | Cosmos-Reason1 | NVIDIA | Physical common sense + embodied CoT |
| 2026 | Gemini 3 / GPT-5.x multimodal | DM / OpenAI | Frontier thinking over images |
 
---
 
## 3. Grounded / region-level VLMs
 
| Year | Work | Org | Output | Vocabulary |
|---|---|---|---|---|
| 2022 | OWL-ViT | Google | boxes | Open, CLIP-text queries |
| 2022 | GLIP | Microsoft | boxes | Detection as phrase grounding |
| 2023 | **Grounding DINO** | IDEA | boxes | Open-set, text-conditioned DETR |
| 2023 | **SAM** | Meta | masks (class-agnostic) | Promptable, no semantics |
| 2023 | Kosmos-2 | Microsoft | boxes inline in text | Grounded interleaved generation |
| 2023 | Shikra | SenseTime | boxes as text coords | Referential dialogue |
| 2023 | Ferret | Apple | free-form regions | Point/box/scribble refer |
| 2024 | **SAM 2** | Meta | masks over video | Streaming memory |
| 2024 | Florence-2 | Microsoft | boxes, masks, captions, OCR | One seq2seq for all tasks |
| 2024 | **Molmo (pointing)** | AI2 | **2D points** | Points > boxes for affordance |
| 2024 | RoboPoint | UW/NVIDIA | affordance points | Robot-targeted grounding |
| 2024 | DINO-X | IDEA | boxes, masks, poses | Unified open-world |
| 2025 | Qwen2.5-VL / Qwen3-VL | Alibaba | 2D **and 3D** boxes | Grounding as a first-class output |
| 2026 | Molmo2-VideoPoint | AI2 | points over video | Pointing + counting, temporal |
 
---
 
## 4. Unified understanding + generation
 
Sub-axis that decides everything: **how images are represented for generation** — discrete tokens
(AR), continuous latents (diffusion/flow), or decoupled encoders per task.
 
| Year | Model | Org | Image repr | Und/Gen pathways | Arch |
|---|---|---|---|---|---|
| 2023 | Emu | BAAI | continuous regress | shared | Mono |
| 2023 | CM3leon | Meta | discrete tokens | shared | Mono AR |
| 2024 | **Chameleon** | Meta | **discrete tokens** | shared, **early fusion** | Mono AR |
| 2024 | **Transfusion** | Meta | continuous (diffusion) | shared backbone, two losses | Mono |
| 2024 | Emu3 | BAAI | discrete tokens | shared | Mono AR (also video) |
| 2024 | Show-o | NUS | discrete + masked | shared | Mono |
| 2024 | VILA-U | NVIDIA/MIT | unified discrete tokenizer | shared | Mono |
| 2024 | **Janus** | DeepSeek | **decoupled** enc per task | split encoders, one transformer | Mono |
| 2024 | JanusFlow | DeepSeek | rectified flow | decoupled | Mono |
| 2025 | Janus-Pro | DeepSeek | decoupled | decoupled | Mono |
| 2025 | MetaQueries | — | learnable queries → diffusion | frozen MLLM + diff decoder | Hier |
| 2025 | **BAGEL** | ByteDance | SigLIP (und) + VAE (gen) | **MoT hard-routed experts** | MoT |
| 2025 | Mogao | ByteDance | hybrid | MoT | MoT |
| 2025 | OneCAT | — | unified | decoder-only native | Mono |
| 2025 | Show-o2 | NUS | unified | shared (+ video) | Mono |
| 2025 | BLIP3-o | Salesforce | diffusion over CLIP feats | adapter-connected | Hier |
| 2025 | Emu3.5 | BAAI | native multimodal | shared, interleaved | Mono |
| 2026 | SenseNova-U1 | SenseTime | NEO-unify | shared | Mono |
| 2026 | Kimi K2.5 | Moonshot | native multimodal | shared | Mono |
 
Note: BAGEL is the one to know cold — it's the world model inside π0.7's subgoal generator in the
robotics doc.
 
---
 
## 5. Video VLMs
 
Binding constraint is token budget and temporal position encoding, not semantics.
 
| Year | Model | Org | Temporal handling |
|---|---|---|---|
| 2023 | Video-LLaMA | Alibaba | Video Q-Former |
| 2023 | VideoChat | Shanghai AI Lab | Video + text foundation |
| 2023 | Video-LLaVA | PKU | Unified image/video projection |
| 2024 | LLaVA-NeXT-Video | UW-Madison | Frame tiling + AnyRes |
| 2024 | InternVideo2 | Shanghai AI Lab | Video encoder at scale |
| 2024 | LLaVA-OneVision | ByteDance/NTU | Single/multi-image/video transfer |
| 2024 | Tarsier2 | ByteDance | Fine-grained description via distillation |
| 2024 | Apollo | Meta | Systematic study of video-LLM design |
| 2024 | LongVA / LongVILA | NTU / NVIDIA | Long-context frame scaling |
| 2024 | Gemini 1.5 | Google DM | 1M+ ctx, hours of video |
| 2025 | Qwen2.5-VL | Alibaba | Dynamic FPS + absolute time encoding |
| 2025 | Keye-VL | Kwai | Short-video-native |
| 2025 | Qwen3-VL | Alibaba | Long-video temporal **localization** |
| 2026 | Molmo2 | AI2 | Video understanding + grounding, open data |
 
---
 
## 6. Document / OCR / GUI-agent VLMs
 
| Year | Work | Org | Emits |
|---|---|---|---|
| 2022 | Donut | NAVER | Structured JSON, OCR-free |
| 2023 | Pix2Struct | Google | Screenshot → structured text |
| 2023 | Nougat | Meta | Academic PDF → markdown |
| 2023 | CogAgent | Zhipu | GUI actions, high-res |
| 2024 | mPLUG-DocOwl 1.5 / 2 | Alibaba | Document understanding |
| 2024 | SeeClick / ShowUI / OS-Atlas | various | GUI grounding coordinates |
| 2024 | Claude computer use | Anthropic | Screen actions |
| 2025 | UI-TARS | ByteDance | Native GUI agent, end-to-end |
| 2025 | Operator / CUA | OpenAI | Browser actions |
| 2025 | olmOCR | AI2 | Open PDF → text pipeline |
| 2025 | DeepSeek-OCR | DeepSeek | Optical **context compression** (text as pixels) |
| 2025 | Qwen2.5-VL / Qwen3-VL agent | Alibaba | GUI operation as a built-in capability |
 
---
 
## 7. Spatial / embodied VLMs — the bridge to VLA
 
These are the direct ancestors of the "embodied reasoning" family in
`robotics_foundation_models_reference.md` §5.
 
| Year | Work | Org | Emits |
|---|---|---|---|
| 2023 | PaLM-E | Google | Multimodal plan text |
| 2024 | **SpatialVLM** | Google DM | Metric spatial QA |
| 2024 | SpatialRGPT | NVIDIA | Region-level 3D relations |
| 2024 | RoboPoint | UW/NVIDIA | Affordance points |
| 2024 | Molmo | AI2 | Pointing |
| 2025 | Cosmos-Reason1 | NVIDIA | Physical common sense |
| 2025 | Gemini Robotics-ER 1.5 | Google DM | Points, trajectories, plans |
| 2025 | RoboBrain | BAAI | Task planning + affordance |
| 2025 | MolmoAct | AI2 | Action reasoning traces |
| 2026 | Pelican-VL | — | Embodied "brain" model |
| 2026 | Cosmos-Reason2 | NVIDIA | Action tokens for decomposition |
| 2026 | Gemini Robotics-ER 2 | Google DM | Spatial plans, multi-robot coordination |
 
---
 
## 8. Substrate — data and benchmarks
 
**Pretraining data**
 
| Year | Dataset | Scale / note |
|---|---|---|
| 2018 | Conceptual Captions 3M/12M | Web alt-text, cleaned |
| 2021 | LAION-400M | Open CLIP-scale corpus |
| 2022 | LAION-5B | The open scaling substrate |
| 2022 | COYO-700M | Kakao alternative |
| 2023 | DataComp | Benchmark for *data curation* itself |
| 2023 | **OBELICS** | Open interleaved image-text documents |
| 2023 | MMC4 | Interleaved C4 |
| 2023 | **LLaVA-Instruct-150K** | GPT-4-generated visual instructions |
| 2023 | ShareGPT4V | Dense high-quality captions |
| 2024 | The Cauldron | 50-dataset instruction mixture |
| 2024 | **PixMo** | Molmo's fully-open data, human-collected pointing |
| 2024 | Infinity-MM | Large open instruction corpus |
 
**Benchmarks**
 
| Focus | Benchmarks |
|---|---|
| General VQA | VQAv2, GQA, OKVQA, MM-Vet, MMBench, SEED-Bench, MME |
| Knowledge / reasoning | **MMMU**, MMMU-Pro, MathVista, ScienceQA, AI2D |
| Perception limits | **BLINK**, MMVP, CV-Bench, RealWorldQA |
| Hallucination | POPE, HallusionBench |
| Text in images | TextVQA, DocVQA, ChartQA, **OCRBench** |
| Grounding | RefCOCO/+/g, ODinW |
| Video | **Video-MME**, MVBench, LongVideoBench, EgoSchema |
| Spatial / embodied | VSI-Bench, MolmoSpaces, ERQA |
| Unified und+gen | Uni-MMMU, MME-Unify, GIR-Bench, WISE |
 
---
 
## The one-line state of play (July 2026)
 
Architecture is converging on **native early fusion** — one weight set trained multimodally from
the start, no bolt-on encoder — with Qwen3.5 the first family to ship it across every size.
Open-weight quality (Qwen3-VL/3.5, InternVL3.5, GLM-4.6V) is at or near frontier proprietary on
most benchmarks. The live research questions are (a) unified understanding+generation, where the
image-representation choice is unsettled, (b) whether generic VLM capability transfers to embodied
control — recent evidence says spatial objectives predict downstream VLA performance far better
than general vision benchmarks do.
 
---
 
## Confidence notes
 
- **Rows through 2024** are well-established; safe from memory.
- **2025–2026 rows** are compiled from the Overshoot open-VLM survey (Mar 2026), the unified-
  multimodal survey (2505.02567, rev. Jan 2026), and the native-multimodal roadmap (2605.25343).
  Names, orgs, dates, and axis values are reliable; connector/encoder details for the newest
  closed models (Gemini 3, GPT-5.x, Llama 4 internals) are not publicly specified — em-dashes in
  those cells mean "not disclosed," not "not applicable."
- **Qwen3.5's early-fusion claim** is the load-bearing 2026 fact in this doc and is worth
  verifying against the model card before asserting it in an interview.
- Parameter counts for MoE models are shown total/active where known; several are omitted rather
  than guessed.
 