# VLA On-Device Inference — Project TODO

## Phase 1: Literature Review

### Primary Papers (read these first — highest overlap with document scope)
- [ ] Read **Embodied Foundation Models at the Edge** (arxiv 2603.16952)
  - Focus: 8 deployment barriers, autoregressive vs diffusion memory/compute characterization, co-design levers
- [ ] Read **Cross-Platform Scaling of VLA Models: Edge to Cloud GPUs** (arxiv 2509.11480)
  - Focus: empirical latency/throughput/memory across Jetson-class → datacenter GPUs, 5 VLA architectures
- [ ] Read **RooflineBench for On-Device LLMs** (arxiv 2602.11506)
  - Focus: roofline methodology for edge devices, operational intensity metrics — adapt to VLA context
- [ ] Read **LLM Inference Unveiled: Roofline Model Insights** (arxiv 2402.16363)
  - Focus: prefill vs decode roofline breakdown, memory-bandwidth bound decode — foundational methodology

### Secondary Papers (efficiency techniques + model landscape)
- [ ] Read **A Survey on Efficient VLA Models** (arxiv 2510.24795)
  - Focus: taxonomy of efficiency techniques — architecture, quantization, distillation, action tokenization
- [ ] Read **Efficient VLA Models for Embodied Manipulation** (arxiv 2510.17111)
  - Focus: compare taxonomy with above, note any non-overlapping coverage
- [ ] Skim **Lite VLA on CPU-Bound Edge Robots** (arxiv 2511.05642)
  - Focus: what "CPU-bound" means in practice, architecture choices made
- [ ] Skim **QVLA: Quantization for VLA Models** (arxiv 2602.03782)
  - Focus: which layers/channels are quantization-sensitive, task success impact
- [ ] Skim **BitVLA: 1-bit VLA** (arxiv 2506.07530)
  - Focus: quality retention at extreme quantization, what breaks first

### Hardware Platform References
- [ ] Review NVIDIA Jetson platform specs: Orin NX, AGX Orin, AGX Thor
  - Extract: peak TFLOPS (FP16/INT8), memory bandwidth, TDP, typical robot use cases
- [ ] Review Qualcomm Robotics RB-series or Snapdragon compute specs (if targeting non-NVIDIA edge)
- [ ] Note ridge points for each platform — this anchors the roofline analysis

---

## Phase 2: Synthesis + Framing

### Compute Profile Section
- [ ] Tabulate edge robot platform specs (FLOPS, bandwidth, TDP, ridge point)
- [ ] Map current VLA model families to parameter counts + memory footprints
- [ ] Identify which models fit in which platform's memory budget

### Roofline Analysis Section (the gap no existing paper fills)
- [ ] Apply roofline methodology to VLA inference phases:
  - Prefill (prompt + image tokens) — likely compute-bound at batch > 1
  - Decode (autoregressive action token generation) — memory-bandwidth bound
  - Diffusion denoising steps (if covering diffusion-based VLAs) — compute-bound
- [ ] Estimate arithmetic intensity for key VLA operations (attention, FFN, vision encoder) at typical robot batch sizes (usually 1)
- [ ] Derive model capacity upper bounds at each platform's bandwidth/compute limits given latency budget (e.g., 10 Hz control loop = 100ms)

### Model Landscape Section
- [ ] Catalog current robotic foundation models: RT-2, OpenVLA, Octo, π0, GR00T, etc.
  - For each: parameter count, architecture type (autoregressive vs diffusion), reported inference speed, target hardware
- [ ] Note which are edge-deployable today vs cloud-only

### Quality Degradation Section (stretch goal)
- [ ] Look for any papers that report task success rate vs quantization level for VLA models
- [ ] Look for latency → task success correlation (does slower inference hurt manipulation quality?)
- [ ] If no data exists, note this as an open problem worth calling out

### Optimization Levers Section
- [ ] Organize levers by what they improve (memory footprint vs latency vs energy):
  - Quantization (INT8/INT4/FP8, per-channel vs per-tensor)
  - Speculative decoding (applicable to autoregressive VLAs)
  - KV cache management / MLA architectures
  - Action chunking / non-autoregressive decoding
  - Hierarchical decomposition (fast low-level controller + slow semantic VLM)
  - Vision encoder compression / token pruning
  - Model distillation into smaller student VLAs

---

## Phase 3: Document Writing

- [ ] Decide on document format (internal design doc vs paper-style writeup)
- [ ] Write Section 1: Edge robot compute landscape (hardware profiles + roofline setup)
- [ ] Write Section 2: VLA model capacity analysis (what fits, what's the bottleneck)
- [ ] Write Section 3: Current model landscape (what exists, edge vs cloud)
- [ ] Write Section 4: Quality vs compute tradeoffs (empirical evidence + gaps)
- [ ] Write Section 5: Optimization levers (organized by lever type + impact)
- [ ] Write intro + framing (do this last)
- [ ] Review against existing papers — make sure novel contributions are clearly distinguished

---

## Notes

- The **roofline analysis applied to VLA model classes** is the clearest gap in existing literature — lean into this
- Existing surveys cover optimization techniques well; don't rehash, cite and extend
- 2603.16952 and 2509.11480 are must-reads before writing anything
- Keep paper summaries as separate files in this folder (e.g., `papers/2603.16952-summary.md`)
