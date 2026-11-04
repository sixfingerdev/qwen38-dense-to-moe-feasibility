# Technical Research Report

## SixFinger Qwen3.8-27B: Training-Free Dense-to-MoE Feasibility Investigation

**Report date:** 4 November 2026  
**Author:** SixFingerDev  
**Project status:** concluded; no deployable converted model

### Abstract

We studied a proposed, training-free transformation of the dense SwiGLU feed-forward sublayers of Qwen3.8-27B into shared and routed expert groups. The original research objective was threefold: near-lossless functional quality, lower inference-time GPU memory, and faster actual inference. The investigation used original first-layer Qwen weights and held-out first-layer activation vectors produced from coding prompts. A disjoint expert decomposition reproduces the original FFN if all parts are evaluated, but this identity provides no sparse-computation benefit. Sparse heuristics including frozen means, fixed importance rankings, local Jacobian-inspired corrections, and reduced-rank PCA selection displayed significant relative output error on held-out activations. At a 75% retained neuron fraction, the best documented static mean-compensated method yielded 13.949% mean relative FFN L2 error; a rank-80 PCA-sketch selector yielded 16.595%. An oracle with access to full activations achieved 2.092% in a related comparison, but the oracle is not inference-efficient. No full-model generation, benchmark accuracy, or GPU-residency study was completed. Therefore the original deployment objective was not achieved, and no converted checkpoint is being released.

### 1. Research question and acceptance gates

The target was to convert `Qwen/Qwen3.8-27B` without distillation, backpropagation, or parameter fine-tuning. Accepting a new architecture requires *all* of the following, measured against a matched dense baseline:

1. A self-contained and reloadable implementation or checkpoint with a documented sparse execution policy.
2. Quality assessed on independent held-out language-model tasks, including coding and general-domain cases, with a predeclared acceptable degradation threshold.
3. Peak **device VRAM** accounting for model weights, temporary buffers, routing state, and KV cache; CPU/RAM residency should be reported separately.
4. Representative end-to-end prefill and decode latency/throughput under the same hardware, batch, context, tokenizer and precision conditions.
5. No reliance on an oracle that evaluates the very experts intended to be skipped.

None of these conditions can be inferred solely from local FFN output reconstruction.

### 2. Model and mathematical starting point

The target Qwen model exposes a `qwen3_5_text` language backbone with 64 decoder blocks, hidden size \(H=5120\), and intermediate FFN width \(M=17408\). For an FFN input row vector \(x\), gate/up weights \(G,U\in\mathbb{R}^{M\times H}\), and down weights \(D\in\mathbb{R}^{H\times M}\), the FFN computes:

\[
f(x)=D\left[\operatorname{SiLU}(Gx)\odot(Ux)\right].
\]

Let the intermediate channels be partitioned into disjoint index sets \(I_0,\ldots,I_E\), possibly assigning \(I_0\) to an always-on shared group. Since the output projection is linear in its intermediate input,

\[
f(x)=\sum_{e=0}^{E} D_{:,I_e}\left[\operatorname{SiLU}(G_{I_e,:}x)\odot(U_{I_e,:}x)\right].
\]

This is an exact identity in real arithmetic. In floating-point implementations, reordering the summation can change low-order bits. Evaluating every group retains essentially the original FFN arithmetic and aggregate weights. It does not by itself reduce resident GPU weight memory.

With only a subset \(S(x)\) of expert channels evaluated, the uncorrected approximation is

\[
\hat{f}_0(x)=\sum_{e\in S(x)}D_{:,I_e}\phi_e(x),
\quad\phi_e(x)=\operatorname{SiLU}(G_{I_e,:}x)\odot(U_{I_e,:}x).
\]

The dropped contribution is the sum over omitted groups. A frozen mean vector \(\mu\), derived only from calibration activations, supports the alternative

\[
\hat{f}_{\mu}(x)=D\mu+\sum_{e\in S(x)}D_{:,I_e}(\phi_e(x)-\mu_{I_e}).
\]

For arbitrary unseen inputs, this formula is exact only if the omitted deviations cancel to zero. No general cancellation theorem was established, and the experiments did not show sufficiently small error.

### 3. Experimental data and methodology

- **Weights:** real pretrained Qwen3.8-27B first decoder FFN tensors; the experiments deliberately avoid loading the entire model.
- **Calibration:** 91 first-layer FFN input token vectors from 8 public, illustrative coding prompts.
- **Holdout:** 47 first-layer FFN input token vectors from separate, illustrative coding prompts.
- **Optimization:** zero gradient updates, optimizer steps, or changes to pretrained FFN weights. Frozen statistics and PCA basis calculations are calibration operations, not language-model training.
- **Metric:** per-token L2 norm of the difference between approximate and exact original FFN output, divided by the L2 norm of the exact output with an epsilon safeguard. Reported aggregate values are mean, median, and p90 over held-out tokens.
- **Scope:** one decoder FFN output, not cumulative error propagation through 64 layers and not full-model task performance.

The calibration data are small and topically narrow. Token vectors are correlated within each prompt. The 91/47 token counts should not be presented as 138 independent downstream tasks.

### 4. Candidate approaches

#### 4.1 Disjoint neuron-to-expert decomposition

We distributed SwiGLU neuron triplets into fixed expert partitions. Reconstructing every group provides an algebraic unit test. This is a structural conversion, not evidence of sparse acceleration. Subsetting whole groups introduces approximation errors; selecting groups by their fully computed contribution norms is an oracle rather than a cheap router.

#### 4.2 Static importance and mean compensation

For intermediate activation \(z_j(x)\), a frozen statistic ranks channels using a measure related to \(\lVert D_{:,j}\rVert_2 \cdot \mathbb{E}_{cal}|z_j(x)-\mu_j|\). The important channels are fixed across holdout tokens. A precomputed \(D\mu\) term compensates for omitted mean contributions. No learned routing parameters or weight changes are required. The best observed mean errors from this family were 41.845%, 26.254%, and 13.949% at 25%, 50%, and 75% channel retention.

#### 4.3 Jacobian and curvature approximations

We investigated first-order local FFN approximation and nonlinear residual correction around anchor hidden states. Anchor proximity can be poor outside a very narrow input neighborhood; local linearization is not a global substitute for a nonlinear FFN. Several earlier variants used synthetic neighborhoods and were not validated as full-model inference methods. A residual oracle uses information that would itself require evaluating omitted activations, so it provides a diagnostic upper bound on selection quality rather than a practical route to saved computation.

#### 4.4 Frozen PCA-sketch routing

The calibration inputs were centered and decomposed by SVD. At rank 80, the calibration PCA captured approximately 99.6% of observed input variance. An input-dependent proxy for intermediate activations provided top-channel scores without obtaining all target activations at routing time. Although geometrically compact for this small calibration set, this sketch did not recover the much lower error of the non-deployable oracle. Rank-80 mean errors remained 43.335%, 29.346%, and 16.595% at 25%, 50%, and 75% retention.

**Important implementation caveat:** the diagnostic reconstruction code in the notebook uses full held-out activations to compute correctness after choosing channels; no end-to-end latency for this selector was measured. This must not be advertised as a shipped sparse implementation.

### 5. Results and uncertainty

See [`RESULTS.md`](RESULTS.md) for the full quantitative table and [`results/real_first_ffn_holdout.csv`](results/real_first_ffn_holdout.csv) for machine-readable records.

The measured local errors are large enough that we cannot describe the tested sparse conversions as near-lossless. Crucially, they are **not language-model accuracy regressions**. No inference result exists from which to infer human preference, coding benchmark score, model perplexity, hallucination rate, or multimodal ability.

The oracle result highlights an actionable research question: useful sparse channel sets can be identified when all activations are known, but a fast, sufficiently accurate selector using substantially less computation was not found in the tested schemes. The gap is relevant; it is not a proof that all possible training-free methods must fail.

### 6. Compute and VRAM considerations

The 64 FFNs contain approximately \(3\cdot5120\cdot17408\cdot64=17{,}112{,}760{,}320\) weight parameters. Sparse execution could avoid some FFN multiply-add work in principle, but the whole model also contains attention, embeddings, and other operations. The fraction of FFN neurons retained cannot be mapped directly to full-model speedup. Moreover:

- Renaming or repartitioning existing weights does not change their total bytes.
- Retaining all experts in GPU memory yields little or no reduction in the model-weight component of peak VRAM.
- Storing inactive experts on CPU, disk, or remote memory reduces GPU residency only by introducing transfers, which can increase decode latency, especially under dynamic token-level routing.
- Quantization and structured low-rank approaches are different compression mechanisms and must be separately controlled if included.
- Isolated single-layer synthetic kernel timings are insufficient evidence of end-to-end inference improvements.

The project did not implement and benchmark a full-model expert offload runtime. We therefore report **no verified peak-VRAM reduction and no verified overall tokens-per-second gain**.

### 7. Reproducibility

1. Use the original upstream model, its applicable licensing terms, and a compatible Qwen3.5/3.8 Transformers implementation.
2. Consult the [Kaggle laboratory notebook](https://www.kaggle.com/code/antigravityiin/sixfinger-qwen3-8-moefication-lab/edit/run/365661914), **currently private and not generally accessible**, for first-layer weight loading, prompt construction, and frozen-weight experiments (stages 54, 56, and 57 are central to the published quantitative table).
3. Do not treat the notebook's historical exploratory cells as an automatically replayable pipeline. Some cells rely on existing objects in a particular session, original large shard downloads, and GPU activation.
4. For a dependency-light mathematical sanity check, run the separate NumPy reference:

   ```bash
   python -m pip install -r requirements.txt
   python sixfinger_moe_prototype.py --demo
   python -m unittest discover -s tests -v
   ```

5. Clearly distinguish synthetic NumPy reference output from actual Qwen FFN measurements.

This repository does not include restricted/private logs, user conversation datasets, access tokens, or upstream model weights.

### 8. Research stopping decision

The study ended without conversion because the explored selectors did not simultaneously achieve low reconstruction error and a deployable, cheap inference path, and because full-model evidence for quality, memory, and speed was absent. Publishing an unverified 27B sparse checkpoint as a finished, high-quality MoE would conflate algebraic weight partitioning with an actual accelerated inference model.

Reopening the question would require a predeclared quality tolerance, substantially more representative hidden-state calibration, optimized inference routing/offloading, adequate disk and GPU resources, and fully paired end-to-end evaluations. Nothing in this report claims that the approach is theoretically impossible; it reports the limits of the specific completed tests.

### 9. Source materials

- Official upstream weights and model card: https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen project: https://github.com/QwenLM/Qwen3.8
- SixFinger Kaggle experimental notebook: https://www.kaggle.com/code/antigravityiin/sixfinger-qwen3-8-moefication-lab/edit/run/365661914

### 10. Suggested citation

SixFingerDev. (2026). *SixFinger Qwen3.8-27B Dense-to-MoE Feasibility Study: A Training-Free, First-Layer Negative Result*. Research code and report. Version 1.0.
