# SixFinger Qwen3.8-27B Dense-to-MoE Feasibility Study

**Status: concluded research experiment — negative result for the stated deployment objective.**

This repository documents an independent, training-free investigation of whether the dense feed-forward networks (FFNs) in [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) can be reorganized into sparse mixture-of-experts (MoE) computations while preserving model behavior and reducing real inference memory and latency.

Our investigation established exact **algebraic partitionability** of a dense SwiGLU FFN when every partition is evaluated. However, the explored inexpensive sparse selection and compensation schemes exhibited substantial errors on real, held-out first-layer FFN inputs. We did **not** produce an end-to-end MoE checkpoint, establish near-lossless full-model behavior, or demonstrate simultaneous reductions in actual model VRAM and generation latency.

We are publishing both the positive mathematical findings and the negative empirical findings to distinguish a promising architectural idea from a validated model conversion.

> **Not a model release.** No Qwen-derived weights, quantized checkpoint, production routing engine, or deployable MoE model is contained here. A numerical FFN output error is not an accuracy score, and a single-layer kernel microbenchmark is not an end-to-end throughput result.

## Research question

Can an existing 27-billion-parameter dense model be transformed, **without gradient-based training or distillation**, into a sparse expert architecture that:

1. Preserves nearly all of the original model's useful behavior;
2. Reduces peak GPU memory required during inference; and
3. Improves measured prefill/decode performance relative to a fair dense baseline?

The tests reported here do **not** establish these three properties together.

## Model and scope

| Property | Experimental scope |
| --- | --- |
| Target checkpoint | `Qwen/Qwen3.8-27B` |
| Architecture | Qwen3.8 / `qwen3_5_text` dense language backbone within a multimodal model |
| Language decoder blocks | 64 |
| Hidden width | 5,120 |
| SwiGLU intermediate width | 17,408 |
| Approximate dense FFN parameter count | 17.113 billion, across 64 blocks |
| Experimentally inspected weights | Original first decoder-block FFN (`gate_proj`, `up_proj`, `down_proj`) |
| Real calibration set | 91 FFN input token vectors, illustrative coding prompts |
| Disjoint real holdout set | 47 FFN input token vectors, different coding prompts |
| Weight optimization | None; no backpropagation or fine-tuning |
| Validation level | First FFN layer, local relative output error; limited isolated speed tests |
| Full-model LM evaluation | **Not performed** |

The calibration/holdout split is small, coding-biased, and drawn from one decoder layer. It must not be described as representative of Qwen's full multilingual, multimodal, reasoning, or agentic capabilities.

## Main observations

The principal experiment measures a **mean relative FFN output L2 error** on 47 held-out real token vectors:

\[
E = \operatorname{mean}_t\frac{\lVert \hat{f}(x_t)-f(x_t)\rVert_2}{\max(\lVert f(x_t)\rVert_2,\epsilon)}.
\]

The active fraction refers to intermediate FFN neuron channels *in this layer* and is not the fraction of the entire model's total inference compute or memory.

| Active FFN channels | Frozen mean compensation + best static selection | PCA-sketch router (rank 80) | Non-deployable oracle (diagnostic) |
| ---: | ---: | ---: | ---: |
| 25% | 41.845% | 43.335% | 29.123% |
| 50% | 26.254% | 29.346% | 10.128% |
| 75% | 13.949% | 16.595% | 2.092% |

The oracle uses the complete dense activation to select important channels. It **cannot** provide compute savings in the measured form and is included only to assess the potential selection gap. See [RESULTS.md](RESULTS.md) and [results/real_first_ffn_holdout.csv](results/real_first_ffn_holdout.csv) for the precise setups, medians, p90s, and provenance.

### Interpretation

- **Exact partitioning is possible:** dividing the gate/up/down FFN neuron triplets into disjoint groups and summing every group reproduces the dense FFN up to floating-point accumulation differences.
- **Exact partitioning is not sparse acceleration:** evaluating all groups generally retains the original dense arithmetic; keeping all weights GPU-resident also retains their weight-memory footprint.
- **Omitting experts changes the function:** zeroing or approximating uncomputed contributions produced substantial local errors under the tested methods.
- **A cheap router remains the critical gap:** rank-80 PCA accounted for about 99.6% of the observed calibration-input variance but still failed to reproduce the oracle's channel choices sufficiently well.
- **No downstream quality claim is justified:** a 13.949% local FFN output error does not mathematically imply a 13.949% loss on a language-model benchmark; no such benchmark was run.

## Reproduction and available code

The original [Kaggle laboratory notebook](https://www.kaggle.com/code/antigravityiin/sixfinger-qwen3-8-moefication-lab/edit/run/365661914) is the source of the real-weight experiments. **The Kaggle notebook was private at the time this report was prepared and its link may require owner access.** Its later stages contain frozen-weight first-layer calibration, held-out sparse selection, mean compensation, and PCA-sketch diagnostics. Historical stages explore additional non-deployable or synthetic probes. Some notebook cells depend on earlier execution state, an available GPU, and downloads of original Qwen shards. The notebook is an experimental record, **not** a turnkey independent pipeline.

This repository also contains a small independent reference implementation that runs without Qwen's multi-gigabyte weights:

```bash
python -m pip install -r requirements.txt
python sixfinger_moe_prototype.py --demo
python -m unittest discover -s tests -v
```

`sixfinger_moe_prototype.py` demonstrates expert partitioning, a frozen cheap representative-neuron routing heuristic, and optional mean-based compensation on **synthetic** weights. Its synthetic numbers must not be substituted for the Qwen measurements above. It is neither a fused GPU kernel nor a model conversion/export tool.

The reference code makes the inference distinction explicit:

```text
Dense SwiGLU FFN
  -> fixed, disjoint neuron groups
  -> all groups computed: approximately exact, not sparse
  -> subset computed: approximate, local error must be measured
```

## What was and was not proven

| Claim | Evidence status |
| --- | --- |
| SwiGLU can be decomposed into expert sums | **Established algebraically; checked numerically** |
| Original first-block Qwen weights can be examined without loading full 27B checkpoint | **Established in Kaggle notebook** |
| No-gradient calibration and routing heuristics can be evaluated | **Established locally** |
| 25–75% active FFN neurons preserve near-zero local error | **Not observed** |
| Proposed sparse router preserves full Qwen benchmark accuracy | **Not evaluated** |
| Proposed system decreases full-model peak VRAM | **Not evaluated** |
| Proposed system increases end-to-end tokens/second | **Not evaluated** |
| Self-contained sparse MoE checkpoint is available | **No** |

## Limitations and responsible interpretation

1. The most recent quantitative results cover **one FFN**; the model contains 64 decoder blocks.
2. Token vectors came from small, hand-authored coding examples; this is not a broad generalization test.
3. The data-free/small-calibration experiment should not be conflated with trained expert routing or language-model distillation.
4. The dynamic routing approximation was examined for **selection quality** but not as an optimized deployable CUDA kernel.
5. Partitioning pretrained weights does not itself shrink the total stored model. Reduced GPU residency would need explicit offloading, paging, compression, or quantization, with an accompanying bandwidth/latency trade-off.
6. Some earlier microbenchmarks isolate a single FFN and use synthetic inputs; they cannot be extrapolated to end-to-end generation.
7. The study makes **no new research priority, novelty, state-of-the-art, or accuracy claim**.

## Artifacts

- [`REPORT.md`](REPORT.md): research design, mathematical decomposition, measurement semantics, interpretation, and stopping criteria.
- [`RESULTS.md`](RESULTS.md): tables with calibration/holdout sizes and quantified error measurements.
- [`results/real_first_ffn_holdout.csv`](results/real_first_ffn_holdout.csv): machine-readable summary of verified held-out measurements.
- [`sixfinger_moe_prototype.py`](sixfinger_moe_prototype.py): self-contained NumPy reference prototype, separate from real Qwen benchmarking.
- [`tests/test_reference.py`](tests/test_reference.py): deterministic property tests for the reference.
- [`CITATION.cff`](CITATION.cff): recommended citation metadata.
- [`OZET_TR.md`](OZET_TR.md): professional Turkish summary of the experiment and limitations.

## Original work and citation

This report is independent of the upstream Qwen team. **Qwen weights are not redistributed.** For the original model architecture, license, and model card, see [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). See `CITATION.cff` for attribution of this experimental record. The prototype code is MIT-licensed; the research text and measurement table are distributed under CC BY 4.0.

**Conclusion:** Partitioning a dense pretrained FFN into experts is straightforward; obtaining a **cheap, accurate sparse selector** that also improves *measured* end-to-end performance is not. The explored training-free methods did not meet the original near-lossless, lower-VRAM, faster-inference objective. This project is intentionally closed as a documented feasibility study rather than marketed as a converted model.
