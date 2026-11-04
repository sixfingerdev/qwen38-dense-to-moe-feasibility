---
language:
- en
license: cc-by-4.0
pretty_name: SixFinger Qwen3.8-27B Dense-to-MoE Feasibility Results
tags:
- qwen3.8
- mixture-of-experts
- model-sparsity
- training-free
- research
- negative-results
task_categories:
- text-generation
size_categories:
- n<1K
---

# SixFinger Qwen3.8-27B Dense-to-MoE Feasibility Results

**Repository type: Dataset / research results. Not a model.**

This dataset records a limited, training-free experiment on sparsifying the original **first** SwiGLU feed-forward sublayer of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

The research goal was to create a near-lossless sparse MoE version of the dense 27B model that would require less inference GPU memory and run faster. **The target was not achieved.** No full Qwen MoE checkpoint was constructed, released, or benchmarked.

## Dataset contents

`real_first_ffn_holdout.csv` contains aggregate local FFN output reconstruction errors from two frozen-weight methods and a non-deployable oracle. Each record includes the experiment stage, selection method, active intermediate-channel fraction, calibration/holdout token counts, mean/median/p90 relative L2 output error, and whether the method is an oracle.

The data were measured on **91 calibration and 47 disjoint held-out real coding-text first-layer FFN activation vectors**. They are very small and not representative of downstream general-purpose language-model performance. No human messages, proprietary user logs, or model weights are distributed.

## Headline results

| Active channels | Static + frozen mean, best ranking | PCA-sketch rank 80 | Full-activation oracle, not deployable |
| ---: | ---: | ---: | ---: |
| 25% | 41.845% | 43.335% | 29.123% |
| 50% | 26.254% | 29.346% | 10.128% |
| 75% | 13.949% | 16.595% | 2.092% |

All values are **mean relative L2 errors of the first FFN output**, not accuracy regressions. The oracle calculates all intermediate activations before selection and therefore cannot support a real inference speedup in this form.

## Method

We partitioned dense SwiGLU neuron triplets into disjoint groups, which algebraically recovers dense behavior when all groups are run. For sparse selection, we investigated calibration-based static importance, precomputed mean compensation, Jacobian-based diagnostics, and frozen PCA-sketch routing. No gradient training, fine-tuning, or distillation was performed. The rank-80 sketch captured approximately 99.6% of the small calibration set's variance, but it did not close the gap to oracle selection.

Simply partitioning weights does not lower their total size or the GPU memory used if all remain resident. The experiment did not deliver an offloading engine or validate an end-to-end tokens/s improvement.

## Reproducibility and code

- [Complete technical report and independent reference prototype](https://github.com/sixfingerdev/qwen38-dense-to-moe-feasibility)
- [Original Kaggle experimental notebook (private; owner access may be required)](https://www.kaggle.com/code/antigravityiin/sixfinger-qwen3-8-moefication-lab/edit/run/365661914)
- [Original pretrained Qwen model](https://huggingface.co/Qwen/Qwen3.8-27B)

The public code reference is a NumPy-only synthetic demonstration of exact all-expert partitioning and heuristic sparse routing; its outputs are **not** Qwen accuracy measurements. The underlying real-weight Kaggle notebook was private when this dataset card was prepared. The aggregate CSV is publicly distributable but does not by itself constitute a complete rerunnable large-model experiment.

## Appropriate uses and limitations

Suitable for discussing negative experimental results and forming follow-up research hypotheses. **Not suitable** as a pretrained model, inference endpoint, ready-to-run MoE engine, benchmark quality result, or proof of VRAM/latency savings. The original study is concluded without claiming near-lossless conversion.

**Citation:** SixFingerDev (2026), *SixFinger Qwen3.8-27B Dense-to-MoE Feasibility Study*. Research report, version 1.0.

**Licensing:** Research report and aggregate results CC BY 4.0. Independent reference software MIT. Qwen model rights remain with the upstream authors; no upstream weights are redistributed.
