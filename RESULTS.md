# Results: real first-FFN held-out diagnostics

## Provenance and interpretation

These numbers were recorded in the Kaggle notebook's stages 54, 56, and 57 on **4 November 2026**. They refer to the frozen **first decoder-block FFN of Qwen3.8-27B**, not to the entire language model.

The test comprised **47 held-out real coding-text token vectors** distinct from **91 calibration token vectors**. The percentages below are relative L2 errors of approximate FFN *output vectors*. They are **not percentages of model benchmark score lost**. Intermediate channel fractions are **not** end-to-end compute fractions.

## Frozen static selection, with precomputed mean correction

Each row selects a fixed set of important intermediate neuron channels based on the calibration sample. The selected channels are used for all test inputs. A precomputed mean FFN response compensates for the omitted channels.

| Retained channels | Ranking method | Mean error | Median error | p90 error |
| ---: | --- | ---: | ---: | ---: |
| 25% | Raw activation importance | 42.717% | 44.126% | 64.098% |
| 25% | Centered activation importance | **41.845%** | 42.960% | 63.266% |
| 50% | Raw activation importance | 27.170% | 27.280% | 44.662% |
| 50% | Centered activation importance | **26.254%** | 24.593% | 41.486% |
| 75% | Raw activation importance | **13.949%** | 12.520% | 21.685% |
| 75% | Centered activation importance | 14.756% | 13.009% | 22.019% |

Without mean correction, the uncorrected versions were generally worse. The mean-vector-only output (0% computed channels) had a 101.023% mean local error.

## PCA-sketch router, rank 80, with frozen mean correction

The selection is input-dependent, using a calibration-derived rank-80 sketch. Approximately **99.6%** of calibration input variance was captured at rank 80. A diagnostic oracle that sees the full exact intermediate activation is included solely for comparison.

| Retained channels | Router | Mean error | Median error | p90 error | Deployable cheap routing? |
| ---: | --- | ---: | ---: | ---: | --- |
| 25% | PCA sketch, rank 80 | 43.335% | 41.506% | 69.442% | Candidate, not speed-benchmarked |
| 25% | Full-activation oracle | 29.123% | 27.863% | 46.446% | **No** |
| 50% | PCA sketch, rank 80 | 29.346% | 29.197% | 52.797% | Candidate, not speed-benchmarked |
| 50% | Full-activation oracle | 10.128% | 9.439% | 14.800% | **No** |
| 75% | PCA sketch, rank 80 | 16.595% | 15.559% | 31.074% | Candidate, not speed-benchmarked |
| 75% | Full-activation oracle | 2.092% | 1.991% | 2.950% | **No** |

Ranks 16 and 48 yielded worse sketch mean errors than rank 80 at the three reported active fractions; see the Kaggle stage-57 output for the full rank sweep. The oracle values are identical across sketch ranks because its selection does not use the sketch.

## Important non-results

There are **no measured end-to-end Qwen benchmark scores** for these sparse variants. There are **no actual full-model VRAM or generation throughput measurements** for a converted checkpoint. The GPU kernel and synthetic neighborhood experiments earlier in the laboratory notebook isolate local operations and must not be presented as a successfully accelerated 27B model.

## Conclusions that follow and do not follow

The data support the conclusion that the tested no-gradient sparse selection heuristics have significant first-layer reconstruction error at 25–75% active intermediate channels. The data **do not** prove that every possible future training-free approach would fail or quantify any downstream language-model accuracy drop. The project is published as a completed feasibility study with negative evidence for the original deployment objective.
