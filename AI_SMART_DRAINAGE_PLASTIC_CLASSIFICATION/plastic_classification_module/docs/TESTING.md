# What is tested, and how

| Kind | Where | What it proves | What it does NOT prove |
|---|---|---|---|
| **SIMULATED** (fake backend, fake Qwen fn, synthetic images made inside tmp dirs) | `tests/simulated/` | decision logic: recyclable/non-recyclable/low-confidence/manual-review, empty & invalid crops, model-unavailable, Qwen fallback, RAM block, Qwen not woken needlessly, duplicate removal, leakage-free splits, label validation, metric code | anything about real accuracy |
| **REAL model** | `tests/real/` (skipped without weights + real test split) | `evaluate_plastic.py` runs on real images with your trained weights and prints real metrics | no accuracy threshold is asserted; judge the printed metrics yourself |

`train_plastic.py` was written but could NOT be executed where this module was authored (torch could not be installed there).
Run it on Colab; if it errors, that is a bug to report, not a model result.
