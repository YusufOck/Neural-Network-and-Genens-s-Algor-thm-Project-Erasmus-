# NNGA Multi-Cancer Project — Completion and Delivery Plan

**Updated:** 26 November 2026<br>
**Student:** Mehmet Yusuf Ocak — Erasmus

## Objective

Build and evaluate a reproducible neural-network pipeline for the supplied Multi Cancer dataset,
using 26 detailed leaf labels and deriving the professor-requested eight parent-group result from
each leaf prediction. Add a bounded genetic-algorithm hyperparameter experiment and compare it
with equal-budget random search.

## Completed scientific work

- [x] Full 130,000-image integrity and metadata audit.
- [x] Explicit 8-parent / 26-leaf taxonomy with safe label semantics.
- [x] Exact duplicate audit using SHA-256.
- [x] Near-duplicate method development using pHash, robust retrieval, ORB/RANSAC, ECC,
  phase correlation, SSIM, NMI, correlation, and visual review.
- [x] Fixed 70/15/15, 26-leaf-stratified, known-family-grouped split.
- [x] Frozen-head ImageNet ResNet18 baseline.
- [x] SmallCNN from-scratch baseline.
- [x] ResNet18 last-block fine-tuning ablation.
- [x] Per-class metrics, confusion matrices, prediction export, and paired comparisons.
- [x] Full best-model error and confidence audit.
- [x] Frozen-sample Grad-CAM audit.
- [x] Equal-budget GA versus random-search proxy experiment.
- [x] Full train/validation confirmation of both search winners.
- [x] Shared 19,500-image test evaluation and exact McNemar comparison.
- [x] English academic final report.
- [x] English professor-facing GitHub documentation.
- [x] Hash-verified live-inference CLI and locked Colab demonstration notebook.
- [x] Complete compact graphics catalogue for all primary and GA/random experiments.

## Current principal result

The strongest model is the manually configured ResNet18 last-block fine-tuning run:

- leaf accuracy: 99.6462%;
- leaf macro-F1: 99.6462%;
- balanced accuracy: 99.6462%;
- derived-parent accuracy: 99.9897%;
- 69 leaf errors and two parent errors on 19,500 fixed test images.

The GA-selected and random-selected models are statistically indistinguishable on the shared test
set: 52 GA-only correct versus 51 random-only correct, exact McNemar `p = 1.0`.

## Deliverables

| Deliverable | Status | Location |
|---|---|---|
| Source code and tests | Complete | `src/`, `tests/` |
| Reproducible Colab notebooks | Complete | `notebooks/` |
| Split and experiment evidence | Complete | `research/` |
| Visual evidence | Complete | `research/final_report/figures/` and model-result folders |
| English final report | Complete | `deliverables/NNGA_Final_Report_Mehmet_Yusuf_Ocak_Erasmus.docx` |
| English GitHub documentation | Complete | `README.md`, `docs/`, `research/README.md` |
| Live demo implementation | Complete; full 37-case Drive replay pending | `src/predict_image.py`, `notebooks/resnet18_live_demo_colab.ipynb` |
| Presentation | Deferred | Ask the professor whether presenting directly from the report is acceptable |

## Remaining delivery actions

1. Run the live-demo notebook once against the Drive checkpoint and retain its evidence ZIP.
2. Review the English report and repository documentation with the student.
3. Ask the professor whether a separate slide deck is required or whether the report may be used
   during the defense.
4. If requested, create the presentation from the already verified report figures and results.
5. Prepare a short oral-defense script and technical question/answer sheet.
6. Perform one final repository-link, report-render, and Git cleanliness check before submission.

## Frozen scientific rules

- Do not regenerate or edit the accepted split after observing test results.
- Do not overwrite completed run directories or Drive checkpoints.
- Do not choose models or hyperparameters from test results.
- Do not call the split patient-level or guaranteed leakage-free.
- Do not describe parent accuracy as cancer-presence or cancer-stage accuracy.
- Do not describe Grad-CAM as medical localization.
- Do not claim that GA is generally superior to random search from this single-seed experiment.

For the full evidence chain, read [Methods and Validation](docs/METHODS_AND_VALIDATION.md) and
[Experiments and Results](docs/EXPERIMENTS_AND_RESULTS.md).
