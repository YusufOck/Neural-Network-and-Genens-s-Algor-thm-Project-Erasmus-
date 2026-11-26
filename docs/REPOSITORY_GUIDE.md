# Repository and Evidence Guide

This guide answers three practical questions:

1. Where is each part of the implementation?
2. Where are the verified numerical results?
3. Which files should be shown as visual evidence during a professor review?

## Top-level structure

```text
NNGA/
├── README.md                         Main English project overview
├── PROJECT_PLAN.md                   Completion and delivery plan
├── requirements.txt                 Local Python dependencies
├── src/                             Python implementation
├── tests/                           Automated verification tests
├── config/                          Frozen experiment/taxonomy contracts
├── notebooks/                       Colab entrypoints
├── research/
│   ├── splits/                      Saved split/proxy manifests
│   ├── model_results/               Verified compact model evidence
│   ├── final_report/figures/        Publication-ready figures
│   └── *.md / *.csv                 Detailed historical research record
├── deliverables/                    Final English report
├── archive/                         Local dataset; not tracked by Git
└── artifacts/                       Large local audit outputs; not tracked by Git
```

## Core implementation

| File | Responsibility |
|---|---|
| [`src/train.py`](../src/train.py) | Unified SmallCNN/ResNet18/DenseNet121/EfficientNet-B0 training and evaluation |
| [`src/training_support.py`](../src/training_support.py) | Models, transforms, metrics, checkpointing, reports, curves, and prediction export |
| [`src/data_index.py`](../src/data_index.py) | Dataset discovery and accepted-image indexing |
| [`src/fixed_split_manifest.py`](../src/fixed_split_manifest.py) | Strict loading and SHA-256 validation of the fixed split |
| [`src/build_practical_leakage_reduced_split.py`](../src/build_practical_leakage_reduced_split.py) | Known-family construction and balanced split assignment |
| [`src/analyze_model_errors.py`](../src/analyze_model_errors.py) | Read-only error, confidence, calibration, and contact-sheet audit |
| [`src/run_gradcam_audit.py`](../src/run_gradcam_audit.py) | Hash-locked, frozen-sample Grad-CAM generation |
| [`src/run_ga_random_search.py`](../src/run_ga_random_search.py) | Equal-budget GA and random proxy search |
| [`src/run_ga_random_full_confirmation.py`](../src/run_ga_random_full_confirmation.py) | Full train/validation confirmation of frozen search winners |
| [`src/evaluate_ga_random_shared_test.py`](../src/evaluate_ga_random_shared_test.py) | Shared-test inference and paired McNemar comparison |

## Dataset and duplicate-audit code

| File | Responsibility |
|---|---|
| [`src/audit_dataset.py`](../src/audit_dataset.py) | Counts, dimensions, formats, corruption checks |
| [`src/audit_exact_duplicates.py`](../src/audit_exact_duplicates.py) | Byte-identical SHA-256 grouping |
| [`src/audit_perceptual_duplicates.py`](../src/audit_perceptual_duplicates.py) | Pilot pHash calculation and visual samples |
| [`src/score_duplicate_development.py`](../src/score_duplicate_development.py) | ORB/RANSAC, ECC, phase, SSIM, NMI, and correlation evidence |
| [`src/evaluate_duplicate_holdout.py`](../src/evaluate_duplicate_holdout.py) | Frozen-rule holdout evaluation |
| [`src/benchmark_candidate_generation_v2.py`](../src/benchmark_candidate_generation_v2.py) | Robust multi-view ResNet18 candidate-retrieval pilot |
| [`src/run_candidate_generator_v2_3_full_retrieval.py`](../src/run_candidate_generator_v2_3_full_retrieval.py) | Restart-safe full-data candidate retrieval |

The candidate-generator v2.4-v2.6 scripts remain in `src/` as a complete audit trail of hypotheses,
diagnostics, prospective reviews, and rejected zero-miss rules. Their failures are not broken
software; they document why the final project adopted a transparent practical split rather than
continuing to adapt retrieval rules indefinitely.

## Fixed split evidence

| Evidence | Location |
|---|---|
| Authoritative split | [`research/splits/practical_leakage_reduced_split_v1.csv`](../research/splits/practical_leakage_reduced_split_v1.csv) |
| Machine-readable report | [`research/splits/practical_leakage_reduced_split_v1_report.json`](../research/splits/practical_leakage_reduced_split_v1_report.json) |
| Frozen config | [`config/practical_leakage_reduced_split_v1.json`](../config/practical_leakage_reduced_split_v1.json) |
| Visual summary | [`research/final_report/figures/07_split_and_components.png`](../research/final_report/figures/07_split_and_components.png) |

The split manifest SHA-256 is
`999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`.

## Model-result folders

### Frozen-head ResNet18

Folder: [`research/model_results/resnet18_26leaf_baseline_v1/`](../research/model_results/resnet18_26leaf_baseline_v1/)

- `test_metrics.json`: headline test metrics and split hash;
- `history.csv`: epoch-by-epoch train/validation measurements;
- `learning_curves.png`: loss, accuracy, and macro-F1 curves;
- `leaf_confusion_matrix.png`: 26-class matrix;
- `parent_confusion_matrix.png`: derived eight-group matrix;
- `classification_reports.json`: per-class precision, recall, and F1;
- `artifact_hashes.json`: review-package integrity record.

### SmallCNN

Folder: [`research/model_results/smallcnn_26leaf_baseline_v1/`](../research/model_results/smallcnn_26leaf_baseline_v1/)

The folder follows the same structure and additionally includes a paired comparison against the
frozen-head ResNet18.

### Best model: last-block fine-tuned ResNet18

Folder: [`research/model_results/resnet18_lastblock_finetune_v1/`](../research/model_results/resnet18_lastblock_finetune_v1/)

- `test_metrics.json`: verified 26-leaf and eight-parent metrics;
- `history.csv`: epoch history;
- `paired_comparison.json`: paired comparison with earlier baselines;
- `verification_summary.json`: independent package and prediction checks;
- `error_analysis_v1/`: all 69 errors, calibration outputs, and 12 contact sheets;
- `gradcam_v1/`: 37 Grad-CAM panels summarized across seven contact sheets.

The large `best_model.pth` checkpoint is stored in Google Drive under
`MyDrive/NNGA_OUTPUTS/resnet18_lastblock_finetune_v1/` and is intentionally excluded from Git.

### GA versus random search

Folder: [`research/model_results/ga_random_shared_test_v1/`](../research/model_results/ga_random_shared_test_v1/)

The compact `verification_summary.json` contains both test metrics, checkpoint hashes, prediction
integrity checks, paired correctness counts, and the exact McNemar result.

## Visual evidence to show the professor

Open these files in order:

1. [Project workflow](../research/final_report/figures/01_project_workflow.png)
2. [Dataset modalities](../research/final_report/figures/02_dataset_modalities.png)
3. [Duplicate examples](../research/final_report/figures/05_duplicate_examples.png)
4. [Fixed split and components](../research/final_report/figures/07_split_and_components.png)
5. [Model comparison](../research/final_report/figures/09_model_comparison.png)
6. [Best-model learning curves](../research/final_report/figures/10_best_training_curves.png)
7. [26-leaf confusion matrix](../research/final_report/figures/11_best_leaf_confusion_matrix.png)
8. [Correct and incorrect predictions](../research/final_report/figures/17_prediction_examples.png)
9. [Grad-CAM examples](../research/final_report/figures/15_gradcam_examples.jpg)
10. [GA shared-test result](../research/final_report/figures/20_ga_shared_test.png)

For every one of the 69 best-model errors, use
[`error_analysis_v1/contact_sheets/`](../research/model_results/resnet18_lastblock_finetune_v1/error_analysis_v1/contact_sheets/).

For all seven Grad-CAM sheets, use
[`gradcam_v1/contact_sheets/`](../research/model_results/resnet18_lastblock_finetune_v1/gradcam_v1/contact_sheets/).

## Colab notebooks

| Notebook | Purpose |
|---|---|
| [`resnet18_26leaf_baseline_colab.ipynb`](../notebooks/resnet18_26leaf_baseline_colab.ipynb) | Frozen-head transfer-learning baseline |
| [`smallcnn_26leaf_baseline_colab.ipynb`](../notebooks/smallcnn_26leaf_baseline_colab.ipynb) | From-scratch baseline |
| [`resnet18_lastblock_finetune_colab.ipynb`](../notebooks/resnet18_lastblock_finetune_colab.ipynb) | Best-model last-block fine-tuning |
| [`resnet18_lastblock_gradcam_colab.ipynb`](../notebooks/resnet18_lastblock_gradcam_colab.ipynb) | Grad-CAM audit |
| [`ga_resnet18_lastblock_search_colab.ipynb`](../notebooks/ga_resnet18_lastblock_search_colab.ipynb) | GA/random proxy search |
| [`ga_random_full_confirmation_colab.ipynb`](../notebooks/ga_random_full_confirmation_colab.ipynb) | Full validation confirmation |
| [`ga_random_shared_test_colab.ipynb`](../notebooks/ga_random_shared_test_colab.ipynb) | Frozen shared-test evaluation |

## Final deliverable

The English report is
[`deliverables/NNGA_Final_Report_Mehmet_Yusuf_Ocak_Erasmus.docx`](../deliverables/NNGA_Final_Report_Mehmet_Yusuf_Ocak_Erasmus.docx).

It contains the complete academic narrative, 20 figures, 15 references, detailed methods,
numerical results, limitations, and appendices.
