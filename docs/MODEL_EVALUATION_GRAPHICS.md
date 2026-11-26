# Model Evaluation and Graphics Catalogue

This catalogue answers two questions: which measurements exist for each model, and where the
professor-facing evidence is stored.

## What each measurement shows

| Evidence | Interpretation |
|---|---|
| Accuracy | Fraction of all images classified correctly |
| Macro-F1 | Mean of the 26 per-class F1 scores, giving every leaf equal weight |
| Balanced accuracy | Mean per-class recall; equivalent to ordinary accuracy here because every leaf has the same test count |
| Precision / recall / F1 per class | Which classes produce false positives or false negatives |
| Learning curves | Train and validation loss, accuracy, and macro-F1 across epochs; used to inspect convergence and overfitting |
| Leaf confusion matrix | All 26 true-versus-predicted leaf counts |
| Parent confusion matrix | The same predictions mapped to eight parent groups |
| Reliability diagram / ECE | Whether confidence agrees with observed correctness |
| Error contact sheets | The actual images behind every best-model error |
| Grad-CAM sheets | Coarse regions influencing selected class scores |
| Paired comparison | Which shared test images were uniquely correct for one model |

The test set is evaluated once after validation-based checkpoint selection. Therefore there is no
per-epoch **test** curve. Monitoring the test set every epoch would make it part of model selection.

## Coverage by experiment

| Experiment | Headline metrics | Learning curves/history | Leaf/parent matrices | Per-class report | Additional evidence |
|---|---|---|---|---|---|
| Frozen-head ResNet18 | Yes | Yes | Yes | Yes | Review hashes and console log |
| SmallCNN | Yes | Yes | Yes | Yes | Paired baseline comparison |
| Last-block ResNet18 | Yes | Yes | Yes | Yes | 69-error audit, calibration, predictions, Grad-CAM |
| GA proxy search | Validation macro-F1 for 12 GA + 12 random candidates | Per-trial histories; aggregate result table | Not applicable | Not applicable | Equal-budget search summary |
| GA/random full confirmation | Full-validation macro-F1 | One curve/history per winner | Not applicable before test | Not applicable before test | Frozen winner decision |
| GA/random shared test | Accuracy, macro-F1, balanced accuracy | Training already completed | Yes, both candidates | Yes, both candidates | Exact paired McNemar comparison |

## Frozen-head ResNet18

Folder: [`research/model_results/resnet18_26leaf_baseline_v1/`](../research/model_results/resnet18_26leaf_baseline_v1/)

- [learning curves](../research/model_results/resnet18_26leaf_baseline_v1/learning_curves.png)
- [26-leaf confusion matrix](../research/model_results/resnet18_26leaf_baseline_v1/leaf_confusion_matrix.png)
- [eight-parent confusion matrix](../research/model_results/resnet18_26leaf_baseline_v1/parent_confusion_matrix.png)
- [test metrics](../research/model_results/resnet18_26leaf_baseline_v1/test_metrics.json)
- [per-class report](../research/model_results/resnet18_26leaf_baseline_v1/classification_reports.json)
- [epoch history](../research/model_results/resnet18_26leaf_baseline_v1/history.csv)

## SmallCNN

Folder: [`research/model_results/smallcnn_26leaf_baseline_v1/`](../research/model_results/smallcnn_26leaf_baseline_v1/)

- [learning curves](../research/model_results/smallcnn_26leaf_baseline_v1/learning_curves.png)
- [26-leaf confusion matrix](../research/model_results/smallcnn_26leaf_baseline_v1/leaf_confusion_matrix.png)
- [eight-parent confusion matrix](../research/model_results/smallcnn_26leaf_baseline_v1/parent_confusion_matrix.png)
- [test metrics](../research/model_results/smallcnn_26leaf_baseline_v1/test_metrics.json)
- [per-class report](../research/model_results/smallcnn_26leaf_baseline_v1/classification_reports.json)
- [epoch history](../research/model_results/smallcnn_26leaf_baseline_v1/history.csv)

## Best model: last-block fine-tuned ResNet18

Folder: [`research/model_results/resnet18_lastblock_finetune_v1/`](../research/model_results/resnet18_lastblock_finetune_v1/)

- [learning curves](../research/model_results/resnet18_lastblock_finetune_v1/learning_curves.png)
- [26-leaf confusion matrix](../research/model_results/resnet18_lastblock_finetune_v1/leaf_confusion_matrix.png)
- [eight-parent confusion matrix](../research/model_results/resnet18_lastblock_finetune_v1/parent_confusion_matrix.png)
- [test metrics](../research/model_results/resnet18_lastblock_finetune_v1/test_metrics.json)
- [per-class report](../research/model_results/resnet18_lastblock_finetune_v1/classification_reports.json)
- [epoch history](../research/model_results/resnet18_lastblock_finetune_v1/history.csv)
- [reliability diagram](../research/model_results/resnet18_lastblock_finetune_v1/error_analysis_v1/reliability_diagram.png)
- [all error contact sheets](../research/model_results/resnet18_lastblock_finetune_v1/error_analysis_v1/contact_sheets/)
- [all Grad-CAM contact sheets](../research/model_results/resnet18_lastblock_finetune_v1/gradcam_v1/contact_sheets/)

The report-ready versions are [best-model curves](../research/final_report/figures/10_best_training_curves.png),
[leaf matrix](../research/final_report/figures/11_best_leaf_confusion_matrix.png),
[parent matrix](../research/final_report/figures/12_best_parent_confusion_matrix.png),
[prediction examples](../research/final_report/figures/17_prediction_examples.png), and
[Grad-CAM examples](../research/final_report/figures/15_gradcam_examples.jpg).

## Genetic algorithm and random search

The proxy candidate results are in
[`ga_resnet18_lastblock_search_v1/`](../research/model_results/ga_resnet18_lastblock_search_v1/),
including `search_results.csv` and `search_summary.json`. The report comparison is
[GA proxy comparison](../research/final_report/figures/19_ga_proxy_comparison.png).

Full-validation winner evidence is in
[`ga_random_full_confirmation_v1/`](../research/model_results/ga_random_full_confirmation_v1/):

- [GA winner learning curve](../research/model_results/ga_random_full_confirmation_v1/candidates/ga/learning_curves.png)
- [random winner learning curve](../research/model_results/ga_random_full_confirmation_v1/candidates/random/learning_curves.png)
- both histories and the frozen `confirmation_summary.json`.

Shared-test evidence is in
[`ga_random_shared_test_v1/`](../research/model_results/ga_random_shared_test_v1/):

- [GA leaf matrix](../research/model_results/ga_random_shared_test_v1/candidates/ga/leaf_confusion_matrix.png)
- [GA parent matrix](../research/model_results/ga_random_shared_test_v1/candidates/ga/parent_confusion_matrix.png)
- [random leaf matrix](../research/model_results/ga_random_shared_test_v1/candidates/random/leaf_confusion_matrix.png)
- [random parent matrix](../research/model_results/ga_random_shared_test_v1/candidates/random/parent_confusion_matrix.png)
- test metrics and per-class reports for both candidates;
- `shared_test_summary.json` with the paired correctness counts and McNemar result.

The professor-facing summary is
[GA shared-test comparison](../research/final_report/figures/20_ga_shared_test.png).

## Completeness statement

All three primary architecture experiments have accuracy, macro-F1, balanced accuracy,
epoch-by-epoch learning evidence, per-class precision/recall/F1, and both leaf and parent confusion
matrices. The GA/random experiment has proxy-search evidence, full-validation learning curves for
both frozen winners, and complete shared-test metrics and matrices. The best model additionally has
confidence calibration, every error image, representative correct predictions, and Grad-CAM.
