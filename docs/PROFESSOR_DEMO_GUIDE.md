# Professor Demonstration Guide

This is a short navigation sequence for demonstrating the completed project directly from the
repository and final report. A slide deck is intentionally deferred until the professor confirms
whether a separate presentation is required.

## Suggested 10-minute order

### 1. Define the task

Open [`README.md`](../README.md) and explain:

- 130,000 images;
- eight heterogeneous parent groups;
- 26 detailed leaf labels;
- one 26-class model with deterministic parent mapping.

Show [dataset modalities](../research/final_report/figures/02_dataset_modalities.png).

### 2. Explain why the split required special care

Show [duplicate examples](../research/final_report/figures/05_duplicate_examples.png), then
[split construction](../research/final_report/figures/07_split_and_components.png).

Key sentence:

> Because the downloaded dataset was already augmented, we grouped every known exact duplicate
> and frozen reviewed relation before partition assignment. Every component remains entirely in
> train, validation, or test, and every model uses the same saved manifest.

### 3. Show the controlled model comparison

Show [model comparison](../research/final_report/figures/09_model_comparison.png) and explain the
progression:

1. frozen ImageNet features plus a new head;
2. compact CNN trained from scratch;
3. adaptation of the final ResNet18 block;
4. GA/random hyperparameter variants.

### 4. Show numerical and visual evidence

Open:

- [best-model training curves](../research/final_report/figures/10_best_training_curves.png);
- [26-leaf confusion matrix](../research/final_report/figures/11_best_leaf_confusion_matrix.png);
- [representative predictions](../research/final_report/figures/17_prediction_examples.png);
- [all error sheets](../research/model_results/resnet18_lastblock_finetune_v1/error_analysis_v1/contact_sheets/).

State the main result: 99.6462% leaf accuracy and macro-F1 on 19,500 fixed test images, with 69
leaf errors. Do not use parent accuracy as a replacement for leaf performance.

### 5. Explain Grad-CAM

Show [Grad-CAM examples](../research/final_report/figures/15_gradcam_examples.jpg) and the
[complete sheets](../research/model_results/resnet18_lastblock_finetune_v1/gradcam_v1/contact_sheets/).

Explain that the heatmap shows coarse influence on a class score, not a tumor bounding box or
pathologist-confirmed lesion.

### 6. Explain the genetic algorithm

Show [GA workflow](../research/final_report/figures/18_ga_workflow.png) and
[shared-test comparison](../research/final_report/figures/20_ga_shared_test.png).

Key conclusion:

> Under the same 12-candidate budget, GA won the proxy and validation comparison. On shared test
> it was better by one image, but exact paired McNemar p=1.0; therefore no statistically
> distinguishable advantage was established.

### 7. Finish with limitations

Explain that the work is a strong dataset benchmark but not clinical validation because modalities
and data sources differ, patient IDs are unavailable, and unknown near relatives may remain.

### 8. Optional live inference

Open the [live demo notebook in Colab](https://colab.research.google.com/github/YusufOck/Neural-Network-and-Genens-s-Algor-thm-Project-Erasmus-/blob/main/notebooks/resnet18_live_demo_colab.ipynb).
First show the frozen `37/37` replay result, then classify one chosen image and display its saved
prediction card. The replay is stronger evidence than showing only a hand-picked correct example.
See the complete [Live Inference Demonstration guide](LIVE_DEMO.md).

## Files to keep open before meeting

1. [`deliverables/NNGA_Final_Report_Mehmet_Yusuf_Ocak_Erasmus.docx`](../deliverables/NNGA_Final_Report_Mehmet_Yusuf_Ocak_Erasmus.docx)
2. [`README.md`](../README.md)
3. [`docs/REPOSITORY_GUIDE.md`](REPOSITORY_GUIDE.md)
4. [`research/final_report/figures/`](../research/final_report/figures/)
5. [`error_analysis_v1/contact_sheets/`](../research/model_results/resnet18_lastblock_finetune_v1/error_analysis_v1/contact_sheets/)
6. [`gradcam_v1/contact_sheets/`](../research/model_results/resnet18_lastblock_finetune_v1/gradcam_v1/contact_sheets/)
7. [`docs/MODEL_EVALUATION_GRAPHICS.md`](MODEL_EVALUATION_GRAPHICS.md)

## If the professor asks for code

- training: [`src/train.py`](../src/train.py);
- split: [`src/build_practical_leakage_reduced_split.py`](../src/build_practical_leakage_reduced_split.py);
- Grad-CAM: [`src/run_gradcam_audit.py`](../src/run_gradcam_audit.py);
- GA search: [`src/run_ga_random_search.py`](../src/run_ga_random_search.py);
- live inference: [`src/predict_image.py`](../src/predict_image.py);
- notebooks: [`notebooks/`](../notebooks/);
- tests: [`tests/`](../tests/).
