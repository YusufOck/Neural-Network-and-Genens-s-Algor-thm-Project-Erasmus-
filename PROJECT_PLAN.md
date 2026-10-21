# NNGA Multi-Cancer Project Plan

Updated: 2026-10-19

Current completion state and acceptance boundaries are tracked in
[research/PROJECT_STATUS_TR.md](research/PROJECT_STATUS_TR.md). In particular, a completed
two-subclass pHash pilot is not a completed 26-subclass near-duplicate audit.
The planned hybrid near-duplicate method and its calibration requirements are defined in
[research/NEAR_DUPLICATE_METHOD_PLAN_TR.md](research/NEAR_DUPLICATE_METHOD_PLAN_TR.md).

## 1. Project question

Professor-aligned baseline question:

> Can a reproducible neural-network pipeline classify each image into one of the eight
> top-level Multi Cancer dataset groups with at least 85% test accuracy and macro-F1?

The eight classes match the professor's repeated wording, but they are not eight consistent
cancer diagnoses. They are source/organ groups, and several contain both normal/benign and
diseased images. The current working direction is a 26-leaf prediction whose output is also
collapsed to the eight parent groups for evaluation. This preserves both levels without claiming
labels that the dataset does not contain.

This is an educational dataset-classification project, not a clinically validated diagnostic
system. The top-level classes mix organs, imaging modalities, benign tissue, normal tissue and
malignant subtypes.

See [research/LABEL_SEMANTICS_AUDIT_TR.md](research/LABEL_SEMANTICS_AUDIT_TR.md) before making
any target-label decision. An eight-class score must be described as top-level dataset-group
classification, not healthy/cancer classification or cancer-stage prediction.

## 2. Why the first 50% is not a result

The completed 320-image run was a one-epoch SmallCNN smoke test. Its purpose was to verify:

- CUDA and the RTX 3050 work;
- the dataset can be indexed;
- training, validation, testing and artifact saving run end to end.

It used only 224 training and 48 test images. It is excluded from the report's model-comparison
table.

## 3. Evidence collected before training

- All 130,000 accepted images open successfully.
- Two extra `*.bak.jpg` files are excluded.
- All accepted images are RGB, but the README's uniform-format claim is not exact: 125,000 files
  contain JPEG data, while all 5,000 `breast_malignant` files have PNG content under `.jpg` names.
- Three `all_early` images are 640x640; the other 129,997 are 512x512.
- SHA-256 found 1,318 exact-duplicate groups containing 1,355 redundant copies. They occur only
  in the five Lung/Colon leaves and `kidney_normal`; none cross leaf labels.
- The dataset was already augmented by its compiler.
- Existing work mostly solves either eight separate organ-specific subtype problems or one
  26-subtype problem; this is not identical to the professor's requested eight-class task.
- Published notebook results commonly use random image-level splits, which may leak augmented
  variants across train and test.

See [research/EXISTING_WORKS.md](research/EXISTING_WORKS.md).
The complete source and label inventory is in
[research/DATASET_MASTER_AUDIT_TR.md](research/DATASET_MASTER_AUDIT_TR.md).

## 4. Experimental phases

### Phase 0 - Confirm target semantics

1. Review the 8/26 label map with Mehmet.
2. Define a structured output for every leaf: parent group, detailed dataset label, and only the
   health/tumor meaning explicitly supported by that source dataset.
3. Do not contact the professor yet. First prepare and understand the concrete 26-leaf plus
   eight-parent plan; ask for scope confirmation later, before final training/report claims, if
   still necessary.

### Phase A - Split and leakage audit

1. Keep every verified exact-duplicate group in one partition.
2. Estimate near-duplicate/augmentation image families with a separately validated perceptual
   method.
3. Save a fixed stratified 70/15/15 split for literature comparison.
4. If reliable duplicate families exist, save a group-aware split that keeps a family in only
   one partition.

### Phase B - Reproduce an existing approach

Adapt the inspected 26-class frozen-backbone setup to the local pipeline:

- image size 128;
- batch size 16;
- frozen ImageNet backbone;
- DenseNet121 and/or ResNet18;
- initially a small subset, then full data only if the pilot is correct.

This establishes whether the reported 26-class results are reproducible on this computer.

### Phase C - Professor-aligned eight-group comparison

Run all models on the same saved split:

1. SmallCNN from scratch.
2. ResNet18 transfer learning.
3. DenseNet121 transfer learning.
4. EfficientNet-B0 transfer learning.

Start with a frozen backbone; fine-tune the last block only after the classifier-head baseline.

### Phase D - Evaluation

For every final run, save:

- accuracy;
- macro-F1;
- balanced accuracy;
- precision/recall/F1 per class;
- confusion matrix;
- learning curves;
- runtime and configuration.

Acceptance target: both test accuracy and macro-F1 at or above 0.85. A high score is not accepted
without checking for leakage and inspecting per-class recall.

### Phase E - Subtype and original contribution

Subject to the professor's scope answer, the strongest defensible contribution is:

- controlled comparison of multiple architectures;
- a 26-leaf or hierarchical top-level/subtype extension that distinguishes normal/benign and
  disease subtypes where the dataset actually provides those labels;
- naive versus leakage-aware split analysis;
- Grad-CAM analysis of correct and incorrect predictions;
- optionally, a small genetic algorithm for learning rate, dropout, weight decay and fine-tuning
  policy, compared with random search under the same trial budget.

### Phase F - Report and defense

Prepare a 15-20 page English report and presentation. Clearly label:

- borrowed methods;
- locally reproduced baselines;
- original experiments and conclusions;
- dataset and clinical-validity limitations.

## 5. Immediate next action

The unified 8/26-class trainer and both short pilots are complete. The eight-class ResNet18 pilot
reached 97.08% test accuracy and 97.02% macro-F1, while the reduced 26-class DenseNet121
reproduction reached 78.21% and 76.18% respectively. These are documented in
[research/PILOT_RESULTS.md](research/PILOT_RESULTS.md).

Do not start a new training run yet. The explicit 26-leaf taxonomy artifact, full-dataset
image-metadata audit, full-dataset exact-duplicate audit and two-subclass pHash **pilot** are
complete within their stated scopes. Mehmet's complete label-semantics review and the
26-subclass near-duplicate audit are not complete. The pHash pilot found a
strong modality effect: MRI produces many false-positive candidates even at low distances, while
the reviewed lung-histology candidates remain strongly geometrically consistent through distance
12. No global threshold has been selected. The hybrid method plan is approved and an all-26-leaf
calibration-set v1 draft has been generated: 1,342 pairs containing 1,064 controlled/SHA positives
and 278 unreviewed candidates. Traceable contact sheets and a review worksheet now cover all 278
Pair IDs with zero mapping errors. The immediate next action is to label only the 139 development
candidates in small batches; the 139 holdout candidates remain unseen until the method rules are
frozen. None of those rows is currently assumed to be a true negative. The existing 120 pairs remain a two-subclass, pHash-biased legacy pilot and
are not the final calibration set. Candidate CSV, review worksheet and all 20 pilot contact sheets
share stable `P0001`–`P0120` Pair IDs. Calibrate pHash candidate generation, ORB/RANSAC, ECC/phase
correlation and aligned SSIM/NMI using controlled positives, hard/random negatives and hold-out
pairs across blood-smear microscopy, MRI, H&E histopathology, Pap-smear microscopy and CT. Only
after every subclass is covered
and compatible modalities/source groups have been checked for cross-subclass perceptual matches
may augmentation families and a fixed reusable split be frozen. Professor contact remains deferred.

