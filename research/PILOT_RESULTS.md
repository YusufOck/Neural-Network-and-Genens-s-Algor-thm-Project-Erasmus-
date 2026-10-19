# Local Pilot Results

Date: 2026-10-14

These runs test feasibility and reproduction. They are not the final model comparison.

| Status | Target | Model | Data | Epochs | Test accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|---:|
| Reproduction pilot | 26 subclasses | Frozen ImageNet DenseNet121 | 100/class | 2 | 78.21% | 76.18% |
| Assigned-task pilot | 8 main classes | Frozen ImageNet ResNet18 | 200/class | 3 | 97.08% | 97.02% |

## Interpretation

The 26-class pilot adapts the inspected RoupB setup. Its smaller dataset and two-epoch budget are
not comparable to the repository's full 5,000/class, ten-epoch run; therefore the repository's
reported 92.21% DenseNet121 result is not yet fully reproduced.

The eight-class pilot comfortably exceeds the project's 85% target. However, the top-level
classes have conspicuously different modalities and the source data was already augmented. A
random image-level split may therefore overestimate generalization. The next experiment is not
"train longer immediately"; it is an exact/near-duplicate and split-leakage audit.

## Artifacts

- `artifacts/pilots/reproduction-26-densenet121-100x2/`
- `artifacts/pilots/assigned-8-resnet18-200x3/`

Each directory contains the saved split manifest, configuration, best checkpoint, history,
classification report, metrics JSON and confusion matrix.

