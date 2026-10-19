# Existing Work on the Same Dataset

Updated: 2026-10-14

The external source snapshots used for this review are stored under `research/external/` and are
ignored by version control. No third-party notebook has been executed blindly.

## 1. Dataset-owner ResNet50V2 notebook

Source: https://www.kaggle.com/code/obulisainaren/multiple-cancer-classification-resnet50

What it actually does:

- loops over the eight top-level folders;
- trains a separate classifier for the 2-5 subtypes inside each folder;
- uses ImageNet-pretrained ResNet50V2 with a frozen backbone;
- resizes to 224x224, uses generator batch size 32 and trains 20 epochs;
- uses an 80/20 training-validation split;
- does not create one eight-class model and does not define an independent test set.

The downloaded source contains no stored execution output, so no accuracy from this notebook is
treated as verified.

## 2. RoupB/cancer-app

Source: https://github.com/RoupB/cancer-app

What it does:

- creates a single 26-subtype classification problem;
- randomly splits each subtype 70%/15%/15%;
- uses up to all 5,000 images per subtype;
- resizes to 128x128;
- freezes ImageNet backbones and trains the classifier for 10 epochs;
- compares multiple architectures and includes Grad-CAM/LIME and statistical comparison code.

Reported test accuracy found in stored notebook outputs:

| Model | Reported test accuracy |
|---|---:|
| MobileNetV3 | 84.90% |
| ResNet18 | 88.02% |
| VGG16 | 88.44% |
| EfficientNet-B0 | 89.08% |
| MobileNetV2 | 90.46% |
| DenseNet121 | 92.21% |
| ViT-B16 | 94.82% |

These values are **reported, not locally reproduced**. Because the source dataset was already
augmented and the repository performs a random image-level split, near-duplicate leakage is a
plausible source of optimistic performance.

A reduced local reproduction has now been run with frozen ImageNet DenseNet121, 100 images per
subtype, 128px input, batch size 16 and two epochs. It achieved 78.21% test accuracy and 76.18%
macro-F1. This verifies the adapted pipeline but does not yet reproduce the full-data 92.21%
reported accuracy.

## 3. Samayun subtype ensemble

Source: https://www.kaggle.com/code/samayun/muti-cancer-subtype-classification

What it does:

- labels images using their immediate parent folder, producing 26 classes;
- defines VGG16, DenseNet201, ResNet50V2, EfficientNet-B0 and a custom ViT;
- defines hard voting, soft voting and an XGBoost/logistic-regression stacking model;
- contains Grad-CAM, LIME and SHAP hooks;
- currently configures 3% of the data, 224x224, batch size 32 and 50 epochs.

The downloaded source contains no execution output. Its ensemble is therefore a design to test,
not an established result. It is also too large to use as the first experiment on a 4 GB laptop
GPU.

## Conclusion for this project

The first fair reproduction target is the simple frozen-backbone 26-class setup, especially
DenseNet121 or ResNet18. The assigned project remains the professor-aligned eight-class problem.
Both can share one PyTorch pipeline, saved splits and evaluation code. Ensemble and ViT work
comes only after individual models are reproduced correctly.

