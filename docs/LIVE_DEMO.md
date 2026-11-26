# Live Inference Demonstration

The live demonstration classifies one selected dataset image with the completed best model. It is
an inference-only workflow: it does not train, fine-tune, select, or modify a model.

## Open the notebook

[Open the live demo directly in Google Colab](https://colab.research.google.com/github/YusufOck/Neural-Network-and-Genens-s-Algor-thm-Project-Erasmus-/blob/main/notebooks/resnet18_live_demo_colab.ipynb)

The notebook expects these existing Drive files:

- `MyDrive/NNGA_DATA/Multi Cancer.zip`;
- `MyDrive/NNGA_OUTPUTS/resnet18_lastblock_finetune_v1/best_model.pth`.

The model checkpoint is accepted only when its SHA-256 is
`538f2891075f0870e450cd6aa7eda0a5e06ecb73eefd14a4f6a0cd869759d54d`.

## What the demonstration does

1. Mounts Google Drive and updates the Git repository.
2. Verifies the checkpoint, inference script, configuration, and frozen sample manifest by
   SHA-256.
3. Extracts the dataset to Colab's temporary disk without editing the Drive archive.
4. Replays 37 previously frozen test cases.
5. Classifies one selected image and displays a prediction card.
6. Saves a hash-manifested evidence ZIP to Drive and downloads it.

For one image, [`src/predict_image.py`](../src/predict_image.py):

- decodes the image as RGB;
- resizes/crops it to the same 224×224 input used during evaluation;
- applies the same ImageNet normalization;
- loads the saved ResNet18 weights without downloading or substituting weights;
- computes 26 softmax probabilities;
- returns the highest-probability leaf class and top-five probabilities;
- deterministically maps the predicted leaf to its parent group;
- records the image and checkpoint hashes;
- creates `prediction.json` and `prediction_card.png`.

If the selected image is inside the labeled dataset, the card also shows the folder-derived true
leaf and whether the prediction matches it. For an external image, no ground-truth correctness
claim is made.

## How to prove that it works

The professor should not be asked to trust one visually selected successful example. The notebook
first runs a frozen replay:

- 37 test images were selected before this live-demo implementation;
- the set contains one correct control from each of the 26 leaf classes and 11 documented errors;
- the checkpoint must reproduce all 37 saved leaf predictions;
- the absolute confidence difference must remain within `0.0001`;
- any failed row stops the notebook with a non-zero result.

Successful output contains:

```text
PROOF STATUS: 37/37 frozen predictions reproduced.
```

The permanent evidence is:

| File | Meaning |
|---|---|
| `frozen_replay/verification_report.json` | Overall pass/fail status, hashes, device, row counts, and maximum confidence difference |
| `frozen_replay/verification_rows.csv` | Expected versus reproduced class and confidence for every frozen image |
| `single_prediction/prediction.json` | Machine-readable result for the demonstrated image |
| `single_prediction/prediction_card.png` | Professor-facing image and prediction card |
| `evidence_hashes.json` | SHA-256 and byte size of every evidence file |

These files are packaged as
`MyDrive/NNGA_OUTPUTS/resnet18_live_demo_v1_review_package.zip`.

There is already independent evidence that the same checkpoint/preprocessing reproduced the same
37 predictions during the completed Grad-CAM audit: the recorded maximum confidence difference
was `3.4571e-06`. The new notebook repeats that check through the standalone live-inference path.

## What to say during the demonstration

> This image is passed through the same preprocessing and the same hash-verified checkpoint used
> for the fixed test evaluation. The model predicts one of the 26 detailed dataset classes. The
> eight-group result is derived from that leaf prediction. Before showing this selected example,
> I replayed 37 frozen cases and required all saved predictions to be reproduced.

The output is dataset-specific educational classification. It must not be described as clinical
diagnosis, universal cancer detection, cancer staging, or lesion localization.

## Local verification status

The inference program has automated tests for deterministic prediction, prediction-card creation,
manifest replay, checkpoint-hash rejection, and class-order rejection. The actual 37-image replay
must be executed in Colab because the 44.7 MB best checkpoint is intentionally stored in Drive,
not Git.
