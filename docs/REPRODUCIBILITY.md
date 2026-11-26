# Reproducibility Guide

## Supported environments

Local Windows commands use Python 3.10 explicitly:

```powershell
py -3.10 --version
py -3.10 -m pip install -r requirements.txt
```

The completed GPU experiments ran on Google Colab with a Tesla T4. Their runtime/package versions
are stored inside the run configurations and review artifacts; they should not be confused with
the local Python version.

## Dataset placement

Download the Kaggle Multi Cancer archive separately. The dataset is excluded from Git because it
is approximately 8.6 GiB.

Expected local structure:

```text
archive/
└── Multi Cancer/
    └── Multi Cancer/
        ├── ALL/
        ├── Brain Cancer/
        ├── Breast Cancer/
        ├── Cervical Cancer/
        ├── Kidney Cancer/
        ├── Lung and Colon Cancer/
        ├── Lymphoma/
        └── Oral Cancer/
```

## Verify the dataset

```powershell
py -3.10 .\src\audit_dataset.py --verify-all
py -3.10 .\src\audit_exact_duplicates.py
```

Expected accepted-image total: 130,000 after excluding two `*.bak.jpg` files.

## Run automated tests

```powershell
py -3.10 -m pytest -q
```

The tests cover dataset contracts, split invariants, training behavior, notebook contracts,
artifact hashes, candidate-generation logic, Grad-CAM selection, and GA/test isolation.

## Fixed split requirement

Reportable model training must use:

```powershell
--split-manifest .\research\splits\practical_leakage_reduced_split_v1.csv
```

The trainer refuses a manifest whose SHA-256 differs from
`999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`.

Do not regenerate a new random split for model comparison.

## Training implementation

The authoritative CLI is `src/train.py`. Its complete options are available with:

```powershell
py -3.10 .\src\train.py --help
```

The locked Colab notebooks record the exact reportable commands and protect completed output
directories. Use the notebooks for faithful reproduction rather than inventing new command-line
parameters.

### Frozen-head ResNet18

Notebook:
[`notebooks/resnet18_26leaf_baseline_colab.ipynb`](../notebooks/resnet18_26leaf_baseline_colab.ipynb)

### SmallCNN

Notebook:
[`notebooks/smallcnn_26leaf_baseline_colab.ipynb`](../notebooks/smallcnn_26leaf_baseline_colab.ipynb)

### Last-block fine-tuned ResNet18

Notebook:
[`notebooks/resnet18_lastblock_finetune_colab.ipynb`](../notebooks/resnet18_lastblock_finetune_colab.ipynb)

This run requires the prior frozen-head best checkpoint. The notebook validates its SHA-256 before
training.

## Post-hoc analysis

### Error and confidence audit

Implementation: [`src/analyze_model_errors.py`](../src/analyze_model_errors.py)

The tool consumes the verified review ZIP and local dataset, checks predictions/probabilities,
extracts all errors, and creates contact sheets plus calibration measurements. It is read-only
with respect to model weights.

### Grad-CAM

Notebook:
[`notebooks/resnet18_lastblock_gradcam_colab.ipynb`](../notebooks/resnet18_lastblock_gradcam_colab.ipynb)

Implementation: [`src/run_gradcam_audit.py`](../src/run_gradcam_audit.py)

The checkpoint, split, predictions, and preselected sample manifest are all hash-verified.

## Genetic algorithm experiment

Run in this order:

1. [`ga_resnet18_lastblock_search_colab.ipynb`](../notebooks/ga_resnet18_lastblock_search_colab.ipynb)
2. [`ga_random_full_confirmation_colab.ipynb`](../notebooks/ga_random_full_confirmation_colab.ipynb)
3. [`ga_random_shared_test_colab.ipynb`](../notebooks/ga_random_shared_test_colab.ipynb)

The first two stages contain no test evaluation. The third evaluates already selected checkpoints
without retraining or test-based reselection.

## Expected output contract

A standard classifier run writes:

- `best_model.pth` and `last_checkpoint.pth`;
- `test_metrics.json`;
- `test_predictions.csv`;
- `history.csv`;
- `classification_reports.json`;
- leaf and parent confusion matrices;
- learning curves;
- run/split configuration and hash manifests.

Large checkpoints and full prediction packages are kept in Google Drive under `NNGA_OUTPUTS` and
excluded from Git. Compact verified metrics, figures, and audit summaries are tracked under
`research/model_results/`.

## Safety rules for completed experiments

- Never overwrite an existing completed output directory.
- Resume an interrupted run only with the identical frozen contract.
- Use validation—not test—to select checkpoints or hyperparameters.
- Record any changed model, seed, split, or budget as a new experiment.
- Do not report smoke-test metrics as project performance.
