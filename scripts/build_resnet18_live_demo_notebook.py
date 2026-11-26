from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "resnet18_live_demo_colab.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).strip().splitlines(True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


cells = [
    markdown(
        """
        # ResNet18 live single-image demonstration and frozen replay

        This notebook performs **inference only** with the completed 26-leaf ResNet18 checkpoint.
        It never trains, fine-tunes, or changes model weights. It provides two independent pieces
        of evidence for a professor demonstration:

        1. replay the 37 frozen Grad-CAM audit samples and require exact class reproduction;
        2. classify one chosen dataset image and save a readable prediction card.

        The checkpoint, inference code, configuration, and frozen sample manifest are verified by
        SHA-256 before inference. The output is educational dataset classification, not clinical
        diagnosis or cancer-stage prediction.
        """
    ),
    code(
        """
        from pathlib import Path
        import hashlib, json, shutil, subprocess, sys, time, zipfile

        assert 'google.colab' in sys.modules, 'This notebook is intended for Google Colab.'
        from google.colab import drive, files
        drive.mount('/content/drive')
        print('Google Drive mounted.')
        """
    ),
    markdown("## 1. Load the locked project, checkpoint, and dataset"),
    code(
        """
        REPOSITORY_URL = 'https://github.com/YusufOck/Neural-Network-and-Genens-s-Algor-thm-Project-Erasmus-.git'
        PROJECT_ROOT = Path('/content/NNGA')
        ZIP_PATH = Path('/content/drive/MyDrive/NNGA_DATA/Multi Cancer.zip')
        LOCAL_DATA_BASE = Path('/content/NNGA_DATA')
        EXTRACT_MARKER = LOCAL_DATA_BASE / '.multi_cancer_extract_complete'
        CHECKPOINT = Path('/content/drive/MyDrive/NNGA_OUTPUTS/resnet18_lastblock_finetune_v1/best_model.pth')
        OUTPUT_ROOT = Path('/content/drive/MyDrive/NNGA_OUTPUTS/resnet18_live_demo_v1')
        VERIFICATION_OUTPUT = OUTPUT_ROOT / 'frozen_replay'
        SINGLE_OUTPUT = OUTPUT_ROOT / 'single_prediction'
        REVIEW_ZIP = OUTPUT_ROOT.parent / 'resnet18_live_demo_v1_review_package.zip'
        EXPECTED_GROUPS = ('ALL', 'Brain Cancer', 'Breast Cancer', 'Cervical Cancer', 'Kidney Cancer', 'Lung and Colon Cancer', 'Lymphoma', 'Oral Cancer')
        FROZEN_CHECKPOINT_SHA256 = '538f2891075f0870e450cd6aa7eda0a5e06ecb73eefd14a4f6a0cd869759d54d'
        FROZEN_PREDICTOR_SHA256 = '79b18d906e0580f0931441a597cec6b1e9f9f5b51e56158d4a3a2d0ce945a0b2'
        FROZEN_CONFIG_SHA256 = '546d6d2720ececb4f9dad4be0a66800ab0f71bc7d19f176a3f160fe5862b2424'
        FROZEN_MANIFEST_SHA256 = '3f797979b4d75b092dfcb4e2348ea1705ed34d35da884ac999c68abef8c87add'

        if (PROJECT_ROOT / '.git').is_dir():
            subprocess.run(['git', '-C', str(PROJECT_ROOT), 'pull', '--ff-only'], check=True)
        else:
            subprocess.run(['git', 'clone', REPOSITORY_URL, str(PROJECT_ROOT)], check=True)
        commit = subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        PREDICTOR = PROJECT_ROOT / 'src/predict_image.py'
        CONFIG = PROJECT_ROOT / 'config/live_demo_resnet18_lastblock_v1.json'
        MANIFEST = PROJECT_ROOT / 'research/gradcam/resnet18_lastblock_gradcam_v1/sample_manifest.csv'

        def sha256(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        assert CHECKPOINT.is_file(), f'Checkpoint not found: {CHECKPOINT}'
        assert sha256(CHECKPOINT) == FROZEN_CHECKPOINT_SHA256, 'Checkpoint SHA-256 mismatch.'
        assert sha256(PREDICTOR) == FROZEN_PREDICTOR_SHA256, 'Inference code SHA-256 mismatch.'
        assert sha256(CONFIG) == FROZEN_CONFIG_SHA256, 'Live-demo config SHA-256 mismatch.'
        assert sha256(MANIFEST) == FROZEN_MANIFEST_SHA256, 'Verification manifest SHA-256 mismatch.'
        assert ZIP_PATH.is_file(), f'Dataset ZIP not found: {ZIP_PATH}'
        free_gib = shutil.disk_usage('/content').free / (1024 ** 3)
        assert free_gib >= 12.0, 'At least 12 GiB of free Colab disk is required.'
        if not EXTRACT_MARKER.is_file():
            LOCAL_DATA_BASE.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            print('Extracting the dataset ZIP; output may be quiet for several minutes...')
            subprocess.run(['unzip', '-q', '-o', str(ZIP_PATH), '-d', str(LOCAL_DATA_BASE)], check=True)
            EXTRACT_MARKER.write_text('complete\\n', encoding='utf-8')
            print(f'ZIP extraction completed in {(time.monotonic() - started) / 60:.1f} minutes.')
        dataset_candidates = (LOCAL_DATA_BASE / 'Multi Cancer' / 'Multi Cancer', LOCAL_DATA_BASE / 'Multi Cancer', LOCAL_DATA_BASE)
        DATASET_ROOT = next((p for p in dataset_candidates if all((p / group).is_dir() for group in EXPECTED_GROUPS)), None)
        assert DATASET_ROOT is not None, 'Could not find a dataset root containing all eight parent folders.'
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        print('Git commit:', commit)
        print('Dataset:', DATASET_ROOT)
        print('Checkpoint SHA-256:', sha256(CHECKPOINT))
        print('Permanent output:', OUTPUT_ROOT)
        """
    ),
    markdown(
        """
        ## 2. Runtime check

        A GPU makes inference faster but is not scientifically required. This cell performs no training.
        """
    ),
    code(
        """
        import torch, torchvision, numpy, PIL
        print('Python:', sys.version.split()[0])
        print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__)
        print('NumPy:', numpy.__version__, '| Pillow:', PIL.__version__)
        print('CUDA:', torch.cuda.is_available())
        print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
        """
    ),
    markdown(
        """
        ## 3. Frozen 37-image verification

        This is the professor-facing proof step. The manifest was frozen before this live-demo
        implementation and contains 26 correct class controls plus 11 documented error cases.
        Success requires 37/37 class predictions to match and every confidence difference to remain
        within the frozen tolerance. Set the authorization flag to `True` and run once.
        """
    ),
    code(
        """
        AUTHORIZE_FROZEN_REPLAY = False
        assert AUTHORIZE_FROZEN_REPLAY, 'Frozen replay locked; set AUTHORIZE_FROZEN_REPLAY = True.'
        command = [
            sys.executable, '-u', str(PREDICTOR),
            '--checkpoint', str(CHECKPOINT), '--config', str(CONFIG),
            '--verify-manifest', str(MANIFEST), '--dataset-root', str(DATASET_ROOT),
            '--output-dir', str(VERIFICATION_OUTPUT), '--device', 'auto',
        ]
        print('Verification command:', ' '.join(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        report = json.loads((VERIFICATION_OUTPUT / 'verification_report.json').read_text(encoding='utf-8'))
        assert report['verification_rows'] == 37
        assert report['passed_rows'] == 37 and report['failed_rows'] == 0 and report['all_passed']
        assert report['checkpoint_sha256'] == FROZEN_CHECKPOINT_SHA256
        print('PROOF STATUS: 37/37 frozen predictions reproduced.')
        print('Maximum confidence difference:', report['maximum_confidence_abs_difference'])
        print('Report:', VERIFICATION_OUTPUT / 'verification_report.json')
        print('Row evidence:', VERIFICATION_OUTPUT / 'verification_rows.csv')
        """
    ),
    markdown(
        """
        ## 4. Live single-image prediction

        Choose any path inside the dataset. The saved card shows the image, predicted 26-leaf label,
        derived parent group, confidence, top-five probabilities, and—because this example comes from
        the labeled dataset—whether the predicted leaf matches the folder label.
        """
    ),
    code(
        """
        DEMO_RELATIVE_PATH = 'Brain Cancer/brain_glioma/brain_glioma_0004.jpg'
        DEMO_IMAGE = DATASET_ROOT / Path(DEMO_RELATIVE_PATH)
        assert DEMO_IMAGE.is_file(), f'Demo image not found: {DEMO_IMAGE}'
        if SINGLE_OUTPUT.exists():
            shutil.rmtree(SINGLE_OUTPUT)
        command = [
            sys.executable, '-u', str(PREDICTOR),
            '--checkpoint', str(CHECKPOINT), '--config', str(CONFIG),
            '--image', str(DEMO_IMAGE), '--dataset-root', str(DATASET_ROOT),
            '--output-dir', str(SINGLE_OUTPUT), '--device', 'auto', '--top-k', '5',
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        prediction = json.loads((SINGLE_OUTPUT / 'prediction.json').read_text(encoding='utf-8'))
        print('Predicted leaf:', prediction['predicted_leaf'])
        print('Derived parent:', prediction['predicted_parent_group'])
        print('Confidence:', prediction['confidence'])
        print('Known folder label:', prediction['true_leaf_from_dataset_path'])
        print('Leaf correct:', prediction['leaf_correct_when_known'])
        from IPython.display import display, Image as DisplayImage
        display(DisplayImage(filename=str(SINGLE_OUTPUT / 'prediction_card.png')))
        """
    ),
    markdown("## 5. Build and download the evidence package"),
    code(
        """
        required = [
            VERIFICATION_OUTPUT / 'verification_report.json',
            VERIFICATION_OUTPUT / 'verification_rows.csv',
            SINGLE_OUTPUT / 'prediction.json',
            SINGLE_OUTPUT / 'prediction_card.png',
        ]
        missing = [str(path) for path in required if not path.is_file()]
        assert not missing, f'Run both verification and single-image cells first. Missing: {missing}'
        evidence_hashes = {str(path.relative_to(OUTPUT_ROOT)): {'sha256': sha256(path), 'size_bytes': path.stat().st_size} for path in required}
        hash_path = OUTPUT_ROOT / 'evidence_hashes.json'
        hash_path.write_text(json.dumps(evidence_hashes, indent=2), encoding='utf-8')
        with zipfile.ZipFile(REVIEW_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in required + [hash_path]:
                archive.write(path, arcname=str(path.relative_to(OUTPUT_ROOT)))
        print('Evidence ZIP:', REVIEW_ZIP)
        print('Evidence ZIP SHA-256:', sha256(REVIEW_ZIP))
        files.download(str(REVIEW_ZIP))
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUTPUT)
