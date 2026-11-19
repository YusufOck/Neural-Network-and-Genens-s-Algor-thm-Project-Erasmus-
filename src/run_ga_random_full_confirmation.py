"""Confirm the frozen GA and random-search winners on full train/validation data.

The existing hash-locked validation-only trainer path is reused with a manifest containing every
frozen train and validation row and zero test rows. The runner is restart-safe and never requests
test evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


GENES = ("learning_rate", "weight_decay", "dropout")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_config(config: dict) -> None:
    contract = config["model_contract"]
    if config["experiment_type"] != "full_data_validation_only_hyperparameter_confirmation":
        raise ValueError("Gecersiz confirmation experiment type")
    if contract["checkpoint_selection"] != "best_validation_macro_f1":
        raise ValueError("Checkpoint secimi validation macro-F1 olmalidir")
    if config["test_policy"] != {
        "test_opened_during_confirmation": False,
        "test_loader_constructed": False,
        "test_metrics_allowed": False,
        "later_test_evaluation_requires_separate_approval": True,
    }:
        raise ValueError("Confirmation test politikasi degismis")
    counts = config["frozen_inputs"]["counts"]
    if counts != {
        "train": 91_000,
        "val": 19_500,
        "test": 0,
        "train_per_leaf": 3500,
        "val_per_leaf": 750,
        "leaf_count": 26,
    }:
        raise ValueError("Full confirmation sayim sozlesmesi degismis")
    candidates = config["candidates"]
    if len(candidates) != 2 or {row["method"] for row in candidates} != {"ga", "random"}:
        raise ValueError("Tam olarak bir GA ve bir random kazanan gerekli")
    if len({row["candidate_id"] for row in candidates}) != 2:
        raise ValueError("Confirmation candidate ID'leri benzersiz degil")
    for row in candidates:
        for gene in GENES:
            if not math.isfinite(float(row[gene])):
                raise ValueError(f"Gecersiz candidate geni: {gene}")


def validate_source_search(summary: dict, config: dict) -> None:
    if summary.get("status") != "ga_random_search_v1_complete":
        raise ValueError("Kaynak GA/random search tamamlanmis degil")
    if summary.get("test_evaluated") is not False:
        raise ValueError("Kaynak search test degerlendirmesi yapmis")
    if summary.get("completed_trials") != {"ga": 12, "random": 12}:
        raise ValueError("Kaynak search 12+12 butceyi tamamlamamis")
    configured = {row["method"]: row for row in config["candidates"]}
    for method in ("ga", "random"):
        source = summary["winners"][method]
        expected = configured[method]
        if source["candidate_id"] != expected["candidate_id"]:
            raise ValueError(f"Kaynak {method} winner ID uyusmuyor")
        for key in (*GENES, "fitness"):
            expected_key = "proxy_validation_macro_f1" if key == "fitness" else key
            if not math.isclose(
                float(source[key]), float(expected[expected_key]), rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError(f"Kaynak {method} winner alani uyusmuyor: {key}")
        if source["summary_sha256"] != expected["source_summary_sha256"]:
            raise ValueError(f"Kaynak {method} trial summary hash'i uyusmuyor")


def candidate_payload(candidate: dict) -> dict:
    return {
        "schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "method": candidate["method"],
        **{gene: float(candidate[gene]) for gene in GENES},
        "proxy_validation_macro_f1": float(candidate["proxy_validation_macro_f1"]),
        "source_summary_sha256": candidate["source_summary_sha256"],
    }


def candidate_command(
    args: argparse.Namespace, config: dict, candidate: dict, output_dir: Path
) -> List[str]:
    contract = config["model_contract"]
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "train.py"),
        "--data-dir",
        str(Path(args.data_dir).resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--model",
        contract["model"],
        "--target-level",
        contract["target_level"],
        "--epochs",
        str(contract["maximum_epochs"]),
        "--batch-size",
        str(contract["batch_size"]),
        "--image-size",
        str(contract["image_size"]),
        "--normalization",
        contract["normalization"],
        "--augmentation",
        contract["augmentation"],
        "--learning-rate",
        f"{float(candidate['learning_rate']):.12g}",
        "--weight-decay",
        f"{float(candidate['weight_decay']):.12g}",
        "--dropout",
        f"{float(candidate['dropout']):.12g}",
        "--split-manifest",
        str(Path(args.split_manifest).resolve()),
        "--search-proxy-manifest",
        str(Path(args.confirmation_manifest).resolve()),
        "--search-proxy-manifest-sha256",
        config["frozen_inputs"]["confirmation_manifest_sha256"],
        "--num-workers",
        str(args.workers),
        "--seed",
        str(contract["seed"]),
        "--early-stopping-patience",
        str(contract["early_stopping_patience"]),
        "--pretrained",
        "--fine-tune",
        "--experiment-type",
        "search_trial",
        "--init-checkpoint",
        str(Path(args.init_checkpoint).resolve()),
        "--init-checkpoint-sha256",
        config["frozen_inputs"]["init_checkpoint_sha256"],
    ]
    if (output_dir / "last_checkpoint.pth").is_file():
        command.append("--resume")
    return command


def run_and_tee(command: Sequence[str], log_path: Path, cwd: Path) -> None:
    mode = "a" if "--resume" in command else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def validate_candidate_summary(summary: dict, candidate: dict, config: dict) -> float:
    if summary.get("status") != "hyperparameter_search_trial_complete":
        raise ValueError("Confirmation trainer summary tamamlanmis degil")
    if summary.get("test_evaluated") is not False:
        raise ValueError("Confirmation adayi test degerlendirmesi yapmis")
    expected_sizes = {
        key: config["frozen_inputs"]["counts"][key]
        for key in ("train", "val", "test")
    }
    if summary.get("split_sizes") != expected_sizes:
        raise ValueError(f"Confirmation split boyutlari uyusmuyor: {summary.get('split_sizes')}")
    if summary.get("search_proxy_manifest_sha256") != config["frozen_inputs"][
        "confirmation_manifest_sha256"
    ]:
        raise ValueError("Confirmation manifest hash'i summary ile uyusmuyor")
    if summary.get("initialization", {}).get("checkpoint_sha256") != config[
        "frozen_inputs"
    ]["init_checkpoint_sha256"]:
        raise ValueError("Confirmation init checkpoint hash'i uyusmuyor")
    for gene in GENES:
        if not math.isclose(
            float(summary["hyperparameters"][gene]),
            float(candidate[gene]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Confirmation hyperparameter uyusmazligi: {gene}")
    epochs = int(summary["epochs_completed"])
    if not 1 <= epochs <= int(config["model_contract"]["maximum_epochs"]):
        raise ValueError(f"Gecersiz confirmation epoch sayisi: {epochs}")
    return float(summary["best_validation_macro_f1"])


def run_candidate(args: argparse.Namespace, config: dict, candidate: dict, root: Path) -> dict:
    payload = candidate_payload(candidate)
    candidate_dir = root / "candidates" / candidate["method"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / "candidate.json"
    if candidate_path.exists():
        if load_json(candidate_path) != payload:
            raise ValueError(f"Confirmation candidate contract degisti: {candidate['method']}")
    else:
        write_json_atomic(candidate_path, payload)

    result_path = candidate_dir / "confirmation_result.json"
    summary_path = candidate_dir / "search_trial_summary.json"
    if result_path.is_file():
        result = load_json(result_path)
        summary = load_json(summary_path)
        fitness = validate_candidate_summary(summary, payload, config)
        for key, expected in payload.items():
            actual = result.get(key)
            if isinstance(expected, float):
                if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15):
                    raise ValueError(f"Kayitli confirmation result uyusmuyor: {key}")
            elif actual != expected:
                raise ValueError(f"Kayitli confirmation result uyusmuyor: {key}")
        if result.get("test_evaluated") is not False:
            raise ValueError("Kayitli confirmation result test degerlendirmesi iceriyor")
        if not math.isclose(
            float(result.get("full_validation_macro_f1")),
            fitness,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Kayitli confirmation result fitness degeri uyusmuyor")
        if result.get("summary_sha256") != sha256_file(summary_path):
            raise ValueError("Kayitli confirmation result summary hash'i uyusmuyor")
        return result
    if not summary_path.is_file():
        partial_without_checkpoint = any(
            (candidate_dir / name).exists()
            for name in ("best_model.pth", "history.csv", "run_config.json")
        ) and not (candidate_dir / "last_checkpoint.pth").is_file()
        if partial_without_checkpoint:
            raise RuntimeError(
                "Confirmation adayi yarim artifact iceriyor fakat resume checkpoint yok; "
                f"otomatik overwrite yasak: {candidate_dir}"
            )
        command = candidate_command(args, config, payload, candidate_dir)
        print("Confirmation komutu:", " ".join(command))
        run_and_tee(
            command,
            candidate_dir / "console.log",
            Path(__file__).resolve().parents[1],
        )
    summary = load_json(summary_path)
    fitness = validate_candidate_summary(summary, payload, config)
    result = {
        **payload,
        "full_validation_macro_f1": fitness,
        "best_epoch": int(summary["best_epoch"]),
        "epochs_completed": int(summary["epochs_completed"]),
        "test_evaluated": False,
        "summary_sha256": sha256_file(summary_path),
    }
    write_json_atomic(result_path, result)
    return result


def collect_results(root: Path) -> List[dict]:
    paths = sorted((root / "candidates").glob("*/confirmation_result.json"))
    return [load_json(path) for path in paths]


def write_results(root: Path, rows: Sequence[dict]) -> None:
    fields = [
        "candidate_id",
        "method",
        *GENES,
        "proxy_validation_macro_f1",
        "full_validation_macro_f1",
        "best_epoch",
        "epochs_completed",
        "test_evaluated",
        "source_summary_sha256",
        "summary_sha256",
    ]
    temporary = root / "confirmation_results.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])
    temporary.replace(root / "confirmation_results.csv")


def verify_frozen_inputs(args: argparse.Namespace, config: dict) -> dict:
    paths = {
        "split_manifest_sha256": Path(args.split_manifest).resolve(),
        "confirmation_manifest_sha256": Path(args.confirmation_manifest).resolve(),
        "init_checkpoint_sha256": Path(args.init_checkpoint).resolve(),
        "source_search_summary_sha256": Path(args.source_search_summary).resolve(),
    }
    checks = {name: sha256_file(path) for name, path in paths.items()}
    for name, actual in checks.items():
        expected = config["frozen_inputs"][name]
        if actual != expected:
            raise ValueError(
                f"Frozen confirmation input hash mismatch: {name}, "
                f"expected={expected}, actual={actual}"
            )
    validate_source_search(load_json(paths["source_search_summary_sha256"]), config)
    return checks


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    validate_config(config)
    if args.authorize != config["authorization_token"]:
        raise ValueError("Full confirmation kilitli; frozen authorization token gerekli")
    checks = verify_frozen_inputs(args, config)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "status": "ga_random_full_confirmation_v1_contract",
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "trainer_sha256": sha256_file(Path(__file__).resolve().parent / "train.py"),
        "frozen_inputs": checks,
        "candidate_ids": [row["candidate_id"] for row in config["candidates"]],
        "test_opened_during_confirmation": False,
    }
    contract_path = output_root / "run_contract.json"
    if contract_path.exists():
        if load_json(contract_path) != contract:
            raise ValueError("Mevcut confirmation run contract yeni sozlesmeyle uyusmuyor")
    else:
        write_json_atomic(contract_path, contract)
    frozen_config_path = output_root / "frozen_config.json"
    if frozen_config_path.exists():
        if sha256_file(frozen_config_path) != contract["config_sha256"]:
            raise ValueError("Output icindeki frozen confirmation config hash'i degismis")
    else:
        shutil.copyfile(config_path, frozen_config_path)
        if sha256_file(frozen_config_path) != contract["config_sha256"]:
            raise RuntimeError("Kopyalanan confirmation config hash'i degisti")

    for candidate in config["candidates"]:
        run_candidate(args, config, candidate, output_root)
        write_results(output_root, collect_results(output_root))

    results = collect_results(output_root)
    complete = len(results) == 2 and {row["method"] for row in results} == {"ga", "random"}
    winner = (
        max(
            results,
            key=lambda row: (
                float(row["full_validation_macro_f1"]),
                row["method"] == "ga",
            ),
        )
        if complete
        else None
    )
    baseline = config["manual_baseline_reference"]
    summary = {
        "schema_version": 1,
        "status": (
            "ga_random_full_confirmation_v1_complete"
            if complete
            else "ga_random_full_confirmation_v1_partial"
        ),
        "experiment_type": config["experiment_type"],
        "test_evaluated": False,
        "completed_candidates": len(results),
        "expected_candidates": 2,
        "results": {row["method"]: row for row in results},
        "validation_winner": winner,
        "manual_baseline_reference": baseline,
        "deltas_vs_manual_baseline": {
            row["method"]: float(row["full_validation_macro_f1"])
            - float(baseline["best_validation_macro_f1"])
            for row in results
        },
        "next_stage": "Review full-validation evidence before any separately approved test evaluation.",
        "warning": "Validation-only full-data confirmation; not test performance or clinical evidence.",
    }
    write_results(output_root, results)
    write_json_atomic(output_root / "confirmation_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Full-data validation-only confirmation of GA and random winners"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--split-manifest",
        default=str(root / "research/splits/practical_leakage_reduced_split_v1.csv"),
    )
    parser.add_argument(
        "--confirmation-manifest",
        default=str(root / "research/splits/ga_full_confirmation_v1.csv"),
    )
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--source-search-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config", default=str(root / "config/ga_resnet18_full_confirmation_v1.json")
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--authorize", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
