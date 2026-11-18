"""Run the frozen equal-budget GA versus random-search proxy experiment.

Every candidate launches ``train.py`` in validation-only proxy mode. Test images are never
loaded. The runner is restart-safe: completed trial summaries are reused, and interrupted
candidate training resumes from its own ``last_checkpoint.pth``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


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


def canonical_candidate(candidate: dict) -> dict:
    return {gene: float(f"{float(candidate[gene]):.12g}") for gene in GENES}


def candidate_key(candidate: dict) -> str:
    normalized = canonical_candidate(candidate)
    return "|".join(f"{gene}={normalized[gene]:.12g}" for gene in GENES)


def sample_value(rng: random.Random, spec: dict) -> float:
    minimum = float(spec["minimum"])
    maximum = float(spec["maximum"])
    if not minimum < maximum:
        raise ValueError(f"Gecersiz search araligi: {spec}")
    if spec["scale"] == "log10":
        return 10 ** rng.uniform(math.log10(minimum), math.log10(maximum))
    if spec["scale"] == "linear":
        return rng.uniform(minimum, maximum)
    raise ValueError(f"Desteklenmeyen search olcegi: {spec['scale']}")


def sample_candidate(config: dict, seed: int) -> dict:
    rng = random.Random(seed)
    return canonical_candidate(
        {
            gene: sample_value(rng, config["search_space"][gene])
            for gene in GENES
        }
    )


def in_search_space(candidate: dict, config: dict) -> bool:
    return all(
        float(config["search_space"][gene]["minimum"])
        <= float(candidate[gene])
        <= float(config["search_space"][gene]["maximum"])
        for gene in GENES
    )


def tournament_select(archive: Sequence[dict], rng: random.Random, size: int) -> dict:
    if not archive:
        raise ValueError("GA parent secimi icin tamamlanmis aday yok")
    sampled = [rng.choice(list(archive)) for _ in range(size)]
    return max(
        sampled,
        key=lambda row: (float(row["fitness"]), -int(row["sequence"])),
    )


def mutate_value(value: float, spec: dict, rng: random.Random, sigma_fraction: float) -> float:
    minimum = float(spec["minimum"])
    maximum = float(spec["maximum"])
    if spec["scale"] == "log10":
        low = math.log10(minimum)
        high = math.log10(maximum)
        mutated = math.log10(value) + rng.gauss(0.0, sigma_fraction * (high - low))
        return 10 ** min(max(mutated, low), high)
    mutated = value + rng.gauss(0.0, sigma_fraction * (maximum - minimum))
    return min(max(mutated, minimum), maximum)


def make_ga_offspring(
    config: dict,
    round_index: int,
    slot_index: int,
    archive: Sequence[dict],
    existing_keys: set[str],
) -> dict:
    ga = config["ga"]
    for attempt in range(100):
        rng = random.Random(
            int(ga["seed"]) + round_index * 100_000 + slot_index * 1_000 + attempt
        )
        first = tournament_select(archive, rng, int(ga["tournament_size"]))
        second = tournament_select(archive, rng, int(ga["tournament_size"]))
        child = {}
        mutated_any = False
        for gene in GENES:
            value = float(first[gene] if rng.random() < 0.5 else second[gene])
            if rng.random() < float(ga["mutation_probability_per_gene"]):
                value = mutate_value(
                    value,
                    config["search_space"][gene],
                    rng,
                    float(ga["mutation_sigma_fraction"]),
                )
                mutated_any = True
            child[gene] = value
        if not mutated_any:
            forced_gene = GENES[rng.randrange(len(GENES))]
            child[forced_gene] = mutate_value(
                float(child[forced_gene]),
                config["search_space"][forced_gene],
                rng,
                float(ga["mutation_sigma_fraction"]),
            )
        child = canonical_candidate(child)
        if candidate_key(child) not in existing_keys and in_search_space(child, config):
            return child
    raise RuntimeError("100 denemede benzersiz GA offspring uretilemedi")


def make_initial_ga_candidate(config: dict, slot_index: int, existing_keys: set[str]) -> dict:
    base_seed = int(config["ga"]["seed"])
    for attempt in range(100):
        candidate = sample_candidate(config, base_seed + slot_index * 1_000 + attempt)
        if candidate_key(candidate) not in existing_keys:
            return candidate
    raise RuntimeError("Benzersiz ilk GA adayi uretilemedi")


def make_random_candidate(config: dict, index: int, existing_keys: set[str]) -> dict:
    base_seed = int(config["random_search"]["seed"])
    for attempt in range(100):
        candidate = sample_candidate(config, base_seed + index * 1_000 + attempt)
        if candidate_key(candidate) not in existing_keys:
            return candidate
    raise RuntimeError("Benzersiz random-search adayi uretilemedi")


def validate_config(config: dict) -> None:
    budget = config["equal_budget"]
    population = int(budget["population_size"])
    rounds = int(budget["ga_rounds"])
    ga_trials = int(budget["ga_trials"])
    random_trials = int(budget["random_trials"])
    if population * rounds != ga_trials:
        raise ValueError("GA budget population_size * ga_rounds ile eslesmiyor")
    if ga_trials != random_trials:
        raise ValueError("GA ve random-search trial butceleri esit degil")
    if config["test_policy"]["test_opened_during_search"] is not False:
        raise ValueError("Search sirasinda test acilamaz")
    proxy_counts = config["frozen_inputs"]["proxy_counts"]
    if proxy_counts["test"] != 0:
        raise ValueError("Search proxy test satiri iceremez")
    if config["model_contract"]["fitness"] != "best_validation_macro_f1":
        raise ValueError("Frozen fitness validation macro-F1 olmalidir")
    for gene in GENES:
        sample_value(random.Random(1), config["search_space"][gene])


def candidate_payload(
    candidate_id: str,
    method: str,
    sequence: int,
    candidate: dict,
    round_index: int | None = None,
    slot_index: int | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "method": method,
        "sequence": sequence,
        "round_index": round_index,
        "slot_index": slot_index,
        **canonical_candidate(candidate),
    }


def trial_command(args: argparse.Namespace, config: dict, candidate: dict, trial_dir: Path) -> List[str]:
    contract = config["model_contract"]
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "train.py"),
        "--data-dir",
        str(Path(args.data_dir).resolve()),
        "--output-dir",
        str(trial_dir.resolve()),
        "--model",
        contract["model"],
        "--target-level",
        contract["target_level"],
        "--epochs",
        str(contract["epochs_per_trial"]),
        "--batch-size",
        str(contract["batch_size"]),
        "--image-size",
        str(contract["image_size"]),
        "--normalization",
        contract["normalization"],
        "--augmentation",
        contract["augmentation"],
        "--learning-rate",
        f"{candidate['learning_rate']:.12g}",
        "--weight-decay",
        f"{candidate['weight_decay']:.12g}",
        "--dropout",
        f"{candidate['dropout']:.12g}",
        "--split-manifest",
        str(Path(args.split_manifest).resolve()),
        "--search-proxy-manifest",
        str(Path(args.proxy_manifest).resolve()),
        "--search-proxy-manifest-sha256",
        config["frozen_inputs"]["proxy_manifest_sha256"],
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
    if (trial_dir / "last_checkpoint.pth").is_file():
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


def validate_trial_summary(summary: dict, candidate: dict, config: dict) -> float:
    if summary.get("status") != "hyperparameter_search_trial_complete":
        raise ValueError("Trial summary tamamlanmis degil")
    if summary.get("test_evaluated") is not False:
        raise ValueError("Search trial test degerlendirmesi yapmis")
    runtime_sizes = summary.get("split_sizes")
    expected_sizes = {
        key: config["frozen_inputs"]["proxy_counts"][key]
        for key in ("train", "val", "test")
    }
    if runtime_sizes != expected_sizes:
        raise ValueError(f"Trial proxy boyutlari uyusmuyor: {runtime_sizes}")
    for gene in GENES:
        actual = float(summary["hyperparameters"][gene])
        if not math.isclose(actual, float(candidate[gene]), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Trial hyperparameter uyusmazligi: {gene}")
    return float(summary["best_validation_macro_f1"])


def run_trial(
    args: argparse.Namespace,
    config: dict,
    output_root: Path,
    payload: dict,
) -> dict:
    trial_dir = output_root / "trials" / payload["candidate_id"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = trial_dir / "candidate.json"
    if candidate_path.exists():
        if load_json(candidate_path) != payload:
            raise ValueError(f"Candidate contract degisti: {payload['candidate_id']}")
    else:
        write_json_atomic(candidate_path, payload)
    result_path = trial_dir / "trial_result.json"
    if result_path.is_file():
        result = load_json(result_path)
        validate_trial_summary(load_json(trial_dir / "search_trial_summary.json"), payload, config)
        return result

    summary_path = trial_dir / "search_trial_summary.json"
    if not summary_path.is_file():
        partial_without_checkpoint = any(
            (trial_dir / name).exists()
            for name in ("best_model.pth", "history.csv", "run_config.json")
        ) and not (trial_dir / "last_checkpoint.pth").is_file()
        if partial_without_checkpoint:
            raise RuntimeError(
                "Trial yarim artifact iceriyor fakat resume checkpoint yok; "
                f"otomatik overwrite yasak: {trial_dir}"
            )
        command = trial_command(args, config, payload, trial_dir)
        print("Trial komutu:", " ".join(command))
        run_and_tee(command, trial_dir / "console.log", Path(__file__).resolve().parents[1])
    summary = load_json(summary_path)
    fitness = validate_trial_summary(summary, payload, config)
    result = {
        **payload,
        "fitness": fitness,
        "best_epoch": int(summary["best_epoch"]),
        "epochs_completed": int(summary["epochs_completed"]),
        "test_evaluated": False,
        "summary_sha256": sha256_file(summary_path),
    }
    write_json_atomic(result_path, result)
    return result


def collect_results(output_root: Path) -> List[dict]:
    results = []
    trials_dir = output_root / "trials"
    if not trials_dir.is_dir():
        return results
    for path in sorted(trials_dir.glob("*/trial_result.json")):
        results.append(load_json(path))
    return sorted(results, key=lambda row: (row["method"], int(row["sequence"])))


def write_results(output_root: Path, results: Sequence[dict]) -> None:
    fields = [
        "candidate_id",
        "method",
        "sequence",
        "round_index",
        "slot_index",
        *GENES,
        "fitness",
        "best_epoch",
        "epochs_completed",
        "test_evaluated",
        "summary_sha256",
    ]
    temporary = output_root / "search_results.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in results])
    temporary.replace(output_root / "search_results.csv")


def verify_frozen_inputs(args: argparse.Namespace, config: dict) -> dict:
    frozen = config["frozen_inputs"]
    checks = {
        "split_manifest_sha256": sha256_file(Path(args.split_manifest).resolve()),
        "proxy_manifest_sha256": sha256_file(Path(args.proxy_manifest).resolve()),
        "init_checkpoint_sha256": sha256_file(Path(args.init_checkpoint).resolve()),
    }
    for key, actual in checks.items():
        if actual != frozen[key]:
            raise ValueError(
                f"Frozen input hash mismatch: {key}, expected={frozen[key]}, actual={actual}"
            )
    return checks


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    validate_config(config)
    if args.authorize != config["authorization_token"]:
        raise ValueError("GA/random search kilitli; frozen authorization token gerekli")
    checks = verify_frozen_inputs(args, config)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "status": "ga_random_search_v1_contract",
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "trainer_sha256": sha256_file(Path(__file__).resolve().parent / "train.py"),
        "frozen_inputs": checks,
        "equal_budget": config["equal_budget"],
        "test_opened_during_search": False,
    }
    contract_path = output_root / "run_contract.json"
    if contract_path.exists():
        if load_json(contract_path) != contract:
            raise ValueError("Mevcut GA run contract yeni sozlesmeyle uyusmuyor")
    else:
        write_json_atomic(contract_path, contract)
    frozen_config_path = output_root / "frozen_config.json"
    if frozen_config_path.exists():
        if sha256_file(frozen_config_path) != contract["config_sha256"]:
            raise ValueError("Output icindeki frozen config hash'i degismis")
    else:
        shutil.copyfile(config_path, frozen_config_path)
        if sha256_file(frozen_config_path) != contract["config_sha256"]:
            raise RuntimeError("Kopyalanan frozen config hash'i degisti")

    population = int(config["equal_budget"]["population_size"])
    rounds = int(config["equal_budget"]["ga_rounds"])
    if args.method in {"ga", "both"}:
        existing_keys: set[str] = set()
        for round_index in range(rounds):
            for slot_index in range(population):
                sequence = round_index * population + slot_index + 1
                candidate_id = f"GA_G{round_index + 1:02d}_C{slot_index + 1:02d}"
                candidate_path = output_root / "trials" / candidate_id / "candidate.json"
                if candidate_path.is_file():
                    payload = load_json(candidate_path)
                elif round_index == 0:
                    candidate = make_initial_ga_candidate(config, slot_index, existing_keys)
                    payload = candidate_payload(
                        candidate_id,
                        "ga",
                        sequence,
                        candidate,
                        round_index + 1,
                        slot_index + 1,
                    )
                else:
                    archive = [
                        row
                        for row in collect_results(output_root)
                        if row["method"] == "ga" and int(row["round_index"]) <= round_index
                    ]
                    if len(archive) != round_index * population:
                        raise RuntimeError("Onceki GA round tamamlanmadan offspring uretilemez")
                    candidate = make_ga_offspring(
                        config,
                        round_index,
                        slot_index,
                        archive,
                        existing_keys,
                    )
                    payload = candidate_payload(
                        candidate_id,
                        "ga",
                        sequence,
                        candidate,
                        round_index + 1,
                        slot_index + 1,
                    )
                key = candidate_key(payload)
                if key in existing_keys:
                    raise ValueError(f"Tekrarlanan GA adayi: {candidate_id}")
                existing_keys.add(key)
                run_trial(args, config, output_root, payload)
                write_results(output_root, collect_results(output_root))

    if args.method in {"random", "both"}:
        existing_keys = set()
        for index in range(int(config["equal_budget"]["random_trials"])):
            sequence = index + 1
            candidate_id = f"RANDOM_{sequence:03d}"
            candidate_path = output_root / "trials" / candidate_id / "candidate.json"
            if candidate_path.is_file():
                payload = load_json(candidate_path)
            else:
                candidate = make_random_candidate(config, index, existing_keys)
                payload = candidate_payload(candidate_id, "random", sequence, candidate)
            key = candidate_key(payload)
            if key in existing_keys:
                raise ValueError(f"Tekrarlanan random adayi: {candidate_id}")
            existing_keys.add(key)
            run_trial(args, config, output_root, payload)
            write_results(output_root, collect_results(output_root))

    results = collect_results(output_root)
    by_method = {
        method: [row for row in results if row["method"] == method]
        for method in ("ga", "random")
    }
    expected = {
        "ga": int(config["equal_budget"]["ga_trials"]),
        "random": int(config["equal_budget"]["random_trials"]),
    }
    completed_methods = [method for method in ("ga", "random") if len(by_method[method]) == expected[method]]
    summary = {
        "schema_version": 1,
        "status": (
            "ga_random_search_v1_complete"
            if len(completed_methods) == 2
            else "ga_random_search_v1_partial"
        ),
        "experiment_type": config["experiment_type"],
        "test_evaluated": False,
        "completed_trials": {method: len(rows) for method, rows in by_method.items()},
        "expected_trials": expected,
        "equal_budget_complete": len(completed_methods) == 2,
        "winners": {
            method: max(rows, key=lambda row: (float(row["fitness"]), -int(row["sequence"])))
            if rows
            else None
            for method, rows in by_method.items()
        },
        "next_stage": "Separate approval for full-data confirmation of both winners.",
        "warning": "Proxy validation search result; not test performance or clinical evidence.",
    }
    write_results(output_root, results)
    write_json_atomic(output_root / "search_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Frozen GA vs random-search runner")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--split-manifest",
        default=str(root / "research/splits/practical_leakage_reduced_split_v1.csv"),
    )
    parser.add_argument(
        "--proxy-manifest",
        default=str(root / "research/splits/ga_search_proxy_v1.csv"),
    )
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config", default=str(root / "config/ga_resnet18_lastblock_v1.json")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--method", choices=["ga", "random", "both"], default="both")
    parser.add_argument("--authorize", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
