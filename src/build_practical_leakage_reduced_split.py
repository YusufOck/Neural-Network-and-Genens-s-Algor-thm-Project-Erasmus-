from __future__ import annotations

"""Build the fixed practical leakage-reduced 70/15/15 split.

The split keeps byte-identical files and every confirmed/conservatively linked
review pair in one connected component. It does not claim that every unknown
near duplicate or patient/series relative has been discovered.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from data_index import discover_records, project_root, resolve_dataset_root


SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = (
    "relative_path",
    "main_class",
    "subclass",
    "split",
    "family_id",
    "family_size",
    "family_sources",
)


class DisjointSet:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in self.parent}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Known-family-aware sabit 70/15/15 split üretir")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default=str(root / "config" / "practical_leakage_reduced_split_v1.json"))
    parser.add_argument("--output-manifest", default=str(root / "research" / "splits" / "practical_leakage_reduced_split_v1.csv"))
    parser.add_argument("--output-report", default=str(root / "research" / "splits" / "practical_leakage_reduced_split_v1_report.json"))
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_sha256(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object bekleniyordu: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV basligi bulunamadi: {path}")
        return [dict(row) for row in reader]


def normalize_relative(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if not normalized or normalized.startswith("/") or ":" in normalized.split("/", 1)[0]:
        raise ValueError(f"Gecersiz relative path: {value!r}")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if ".." in parts:
        raise ValueError(f"Parent traversal yasak: {value!r}")
    return "/".join(parts)


def index_by_pair_id(rows: Sequence[Mapping[str, str]], name: str) -> dict[str, Mapping[str, str]]:
    result: dict[str, Mapping[str, str]] = {}
    for row in rows:
        pair_id = row.get("pair_id", "").strip()
        if not pair_id or pair_id in result:
            raise ValueError(f"{name}: bos/tekrarlanan Pair ID {pair_id!r}")
        result[pair_id] = row
    return result


def add_review_edge(
    edges: dict[tuple[str, str], set[str]],
    row: Mapping[str, str],
    source: str,
) -> None:
    left = normalize_relative(row.get("left_path", ""))
    right = normalize_relative(row.get("right_path", ""))
    if left == right:
        raise ValueError(f"Kendi kendine review edge: {left}")
    key = tuple(sorted((left, right)))
    edges.setdefault(key, set()).add(source)


def collect_review_edges(root: Path, config: Mapping[str, object]) -> dict[tuple[str, str], set[str]]:
    evidence = config["evidence"]
    paths = {name: root / str(info["path"]) for name, info in evidence.items()}
    edges: dict[tuple[str, str], set[str]] = {}

    natural_paths = index_by_pair_id(read_csv(paths["natural_pair_paths"]), "natural paths")
    for reference in read_csv(paths["natural_final_references"]):
        if reference.get("final_reference") == "same_source":
            pair_id = reference["pair_id"]
            if pair_id not in natural_paths:
                raise ValueError(f"Natural path bulunamadi: {pair_id}")
            add_review_edge(edges, natural_paths[pair_id], "natural_confirmed_same_source")

    v25_rows = read_csv(paths["v2_5_unblinded_pairs"])
    v25_by_id = index_by_pair_id(v25_rows, "v2.5 pairs")
    for row in v25_rows:
        verdict = row.get("visual_verdict")
        if verdict == "same_source_candidate":
            add_review_edge(edges, row, "v2_5_review_same_source")
        elif verdict == "uncertain":
            add_review_edge(edges, row, "v2_5_review_uncertain_conservative")

    for reference in read_csv(paths["v2_6_same2d_final_references"]):
        if reference.get("final_reference") == "same_source":
            pair_id = reference["pair_id"]
            if pair_id not in v25_by_id:
                raise ValueError(f"V2.6 development path bulunamadi: {pair_id}")
            add_review_edge(edges, v25_by_id[pair_id], "v2_6_confirmed_same2d")

    for row in read_csv(paths["v2_6b_unblinded_pairs"]):
        verdict = row.get("visual_verdict")
        if verdict == "same_source_candidate":
            add_review_edge(edges, row, "v2_6b_confirmed_same2d")
        elif verdict == "uncertain":
            add_review_edge(edges, row, "v2_6b_uncertain_conservative")
    return edges


def target_counts(total: int, ratios: Mapping[str, object]) -> dict[str, int]:
    train = round(total * float(ratios["train"]))
    val = round(total * float(ratios["val"]))
    return {"train": train, "val": val, "test": total - train - val}


def assign_leaf_components(
    components: Sequence[Sequence[str]],
    targets: Mapping[str, int],
    seed: int,
    leaf: str,
) -> dict[str, str]:
    stable_seed = int(hashlib.sha256(f"{seed}:{leaf}".encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(stable_seed)
    ordered = [tuple(sorted(component)) for component in components]
    rng.shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    remaining = {split: int(targets[split]) for split in SPLITS}
    assignment: dict[str, str] = {}

    for component in ordered:
        size = len(component)
        fitting = [split for split in SPLITS if remaining[split] >= size]
        choices = fitting if fitting else list(SPLITS)
        chosen = max(
            choices,
            key=lambda split: (
                remaining[split] / max(int(targets[split]), 1),
                remaining[split],
                -SPLITS.index(split),
            ),
        )
        for path in component:
            assignment[path] = chosen
        remaining[chosen] -= size

    if any(value != 0 for value in remaining.values()):
        raise ValueError(f"{leaf}: exact split hedefi tutmadi; kalan={remaining}")
    return assignment


def family_id(paths: Sequence[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(paths)).encode("utf-8")).hexdigest()[:20]
    return f"fam_{digest}"


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def build(
    dataset_root: Path,
    root: Path,
    config: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records = discover_records(dataset_root)
    metadata: dict[str, tuple[str, str]] = {}
    for record in records:
        relative = record.path.relative_to(dataset_root).as_posix()
        if relative in metadata:
            raise ValueError(f"Tekrarlanan dataset path: {relative}")
        metadata[relative] = (record.main_class, record.subclass)

    dsu = DisjointSet(metadata)
    exact_data = read_json(root / str(config["evidence"]["exact_duplicates"]["path"]))
    exact_groups: list[list[str]] = []
    for group in exact_data["duplicate_groups"]:
        paths = [normalize_relative(path) for path in group["paths"]]
        if any(path not in metadata for path in paths):
            raise ValueError("Exact grupta dataset disi path bulundu")
        for path in paths[1:]:
            dsu.union(paths[0], path)
        exact_groups.append(paths)

    review_edges = collect_review_edges(root, config)
    for left, right in review_edges:
        if left not in metadata or right not in metadata:
            raise ValueError(f"Review edge dataset disinda: {left}, {right}")
        if metadata[left][1] != metadata[right][1]:
            raise ValueError(f"Cross-leaf family edge desteklenmiyor: {left}, {right}")
        dsu.union(left, right)

    components_by_root: dict[str, list[str]] = defaultdict(list)
    for path in sorted(metadata):
        components_by_root[dsu.find(path)].append(path)
    components = list(components_by_root.values())

    sources_by_root: dict[str, set[str]] = defaultdict(set)
    for group in exact_groups:
        sources_by_root[dsu.find(group[0])].add("exact_sha256")
    for (left, _right), sources in review_edges.items():
        sources_by_root[dsu.find(left)].update(sources)

    components_by_leaf: dict[str, list[list[str]]] = defaultdict(list)
    for component in components:
        leaves = {metadata[path][1] for path in component}
        if len(leaves) != 1:
            raise ValueError(f"Family birden fazla leaf iceriyor: {sorted(leaves)}")
        components_by_leaf[next(iter(leaves))].append(component)

    seed = int(config["seed"])
    ratios = config["ratios"]
    assignment: dict[str, str] = {}
    targets_by_leaf: dict[str, dict[str, int]] = {}
    for leaf in sorted(components_by_leaf):
        total = sum(len(component) for component in components_by_leaf[leaf])
        targets = target_counts(total, ratios)
        targets_by_leaf[leaf] = targets
        assignment.update(assign_leaf_components(components_by_leaf[leaf], targets, seed, leaf))

    manifest_rows: list[dict[str, object]] = []
    component_ids: dict[str, str] = {}
    for component in components:
        root_path = dsu.find(component[0])
        component_ids[root_path] = family_id(component)
        sources = sources_by_root.get(root_path, set())
        source_text = "+".join(sorted(sources)) if sources else "singleton"
        split_names = {assignment[path] for path in component}
        if len(split_names) != 1:
            raise ValueError("Family splitler arasinda bolundu")
        for path in component:
            main_class, subclass = metadata[path]
            manifest_rows.append(
                {
                    "relative_path": path,
                    "main_class": main_class,
                    "subclass": subclass,
                    "split": assignment[path],
                    "family_id": component_ids[root_path],
                    "family_size": len(component),
                    "family_sources": source_text,
                }
            )
    manifest_rows.sort(key=lambda row: str(row["relative_path"]))

    split_counts = Counter(str(row["split"]) for row in manifest_rows)
    leaf_split_counts: dict[str, dict[str, int]] = {}
    for leaf in sorted(components_by_leaf):
        counts = Counter(
            str(row["split"]) for row in manifest_rows if row["subclass"] == leaf
        )
        leaf_split_counts[leaf] = {split: counts[split] for split in SPLITS}

    exact_cross = sum(
        len({assignment[path] for path in group}) > 1 for group in exact_groups
    )
    review_cross = sum(
        assignment[left] != assignment[right] for left, right in review_edges
    )
    source_edge_counts = Counter(
        source for sources in review_edges.values() for source in sources
    )
    multi_image = [component for component in components if len(component) > 1]
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "practical_leakage_reduced_split_v1_complete",
        "experiment_type": "fixed_known_family_grouped_split",
        "dataset_root": str(dataset_root),
        "policy": {
            "seed": seed,
            "ratios": ratios,
            "stratify_by": "subclass",
            "claim": config["interpretation"]["allowed_claim"],
            "forbidden_claim": config["interpretation"]["forbidden_claim"],
        },
        "input": {
            "images": len(manifest_rows),
            "leaves": len(components_by_leaf),
            "exact_groups": len(exact_groups),
            "review_edges_unique": len(review_edges),
            "review_edges_by_source": dict(sorted(source_edge_counts.items())),
        },
        "families": {
            "total_components": len(components),
            "multi_image_components": len(multi_image),
            "images_in_multi_image_components": sum(len(component) for component in multi_image),
            "maximum_component_size": max(map(len, components)),
        },
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "split_counts_by_leaf": leaf_split_counts,
        "targets_by_leaf": targets_by_leaf,
        "invariants": {
            "all_images_once": len(manifest_rows) == len(metadata) == len(set(metadata)),
            "all_26_leaves_present": len(components_by_leaf) == 26,
            "exact_split_counts_per_leaf": leaf_split_counts == targets_by_leaf,
            "cross_split_exact_groups": exact_cross,
            "cross_split_review_edges": review_cross,
            "cross_split_family_components": 0,
            "all_passed": False,
        },
        "limitations": [
            "This split groups every known exact and reviewed relation, not every unknown near duplicate.",
            "The dataset has no patient/series IDs, so patient-level separation cannot be claimed.",
            "Conservative uncertain links may keep two independent images together; this cannot create train-test leakage.",
        ],
    }
    invariants = report["invariants"]
    invariants["all_passed"] = (
        invariants["all_images_once"]
        and invariants["all_26_leaves_present"]
        and invariants["exact_split_counts_per_leaf"]
        and exact_cross == 0
        and review_cross == 0
    )
    if not invariants["all_passed"]:
        raise ValueError(f"Split invariants failed: {invariants}")
    return manifest_rows, report


def main() -> int:
    args = parse_args()
    root = project_root().resolve()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    for name, info in config["evidence"].items():
        path = root / str(info["path"])
        expected = info.get("sha256")
        actual = file_sha256(path) if expected else normalized_text_sha256(path)
        expected = expected or info["normalized_text_sha256"]
        if actual != expected:
            raise ValueError(f"Evidence hash mismatch: {name}; expected={expected}, actual={actual}")

    dataset_root = resolve_dataset_root(args.data_dir)
    rows, report = build(dataset_root, root, config)
    manifest_path = Path(args.output_manifest).resolve()
    report_path = Path(args.output_report).resolve()
    atomic_write_csv(manifest_path, rows)
    report["artifacts"] = {
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": file_sha256(config_path),
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": file_sha256(manifest_path),
    }
    atomic_write_json(report_path, report)

    print("Practical leakage-reduced split tamamlandi")
    print(f"Goruntu: {report['input']['images']} | leaf: {report['input']['leaves']}")
    print(f"Split: {report['split_counts']}")
    print(f"Family: {report['families']}")
    print(f"Cross-split exact: {report['invariants']['cross_split_exact_groups']}")
    print(f"Cross-split review: {report['invariants']['cross_split_review_edges']}")
    print(f"Manifest: {manifest_path}")
    print(f"Rapor: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
