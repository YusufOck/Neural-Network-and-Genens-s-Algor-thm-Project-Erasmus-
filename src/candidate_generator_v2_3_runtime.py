from __future__ import annotations

"""Reusable, hash-verified runtime helpers for Candidate Generator v2.3.

The frozen candidate rule itself remains in the previously hashed implementation
files. This module adds resumable orchestration and exact blockwise execution without
changing the rule's top-15 or flip-aware pHash<=12 semantics.
"""

import csv
import hashlib
import json
import os
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from duplicate_decision_rule import file_sha256


PAIR_SHIFT = np.uint64(32)
PAIR_MASK = np.uint64((1 << 32) - 1)
EXPECTED_NORMALIZED_EXACT_DUPLICATE_SHA256 = (
    "e8a27465513e948d2cf7a012a4d2eaebea2dd97d8f5c313299f81eb86780ee4a"
)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_pair_keys(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.uint64)
    right64 = np.asarray(right, dtype=np.uint64)
    if left64.shape != right64.shape:
        raise ValueError("Pair left/right shapes do not match.")
    if np.any(left64 >= right64):
        raise ValueError("Pairs must be canonical with left < right.")
    if np.any(right64 > PAIR_MASK):
        raise ValueError("Pair index exceeds uint32 capacity.")
    return (left64 << PAIR_SHIFT) | right64


def decode_pair_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(keys, dtype=np.uint64)
    left = (values >> PAIR_SHIFT).astype(np.uint32)
    right = (values & PAIR_MASK).astype(np.uint32)
    return left, right


def stable_topk_index_set(row: np.ndarray, k: int) -> np.ndarray:
    """Return the same selected index set as stable descending argsort[:k]."""

    values = np.asarray(row)
    if values.ndim != 1:
        raise ValueError("stable_topk_index_set expects a one-dimensional row.")
    if not 1 <= k <= len(values):
        raise ValueError("k must be between 1 and row length.")
    threshold = np.partition(values, len(values) - k)[len(values) - k]
    greater = np.flatnonzero(values > threshold)
    equal = np.flatnonzero(values == threshold)
    needed = k - len(greater)
    if needed < 0:
        raise RuntimeError("Top-k threshold produced too many greater values.")
    return np.concatenate((greater, equal[:needed])).astype(np.int64, copy=False)


def robust_channel_candidate_keys(
    descriptors: np.ndarray,
    global_indices: np.ndarray,
    top_k: int,
    block_size: int,
    device: torch.device,
) -> np.ndarray:
    """Exact cosine directed-top-k union for one frozen descriptor channel."""

    values = np.asarray(descriptors, dtype=np.float32)
    indices = np.asarray(global_indices, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != len(indices):
        raise ValueError("Descriptor/global-index shapes do not match.")
    if len(indices) < 2:
        return np.empty(0, dtype=np.uint64)
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    kept = min(top_k, len(indices) - 1)
    gallery = torch.from_numpy(values).to(device)
    parts: list[np.ndarray] = []
    try:
        for start in range(0, len(indices), block_size):
            stop = min(start + block_size, len(indices))
            similarities = (gallery[start:stop] @ gallery.T).float().cpu().numpy()
            row_offsets = np.arange(stop - start)
            similarities[row_offsets, start + row_offsets] = -np.inf
            left_parts: list[np.ndarray] = []
            right_parts: list[np.ndarray] = []
            for row_offset, local_left in enumerate(range(start, stop)):
                selected = stable_topk_index_set(similarities[row_offset], kept)
                global_left = np.full(kept, indices[local_left], dtype=np.int64)
                global_right = indices[selected]
                left_parts.append(np.minimum(global_left, global_right))
                right_parts.append(np.maximum(global_left, global_right))
            parts.append(
                encode_pair_keys(
                    np.concatenate(left_parts), np.concatenate(right_parts)
                )
            )
    finally:
        del gallery
        if device.type == "cuda":
            torch.cuda.synchronize()
    return np.unique(np.concatenate(parts))


def phash_block_candidate_keys(
    original: np.ndarray,
    flipped: np.ndarray,
    global_indices: np.ndarray,
    start: int,
    stop: int,
    radius: int = 12,
) -> np.ndarray:
    """Exact flip-aware Hamming-radius pairs for a query block.

    Only local right positions greater than the local query position are emitted, so
    blocks are disjoint and their union is already canonical.
    """

    first = np.asarray(original, dtype=np.uint64)
    second = np.asarray(flipped, dtype=np.uint64)
    indices = np.asarray(global_indices, dtype=np.int64)
    if first.shape != second.shape or first.ndim != 1 or len(first) != len(indices):
        raise ValueError("pHash/global-index shapes do not match.")
    if not 0 <= start < stop <= len(indices):
        raise ValueError("Invalid pHash query block bounds.")
    if not 0 <= radius <= 64:
        raise ValueError("pHash radius must be in 0..64.")

    query_original = first[start:stop, None]
    query_flipped = second[start:stop, None]
    distance = np.bitwise_count(query_original ^ first[None, :])
    np.minimum(
        distance,
        np.bitwise_count(query_original ^ second[None, :]),
        out=distance,
    )
    np.minimum(
        distance,
        np.bitwise_count(query_flipped ^ first[None, :]),
        out=distance,
    )
    np.minimum(
        distance,
        np.bitwise_count(query_flipped ^ second[None, :]),
        out=distance,
    )
    query_positions = np.arange(start, stop, dtype=np.int64)[:, None]
    gallery_positions = np.arange(len(indices), dtype=np.int64)[None, :]
    rows, columns = np.nonzero(
        (gallery_positions > query_positions) & (distance <= radius)
    )
    if len(rows) == 0:
        return np.empty(0, dtype=np.uint64)
    left = indices[start + rows]
    right = indices[columns]
    return np.unique(encode_pair_keys(left, right))


def merge_candidate_channels(
    robust_parts: Sequence[np.ndarray], phash_parts: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robust = (
        np.unique(np.concatenate([np.asarray(part, dtype=np.uint64) for part in robust_parts]))
        if robust_parts
        else np.empty(0, dtype=np.uint64)
    )
    phash = (
        np.unique(np.concatenate([np.asarray(part, dtype=np.uint64) for part in phash_parts]))
        if phash_parts
        else np.empty(0, dtype=np.uint64)
    )
    union = np.union1d(robust, phash)
    robust_flag = np.isin(union, robust, assume_unique=True)
    phash_flag = np.isin(union, phash, assume_unique=True)
    return union.astype(np.uint64), robust_flag, phash_flag


def normalized_exact_duplicate_payload(payload: Mapping[str, object]) -> dict[str, object]:
    groups = []
    for raw_group in payload["duplicate_groups"]:  # type: ignore[index]
        group = dict(raw_group)
        groups.append(
            {
                "sha256": str(group["sha256"]),
                "subclasses": sorted(str(value) for value in group["subclasses"]),
                "paths": sorted(
                    str(value).replace("\\", "/") for value in group["paths"]
                ),
            }
        )
    groups.sort(key=lambda group: str(group["sha256"]))
    return {
        "accepted_images": int(payload["accepted_images"]),
        "exact_duplicate_group_count": int(payload["exact_duplicate_group_count"]),
        "files_in_exact_duplicate_groups": int(
            payload["files_in_exact_duplicate_groups"]
        ),
        "redundant_exact_copies": int(payload["redundant_exact_copies"]),
        "cross_subclass_duplicate_group_count": int(
            payload["cross_subclass_duplicate_group_count"]
        ),
        "duplicate_groups": groups,
    }


def verify_exact_duplicate_manifest(path: Path) -> tuple[dict[str, object], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = normalized_exact_duplicate_payload(payload)
    digest = canonical_json_sha256(normalized)
    if digest != EXPECTED_NORMALIZED_EXACT_DUPLICATE_SHA256:
        raise ValueError(
            "Normalized exact-duplicate manifest hash mismatch: "
            f"expected={EXPECTED_NORMALIZED_EXACT_DUPLICATE_SHA256}, actual={digest}"
        )
    return normalized, digest


def exact_duplicate_pair_keys(
    normalized_manifest: Mapping[str, object],
    index_by_relative_path: Mapping[str, int],
) -> np.ndarray:
    keys: list[np.uint64] = []
    for raw_group in normalized_manifest["duplicate_groups"]:  # type: ignore[index]
        group = dict(raw_group)
        present = [
            index_by_relative_path[path]
            for path in group["paths"]
            if path in index_by_relative_path
        ]
        for left, right in combinations(sorted(present), 2):
            keys.append(np.uint64((left << 32) | right))
    return np.unique(np.asarray(keys, dtype=np.uint64))


def verify_exact_duplicate_coverage(
    candidate_keys: np.ndarray, expected_exact_keys: np.ndarray
) -> dict[str, int]:
    candidates = np.asarray(candidate_keys, dtype=np.uint64)
    expected = np.asarray(expected_exact_keys, dtype=np.uint64)
    missing = np.setdiff1d(expected, candidates, assume_unique=True)
    if len(missing):
        left, right = decode_pair_keys(missing[:5])
        raise ValueError(
            "Candidate union misses known exact-duplicate pairs: "
            f"{list(zip(left.tolist(), right.tolist()))}"
        )
    return {
        "expected_exact_duplicate_pairs_in_scope": int(len(expected)),
        "missing_exact_duplicate_pairs": 0,
    }


class ResumeManifest:
    """Hash-register immutable stage artifacts and verify them before resume."""

    def __init__(self, output_dir: Path, contract_sha256: str) -> None:
        self.output_dir = output_dir.resolve()
        self.path = self.output_dir / "stage_manifest.json"
        if self.path.exists():
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            if self.payload.get("contract_sha256") != contract_sha256:
                raise ValueError("Resume manifest belongs to a different run contract.")
        else:
            self.payload: dict[str, object] = {
                "schema_version": 1,
                "contract_sha256": contract_sha256,
                "artifacts": {},
            }
            atomic_json(self.path, self.payload)

    @property
    def artifacts(self) -> dict[str, dict[str, object]]:
        return self.payload["artifacts"]  # type: ignore[return-value]

    def _resolve(self, relative: str) -> Path:
        path = (self.output_dir / relative).resolve()
        if not path.is_relative_to(self.output_dir):
            raise ValueError(f"Artifact path escapes output directory: {relative}")
        return path

    def verify_registered(self) -> None:
        for relative, record in self.artifacts.items():
            path = self._resolve(relative)
            if not path.is_file():
                raise FileNotFoundError(f"Registered stage artifact is missing: {relative}")
            actual = file_sha256(path)
            if actual != record["sha256"]:
                raise ValueError(
                    f"Registered stage artifact hash mismatch: {relative}; "
                    f"expected={record['sha256']}, actual={actual}"
                )

    def is_complete(self, relative: str) -> bool:
        path = self._resolve(relative)
        record = self.artifacts.get(relative)
        if record is None:
            if path.exists():
                raise FileExistsError(
                    f"Unregistered artifact exists; refusing overwrite: {relative}"
                )
            return False
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Completed artifact failed resume verification: {relative}")
        return True

    def register(
        self,
        relative: str,
        stage: str,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        path = self._resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot register missing artifact: {relative}")
        if relative in self.artifacts:
            raise FileExistsError(f"Artifact is already registered: {relative}")
        record: dict[str, object] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "stage": stage,
            "metadata": dict(metadata or {}),
        }
        self.artifacts[relative] = record
        atomic_json(self.path, self.payload)
        return record

    def stage_seconds(self, stage: str) -> float:
        total = 0.0
        for record in self.artifacts.values():
            if record["stage"] == stage:
                total += float(dict(record.get("metadata", {})).get("seconds", 0.0))
        return total
