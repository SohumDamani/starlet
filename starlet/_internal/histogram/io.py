from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

COMPRESSED_SUFFIX = ".npy.gz"
LEGACY_SUFFIX = ".npy"


def histogram_path_candidates(base_path: str | Path) -> list[Path]:
    path = Path(base_path)
    path_str = str(path)

    if path_str.endswith(COMPRESSED_SUFFIX) or path_str.endswith(LEGACY_SUFFIX):
        stem = path_str.removesuffix(COMPRESSED_SUFFIX).removesuffix(LEGACY_SUFFIX)
        return [Path(stem + COMPRESSED_SUFFIX), Path(stem + LEGACY_SUFFIX)]

    return [Path(path_str + COMPRESSED_SUFFIX), Path(path_str + LEGACY_SUFFIX)]


def resolve_histogram_path(base_path: str | Path) -> Path:
    for candidate in histogram_path_candidates(base_path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Histogram not found for {base_path}")


def load_numpy_array(path: str | Path) -> np.ndarray:
    resolved = resolve_histogram_path(path)
    if str(resolved).endswith(COMPRESSED_SUFFIX):
        with gzip.open(resolved, "rb") as handle:
            return np.load(handle, allow_pickle=False)
    return np.load(resolved, allow_pickle=False)


def load_histogram(hist_dir: str | Path) -> np.ndarray:
    return load_numpy_array(Path(hist_dir) / "global")


def load_prefix_histogram(hist_dir: str | Path) -> np.ndarray:
    return load_histogram(hist_dir).cumsum(axis=0).cumsum(axis=1)


def save_numpy_array(path: str | Path, array: np.ndarray) -> Path:
    target = Path(path)
    if not str(target).endswith(COMPRESSED_SUFFIX):
        target = Path(str(target).removesuffix(LEGACY_SUFFIX) + COMPRESSED_SUFFIX)

    with gzip.open(target, "wb", compresslevel=1) as handle:
        np.save(handle, array, allow_pickle=False)
    return target
