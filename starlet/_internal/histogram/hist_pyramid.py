from __future__ import annotations

import json
import logging
import os
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely import from_wkb, get_coordinates
from pyproj import Transformer

from starlet._internal.executor import create_process_executor
from starlet._internal.histogram.io import save_numpy_array
from starlet._internal.tiling.crs import WGS84_CRS, geoparquet_crs
from starlet._internal.tiling.datasource import GeoParquetSource

logger = logging.getLogger(__name__)

# Global Web Mercator extent
LIM = 20037508.342789244
GLOBAL_BBOX = (-LIM, -LIM, LIM, LIM)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class HistConfig:
    grid_size: int = 4096
    out_crs: str = "EPSG:3857"
    dtype: str = "float64"
    parallelism: int = 8


# ---------------------------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------------------------

def _geometry_vertices_array(g) -> np.ndarray:
    if g is None or g.is_empty:
        return np.empty((0, 2), dtype=np.float64)
    coords = get_coordinates(g)
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(coords[:, :2], dtype=np.float64)


def _geometry_vertices_iter(g):
    for x, y in _geometry_vertices_array(g):
        yield (float(x), float(y))


# ---------------------------------------------------------------------------
# PER TILE HISTOGRAM
# ---------------------------------------------------------------------------

def _accumulate_vertices_hist(
    table,
    histogram: np.ndarray,
    geom_col: str,
    bbox_out,
    transformer,
):
    """Transform vertices in batches and increment histogram cells in place."""
    geoms = from_wkb(table[geom_col].to_numpy(zero_copy_only=False))

    minx, miny, maxx, maxy = bbox_out
    inv_w = 1.0 / (maxx - minx)
    inv_h = 1.0 / (maxy - miny)
    grid_size = histogram.shape[0]

    coords = get_coordinates(geoms)
    if coords.size == 0:
        return

    coords = np.asarray(coords[:, :2], dtype=np.float64)
    xs = coords[:, 0]
    ys = coords[:, 1]
    transformed_x, transformed_y = transformer.transform(xs, ys)
    transformed_x = np.asarray(transformed_x, dtype=np.float64)
    transformed_y = np.asarray(transformed_y, dtype=np.float64)
    if transformed_x.ndim == 0:
        transformed_x = np.full_like(xs, float(transformed_x))
    if transformed_y.ndim == 0:
        transformed_y = np.full_like(ys, float(transformed_y))

    finite = np.isfinite(transformed_x) & np.isfinite(transformed_y)
    skipped = int(finite.size - np.count_nonzero(finite))
    if skipped:
        logger.debug(
            "Skipping %d histogram vertices with non-finite transformed coordinates",
            skipped,
        )

    if not np.any(finite):
        return

    transformed_x = transformed_x[finite]
    transformed_y = transformed_y[finite]
    ix = ((transformed_x - minx) * inv_w * grid_size).astype(np.int64)
    iy = ((transformed_y - miny) * inv_h * grid_size).astype(np.int64)
    ix = np.clip(ix, 0, grid_size - 1)
    iy = np.clip(iy, 0, grid_size - 1)
    np.add.at(histogram, (grid_size - 1 - iy, ix), 1)


# ---------------------------------------------------------------------------
# PROCESS SPLIT GROUPS
# ---------------------------------------------------------------------------

def _effective_parallelism(max_workers: int | None, split_count: int) -> int:
    if split_count <= 0:
        return 1
    if max_workers is None:
        return max(1, min(split_count, os.cpu_count() or 1))
    if max_workers <= 0:
        raise ValueError("parallelism must be positive")
    return min(int(max_workers), split_count)


def _split_groups(splits: Sequence, max_workers: int | None) -> List[List]:
    """Divide splits into balanced, non-empty groups for worker processes."""
    worker_count = _effective_parallelism(max_workers, len(splits))
    base_size, remainder = divmod(len(splits), worker_count)
    groups: List[List] = []
    offset = 0
    for index in range(worker_count):
        group_size = base_size + (1 if index < remainder else 0)
        groups.append(list(splits[offset:offset + group_size]))
        offset += group_size
    return groups


def _process_split_group(
    source_path: str,
    splits: Sequence,
    cfg,
    geom_col: str,
) -> np.ndarray:
    """Sequentially accumulate one balanced group of source splits."""
    logger.info("Processing histogram group with %d splits", len(splits))

    source = GeoParquetSource(
        source_path,
        geometry_only=True,
        geom_col=geom_col,
    )
    dtype = np.dtype(cfg.dtype)
    bbox = GLOBAL_BBOX
    base = np.zeros((cfg.grid_size, cfg.grid_size), dtype=dtype)
    transformers = {}

    for split in splits:
        for table in source.iter_tables(split):
            source_crs = geoparquet_crs(table.schema, geom_col) or WGS84_CRS
            transformer = transformers.get(str(source_crs))
            if transformer is None:
                transformer = Transformer.from_crs(source_crs, cfg.out_crs, always_xy=True)
                transformers[str(source_crs)] = transformer
            _accumulate_vertices_hist(
                table.combine_chunks(),
                base,
                geom_col,
                bbox,
                transformer,
            )

    return base


# ---------------------------------------------------------------------------
# GLOBAL SUM AND PREFIX SUM
# ---------------------------------------------------------------------------

def _sum_all_tiles(tile_hists: List[np.ndarray], outdir: Path, dtype="float64") -> Path:

    if not tile_hists:
        raise RuntimeError("No tile histograms generated")

    example = tile_hists[0]
    total = np.zeros_like(example, dtype=dtype)

    for hist in tile_hists:
        if hist.shape != total.shape:
            raise ValueError(
                f"Tile histogram has shape {hist.shape}, expected {total.shape}"
            )
        total += hist

    global_path = save_numpy_array(outdir / "global.npy.gz", total)

    global_json = {
        "filename": global_path.name,
        "dtype": str(total.dtype),
        "grid_size": int(total.shape[0]),
        "shape": list(total.shape),
        "crs": "EPSG:3857",
        "bbox": list(GLOBAL_BBOX),
        "sum": float(total.sum()),
        "nonzero": int(np.count_nonzero(total)),
    }
    with open(outdir / "global.json", "w") as f:
        json.dump(global_json, f, indent=2)

    logger.info(f"Wrote global histogram: {global_path}")

    return global_path


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def build_histograms_for_dir(
    tiles_dir: str,
    outdir: str,
    geom_col="geometry",
    grid_size=4096,
    dtype="float64",
    parallelism=8,
):
    cfg = HistConfig(
        grid_size=grid_size,
        dtype=dtype,
        parallelism=parallelism,
    )

    source = GeoParquetSource(
        tiles_dir,
        geometry_only=True,
        geom_col=geom_col,
    )
    splits = source.create_splits()
    if not splits:
        logger.error("No parquet tiles found")
        return

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    tile_outputs = []
    worker_count = _effective_parallelism(cfg.parallelism, len(splits))
    split_groups = _split_groups(splits, worker_count)
    logger.info(
        "Histogram computation: %d splits grouped into %d worker tasks",
        len(splits),
        len(split_groups),
    )

    with create_process_executor(
        max_workers=worker_count,
        logger=logger,
        context="histogram construction",
    ) as ex:
        futures = {
            ex.submit(_process_split_group, source.path, split_group, cfg, geom_col): split_group
            for split_group in split_groups
        }

        for f in as_completed(futures):
            tile_outputs.append(f.result())

    _sum_all_tiles(tile_outputs, outdir_p, dtype=dtype)
