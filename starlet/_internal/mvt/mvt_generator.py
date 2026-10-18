"""Two-stage dataset-to-MVT generator using intermediate vector tiles."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import heapq
import logging
import math
import multiprocessing
import os
from pathlib import Path
import random
try:
    import resource
except ImportError:  # Windows
    resource = None
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyproj import Transformer
import shapely
from shapely import from_wkb

from starlet._internal.histogram.io import resolve_histogram_path
from starlet._internal.histogram.loader import HistogramLoader
from starlet._internal.config import config_value, resolve_temp_dir
from starlet._internal.mvt.helpers import (
    WORLD_MAXX,
    WORLD_MAXY,
    WORLD_MINX,
    WORLD_MINY,
    mercator_tile_bounds,
)
from starlet._internal.mvt.intermediate_tile import (
    IntermediateVectorTile,
    feature_priority,
    normalize_simplify_tolerance,
    normalize_tile_attributes,
)
from starlet._internal.mvt.pyramid_partitioner import PyramidPartitioner
from starlet._internal.pmtiles.paths import default_pmtiles_path
from starlet._internal.pmtiles.exporter import export_to_pmtiles
from starlet._internal.internal_columns import BBOX_COLS, MVT_EXCLUDED_ATTRIBUTE_COLS
from starlet._internal.server.tiler.parquet_index import ParquetIndex
from starlet._internal.tiling.crs import WEB_MERCATOR_CRS, WGS84_CRS, geoparquet_crs, reproject_geometries
from starlet._internal.tiling.geoparquet_source import GeoParquetSource, GeoParquetSplit

logger = logging.getLogger(__name__)

_INTERNAL_ATTRIBUTE_COLUMNS = set(MVT_EXCLUDED_ATTRIBUTE_COLS)
_SINGLE_TILE_INDEX_CACHE_SIZE = 16
_single_tile_index_cache: "OrderedDict[str, ParquetIndex]" = OrderedDict()
_REDUCE_GROUP_SIZE = 10


@dataclass(frozen=True)
class DatasetMVTGenerationResult:
    outdir: str
    tile_count: int
    zoom_levels: list[int]
    tile_counts_by_zoom: list[int]
    pmtiles_path: str | None = None


@dataclass(frozen=True)
class _MapStageResult:
    intermediate_dir: str
    tile_part_counts: tuple[tuple[int, int], ...]

    @property
    def tile_ids(self) -> list[int]:
        return [tile_id for tile_id, _ in self.tile_part_counts]

    @property
    def total_parts(self) -> int:
        return sum(part_count for _, part_count in self.tile_part_counts)


@dataclass(frozen=True)
class _ReduceTileInput:
    tile_id: int
    mapper_parts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _TableBatch:
    table: pa.Table


_MapInput = GeoParquetSplit | _TableBatch
_MAPPER_PROGRESS_ROWS = 1_000_000
_MAPPER_MEMORY_CHECK_FEATURES = 10_000
_MIB = 1024 * 1024


@dataclass(frozen=True)
class _MapperMemoryBudget:
    high_watermark_bytes: int
    low_watermark_bytes: int
    total_budget_bytes: int

    @property
    def high_watermark_mib(self) -> float:
        return self.high_watermark_bytes / _MIB

    @property
    def low_watermark_mib(self) -> float:
        return self.low_watermark_bytes / _MIB


def _resolve_total_mvt_memory_budget(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "off", "false", "0"}:
            return None
        if normalized == "auto":
            available = _system_available_memory_bytes() or _system_total_memory_bytes()
            if available is None:
                return None
            return int(available * 0.70)
    else:
        if int(value) <= 0:
            return None

    parsed = _parse_memory_size_bytes(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _mvt_memory_budget_for_stage(
    *,
    total_budget_bytes: int | None,
    active_mappers: int,
) -> _MapperMemoryBudget | None:
    if total_budget_bytes is None:
        return None
    active_mappers = max(1, int(active_mappers))
    high = max(_MIB, int(total_budget_bytes / active_mappers))
    low = max(_MIB, int(high * 0.85))
    return _MapperMemoryBudget(
        high_watermark_bytes=high,
        low_watermark_bytes=low,
        total_budget_bytes=total_budget_bytes,
    )


def _parse_memory_size_bytes(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s.isdigit():
            return int(s)
        suffixes = {
            "kb": 1024,
            "mb": 1024 ** 2,
            "gb": 1024 ** 3,
            "tb": 1024 ** 4,
        }
        for suffix, mul in suffixes.items():
            if s.endswith(suffix):
                return int(float(s[: -len(suffix)].strip()) * mul)
    raise ValueError(f"Invalid MVT memory budget: {value!r}")


class _MapperTileCache:
    def __init__(
        self,
        *,
        mapper_index: int,
        intermediate_dir: Path,
        feature_capacity: int,
        memory_budget: _MapperMemoryBudget | None,
        extent: int,
        buffer: int,
    ) -> None:
        self.mapper_index = int(mapper_index)
        self.intermediate_dir = intermediate_dir
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.feature_capacity = int(feature_capacity)
        self.memory_budget = memory_budget
        self.extent = int(extent)
        self.buffer = int(buffer)
        self.tiles: OrderedDict[int, IntermediateVectorTile] = OrderedDict()
        self.feature_counts: dict[int, int] = {}
        self.live_features = 0
        self.tile_part_counts: dict[int, int] = {}
        self.peak_live_features = 0
        self.peak_active_tiles = 0
        self.memory_evictions = 0
        self.memory_spill_events = 0
        self.total_parts = 0
        self._features_since_memory_check = 0

    def get(self, tile_id: int) -> IntermediateVectorTile | None:
        tile = self.tiles.get(tile_id)
        if tile is not None:
            self.tiles.move_to_end(tile_id)
        return tile

    def put(self, tile_id: int, tile: IntermediateVectorTile) -> None:
        self.tiles[tile_id] = tile
        self.tiles.move_to_end(tile_id)
        self.feature_counts[tile_id] = tile.feature_count
        self.peak_active_tiles = max(self.peak_active_tiles, len(self.tiles))

    def account_feature_change(self, tile_id: int) -> None:
        tile = self.tiles[tile_id]
        old_count = self.feature_counts.get(tile_id, 0)
        new_count = tile.feature_count
        delta = new_count - old_count
        self.live_features += delta
        self.feature_counts[tile_id] = new_count
        self.peak_live_features = max(self.peak_live_features, self.live_features)
        if delta > 0:
            self._features_since_memory_check += delta
            self._evict_under_memory_pressure()

    def flush_remaining(self) -> tuple[tuple[int, int], ...]:
        while self.tiles:
            self._write_lru_tile()
        return tuple(sorted(self.tile_part_counts.items()))

    def _evict_under_memory_pressure(self) -> None:
        if self.memory_budget is None:
            return
        if self._features_since_memory_check < _MAPPER_MEMORY_CHECK_FEATURES:
            return
        self._features_since_memory_check = 0
        current_rss, peak_rss = _process_rss_mib()
        current_rss = current_rss if current_rss is not None else peak_rss
        if current_rss is None:
            return
        if current_rss * _MIB <= self.memory_budget.high_watermark_bytes:
            return

        self.memory_spill_events += 1
        spill_start_rss = current_rss
        evictions_before = self.memory_evictions
        while self.tiles and current_rss * _MIB > self.memory_budget.low_watermark_bytes:
            self.memory_evictions += 1
            self._write_lru_tile()
            current_rss, peak_rss = _process_rss_mib()
            current_rss = current_rss if current_rss is not None else peak_rss
            if current_rss is None:
                break
        logger.info(
            "DatasetMVTGenerator mapper %d memory spill: rss_mib=%s "
            "after_rss_mib=%s high_mib=%.1f low_mib=%.1f "
            "evicted_tiles=%d live_features=%d active_tiles=%d parts=%d",
            self.mapper_index,
            _metric(spill_start_rss),
            _metric(current_rss),
            self.memory_budget.high_watermark_mib,
            self.memory_budget.low_watermark_mib,
            self.memory_evictions - evictions_before,
            self.live_features,
            len(self.tiles),
            self.total_parts,
        )

    def _write_lru_tile(self) -> None:
        tile_id, tile = self.tiles.popitem(last=False)
        self.live_features -= self.feature_counts.pop(tile_id, 0)
        if tile.feature_count == 0:
            return
        z, x, y = PyramidPartitioner.decode_tile_id(tile_id)
        part_index = self.tile_part_counts.get(tile_id, 0)
        path = self.intermediate_dir / _intermediate_tile_part_filename(
            z,
            x,
            y,
            part_index,
        )
        tile.write_features(path)
        self.tile_part_counts[tile_id] = part_index + 1
        self.total_parts += 1


class DatasetMVTGenerator:
    """Generate MVT tiles from a Starlet tiled dataset.

    This class is intentionally separate from the existing streaming MVT
    generator while the intermediate-tile workflow is developed.
    """

    def __init__(
        self,
        dataset_dir: str,
        *,
        num_zoom_levels: int,
        threshold: float,
        output_format: str = "mvt",
        outdir: str | None = None,
        pmtiles_path: str | None = None,
        pmtiles_compression: str = "gzip",
        workers: int | None = None,
        feature_capacity: int | None = None,
        mvt_memory_budget: str | int | None = None,
        extent: int | None = None,
        buffer: int | None = None,
        geom_col: str = "geometry",
        seed: int = 42,
        temp_dir: str | None = None,
        tile_attributes: Any = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.tile_attributes = normalize_tile_attributes(
            tile_attributes if tile_attributes is not None else config_value("mvt", "tile_attributes")
        )
        self.simplify_tolerance = config_value("mvt", "simplify_tolerance")
        self.parquet_dir = self.dataset_dir / "parquet_tiles"
        self.hist_path = self.dataset_dir / "histograms" / "global"
        self.num_zoom_levels = int(num_zoom_levels)
        self.threshold = float(threshold)
        self.output_format = output_format.strip().lower()
        self.outdir = Path(outdir) if outdir is not None else self.dataset_dir / "mvt"
        self.pmtiles_path = Path(pmtiles_path) if pmtiles_path is not None else default_pmtiles_path(self.dataset_dir)
        self.pmtiles_compression = pmtiles_compression
        cpu_default = max(1, multiprocessing.cpu_count() - 1)
        self.workers = max(1, int(workers or cpu_default))
        self.feature_capacity = int(
            feature_capacity if feature_capacity is not None else config_value("mvt", "feature_capacity")
        )
        configured_mvt_memory_budget = config_value("mvt", "mvt_memory_budget")
        self.mvt_memory_budget = _resolve_total_mvt_memory_budget(
            mvt_memory_budget
            if mvt_memory_budget is not None
            else configured_mvt_memory_budget
        )
        self.extent = int(extent if extent is not None else config_value("mvt", "extent"))
        self.buffer = int(buffer if buffer is not None else config_value("mvt", "buffer"))
        self.partition_buffer = float(self.buffer) / float(self.extent)
        self.geom_col = geom_col
        self.seed = int(seed)
        self.temp_dir = temp_dir

        if self.num_zoom_levels <= 0:
            raise ValueError("num_zoom_levels must be positive")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if self.output_format not in {"mvt", "pmtiles"}:
            raise ValueError("output_format must be 'mvt' or 'pmtiles'")
        if self.extent <= 0:
            raise ValueError("extent must be positive")

    def run(self) -> DatasetMVTGenerationResult:
        if not self.parquet_dir.is_dir():
            raise FileNotFoundError(f"GeoParquet tile directory not found: {self.parquet_dir}")
        try:
            self.hist_path = resolve_histogram_path(self.hist_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Histogram not found under {self.dataset_dir / 'histograms'}"
            ) from exc

        from starlet._internal.mvt import rust_engine

        if rust_engine.batch_enabled() and _rust_supports_dataset(str(self.dataset_dir)):
            return self._run_rust_pyramid()
        if str(config_value("mvt", "python_pyramid")).lower() != "mapreduce" and _rust_supports_dataset(str(self.dataset_dir)):
            return self._run_python_pyramid()

        source = GeoParquetSource(str(self.parquet_dir), geom_col=self.geom_col)
        map_groups = _create_map_groups(source, self.workers)
        if not map_groups:
            return DatasetMVTGenerationResult(str(self.outdir), 0, [], [], None)

        temp_parent = resolve_temp_dir(self.temp_dir, self.dataset_dir / "tmp")
        with tempfile.TemporaryDirectory(prefix="starlet_mvt_", dir=temp_parent) as temp_dir:
            map_results = self._run_map_stage(map_groups, source, Path(temp_dir))
            self._run_reduce_stage(map_results)

        tile_counts_by_zoom = _discover_tile_counts_by_zoom(self.outdir)
        zoom_levels = [z for z, count in enumerate(tile_counts_by_zoom) if count > 0]
        tile_count = sum(tile_counts_by_zoom)

        pmtiles_path = None
        if self.output_format == "pmtiles":
            if tile_count == 0:
                logger.warning(
                    "No MVT tiles passed threshold %s; skipping PMTiles export. "
                    "Lower the MVT threshold to generate a prebuilt pyramid.",
                    self.threshold,
                )
            else:
                pmtiles_path = str(self.pmtiles_path)
                export_to_pmtiles(
                    mvt_dir=str(self.outdir),
                    output_path=pmtiles_path,
                    tile_type="mvt",
                    compression=self.pmtiles_compression,
                )
            if self.outdir.exists():
                shutil.rmtree(self.outdir)
        return DatasetMVTGenerationResult(
            outdir=str(self.outdir),
            tile_count=tile_count,
            zoom_levels=zoom_levels,
            tile_counts_by_zoom=tile_counts_by_zoom,
            pmtiles_path=pmtiles_path,
        )

    def _enumerate_tiles(self) -> dict[int, list[tuple[int, int, int]]]:
        """Tiles passing the histogram / threshold filter, per zoom, in
        quadtree (spatially local) order."""
        prefix = HistogramLoader(str(self.hist_path)).load()
        partitioner = PyramidPartitioner(
            (WORLD_MINX, WORLD_MINY, WORLD_MAXX, WORLD_MAXY),
            self.num_zoom_levels,
            prefix_histogram=prefix,
            size_threshold=self.threshold,
            buffer=self.partition_buffer,
        )
        tiles_by_zoom: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        stack: list[tuple[int, int, int]] = [(0, 0, 0)]
        while stack:
            z, x, y = stack.pop()
            if not partitioner._tile_passes_filter(PyramidPartitioner.encode_tile_id(z, x, y), z, x, y):
                continue
            tiles_by_zoom[z].append((z, x, y))
            if z < partitioner.max_zoom:
                stack.extend(((z + 1, 2 * x, 2 * y), (z + 1, 2 * x + 1, 2 * y),
                              (z + 1, 2 * x, 2 * y + 1), (z + 1, 2 * x + 1, 2 * y + 1)))
        return tiles_by_zoom

    def _finish(self) -> DatasetMVTGenerationResult:
        tile_counts_by_zoom = _discover_tile_counts_by_zoom(self.outdir)
        zoom_levels = [z for z, count in enumerate(tile_counts_by_zoom) if count > 0]
        tile_count = sum(tile_counts_by_zoom)
        pmtiles_path = None
        if self.output_format == "pmtiles":
            pmtiles_path = str(self.pmtiles_path)
            export_to_pmtiles(
                mvt_dir=str(self.outdir), output_path=pmtiles_path,
                tile_type="mvt", compression=self.pmtiles_compression,
            )
            if self.outdir.exists():
                shutil.rmtree(self.outdir)
        return DatasetMVTGenerationResult(
            outdir=str(self.outdir), tile_count=tile_count, zoom_levels=zoom_levels,
            tile_counts_by_zoom=tile_counts_by_zoom, pmtiles_path=pmtiles_path,
        )

    def _run_python_pyramid(self) -> DatasetMVTGenerationResult:
        """Pure-Python pyramid without intermediate files: every tile is
        generated by the vectorised single-tile path (:mod:`fast_tile`) in a
        process pool, tiles handed out in quadtree order so each worker's
        row-group cache stays hot. Memory is bounded by the per-worker cache
        (``mvt.batch_row_group_cache`` row groups); no temp disk is used."""
        tiles_by_zoom = self._enumerate_tiles()
        self.outdir.mkdir(parents=True, exist_ok=True)
        params = dict(
            feature_capacity=self.feature_capacity, extent=self.extent, buffer=self.buffer,
            tile_attributes=self.tile_attributes, simplify_tolerance=self.simplify_tolerance,
            row_group_cache_size=int(config_value("mvt", "batch_row_group_cache")),
        )
        chunk = 64
        # Zooms with few tiles (each touching most of the dataset) go through
        # the push-mode pass, which streams every row group once for all of
        # them; the rest are pulled tile by tile.
        push_max = int(config_value("mvt", "push_max_tiles"))
        push_zooms = [z for z in sorted(tiles_by_zoom) if len(tiles_by_zoom[z]) <= push_max]
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            if push_zooms:
                from starlet._internal.mvt.python_pyramid import run_push

                push_tiles = [t for z in push_zooms for t in tiles_by_zoom[z]]
                written = run_push(str(self.dataset_dir), str(self.outdir), push_tiles, params, executor, self.workers)
                logger.info("DatasetMVTGenerator[python] push zooms=%s candidates=%d written=%d", push_zooms, len(push_tiles), written)
            for z in sorted(tiles_by_zoom):
                if z in push_zooms:
                    continue
                tiles = tiles_by_zoom[z]
                futures = [
                    executor.submit(_python_pyramid_chunk, str(self.dataset_dir), str(self.outdir), tiles[i:i + chunk], params)
                    for i in range(0, len(tiles), chunk)
                ]
                written = sum(f.result() for f in futures)
                logger.info("DatasetMVTGenerator[python] z=%d candidates=%d written=%d", z, len(tiles), written)
        return self._finish()

    def _run_rust_pyramid(self) -> DatasetMVTGenerationResult:
        """Pull-based pyramid on the Rust engine (``STARLET_ENGINE=rust``).

        The tile *set* is the same as the map/reduce path's: every tile that
        passes the histogram/threshold filter and ends up with at least one
        feature. Each tile is generated independently from bbox-pruned row
        groups (top-k by ``crc32(WKB)``, like the Python pipeline), in
        parallel across all cores, and written straight to ``<z>/<x>/<y>.mvt``
        from Rust; no intermediate files, no reduce stage.
        """
        from starlet._internal.mvt import rust_engine

        prefix = HistogramLoader(str(self.hist_path)).load()
        partitioner = PyramidPartitioner(
            (WORLD_MINX, WORLD_MINY, WORLD_MAXX, WORLD_MAXY),
            self.num_zoom_levels,
            prefix_histogram=prefix,
            size_threshold=self.threshold,
            buffer=self.partition_buffer,
        )
        # Quadtree walk: histogram mass is monotone, so a tile that fails the
        # filter has no descendant that passes it.
        tiles_by_zoom: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        stack: list[tuple[int, int, int]] = [(0, 0, 0)]
        while stack:
            z, x, y = stack.pop()
            if not partitioner._tile_passes_filter(PyramidPartitioner.encode_tile_id(z, x, y), z, x, y):
                continue
            tiles_by_zoom[z].append((z, x, y))
            if z < partitioner.max_zoom:
                stack.extend(((z + 1, 2 * x, 2 * y), (z + 1, 2 * x + 1, 2 * y),
                              (z + 1, 2 * x, 2 * y + 1), (z + 1, 2 * x + 1, 2 * y + 1)))

        ds = rust_engine.dataset(self.dataset_dir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        # Push-mode pyramid: two streaming passes over the row groups per
        # call, memory bounded by per-tile winner *references* plus the tiles
        # still being filled. Zooms are batched so one call never tracks more
        # than _RUST_PYRAMID_MAX_TILES tiles at once.
        bands: list[list[int]] = []
        for z in sorted(tiles_by_zoom):
            if bands and sum(len(tiles_by_zoom[b]) for b in bands[-1]) + len(tiles_by_zoom[z]) <= _RUST_PYRAMID_MAX_TILES:
                bands[-1].append(z)
            else:
                bands.append([z])
        for band in bands:
            tiles = [t for z in band for t in tiles_by_zoom[z]]
            st = ds.write_pyramid(
                str(self.outdir), tiles,
                feature_capacity=self.feature_capacity, extent=self.extent, buffer=self.buffer,
                tile_attributes=self.tile_attributes,
                simplify_tolerance=_tolerance_arg(self.simplify_tolerance, self.extent),
            )
            logger.info(
                "DatasetMVTGenerator[rust] zooms=%s requested=%d written=%d candidates=%d features=%d",
                band, st["tiles_requested"], st["tiles_written"], st["candidates"], st["features"],
            )

        tile_counts_by_zoom = _discover_tile_counts_by_zoom(self.outdir)
        zoom_levels = [z for z, count in enumerate(tile_counts_by_zoom) if count > 0]
        tile_count = sum(tile_counts_by_zoom)
        pmtiles_path = None
        if self.output_format == "pmtiles":
            pmtiles_path = str(self.pmtiles_path)
            export_to_pmtiles(
                mvt_dir=str(self.outdir), output_path=pmtiles_path,
                tile_type="mvt", compression=self.pmtiles_compression,
            )
            if self.outdir.exists():
                shutil.rmtree(self.outdir)
        return DatasetMVTGenerationResult(
            outdir=str(self.outdir), tile_count=tile_count, zoom_levels=zoom_levels,
            tile_counts_by_zoom=tile_counts_by_zoom, pmtiles_path=pmtiles_path,
        )

    def _run_map_stage(
        self,
        map_groups: Sequence[Sequence[_MapInput]],
        source: GeoParquetSource,
        temp_root: Path,
    ) -> list[_MapStageResult]:
        active_mappers = min(self.workers, len(map_groups))
        memory_budget = _mvt_memory_budget_for_stage(
            total_budget_bytes=self.mvt_memory_budget,
            active_mappers=active_mappers,
        )
        logger.info(
            "DatasetMVTGenerator map stage: groups=%d workers=%d "
            "feature_capacity=%d mvt_memory_budget_mib=%s "
            "per_mapper_memory_high_mib=%s per_mapper_memory_low_mib=%s",
            len(map_groups),
            self.workers,
            self.feature_capacity,
            _metric(None if self.mvt_memory_budget is None else self.mvt_memory_budget / _MIB),
            _metric(None if memory_budget is None else memory_budget.high_watermark_mib),
            _metric(None if memory_budget is None else memory_budget.low_watermark_mib),
        )
        try:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(
                        _map_split_group,
                        group,
                        source,
                        str(self.hist_path),
                        self.num_zoom_levels,
                        self.threshold,
                        self.partition_buffer,
                        self.feature_capacity,
                        memory_budget,
                        self.extent,
                        self.buffer,
                        self.seed + index,
                        str(temp_root),
                        index,
                    ): index
                    for index, group in enumerate(map_groups)
                }
                results = []
                for future in as_completed(futures):
                    mapper_index = futures[future]
                    result = future.result()
                    results.append(result)
                    logger.info(
                        "DatasetMVTGenerator map progress: completed=%d/%d "
                        "mapper=%d parts=%d",
                        len(results),
                        len(futures),
                        mapper_index,
                        result.total_parts,
                    )
                return results
        except Exception:
            logger.exception(
                "DatasetMVTGenerator map stage failed. If a worker exited "
                "abruptly, check the preceding mapper RSS/live_features logs "
                "for an OOM kill; lower mvt_memory_budget or worker count."
            )
            raise

    def _run_reduce_stage(self, map_results: list[_MapStageResult]) -> None:
        if not map_results:
            logger.info("DatasetMVTGenerator reduce stage: no intermediate tiles")
            return

        self.outdir.mkdir(parents=True, exist_ok=True)
        tile_locations: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for result in map_results:
            for tile_id, part_count in result.tile_part_counts:
                tile_locations[tile_id].append((result.intermediate_dir, part_count))

        reduce_inputs = [
            _ReduceTileInput(tile_id, tuple(mapper_parts))
            for tile_id, mapper_parts in sorted(tile_locations.items())
        ]
        reduce_groups = _chunk_reduce_inputs(reduce_inputs)
        logger.info(
            "DatasetMVTGenerator reduce stage: tile_ids=%d groups=%d workers=%d",
            len(tile_locations),
            len(reduce_groups),
            self.workers,
        )
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(
                    _reduce_tile_group,
                    tuple(group),
                    str(self.outdir),
                    self.feature_capacity,
                    self.extent,
                    self.buffer,
                    self.tile_attributes,
                    self.simplify_tolerance,
                )
                for group in reduce_groups
                if group
            ]
            for future in as_completed(futures):
                future.result()


def _map_split_group(
    inputs: Sequence[_MapInput],
    source: GeoParquetSource,
    hist_path: str,
    num_zoom_levels: int,
    threshold: float,
    partition_buffer: float,
    feature_capacity: int,
    memory_budget: _MapperMemoryBudget | None,
    extent: int,
    buffer: int,
    seed: int,
    temp_root: str,
    mapper_index: int,
) -> _MapStageResult:
    started = time.monotonic()
    current_rss, peak_rss = _process_rss_mib()
    logger.info(
        "DatasetMVTGenerator mapper %d starting: inputs=%d rss_mib=%s "
        "peak_rss_mib=%s memory_high_mib=%s memory_low_mib=%s",
        mapper_index,
        len(inputs),
        _metric(current_rss),
        _metric(peak_rss),
        _metric(None if memory_budget is None else memory_budget.high_watermark_mib),
        _metric(None if memory_budget is None else memory_budget.low_watermark_mib),
    )
    prefix = HistogramLoader(hist_path).load()
    partitioner = PyramidPartitioner(
        (WORLD_MINX, WORLD_MINY, WORLD_MAXX, WORLD_MAXY),
        num_zoom_levels,
        prefix_histogram=prefix,
        size_threshold=threshold,
        buffer=partition_buffer,
    )
    current_rss, peak_rss = _process_rss_mib()
    logger.info(
        "DatasetMVTGenerator mapper %d initialized: histogram_mib=%.1f "
        "rss_mib=%s peak_rss_mib=%s arrow_mib=%.1f",
        mapper_index,
        int(getattr(prefix, "nbytes", 0)) / _MIB,
        _metric(current_rss),
        _metric(peak_rss),
        pa.total_allocated_bytes() / _MIB,
    )
    intermediate_dir = Path(temp_root) / f"mapper-{mapper_index:06d}"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    tiles = _MapperTileCache(
        mapper_index=mapper_index,
        intermediate_dir=intermediate_dir,
        feature_capacity=feature_capacity,
        memory_budget=memory_budget,
        extent=extent,
        buffer=buffer,
    )

    input_rows = 0
    valid_features = 0
    tile_feature_offers = 0
    retained_offers = 0
    max_batch_bytes = 0
    next_progress_row = _MAPPER_PROGRESS_ROWS
    for table in _iter_map_input_tables(source, inputs):
        input_rows += table.num_rows
        max_batch_bytes = max(max_batch_bytes, int(getattr(table, "nbytes", 0)))
        for geom, attrs, priority in _iter_web_mercator_features(table, source.geom_col):
            valid_features += 1
            bounds = _positive_bounds_tuple(geom.bounds)
            tile_ids = partitioner.overlapping_tile_ids(bounds)
            if not tile_ids:
                continue
            tile_feature_offers += len(tile_ids)
            for tile_id in tile_ids:
                tile = tiles.get(tile_id)
                if tile is None:
                    z, x, y = PyramidPartitioner.decode_tile_id(tile_id)
                    tile = IntermediateVectorTile(
                        z,
                        x,
                        y,
                        feature_capacity=feature_capacity,
                        extent=extent,
                        buffer=buffer,
                    )
                    tiles.put(tile_id, tile)
                retained = tile.add_feature(
                    geom,
                    attrs,
                    priority=priority,
                )
                if retained:
                    retained_offers += 1
                    tiles.account_feature_change(tile_id)
        if input_rows >= next_progress_row:
            _log_mapper_progress(
                mapper_index=mapper_index,
                input_rows=input_rows,
                valid_features=valid_features,
                tile_feature_offers=tile_feature_offers,
                retained_offers=retained_offers,
                max_batch_bytes=max_batch_bytes,
                tiles=tiles,
                started=started,
            )
            next_progress_row = ((input_rows // _MAPPER_PROGRESS_ROWS) + 1) * _MAPPER_PROGRESS_ROWS
    tile_part_counts = tiles.flush_remaining()
    total_parts = sum(part_count for _, part_count in tile_part_counts)
    current_rss, peak_rss = _process_rss_mib()
    logger.info(
        "DatasetMVTGenerator mapper %d finished: input_rows=%d "
        "valid_features=%d tile_feature_offers=%d retained_offers=%d "
        "tile_ids=%d parts=%d memory_evictions=%d memory_spill_events=%d "
        "peak_live_features=%d peak_active_tiles=%d max_batch_mib=%.1f "
        "rss_mib=%s peak_rss_mib=%s arrow_mib=%.1f elapsed_s=%.1f",
        mapper_index,
        input_rows,
        valid_features,
        tile_feature_offers,
        retained_offers,
        len(tile_part_counts),
        total_parts,
        tiles.memory_evictions,
        tiles.memory_spill_events,
        tiles.peak_live_features,
        tiles.peak_active_tiles,
        max_batch_bytes / _MIB,
        _metric(current_rss),
        _metric(peak_rss),
        pa.total_allocated_bytes() / _MIB,
        time.monotonic() - started,
    )
    return _MapStageResult(str(intermediate_dir), tile_part_counts)


def _log_mapper_progress(
    *,
    mapper_index: int,
    input_rows: int,
    valid_features: int,
    tile_feature_offers: int,
    retained_offers: int,
    max_batch_bytes: int,
    tiles: _MapperTileCache,
    started: float,
) -> None:
    current_rss, peak_rss = _process_rss_mib()
    logger.info(
        "DatasetMVTGenerator mapper %d progress: input_rows=%d "
        "valid_features=%d tile_feature_offers=%d retained_offers=%d "
        "live_features=%d active_tiles=%d parts=%d "
        "memory_evictions=%d memory_spill_events=%d "
        "max_batch_mib=%.1f rss_mib=%s peak_rss_mib=%s arrow_mib=%.1f "
        "elapsed_s=%.1f",
        mapper_index,
        input_rows,
        valid_features,
        tile_feature_offers,
        retained_offers,
        tiles.live_features,
        len(tiles.tiles),
        tiles.total_parts,
        tiles.memory_evictions,
        tiles.memory_spill_events,
        max_batch_bytes / _MIB,
        _metric(current_rss),
        _metric(peak_rss),
        pa.total_allocated_bytes() / _MIB,
        time.monotonic() - started,
    )


def _metric(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.1f}"


def _process_rss_mib() -> tuple[float | None, float | None]:
    """Return current and peak RSS for lightweight worker diagnostics."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        if query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / _MIB, counters.PeakWorkingSetSize / _MIB
        return None, None
    current = None
    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            current = resident_pages * os.sysconf("SC_PAGE_SIZE") / _MIB
        except (OSError, ValueError, IndexError):
            pass

    if resource is None:
        return current, None
    try:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS and the BSDs report bytes.
        peak /= _MIB if sys.platform == "darwin" else 1024.0
    except (OSError, ValueError):
        peak = None
    return current, peak


def _system_available_memory_bytes() -> int | None:
    values = []
    if not sys.platform.startswith("linux"):
        return _cgroup_available_memory_bytes()
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                values.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    cgroup_available = _cgroup_available_memory_bytes()
    if cgroup_available is not None:
        values.append(cgroup_available)
    return min(values) if values else None


def _cgroup_available_memory_bytes() -> int | None:
    candidates = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, current_path in candidates:
        try:
            raw_limit = limit_path.read_text().strip()
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
            current = int(current_path.read_text().strip())
        except (OSError, ValueError):
            continue
        # Some cgroup v1 hosts expose a huge sentinel rather than a real limit.
        if limit <= 0 or limit >= 2 ** 60:
            continue
        return max(0, limit - current)
    return None


def _system_total_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (OSError, ValueError, AttributeError):
        return None
    return None


def _iter_map_input_tables(
    source: GeoParquetSource,
    inputs: Sequence[_MapInput],
) -> Iterable[pa.Table]:
    for item in inputs:
        if isinstance(item, _TableBatch):
            yield item.table
        else:
            yield from source.iter_tables(item)


def _python_pyramid_chunk(dataset_dir: str, outdir: str, tiles, params: dict) -> int:
    """Worker: generate a chunk of tiles with the vectorised path and write the
    non-empty ones. The worker's row-group cache persists across chunks."""
    from starlet._internal.mvt import fast_tile

    dataset_path = Path(dataset_dir)
    index = _single_tile_parquet_index(dataset_path / "parquet_tiles")
    out = Path(outdir)
    written = 0
    for (z, x, y) in tiles:
        data, count = fast_tile.generate_tile_ex(dataset_path, (z, x, y), index=index, **params)
        if count == 0:
            continue
        d = out / str(z) / str(x)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{y}.mvt").write_bytes(data)
        written += 1
    return written


def _reduce_tile_group(
    reduce_inputs: Sequence[_ReduceTileInput],
    outdir: str,
    feature_capacity: int,
    extent: int,
    buffer: int,
    tile_attributes: list[str] | None = None,
    simplify_tolerance: Any = None,
) -> None:
    out_path = Path(outdir)
    for reduce_input in reduce_inputs:
        tile_id = reduce_input.tile_id
        z, x, y = PyramidPartitioner.decode_tile_id(tile_id)
        merged = IntermediateVectorTile(
            z,
            x,
            y,
            feature_capacity=feature_capacity,
            extent=extent,
            buffer=buffer,
            rng=random.Random(tile_id),
            tile_attributes=tile_attributes,
            simplify_tolerance=simplify_tolerance,
        )
        first_tile = True
        for intermediate_dir, part_count in reduce_input.mapper_parts:
            for part_index in range(part_count):
                path = Path(intermediate_dir) / _intermediate_tile_part_filename(
                    z,
                    x,
                    y,
                    part_index,
                )
                if not path.exists():
                    continue
                if first_tile:
                    merged.load_features(path)
                    first_tile = False
                else:
                    partial = IntermediateVectorTile(
                        z,
                        x,
                        y,
                        feature_capacity=feature_capacity,
                        extent=extent,
                        buffer=buffer,
                    )
                    partial.load_features(path)
                    merged.merge(partial)

        if merged.feature_count == 0:
            continue
        x_dir = out_path / str(z) / str(x)
        x_dir.mkdir(parents=True, exist_ok=True)
        with open(x_dir / f"{y}.mvt", "wb") as output:
            output.write(merged.encode())


def _intermediate_tile_part_filename(z: int, x: int, y: int, part_index: int) -> str:
    return f"{z}-{x}-{y}-part-{int(part_index):06d}.pyarrow"


_RUST_PYRAMID_MAX_TILES = 1_000_000
_RUST_DATASET_OK: dict[str, bool] = {}


def _rust_supports_dataset(dataset_path: str) -> bool:
    """The Rust engine reads lon/lat (EPSG:4326) partitions; anything else stays on Python."""
    key = str(Path(dataset_path).resolve())
    ok = _RUST_DATASET_OK.get(key)
    if ok is None:
        ok = False
        try:
            files = sorted((Path(dataset_path) / "parquet_tiles").glob("*.parquet"))
            if files:
                from starlet._internal.server.tiler.parquet_index import geometry_column
                from starlet._internal.tiling.crs import geoparquet_crs

                schema = pq.ParquetFile(files[0]).schema_arrow
                geom_col = geometry_column(schema)
                crs = geoparquet_crs(schema, geom_col)
                if crs is None:
                    ok = True
                else:
                    from pyproj import CRS

                    ok = CRS.from_user_input(crs).equals(CRS.from_epsg(4326), ignore_axis_order=True)
        except Exception:
            ok = False
        _RUST_DATASET_OK[key] = ok
    return ok


def generate_single_mvt_tile(
    dataset_path: str,
    tile_id: tuple[int, int, int],
    *,
    feature_capacity: int | None = None,
    extent: int | None = None,
    buffer: int | None = None,
    layer_name: str = "layer0",
    tile_attributes: Any = None,
    attributes: Sequence[str] | None = None,
) -> bytes:
    """Generate one MVT tile directly from an indexed Starlet dataset.

    Uses the optional Rust extension (``starlet_core``) when it is installed
    and the dataset is EPSG:4326, and falls back to the pure-Python pipeline
    otherwise — see :mod:`starlet._internal.mvt.rust_engine`.
    """
    feature_capacity = int(
        feature_capacity if feature_capacity is not None else config_value("mvt", "feature_capacity")
    )
    extent = int(extent if extent is not None else config_value("mvt", "extent"))
    buffer = int(buffer if buffer is not None else config_value("mvt", "buffer"))
    tile_attributes = _restrict_tile_attributes(
        normalize_tile_attributes(
            tile_attributes if tile_attributes is not None else config_value("mvt", "tile_attributes")
        ),
        _normalize_attribute_whitelist(attributes),
    )
    if layer_name == "layer0":
        from starlet._internal.mvt import rust_engine

        if rust_engine.available() and _rust_supports_dataset(dataset_path):
            try:
                z, x, y = tile_id
                return rust_engine.generate_tile(
                    dataset_path, z, x, y,
                    feature_capacity=feature_capacity, extent=extent, buffer=buffer,
                    tile_attributes=tile_attributes,
                    simplify_tolerance=_tolerance_arg(config_value("mvt", "simplify_tolerance"), extent),
                )
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "starlet_core failed for %s %s; falling back to Python", dataset_path, tile_id, exc_info=True
                )
    return _generate_single_mvt_tile_python(
        dataset_path, tile_id,
        feature_capacity=feature_capacity, extent=extent, buffer=buffer, layer_name=layer_name,
        tile_attributes=tile_attributes,
    )


def _restrict_tile_attributes(
    tile_attributes: list[str] | None,
    attributes: frozenset[str] | None,
) -> list[str] | None:
    """Narrow the dataset's tile-attribute policy to a per-request whitelist."""
    if attributes is None:
        return tile_attributes
    if tile_attributes is None:
        return sorted(attributes)
    return [column for column in tile_attributes if column in attributes]


def _tolerance_arg(value: Any, extent: int) -> float | None:
    """Config ``mvt.simplify_tolerance`` -> the Rust engine's argument (None = auto)."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "auto")):
        return None
    return float(value)


def _generate_single_mvt_tile_python(
    dataset_path: str,
    tile_id: tuple[int, int, int],
    *,
    feature_capacity: int | None = None,
    extent: int | None = None,
    buffer: int | None = None,
    layer_name: str = "layer0",
    tile_attributes: Any = None,
) -> bytes:
    """Pure-Python single-tile generation.

    EPSG:4326 datasets take the vectorised path (:mod:`fast_tile`, numpy +
    shapely arrays, byte-compatible with the Rust core); other CRSs fall back
    to the per-feature ``IntermediateVectorTile`` implementation below.
    """
    feature_capacity = int(
        feature_capacity if feature_capacity is not None else config_value("mvt", "feature_capacity")
    )
    extent = int(extent if extent is not None else config_value("mvt", "extent"))
    buffer = int(buffer if buffer is not None else config_value("mvt", "buffer"))
    dataset_dir = Path(dataset_path)
    parquet_dir = dataset_dir / "parquet_tiles"
    if not parquet_dir.is_dir():
        raise FileNotFoundError(f"GeoParquet tile directory not found: {parquet_dir}")
    if _rust_supports_dataset(dataset_path):
        from starlet._internal.mvt import fast_tile

        return fast_tile.generate_tile(
            dataset_path, tile_id,
            feature_capacity=feature_capacity, extent=extent, buffer=buffer,
            tile_attributes=tile_attributes if tile_attributes is not None else config_value("mvt", "tile_attributes"),
            simplify_tolerance=config_value("mvt", "simplify_tolerance"),
            layer_name=layer_name, index=_single_tile_parquet_index(parquet_dir),
        )

    z, x, y = tile_id
    tile_bounds = mercator_tile_bounds(int(z), int(x), int(y))
    query_bounds = _expand_tile_bounds_for_buffer(tile_bounds, extent, buffer)
    index = _single_tile_parquet_index(parquet_dir)
    query_bounds_4326 = index._transform_bbox(query_bounds, WEB_MERCATOR_CRS, WGS84_CRS)
    tile_attributes = normalize_tile_attributes(
        tile_attributes if tile_attributes is not None else config_value("mvt", "tile_attributes")
    )
    attribute_set = frozenset(tile_attributes) if tile_attributes is not None else None

    tile = IntermediateVectorTile(
        int(z),
        int(x),
        int(y),
        feature_capacity=feature_capacity,
        extent=extent,
        buffer=buffer,
        tile_attributes=tile_attributes,
        simplify_tolerance=config_value("mvt", "simplify_tolerance"),
    )

    sampled_features = _sample_single_tile_records(
        index,
        query_bounds_4326,
        feature_capacity,
        tile,
        attributes=attribute_set,
    )
    if sampled_features is None:
        sampled_features = _sample_single_tile_records_legacy(
            index,
            query_bounds_4326,
            feature_capacity,
            tile,
            attributes=attribute_set,
        )

    for geom, attrs, priority in sampled_features:
        tile.add_feature(geom, attrs, priority=priority)

    return tile.encode(layer_name=layer_name)


def _offer(heap, cells, placement, feature_capacity, hash_priority, seq, payload):
    """Shared pre-selection step (mirrors ``IntermediateVectorTile``): the
    top-k larger-than-a-pixel rows by (size, hash) go to ``heap``; every
    other row is a dot candidate for the pixel cell of its bbox centre.
    Heap entries are ``(priority, seq, cell, *payload)``."""
    cell, small, size16 = placement
    priority = IntermediateVectorTile.combine_priority(size16, hash_priority)

    def offer_cell(c, entry):
        current = cells.get(c)
        if current is not None and entry[0] <= current[0]:
            return False
        cells[c] = entry
        return True

    entry = (priority, seq, cell) + payload
    if small:
        return offer_cell(cell, entry)
    if len(heap) < feature_capacity:
        heapq.heappush(heap, entry)
        return True
    if priority <= heap[0][0]:
        return offer_cell(cell, entry)
    evicted = heapq.heapreplace(heap, entry)
    offer_cell(evicted[2], evicted)
    return True


def _sample_single_tile_records(
    index: ParquetIndex,
    query_bounds_4326: tuple[float, float, float, float],
    feature_capacity: int,
    tile: IntermediateVectorTile,
    *,
    attributes: frozenset[str] | None = None,
) -> list[tuple[Any, dict[str, Any], int]] | None:
    """Pre-select raw parquet rows before WKB parsing, with the tile's own
    selection rule: sub-pixel rows (judged from their ``_bbox_*`` columns)
    compete per pixel cell, larger rows in a top-k by priority.

    Rows are ranked by :func:`feature_priority` of their raw WKB bytes — the
    same geometry-intrinsic priority the batch pipeline uses — so adjacent
    on-demand tiles (and pre-generated tiles) make consistent keep/drop
    decisions for a geometry they share. Attribute dicts are only built for
    the winners, after sampling.

    Returns ``None`` when any candidate partition lacks row bbox columns;
    those legacy datasets need the older geometry-based path for correctness.
    """
    feature_capacity = max(1, int(feature_capacity))
    # Entries: (priority, seq, cell, wkb, crs, table, geom_col, row_idx).
    heap: list[tuple] = []
    cells: dict[tuple[int, int], tuple] = {}
    seq = 0

    for path in index.find_intersecting_files(query_bounds_4326):
        names, geom_col, has_bbox, crs = index._schema_info(path)
        if not has_bbox:
            return None

        bbox_native = index._transform_bbox(query_bounds_4326, WGS84_CRS, crs)
        table = _read_bbox_filtered_table(path, bbox_native, geom_col=geom_col, attributes=attributes)
        if table.num_rows == 0:
            continue
        bounds_merc = _mercator_row_bounds(table, crs)
        for row_idx, geometry_wkb in enumerate(table[geom_col].to_pylist()):
            if geometry_wkb is None:
                continue
            priority = feature_priority(geometry_wkb)
            if _offer(heap, cells, tile.place(bounds_merc[row_idx]), feature_capacity, priority, seq,
                      (geometry_wkb, crs, table, geom_col, row_idx)):
                seq += 1

    samples = [
        (geometry_wkb, _row_attrs(table, geom_col, row_idx), crs, priority)
        for (priority, _, _, geometry_wkb, crs, table, geom_col, row_idx) in list(heap) + list(cells.values())
    ]
    return _decode_sampled_features(samples)


def _mercator_row_bounds(table: pa.Table, crs: Any) -> list[tuple[float, float, float, float]]:
    """Per-row ``_bbox_*`` bounds reprojected to Web Mercator (vectorised)."""
    xmin = table["_bbox_xmin"].to_numpy(zero_copy_only=False).astype(float)
    ymin = table["_bbox_ymin"].to_numpy(zero_copy_only=False).astype(float)
    xmax = table["_bbox_xmax"].to_numpy(zero_copy_only=False).astype(float)
    ymax = table["_bbox_ymax"].to_numpy(zero_copy_only=False).astype(float)
    transformer = Transformer.from_crs(crs, WEB_MERCATOR_CRS, always_xy=True)
    x0, y0 = transformer.transform(xmin, ymin)
    x1, y1 = transformer.transform(xmax, ymax)
    return list(zip(np.minimum(x0, x1), np.minimum(y0, y1), np.maximum(x0, x1), np.maximum(y0, y1)))


def _row_attrs(table: pa.Table, geom_col: str, row_idx: int) -> dict[str, Any]:
    """Attribute dict for a single sampled row (winners only)."""
    attrs: dict[str, Any] = {}
    for column in table.column_names:
        if column == geom_col or column in _INTERNAL_ATTRIBUTE_COLUMNS:
            continue
        value = table[column][row_idx].as_py()
        if value is not None:
            attrs[column] = value
    return attrs


def _normalize_attribute_whitelist(attributes: Sequence[str] | None) -> frozenset[str] | None:
    if attributes is None:
        return None
    normalized = frozenset(str(attribute).strip() for attribute in attributes if str(attribute).strip())
    return normalized


def _sample_single_tile_records_legacy(
    index: ParquetIndex,
    query_bounds_4326: tuple[float, float, float, float],
    feature_capacity: int,
    tile: IntermediateVectorTile,
    *,
    attributes: frozenset[str] | None = None,
) -> list[tuple[Any, dict[str, Any], int]]:
    """Pre-select after exact legacy geometry filtering (no bbox columns),
    with the tile's selection rule (pixel cells + top-k).

    Priorities come from the WKB of the (already reprojected) geometries —
    still deterministic per geometry, so tiles over a legacy dataset stay
    mutually consistent. Attribute dicts are built only for winners.
    """
    feature_capacity = max(1, int(feature_capacity))
    # Entries: (priority, seq, cell, geom, col_arrays, row_idx).
    heap: list[tuple] = []
    cells: dict[tuple[int, int], tuple] = {}
    seq = 0

    for gdf in index.iter_query_batches(
        query_bounds_4326,
        target_crs=WEB_MERCATOR_CRS,
        include_feature_id=True,
        attributes=attributes,
    ):
        col_arrays = {
            column: gdf[column].to_numpy()
            for column in gdf.columns
            if column != "geometry" and column not in _INTERNAL_ATTRIBUTE_COLUMNS
        }
        for row_idx, geom in enumerate(gdf.geometry.values):
            if geom is None or geom.is_empty:
                continue
            priority = feature_priority(shapely.to_wkb(geom))
            if _offer(heap, cells, tile.place(geom.bounds), feature_capacity, priority, seq,
                      (geom, col_arrays, row_idx)):
                seq += 1

    return [
        (
            geom,
            _attrs_for_row(col_arrays, row_idx),
            priority,
        )
        for (priority, _, _, geom, col_arrays, row_idx) in list(heap) + list(cells.values())
    ]


def _read_bbox_filtered_table(
    path: Path,
    bbox_native: tuple[float, float, float, float],
    *,
    geom_col: str,
    attributes: frozenset[str] | None = None,
) -> pa.Table:
    minx, miny, maxx, maxy = bbox_native
    flt = (
        (pc.field("_bbox_xmax") >= minx)
        & (pc.field("_bbox_xmin") <= maxx)
        & (pc.field("_bbox_ymax") >= miny)
        & (pc.field("_bbox_ymin") <= maxy)
    )
    columns = None
    if attributes is not None:
        names = pq.ParquetFile(path).schema_arrow.names
        required = {geom_col, *BBOX_COLS}
        columns = [name for name in names if name in required or name in attributes]
    return pq.read_table(path, filters=flt, columns=columns)


def _decode_sampled_features(
    samples: Sequence[tuple[bytes, dict[str, Any], Any, int]],
) -> list[tuple[Any, dict[str, Any], int]]:
    decoded: list[tuple[Any, dict[str, Any], int]] = []
    by_crs: dict[str, list[tuple[bytes, dict[str, Any], Any, int]]] = defaultdict(list)
    for geometry_wkb, attrs, crs, priority in samples:
        by_crs[str(crs)].append((geometry_wkb, attrs, crs, priority))

    for group in by_crs.values():
        geometries = from_wkb([geometry_wkb for geometry_wkb, _, _, _ in group])
        geometries = shapely.make_valid(geometries)
        crs = group[0][2]
        geometries, _ = reproject_geometries(geometries, crs, WEB_MERCATOR_CRS)
        for geom, (_, attrs, _, priority) in zip(geometries, group):
            if geom is not None and not geom.is_empty:
                decoded.append((geom, attrs, priority))
    return decoded


def _expand_tile_bounds_for_buffer(
    bounds: tuple[float, float, float, float],
    extent: int,
    buffer: int,
) -> tuple[float, float, float, float]:
    if extent <= 0:
        raise ValueError("extent must be positive")
    if buffer <= 0:
        return bounds
    minx, miny, maxx, maxy = bounds
    buffer_ratio = float(buffer) / float(extent)
    dx = (maxx - minx) * buffer_ratio
    dy = (maxy - miny) * buffer_ratio
    return (
        max(WORLD_MINX, minx - dx),
        max(WORLD_MINY, miny - dy),
        min(WORLD_MAXX, maxx + dx),
        min(WORLD_MAXY, maxy + dy),
    )


def _single_tile_parquet_index(parquet_dir: Path) -> ParquetIndex:
    key = str(parquet_dir.resolve())
    index = _single_tile_index_cache.get(key)
    if index is not None:
        _single_tile_index_cache.move_to_end(key)
        return index

    index = ParquetIndex(parquet_dir)
    _single_tile_index_cache[key] = index
    _single_tile_index_cache.move_to_end(key)
    while len(_single_tile_index_cache) > _SINGLE_TILE_INDEX_CACHE_SIZE:
        _single_tile_index_cache.popitem(last=False)
    return index


def _iter_web_mercator_features(table: Any, geom_col: str) -> Iterable[tuple[Any, dict[str, Any], int]]:
    source_crs = geoparquet_crs(table.schema, geom_col) or WGS84_CRS
    raw_wkb = table[geom_col].to_numpy(zero_copy_only=False)
    geometries = from_wkb(raw_wkb)
    geometries = shapely.make_valid(geometries)
    geometries, _ = reproject_geometries(geometries, source_crs, WEB_MERCATOR_CRS)

    attr_columns = [
        column
        for column in table.column_names
        if column != geom_col and column not in _INTERNAL_ATTRIBUTE_COLUMNS
    ]
    attrs_by_column = {column: table[column].to_pylist() for column in attr_columns}

    for index, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        attrs = _attrs_for_row(attrs_by_column, index)
        # Priority from the *source* WKB bytes: identical for this feature in
        # every tile/zoom it touches (and in the on-demand serving sampler),
        # which is what makes sampling seam-consistent.
        yield geom, attrs, feature_priority(raw_wkb[index])


def _positive_bounds_tuple(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = map(float, bounds)
    if maxx <= minx:
        maxx = np.nextafter(minx, math.inf)
    if maxy <= miny:
        maxy = np.nextafter(miny, math.inf)
    return minx, miny, maxx, maxy


def _attrs_for_row(attrs_by_column: dict[str, Any], row_idx: int) -> dict[str, Any]:
    return {
        column: values[row_idx]
        for column, values in attrs_by_column.items()
        if values[row_idx] is not None
    }


def _group_splits(splits: Sequence[GeoParquetSplit], num_groups: int) -> list[list[GeoParquetSplit]]:
    if not splits:
        return []
    group_count = max(1, min(int(num_groups), len(splits)))
    groups: list[list[GeoParquetSplit]] = [[] for _ in range(group_count)]
    for index, split in enumerate(splits):
        groups[index % group_count].append(split)
    return groups


_MAP_FALLBACK_MAX_BYTES = 512 * 1024 * 1024


def _create_map_groups(source: GeoParquetSource, num_groups: int) -> list[list[_MapInput]]:
    splits = source.create_splits()
    if len(splits) >= max(1, int(num_groups)):
        return _group_splits(splits, num_groups)

    # The repartitioning fallback below materialises the whole dataset in the
    # driver process; only take it for small inputs. Large datasets with few
    # row groups simply run with fewer map workers.
    try:
        input_bytes = int(source.input_size_bytes())
    except Exception:
        input_bytes = None
    if input_bytes is not None and input_bytes > _MAP_FALLBACK_MAX_BYTES:
        logger.info(
            "DatasetMVTGenerator: %d row-group splits for %d workers but input "
            "is %d bytes (> %d); running with %d map workers instead of "
            "loading the dataset into memory to repartition",
            len(splits),
            max(1, int(num_groups)),
            input_bytes,
            _MAP_FALLBACK_MAX_BYTES,
            len(splits),
        )
        return _group_splits(splits, len(splits))

    tables = [table for split in splits for table in source.iter_tables(split)]
    if not tables:
        return []

    table = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
    logger.info(
        "DatasetMVTGenerator map fallback: %d row-group splits for %d workers; "
        "loaded %d rows into memory and repartitioned by row batches",
        len(splits),
        max(1, int(num_groups)),
        table.num_rows,
    )
    return _group_table_batches(table, num_groups)


def _group_table_batches(table: pa.Table, num_groups: int) -> list[list[_TableBatch]]:
    if table.num_rows == 0:
        return []
    group_count = max(1, min(int(num_groups), table.num_rows))
    batch_size = max(1, (table.num_rows + group_count - 1) // group_count)
    groups: list[list[_TableBatch]] = []
    for start in range(0, table.num_rows, batch_size):
        groups.append([_TableBatch(table.slice(start, min(batch_size, table.num_rows - start)))])
    return groups


def _chunk_reduce_inputs(
    reduce_inputs: Sequence[_ReduceTileInput],
    group_size: int = _REDUCE_GROUP_SIZE,
) -> list[list[_ReduceTileInput]]:
    if not reduce_inputs:
        return []
    chunk_size = max(1, int(group_size))
    return [
        list(reduce_inputs[start : start + chunk_size])
        for start in range(0, len(reduce_inputs), chunk_size)
    ]


def _bucket_tile_ids(tile_ids: Sequence[int], num_buckets: int) -> list[list[int]]:
    bucket_count = max(1, int(num_buckets))
    buckets: list[list[int]] = [[] for _ in range(bucket_count)]
    for tile_id in tile_ids:
        buckets[int(tile_id) % bucket_count].append(tile_id)
    return buckets


def _discover_tile_counts_by_zoom(outdir: Path) -> list[int]:
    if not outdir.exists():
        return []
    counts: dict[int, int] = {}
    for child in outdir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        zoom = int(child.name)
        counts[zoom] = len(list(child.rglob("*.mvt")))
    if not counts:
        return []
    return [counts.get(z, 0) for z in range(max(counts) + 1)]
