"""Two-stage dataset-to-MVT generator using intermediate vector tiles."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import heapq
import logging
import math
import multiprocessing
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely
from shapely import from_wkb

from starlet._internal.histogram.loader import HistogramLoader
from starlet._internal.config import config_value, resolve_temp_dir
from starlet._internal.mvt.helpers import (
    WORLD_MAXX,
    WORLD_MAXY,
    WORLD_MINX,
    WORLD_MINY,
    mercator_tile_bounds,
)
from starlet._internal.mvt.intermediate_tile import IntermediateVectorTile, feature_priority
from starlet._internal.mvt.pyramid_partitioner import PyramidPartitioner
from starlet._internal.pmtiles.paths import default_pmtiles_path
from starlet._internal.pmtiles.exporter import export_to_pmtiles
from starlet._internal.server.tiler.parquet_index import INTERNAL_COLS, ParquetIndex
from starlet._internal.tiling.crs import WEB_MERCATOR_CRS, WGS84_CRS, geoparquet_crs, reproject_geometries
from starlet._internal.tiling.geoparquet_source import GeoParquetSource, GeoParquetSplit

logger = logging.getLogger(__name__)

_INTERNAL_ATTRIBUTE_COLUMNS = {
    "_tile_id",
    "_bbox_xmin",
    "_bbox_ymin",
    "_bbox_xmax",
    "_bbox_ymax",
}
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
    tile_ids: list[int]


@dataclass(frozen=True)
class _ReduceTileInput:
    tile_id: int
    intermediate_dirs: tuple[str, ...]


@dataclass(frozen=True)
class _TableBatch:
    table: pa.Table


_MapInput = GeoParquetSplit | _TableBatch


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
        extent: int | None = None,
        buffer: int | None = None,
        geom_col: str = "geometry",
        seed: int = 42,
        temp_dir: str | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.parquet_dir = self.dataset_dir / "parquet_tiles"
        self.hist_path = self.dataset_dir / "histograms" / "global_prefix.npy"
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
        if not self.hist_path.exists():
            raise FileNotFoundError(f"Prefix histogram not found: {self.hist_path}")

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

    def _run_map_stage(
        self,
        map_groups: Sequence[Sequence[_MapInput]],
        source: GeoParquetSource,
        temp_root: Path,
    ) -> list[_MapStageResult]:
        logger.info(
            "DatasetMVTGenerator map stage: groups=%d workers=%d",
            len(map_groups),
            self.workers,
        )
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(
                    _map_split_group,
                    group,
                    source,
                    str(self.hist_path),
                    self.num_zoom_levels,
                    self.threshold,
                    self.partition_buffer,
                    self.feature_capacity,
                    self.extent,
                    self.buffer,
                    self.seed + index,
                    str(temp_root),
                    index,
                )
                for index, group in enumerate(map_groups)
            ]
            return [future.result() for future in as_completed(futures)]

    def _run_reduce_stage(self, map_results: list[_MapStageResult]) -> None:
        if not map_results:
            logger.info("DatasetMVTGenerator reduce stage: no intermediate tiles")
            return

        self.outdir.mkdir(parents=True, exist_ok=True)
        tile_locations: dict[int, list[str]] = defaultdict(list)
        for result in map_results:
            for tile_id in result.tile_ids:
                tile_locations[tile_id].append(result.intermediate_dir)

        reduce_inputs = [
            _ReduceTileInput(tile_id, tuple(intermediate_dirs))
            for tile_id, intermediate_dirs in sorted(tile_locations.items())
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
    extent: int,
    buffer: int,
    seed: int,
    temp_root: str,
    mapper_index: int,
) -> _MapStageResult:
    prefix = HistogramLoader(hist_path).load()
    partitioner = PyramidPartitioner(
        (WORLD_MINX, WORLD_MINY, WORLD_MAXX, WORLD_MAXY),
        num_zoom_levels,
        prefix_histogram=prefix,
        size_threshold=threshold,
        buffer=partition_buffer,
    )
    tiles: dict[int, IntermediateVectorTile] = {}

    for table in _iter_map_input_tables(source, inputs):
        for geom, attrs, priority in _iter_web_mercator_features(table, source.geom_col):
            bounds = _positive_bounds_tuple(geom.bounds)
            tile_ids = partitioner.overlapping_tile_ids(bounds)
            if not tile_ids:
                continue
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
                    tiles[tile_id] = tile
                tile.add_feature(
                    geom,
                    attrs,
                    priority=priority,
                )
    intermediate_dir = Path(temp_root) / f"mapper-{mapper_index:06d}"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    tile_ids = []
    for tile_id, tile in tiles.items():
        if tile.feature_count == 0:
            continue
        z, x, y = PyramidPartitioner.decode_tile_id(tile_id)
        tile.write_features(intermediate_dir / _intermediate_tile_filename(z, x, y))
        tile_ids.append(tile_id)
    return _MapStageResult(str(intermediate_dir), tile_ids)


def _iter_map_input_tables(
    source: GeoParquetSource,
    inputs: Sequence[_MapInput],
) -> Iterable[pa.Table]:
    for item in inputs:
        if isinstance(item, _TableBatch):
            yield item.table
        else:
            yield from source.iter_tables(item)


def _reduce_tile_group(
    reduce_inputs: Sequence[_ReduceTileInput],
    outdir: str,
    feature_capacity: int,
    extent: int,
    buffer: int,
) -> None:
    out_path = Path(outdir)
    for reduce_input in reduce_inputs:
        tile_id = reduce_input.tile_id
        z, x, y = PyramidPartitioner.decode_tile_id(tile_id)
        filename = _intermediate_tile_filename(z, x, y)
        merged = IntermediateVectorTile(
            z,
            x,
            y,
            feature_capacity=feature_capacity,
            extent=extent,
            buffer=buffer,
            rng=random.Random(tile_id),
        )
        first_tile = True
        for intermediate_dir in reduce_input.intermediate_dirs:
            path = Path(intermediate_dir) / filename
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


def _intermediate_tile_filename(z: int, x: int, y: int) -> str:
    return f"{z}-{x}-{y}.pyarrow"


def generate_single_mvt_tile(
    dataset_path: str,
    tile_id: tuple[int, int, int],
    *,
    feature_capacity: int | None = None,
    extent: int | None = None,
    buffer: int | None = None,
    layer_name: str = "layer0",
    triage: bool = False,
) -> bytes:
    """Generate one MVT tile directly from an indexed Starlet dataset.

    When ``triage=True``, applies numeric quantization and string prefix triage
    (HiFIVE §6.2) before encoding. This reduces tile size without dropping any
    features. Defaults to ``False`` so existing callers are unaffected.
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

    z, x, y = tile_id
    tile_bounds = mercator_tile_bounds(int(z), int(x), int(y))
    query_bounds = _expand_tile_bounds_for_buffer(tile_bounds, extent, buffer)
    index = _single_tile_parquet_index(parquet_dir)
    query_bounds_4326 = index._transform_bbox(query_bounds, WEB_MERCATOR_CRS, WGS84_CRS)

    sampled_features = _sample_single_tile_records(
        index,
        query_bounds_4326,
        feature_capacity,
    )
    if sampled_features is None:
        sampled_features = _sample_single_tile_records_legacy(
            index,
            query_bounds_4326,
            feature_capacity,
        )

    tile = IntermediateVectorTile(
        int(z),
        int(x),
        int(y),
        feature_capacity=feature_capacity,
        extent=extent,
        buffer=buffer,
    )

    for geom, attrs, priority in sampled_features:
        tile.add_feature(geom, attrs, priority=priority)

    if triage:
        tile.triage()

    return tile.encode(layer_name=layer_name)


def _sample_single_tile_records(
    index: ParquetIndex,
    query_bounds_4326: tuple[float, float, float, float],
    feature_capacity: int,
) -> list[tuple[Any, dict[str, Any], int]] | None:
    """Top-k sample raw parquet rows by priority before WKB parsing.

    Rows are ranked by :func:`feature_priority` of their raw WKB bytes — the
    same geometry-intrinsic priority the batch pipeline uses — so adjacent
    on-demand tiles (and pre-generated tiles) make consistent keep/drop
    decisions for a geometry they share. Attribute dicts are only built for
    the winners, after sampling.

    Returns ``None`` when any candidate partition lacks row bbox columns;
    those legacy datasets need the older geometry-based path for correctness.
    """
    feature_capacity = max(1, int(feature_capacity))
    # Min-heap of (priority, seq, wkb, crs, table, geom_col, row_idx).
    heap: list[tuple[int, int, bytes, Any, pa.Table, str, int]] = []
    seq = 0

    for path in index.find_intersecting_files(query_bounds_4326):
        names, geom_col, has_bbox, crs = index._schema_info(path)
        if not has_bbox:
            return None

        bbox_native = index._transform_bbox(query_bounds_4326, WGS84_CRS, crs)
        table = _read_bbox_filtered_table(path, bbox_native)
        if table.num_rows == 0:
            continue
        for row_idx, geometry_wkb in enumerate(table[geom_col].to_pylist()):
            if geometry_wkb is None:
                continue
            priority = feature_priority(geometry_wkb)
            if len(heap) >= feature_capacity and priority <= heap[0][0]:
                continue
            entry = (priority, seq, geometry_wkb, crs, table, geom_col, row_idx)
            seq += 1
            if len(heap) < feature_capacity:
                heapq.heappush(heap, entry)
            else:
                heapq.heapreplace(heap, entry)

    samples = [
        (geometry_wkb, _row_attrs(table, geom_col, row_idx), crs, priority)
        for (priority, _, geometry_wkb, crs, table, geom_col, row_idx) in heap
    ]
    return _decode_sampled_features(samples)


def _row_attrs(table: pa.Table, geom_col: str, row_idx: int) -> dict[str, Any]:
    """Attribute dict for a single sampled row (winners only)."""
    attrs: dict[str, Any] = {}
    for column in table.column_names:
        if column == geom_col or column in INTERNAL_COLS:
            continue
        value = table[column][row_idx].as_py()
        if value is not None:
            attrs[column] = _property_value(value)
    return attrs


def _sample_single_tile_records_legacy(
    index: ParquetIndex,
    query_bounds_4326: tuple[float, float, float, float],
    feature_capacity: int,
) -> list[tuple[Any, dict[str, Any], int]]:
    """Top-k sample after exact legacy geometry filtering (no bbox columns).

    Priorities come from the WKB of the (already reprojected) geometries —
    still deterministic per geometry, so tiles over a legacy dataset stay
    mutually consistent. Attribute dicts are built only for winners.
    """
    feature_capacity = max(1, int(feature_capacity))
    # Min-heap of (priority, seq, geom, col_arrays, row_idx).
    heap: list[tuple[int, int, Any, dict[str, Any], int]] = []
    seq = 0

    for gdf in index.iter_query_batches(query_bounds_4326, target_crs=WEB_MERCATOR_CRS):
        col_arrays = {
            column: gdf[column].to_numpy()
            for column in gdf.columns
            if column != "geometry" and column not in INTERNAL_COLS
        }
        for row_idx, geom in enumerate(gdf.geometry.values):
            if geom is None or geom.is_empty:
                continue
            priority = feature_priority(shapely.to_wkb(geom))
            if len(heap) >= feature_capacity and priority <= heap[0][0]:
                continue
            entry = (priority, seq, geom, col_arrays, row_idx)
            seq += 1
            if len(heap) < feature_capacity:
                heapq.heappush(heap, entry)
            else:
                heapq.heapreplace(heap, entry)

    return [
        (
            geom,
            {
                column: _property_value(values[row_idx])
                for column, values in col_arrays.items()
                if values[row_idx] is not None
            },
            priority,
        )
        for (priority, _, geom, col_arrays, row_idx) in heap
    ]


def _read_bbox_filtered_table(path: Path, bbox_native: tuple[float, float, float, float]) -> pa.Table:
    minx, miny, maxx, maxy = bbox_native
    flt = (
        (pc.field("_bbox_xmax") >= minx)
        & (pc.field("_bbox_xmin") <= maxx)
        & (pc.field("_bbox_ymax") >= miny)
        & (pc.field("_bbox_ymin") <= maxy)
    )
    return pq.read_table(path, filters=flt)


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
        attrs = {
            column: _property_value(values[index])
            for column, values in attrs_by_column.items()
            if values[index] is not None
        }
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


def _property_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


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
