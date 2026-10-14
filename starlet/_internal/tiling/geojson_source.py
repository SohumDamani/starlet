from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass, replace
from decimal import Decimal
import bz2
import json
import logging
import math
import multiprocessing
from numbers import Number
import os
from pathlib import Path
import queue
import random
import re
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from starlet._internal._parallel import pool_context
from starlet._internal.executor import create_process_executor
from starlet._internal.tiling.datasource import (
    DataSource,
    SpatialSample,
    _GEOJSON_SUFFIXES,
    _attach_geoparquet_metadata,
    _combine_spatial_samples,
    _iter_bz2_decompressed_blocks,
    _normalize_decimal_columns,
    _properties_dataframe_to_arrow_table,
    _unify_tabular_schemas,
    _spatial_sample_from_state,
    _split_sample_cap,
    _source_files,
)
from starlet._internal.tiling.partition_reader import GeoJSONPartitionReader, OPEN_BRACE

logger = logging.getLogger(__name__)

_GEOMETRY_MEMBER = re.compile(rb'"geometry"\s*:\s*')
_PROPERTIES_MEMBER = re.compile(rb'"properties"\s*:\s*')
_COORDINATES_MEMBER = re.compile(rb'"coordinates"\s*:\s*\[')
_JSON_NUMBER = re.compile(
    rb'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?'
)


def _describe_geojson_split(path: str, offset: int, length: int) -> str:
    return f"path={path!r}, offset={offset}, length={length}"


def is_geojson_path(path: str) -> bool:
    return path.lower().endswith(_GEOJSON_SUFFIXES)


@dataclass(frozen=True)
class GeoJSONSplit:
    """Byte range to read from one GeoJSON source."""

    path: str
    offset: int
    length: int


class GeoJSONSource(DataSource):
    """
    Streams GeoJSON / GeoJSONL as Arrow Tables, converting geometry to WKB.

    - For standard FeatureCollection GeoJSON, reads byte partitions in parallel.
    - For GeoJSON Lines (one Feature per line), reads and batches by line.
    - Geometry dicts → shapely.shape → WKB bytes (binary Arrow column 'geometry').
    - Attaches minimal GeoParquet metadata (version, primary_column, encoding, crs hint).
    """

    def __init__(
        self,
        path: str,
        batch_rows: int = 1_000,
        src_crs: str = "EPSG:4326",
        keep_null_geoms: bool = False,
    ):
        self.path = str(path)
        self._files = _source_files(self.path, _GEOJSON_SUFFIXES)
        if not self._files:
            raise ValueError(f"No GeoJSON files found in {self.path}")
        self.batch_rows = int(batch_rows)
        self.src_crs = src_crs
        self.keep_null_geoms = keep_null_geoms

        self._schema: Optional[pa.Schema] = None
        self._crs_hint: Optional[str] = _extract_feature_collection_crs_hint(
            _read_geojson_header(self._files[0])
        )

        logger.info(
            "GeoJSONSource opened %s with %d files (batch_rows=%d, src_crs=%s)",
            path, len(self._files), self.batch_rows, self._crs_hint or self.src_crs
        )

    # ---------------- schema ---------------- #
    def infer_schema(self) -> pa.Schema:
        property_types: dict[str, str] = {}
        property_order: list[str] = []
        for split in self.create_splits():
            for features in self._iter_feature_batches_for_split(split):
                _update_geojson_property_types(
                    property_types,
                    property_order,
                    (feature.get("properties") or {} for feature in features),
                )

        properties_schema = _geojson_property_schema(property_types, property_order)
        base = properties_schema.append(pa.field("geometry", pa.binary()))
        return _attach_geoparquet_metadata(base, self._crs_hint or self.src_crs)

    def schema(self) -> pa.Schema:
        assert self._schema is not None, "No schema to return"
        return self._schema

    def set_schema(self, schema: pa.Schema) -> None:
        """Use a schema discovered by an earlier scan of this source."""
        if "geometry" not in schema.names:
            raise ValueError("GeoJSON schema must contain a geometry column")
        self._schema = schema

    def input_size_bytes(self) -> int:
        return sum(file_path.stat().st_size for file_path in self._files)

    # ---------------- iterator ---------------- #
    def create_splits(self, num_splits: Optional[int] = None) -> List[GeoJSONSplit]:
        splits: List[GeoJSONSplit] = []
        for file_path in self._files:
            file_size = file_path.stat().st_size
            if num_splits is None:
                target_partition_size = 32 * 1024 * 1024
                split_count = max(1, (file_size + target_partition_size - 1) // target_partition_size)
            else:
                split_count = max(1, int(num_splits))
            splits.extend(
                GeoJSONSplit(path=str(file_path), offset=offset, length=length)
                for offset, length in _geojson_partition_ranges(file_size, split_count)
            )
        return splits

    def iter_tables(self, *splits: GeoJSONSplit) -> Iterable[pa.Table]:
        batch_index = 0
        crs_value = self._crs_hint or self.src_crs
        schema = self.schema()
        properties_schema = pa.schema(
            [field for field in schema if field.name != "geometry"]
        )

        import geopandas as gpd
        for split in splits:
            for features in self._iter_feature_batches_for_split(split):
                if not features:
                    continue

                gdf = gpd.GeoDataFrame.from_features(features, crs=crs_value)
                geometry_col = pa.array(gdf.geometry.to_wkb(), type=pa.binary())

                props_df = gdf.drop(columns="geometry")
                props_df = _normalize_decimal_columns(props_df)
                props_table = _properties_dataframe_to_arrow_table(
                    props_df,
                    schema=properties_schema,
                )

                table = (
                    pa.table([geometry_col], names=["geometry"])
                    if props_table.num_columns == 0
                    else props_table.append_column("geometry", geometry_col)
                )

                table = table.cast(schema)
                table = table.combine_chunks()

                logger.debug(
                    "GeoJSON batch %d (%d rows) -> %d columns (including 'geometry')",
                    batch_index,
                    table.num_rows,
                    len(table.column_names),
                )
                batch_index += 1
                yield table

    def _iter_feature_batches_for_split(
        self,
        split: Optional[GeoJSONSplit],
    ) -> Iterable[List[Dict[str, Any]]]:
        assert split is not None, "No split to iterate"

        for feature_batch in _iter_feature_json_batches(
            split.path,
            split.offset,
            split.length,
            batch_size=self.batch_rows,
        ):
            yield [json.loads(feature) for feature in feature_batch]

    # ---------------- internal helpers ---------------- #
    @classmethod
    def read_spatial_sample(
        cls,
        path: str,
        *,
        sample_cap: Optional[int],
        seed: int,
        workers: Optional[int],
        src_crs: str = "EPSG:4326",
    ) -> SpatialSample:
        source = cls(path, src_crs=src_crs)
        splits = source.create_splits()
        sample_caps = _split_sample_cap(sample_cap, len(splits))

        logger.info(
            "Reading GeoJSON spatial sample from %s in %d partitions with %s workers",
            path,
            len(splits),
            workers or "auto",
        )

        jobs = [
            (
                idx,
                split.path,
                split.offset,
                split.length,
                sample_caps[idx],
                seed + idx,
            )
            for idx, split in enumerate(splits)
        ]

        total = len(jobs)
        completed = 0

        def report_progress() -> None:
            nonlocal completed
            completed += 1
            percent = 100 * completed // total
            print(f"\r{percent}% ({completed}/{total})", end="", flush=True, file=sys.stderr)

        if workers == 1:
            parts = []
            for job in jobs:
                parts.append(_read_geojson_partition_spatial_sample(*job))
                report_progress()
        else:
            with create_process_executor(
                max_workers=workers,
                logger=logger,
                context="GeoJSON spatial sampling",
            ) as ex:
                futures = [
                    ex.submit(_read_geojson_partition_spatial_sample, *job)
                    for job in jobs
                ]
                parts = []
                for future in as_completed(futures):
                    parts.append(future.result())
                    report_progress()

        if total:
            print(file=sys.stderr)

        sample = _combine_spatial_samples(parts)
        properties_schema = _unify_tabular_schemas(
            part.schema for part in parts if part.schema is not None
        )
        schema = _attach_geoparquet_metadata(
            properties_schema.append(pa.field("geometry", pa.binary())),
            source._crs_hint or source.src_crs,
        )
        return replace(sample, schema=schema)


def _read_geojson_spatial_sample(
    path: str,
    *,
    sample_cap: Optional[int],
    seed: int,
    geojson_workers: Optional[int],
    src_crs: str = "EPSG:4326",
) -> SpatialSample:
    return GeoJSONSource.read_spatial_sample(
        path,
        sample_cap=sample_cap,
        seed=seed,
        workers=geojson_workers,
        src_crs=src_crs,
    )


def convert_geojson_to_geoparquet(
    path: str,
    output_dir: str,
    *,
    workers: Optional[int] = None,
    schema: Optional[pa.Schema] = None,
    src_crs: str = "EPSG:4326",
    row_group_rows: int = 128 * 1024,
    compression: str = "zstd",
) -> List[str]:
    """
    Convert a GeoJSON file (or directory of files) to GeoParquet using all
    available processors.

    One long-lived worker process is started per processor. Workers pull
    splits from a shared queue (so fast workers take more splits) and each
    worker appends everything it converts to its own output file, so the
    result is at most `workers` files named part-NNNNN.parquet in
    `output_dir`, regardless of how many splits the input has.

    All output files share one schema. If `schema` is not given it is
    discovered with a parallel properties-only pre-scan (the same one
    read_spatial_sample does), which is much cheaper than the conversion.

    Returns the paths of the files written.
    """
    source = GeoJSONSource(path, src_crs=src_crs)
    splits = source.create_splits()
    if workers is None:
        workers = _available_processors()
    workers = max(1, min(int(workers), len(splits)))

    if schema is None:
        logger.info("Discovering GeoJSON schema for %s", path)
        schema = GeoJSONSource.read_spatial_sample(
            path, sample_cap=1, seed=0, workers=workers, src_crs=src_crs,
        ).schema

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_paths = [str(output / f"part-{index:05d}.parquet") for index in range(workers)]
    writer_args = (path, src_crs, schema, row_group_rows, compression)

    logger.info(
        "Converting %s to GeoParquet: %d splits over %d worker processes -> %s",
        path, len(splits), workers, output,
    )

    total = len(splits)
    completed = 0

    def report_progress() -> None:
        nonlocal completed
        completed += 1
        percent = 100 * completed // total
        print(f"\r{percent}% ({completed}/{total})", end="", flush=True, file=sys.stderr)

    if workers == 1:
        written = _convert_geojson_splits_to_parquet(
            *writer_args, output_paths[0], iter(splits), lambda _rows: report_progress(),
        )
        if total:
            print(file=sys.stderr)
        return [output_paths[0]] if written else []

    ctx = pool_context() or multiprocessing.get_context()
    split_queue = ctx.Queue()
    event_queue = ctx.Queue()
    for split in splits:
        split_queue.put(split)
    for _ in range(workers):
        split_queue.put(None)

    processes = [
        ctx.Process(
            target=_geojson_to_geoparquet_worker,
            args=(index, *writer_args, output_paths[index], split_queue, event_queue),
            daemon=True,
        )
        for index in range(workers)
    ]
    for process in processes:
        process.start()

    written_paths: List[str] = []
    finished: set[int] = set()
    try:
        while len(finished) < workers:
            try:
                kind, index, payload = event_queue.get(timeout=1.0)
            except queue.Empty:
                for index, process in enumerate(processes):
                    if index not in finished and process.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"GeoParquet conversion worker {index} died "
                            f"with exit code {process.exitcode}"
                        )
                continue

            if kind == "split":
                report_progress()
            elif kind == "done":
                finished.add(index)
                if payload:
                    written_paths.append(output_paths[index])
            else:
                raise RuntimeError(
                    f"GeoParquet conversion worker {index} failed:\n{payload}"
                )
    finally:
        if total:
            print(file=sys.stderr)
        for process in processes:
            if process.is_alive() and len(finished) < workers:
                process.terminate()
            process.join()

    return sorted(written_paths)


def _available_processors() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _geojson_to_geoparquet_worker(
    worker_index: int,
    path: str,
    src_crs: str,
    schema: pa.Schema,
    row_group_rows: int,
    compression: str,
    output_path: str,
    split_queue,
    event_queue,
) -> None:
    """Process entry point: convert splits from `split_queue` until a None sentinel."""
    try:
        splits = iter(split_queue.get, None)
        written = _convert_geojson_splits_to_parquet(
            path,
            src_crs,
            schema,
            row_group_rows,
            compression,
            output_path,
            splits,
            lambda rows: event_queue.put(("split", worker_index, rows)),
        )
        event_queue.put(("done", worker_index, written))
    except BaseException:
        event_queue.put(("error", worker_index, traceback.format_exc()))


def _convert_geojson_splits_to_parquet(
    path: str,
    src_crs: str,
    schema: pa.Schema,
    row_group_rows: int,
    compression: str,
    output_path: str,
    splits: Iterable[GeoJSONSplit],
    on_split_done,
) -> bool:
    """
    Append every split in `splits` to one Parquet file at `output_path`.

    Tables are buffered up to `row_group_rows` so row groups are not as small
    as GeoJSONSource's per-batch tables. The file is only created once there
    is something to write; returns whether it was.
    """
    import pyarrow.parquet as pq

    source = GeoJSONSource(path, src_crs=src_crs)
    source.set_schema(schema)

    writer = None
    pending: List[pa.Table] = []
    pending_rows = 0

    def flush() -> None:
        nonlocal writer, pending, pending_rows
        if not pending:
            return
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema, compression=compression)
        writer.write_table(pa.concat_tables(pending), row_group_size=row_group_rows)
        pending = []
        pending_rows = 0

    try:
        for split in splits:
            split_rows = 0
            for table in source.iter_tables(split):
                pending.append(table)
                pending_rows += table.num_rows
                split_rows += table.num_rows
                if pending_rows >= row_group_rows:
                    flush()
            on_split_done(split_rows)
        flush()
    finally:
        if writer is not None:
            writer.close()
    return writer is not None


def _iter_geojson_geometry_xy(geometry):
    stack = [geometry]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            if v.get("type") == "GeometryCollection":
                stack.extend(reversed(v.get("geometries") or []))
            else:
                coordinates = v.get("coordinates")
                if coordinates is not None:
                    stack.append(coordinates)
        elif isinstance(v, list):
            if len(v) >= 2 and isinstance(v[0], Number) and isinstance(v[1], Number):
                yield float(v[0]), float(v[1])
            else:
                stack.extend(reversed(v))


def iter_geojson_xy(feature_json):
    feature = json.loads(feature_json)
    yield from _iter_geojson_geometry_xy(feature.get("geometry"))


_PROPERTY_NULL = "null"
_PROPERTY_BOOL = "bool"
_PROPERTY_INT = "int"
_PROPERTY_FLOAT = "float"
_PROPERTY_STRING = "string"
_PROPERTY_MAP = "map"
_PROPERTY_LARGE_STRING = "large_string"


def _update_geojson_property_types(
    property_types: dict[str, str],
    property_order: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    for row in rows:
        _update_geojson_property_row(property_types, property_order, row)


def _update_geojson_property_row(
    property_types: dict[str, str],
    property_order: list[str],
    row: dict[str, Any],
) -> None:
    for name, value in row.items():
        field_name = str(name)
        value_type = _geojson_property_value_type(value)
        current_type = property_types.get(field_name)
        if current_type is None:
            property_order.append(field_name)
            property_types[field_name] = value_type
        else:
            property_types[field_name] = _merge_geojson_property_type(
                current_type,
                value_type,
            )


def _geojson_property_value_type(value: Any) -> str:
    if value is None:
        return _PROPERTY_NULL
    if isinstance(value, bool):
        return _PROPERTY_BOOL
    if isinstance(value, int):
        return _PROPERTY_INT
    if isinstance(value, (float, Decimal)):
        return _PROPERTY_FLOAT
    if isinstance(value, str):
        return _PROPERTY_STRING
    if _is_geojson_string_map(value):
        return _PROPERTY_MAP
    return _PROPERTY_LARGE_STRING


def _is_geojson_string_map(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and (item is None or isinstance(item, str))
        for key, item in value.items()
    )


def _merge_geojson_property_type(left: str, right: str) -> str:
    if left == right:
        return left
    if left == _PROPERTY_NULL:
        return right
    if right == _PROPERTY_NULL:
        return left
    if {left, right} <= {_PROPERTY_INT, _PROPERTY_FLOAT}:
        return _PROPERTY_FLOAT
    if _PROPERTY_LARGE_STRING in {left, right}:
        return _PROPERTY_LARGE_STRING
    if _PROPERTY_STRING in {left, right}:
        return _PROPERTY_LARGE_STRING
    return _PROPERTY_LARGE_STRING


def _geojson_property_schema(
    property_types: dict[str, str],
    property_order: list[str],
) -> pa.Schema:
    return pa.schema(
        pa.field(name, _geojson_property_arrow_type(property_types[name]), nullable=True)
        for name in property_order
    )


def _geojson_property_arrow_type(property_type: str) -> pa.DataType:
    if property_type == _PROPERTY_BOOL:
        return pa.bool_()
    if property_type == _PROPERTY_INT:
        return pa.int64()
    if property_type == _PROPERTY_FLOAT:
        return pa.float64()
    if property_type == _PROPERTY_STRING:
        return pa.string()
    if property_type == _PROPERTY_MAP:
        return pa.map_(pa.string(), pa.string())
    if property_type == _PROPERTY_LARGE_STRING:
        return pa.large_string()
    return pa.null()


_TYPE_WIDENING_CHAIN = (_PROPERTY_INT, _PROPERTY_FLOAT, _PROPERTY_STRING, _PROPERTY_LARGE_STRING)
_TYPE_WIDENING_RANK = {property_type: rank for rank, property_type in enumerate(_TYPE_WIDENING_CHAIN)}


def _upgrade_sampled_property_type(
    name: str,
    current: str,
    incoming: str,
    *,
    context: str,
) -> str:
    """
    Upgrade a sampled property's inferred type to accommodate a new observed
    value. Compatible widenings (int -> float -> string -> large_string) are
    applied silently. Anything else (e.g. bool vs int, or a scalar vs a map)
    is incompatible: log a warning and keep the first type seen.
    """
    if current == incoming:
        return current
    if current == _PROPERTY_NULL:
        return incoming
    if incoming == _PROPERTY_NULL:
        return current
    if current in _TYPE_WIDENING_RANK and incoming in _TYPE_WIDENING_RANK:
        return current if _TYPE_WIDENING_RANK[current] >= _TYPE_WIDENING_RANK[incoming] else incoming

    logger.warning(
        "GeoJSON property %r has incompatible types %r and %r (%s); keeping %r",
        name, current, incoming, context, current,
    )
    return current


def _update_sampled_property_types(
    property_types: dict[str, str],
    property_display_names: dict[str, str],
    property_order: list[str],
    properties: dict[str, Any],
    *,
    context: str,
) -> None:
    """Like `_update_geojson_property_row`, but case-insensitive by attribute name."""
    for name, value in properties.items():
        key = str(name).casefold()
        value_type = _geojson_property_value_type(value)
        current_type = property_types.get(key)
        if current_type is None:
            property_order.append(key)
            property_display_names[key] = str(name)
            property_types[key] = value_type
        else:
            property_types[key] = _upgrade_sampled_property_type(
                property_display_names[key],
                current_type,
                value_type,
                context=context,
            )


def _sampled_property_schema(
    property_types: dict[str, str],
    property_order: list[str],
    property_display_names: dict[str, str],
) -> pa.Schema:
    return pa.schema(
        pa.field(
            property_display_names[key],
            _geojson_property_arrow_type(property_types[key]),
            nullable=True,
        )
        for key in property_order
    )


def _extract_geojson_properties(feature_json: bytes) -> dict[str, Any]:
    """
    Parse only a Feature's "properties" object, skipping the rest of the
    Feature -- notably its geometry, which can be far larger (e.g. a complex
    polygon) than the properties it comes with.
    """
    match = _PROPERTIES_MEMBER.search(feature_json)
    if match is None:
        return {}

    start = match.end()
    if start >= len(feature_json) or feature_json[start] != OPEN_BRACE:
        return {}

    try:
        end = GeoJSONPartitionReader._find_json_object_end(feature_json, start)
        return json.loads(feature_json[start:end])
    except ValueError:
        return {}


def _read_geojson_partition_spatial_sample(
    split_index: int,
    path: str,
    offset: int,
    length: int,
    sample_cap: Optional[int],
    seed: int,
) -> SpatialSample:
    rng = random.Random(seed)
    x_sample: List[float] = []
    y_sample: List[float] = []
    n_seen = 0
    n_batches = 0
    property_types: dict[str, str] = {}
    property_display_names: dict[str, str] = {}
    property_order: list[str] = []
    context = _describe_geojson_split(path, offset, length)

    requested_cap = None if sample_cap is None else max(0, int(sample_cap))
    reservoir_cap = max(requested_cap, 1) if requested_cap is not None else 0
    reservoir: list[tuple[float, float]] = []
    reservoir_weight = 0.0
    reservoir_skip = 0

    try:
        for batch in _iter_feature_json_batches(
            path,
            offset,
            length,
            batch_size=1_024,
            decode=False,
        ):
            for feature_json in batch:
                _update_sampled_property_types(
                    property_types,
                    property_display_names,
                    property_order,
                    _extract_geojson_properties(feature_json),
                    context=context,
                )

                coordinate = _first_geojson_coordinate(feature_json)
                if coordinate is None:
                    continue

                x, y = coordinate
                n_seen += 1

                if requested_cap is None:
                    x_sample.append(x)
                    y_sample.append(y)
                elif len(reservoir) < reservoir_cap:
                    reservoir.append((x, y))
                    if len(reservoir) == reservoir_cap:
                        reservoir_weight = _initial_reservoir_weight(
                            rng,
                            reservoir_cap,
                        )
                        reservoir_skip = _next_reservoir_skip(
                            rng,
                            reservoir_weight,
                        )
                elif reservoir_skip:
                    reservoir_skip -= 1
                else:
                    slot = rng.randrange(reservoir_cap)
                    reservoir[slot] = (x, y)
                    reservoir_weight *= _initial_reservoir_weight(
                        rng,
                        reservoir_cap,
                    )
                    reservoir_skip = _next_reservoir_skip(
                        rng,
                        reservoir_weight,
                    )

            n_batches += 1
    except Exception as exc:
        raise ValueError(
            "GeoJSON spatial sampling failed for "
            f"split_index={split_index} ({context})"
        ) from exc

    if requested_cap is not None:
        selected = (
            reservoir
            if len(reservoir) <= requested_cap
            else rng.sample(reservoir, requested_cap)
        )
        x_sample = [entry[0] for entry in selected]
        y_sample = [entry[1] for entry in selected]

    return _spatial_sample_from_state(
        x_sample=x_sample,
        y_sample=y_sample,
        n_seen=n_seen,
        batches_read=n_batches,
        schema=_sampled_property_schema(property_types, property_order, property_display_names),
    )


def _first_geojson_coordinate(feature_json: bytes) -> tuple[float, float] | None:
    """
    Return one representative XY pair by locating "coordinates" and reading
    its first two numbers directly -- no json.loads, and no byte scan across
    the geometry beyond finding these two small matches.
    """
    geometry = _GEOMETRY_MEMBER.search(feature_json)
    if geometry is None:
        return None

    value_start = geometry.end()
    if value_start >= len(feature_json) or feature_json[value_start] != OPEN_BRACE:
        return None  # null geometry, or not an object

    coordinates = _COORDINATES_MEMBER.search(feature_json, value_start)
    if coordinates is None:
        return None
    x_match = _JSON_NUMBER.search(feature_json, coordinates.end())
    if x_match is None:
        return None
    y_match = _JSON_NUMBER.search(feature_json, x_match.end())
    if y_match is None:
        return None
    return float(x_match.group()), float(y_match.group())


def _initial_reservoir_weight(rng: random.Random, capacity: int) -> float:
    random_value = rng.random()
    while random_value == 0.0:
        random_value = rng.random()
    weight = math.exp(math.log(random_value) / capacity)
    return min(weight, math.nextafter(1.0, 0.0))


def _next_reservoir_skip(rng: random.Random, weight: float) -> int:
    random_value = rng.random()
    while random_value == 0.0:
        random_value = rng.random()
    return int(math.log(random_value) / math.log1p(-weight))


def _iter_feature_json_batches(
    path: str,
    offset: int,
    length: int,
    *,
    batch_size: int,
    decode: bool = True,
) -> Iterable[list[str] | list[bytes]]:
    if path.lower().endswith(".bz2"):
        blocks = _iter_bz2_geojson_byte_blocks(path, offset, length)
    else:
        blocks = _iter_geojson_byte_blocks(path, offset, length)

    reader = GeoJSONPartitionReader(blocks)
    features = iter(reader) if decode else reader.iter_bytes()
    yield from _batch_geojson_features(features, batch_size)


def _batch_geojson_features(
    features: Iterable[str] | Iterable[bytes],
    batch_size: int,
) -> Iterable[list[str] | list[bytes]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    batch = []
    for feature in features:
        batch.append(feature)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_geojson_byte_blocks(
    path: str,
    offset: int,
    length: int,
    *,
    block_size: int = 1024 * 1024,
) -> Iterable[bytes]:
    file_size = Path(path).stat().st_size
    split_end = min(offset + length, file_size)
    if length <= 0 or offset >= file_size or split_end <= offset:
        return

    partition_end_emitted = False
    position = offset

    with open(path, "rb") as stream:
        stream.seek(offset)
        while True:
            data = stream.read(block_size)
            if not data:
                break

            chunk_start = position
            position += len(data)

            if not partition_end_emitted and split_end <= position:
                owned_len = split_end - chunk_start
                if owned_len > 0:
                    yield data[:owned_len]
                yield b"\0"
                partition_end_emitted = True
                data = data[owned_len:]

            if data:
                yield data
    if not partition_end_emitted:
        yield b"\0"


def _iter_bz2_geojson_byte_blocks(
    path: str,
    offset: int,
    length: int,
) -> Iterable[bytes]:
    partition_end_emitted = False

    for block in _iter_bz2_decompressed_blocks(
        path,
        offset=offset,
        length=length,
        require_stream_state=True,
    ):
        if not block.owned and not partition_end_emitted:
            yield b"\0"
            partition_end_emitted = True

        yield block.data
    if not partition_end_emitted:
        yield b"\0"


def _geometries_to_wkb(geometries: List[Any]) -> List[Any]:
    """
    Vectorized geometry -> WKB conversion using shapely's GeoJSON reader.

    Converting via shapely.geometry.shape per-feature is expensive for large
    files. Using shapely.from_geojson on an array of compact JSON strings keeps
    the heavy work inside GEOS and removes most Python-level loops.
    """
    from shapely import from_geojson, to_wkb

    wkb: List[Any] = [None] * len(geometries)
    non_null_idx: List[int] = []
    geojson_strings: List[str] = []

    for idx, geom in enumerate(geometries):
        if geom is None:
            continue
        non_null_idx.append(idx)
        geojson_strings.append(json.dumps(geom, separators=(",", ":")))

    if not geojson_strings:
        return wkb

    shapely_geoms = from_geojson(geojson_strings)
    encoded = to_wkb(shapely_geoms, hex=False).tolist()

    for idx, val in zip(non_null_idx, encoded):
        wkb[idx] = val

    return wkb


def _geojson_partition_ranges(file_size: int, num_splits: int) -> List[tuple[int, int]]:
    if file_size <= 0:
        return []

    num_splits = max(1, min(int(num_splits), file_size))
    partition_size = max(1, (file_size + num_splits - 1) // num_splits)
    ranges: List[tuple[int, int]] = []

    for offset in range(0, file_size, partition_size):
        ranges.append((offset, min(partition_size, file_size - offset)))

    return ranges


def _extract_feature_collection_crs_hint(buffer: str) -> Optional[str]:
    """
    Try to read the CRS from the header of a FeatureCollection without loading the whole file.
    Looks for a 'crs' object and returns its 'properties.name' if present.
    """
    if not buffer:
        return None

    idx = buffer.lower().find('"features"')
    if idx == -1:
        return None

    header = buffer[:idx]
    first_brace = header.find("{")
    if first_brace == -1:
        return None

    candidate = header[first_brace:]
    candidate = candidate.rstrip(", \r\n\t")
    candidate = candidate + "}"

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None

    crs = parsed.get("crs")
    if isinstance(crs, dict):
        props = crs.get("properties") or {}
        name = props.get("name")
        if isinstance(name, str):
            return name

    return None


def _read_geojson_header(path: Path, max_bytes: int = 64 * 1024) -> str:
    """
    Read a small prefix from a GeoJSON file so we can inspect top-level CRS metadata.
    """
    if str(path).lower().endswith(".bz2"):
        with bz2.open(path, "rt", encoding="utf-8") as f:
            return f.read(max_bytes)

    with path.open("r", encoding="utf-8") as f:
        return f.read(max_bytes)


def _iter_bz2_geojson_batches(
    path: str,
    offset: int,
    length: int,
    *,
    batch_size: int,
) -> Iterable[list[str]]:
    blocks = _iter_bz2_geojson_byte_blocks(path, offset, length)
    yield from _batch_geojson_features(GeoJSONPartitionReader(blocks), batch_size)


def _iter_feature_collection_batches_from_bytes(
    payload: bytes,
    *,
    batch_size: int,
    owned_output_len: int,
) -> Iterable[list[str]]:
    blocks = [
        payload[:owned_output_len],
        b"\0",
        payload[owned_output_len:],
    ]
    yield from _batch_geojson_features(GeoJSONPartitionReader(blocks), batch_size)
