from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from concurrent.futures import as_completed
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from starlet._internal.executor import create_process_executor
from starlet._internal.tiling.datasource import (
    DataSource,
    SpatialSample,
    _GEOPARQUET_SUFFIXES,
    _combine_spatial_samples,
    _decode_wkb_geometries,
    _reservoir_add,
    _spatial_sample_from_state,
    _split_context,
    _split_sample_cap,
    _source_files,
)
from starlet._internal.tiling.utils_large import ensure_large_types

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeoParquetSplit:
    """Row groups to read from one GeoParquet file."""

    path: str
    row_groups: Tuple[int, ...]


class GeoParquetSource(DataSource):
    def __init__(
        self,
        path: str,
        *,
        geometry_only: bool = False,
        geom_col: str = "geometry",
    ):
        self.path = str(path)
        self.geometry_only = bool(geometry_only)
        self._files = _source_files(self.path, _GEOPARQUET_SUFFIXES)
        if not self._files:
            raise ValueError(f"No GeoParquet files found in {self.path}")

        pf = pq.ParquetFile(str(self._files[0]))
        self._schema = pf.schema_arrow
        self.geom_col = _resolve_geometry_column(self._schema, geom_col)
        self._row_group_counts = {
            str(file_path): pq.ParquetFile(str(file_path)).num_row_groups
            for file_path in self._files
        }
        self._num_row_groups = sum(self._row_group_counts.values())
        logger.info(
            "GeoParquetSource opened %s with %d files and %d row groups "
            "(geometry_only=%s, geom_col=%s)",
            path,
            len(self._files),
            self._num_row_groups,
            self.geometry_only,
            self.geom_col,
        )

    def schema(self) -> pa.Schema:
        logger.debug("GeoParquet source schema metadata: %s", self._schema.metadata)
        return self._schema

    def input_size_bytes(self) -> int:
        return sum(file_path.stat().st_size for file_path in self._files)

    def create_splits(self, num_splits: Optional[int] = None) -> List[GeoParquetSplit]:
        row_groups = [
            (str(file_path), row_group)
            for file_path in self._files
            for row_group in range(self._row_group_counts[str(file_path)])
        ]
        if num_splits is None:
            return [
                GeoParquetSplit(path=path, row_groups=(row_group,))
                for path, row_group in row_groups
            ]

        split_count = max(1, min(int(num_splits), max(1, len(row_groups))))
        chunk_size = max(1, (len(row_groups) + split_count - 1) // split_count)
        splits: List[GeoParquetSplit] = []
        for file_path in self._files:
            groups = list(range(self._row_group_counts[str(file_path)]))
            for index in range(0, len(groups), chunk_size):
                splits.append(
                    GeoParquetSplit(
                        path=str(file_path),
                        row_groups=tuple(groups[index:index + chunk_size]),
                    )
                )
        return splits

    def iter_tables(
        self,
        split: Optional[GeoParquetSplit] = None,
        columns: Optional[List[str]] = None,
    ) -> Iterable[pa.Table]:
        selected_columns = [self.geom_col] if self.geometry_only else columns
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            pf = pq.ParquetFile(source_split.path)
            num_row_groups = self._row_group_counts.get(source_split.path, pf.num_row_groups)
            for row_group in source_split.row_groups:
                logger.debug(
                    "Reading row group %d/%d from %s",
                    row_group,
                    num_row_groups,
                    source_split.path,
                )
                yield pf.read_row_group(row_group, columns=selected_columns)

    @classmethod
    def read_spatial_sample(
        cls,
        path: str,
        *,
        geom_col: str = "geometry",
        sample_cap: Optional[int],
        seed: int,
        workers: Optional[int],
    ) -> SpatialSample:
        """Sample GeoParquet row-group splits in parallel processes."""
        source = cls(path, geometry_only=True, geom_col=geom_col)
        geom_col = source.geom_col
        splits = source.create_splits()
        sample_caps = _split_sample_cap(sample_cap, len(splits))

        logger.info(
            "Reading GeoParquet spatial sample from %s in %d row-group partitions "
            "with %s process workers",
            path,
            len(splits),
            workers or "auto",
        )

        with create_process_executor(
            max_workers=workers,
            logger=logger,
            context="GeoParquet spatial sampling",
        ) as executor:
            futures = [
                executor.submit(
                    _read_geoparquet_split_spatial_sample,
                    path,
                    split,
                    geom_col,
                    sample_caps[index],
                    seed + index,
                )
                for index, split in enumerate(splits)
            ]
            parts: List[SpatialSample] = []
            for future in as_completed(futures):
                parts.append(future.result())
            return _combine_spatial_samples(parts)


def _resolve_geometry_column(schema: pa.Schema, requested: str) -> str:
    """Return the requested geometry column or discover the GeoParquet primary column."""
    if requested in schema.names:
        return requested

    metadata = schema.metadata or {}
    raw_geo = metadata.get(b"geo")
    if raw_geo:
        try:
            geo = json.loads(raw_geo.decode("utf-8"))
        except Exception:
            geo = {}
        primary = geo.get("primary_column")
        if isinstance(primary, str) and primary in schema.names:
            logger.info(
                "Using GeoParquet primary geometry column %r instead of missing %r",
                primary,
                requested,
            )
            return primary
        columns = geo.get("columns") or {}
        for name, column_meta in columns.items():
            if name in schema.names and (column_meta or {}).get("encoding") == "WKB":
                logger.info(
                    "Using GeoParquet geometry column %r instead of missing %r",
                    name,
                    requested,
                )
                return name

    geoarrow_columns = [
        field.name
        for field in schema
        if (field.metadata or {}).get(b"ARROW:extension:name") == b"geoarrow.wkb"
    ]
    if geoarrow_columns:
        logger.info(
            "Using GeoArrow WKB geometry column %r instead of missing %r",
            geoarrow_columns[0],
            requested,
        )
        return geoarrow_columns[0]

    common_names = ("geometry", "geom", "wkb_geometry", "SHAPE")
    for name in common_names:
        if name in schema.names:
            logger.info(
                "Using likely geometry column %r instead of missing %r",
                name,
                requested,
            )
            return name

    raise ValueError(
        f"Geometry column {requested!r} was not found and no GeoParquet "
        f"primary geometry column could be detected. Available columns: {schema.names}"
    )


def _read_geoparquet_spatial_sample(
    path: str,
    *,
    geom_col: str,
    sample_cap: Optional[int],
    seed: int,
    geoparquet_workers: Optional[int],
) -> SpatialSample:
    return GeoParquetSource.read_spatial_sample(
        path,
        geom_col=geom_col,
        sample_cap=sample_cap,
        seed=seed,
        workers=geoparquet_workers,
    )


def _read_geoparquet_split_spatial_sample(
    path: str,
    split: GeoParquetSplit,
    geom_col: str,
    sample_cap: Optional[int],
    seed: int,
) -> SpatialSample:
    """Read one GeoParquet row-group split for parallel spatial sampling."""
    source = GeoParquetSource(path, geometry_only=True, geom_col=geom_col)
    rng = np.random.default_rng(seed)
    x_sample: List[float] = []
    y_sample: List[float] = []
    n_seen = 0
    n_batches = 0

    for table in source.iter_tables(split):
        table = table.combine_chunks()
        if table.num_rows == 0:
            continue
        n_batches += 1
        table = ensure_large_types(table, geom_col)
        geometries = _decode_wkb_geometries(
            table[geom_col].to_numpy(zero_copy_only=False),
            geom_col=geom_col,
            context=_split_context(path, split),
        )

        for geom in geometries:
            if geom is None or geom.is_empty:
                continue
            centroid = geom.centroid
            n_seen += 1
            _reservoir_add(
                rng=rng,
                sample_cap=sample_cap,
                x_sample=x_sample,
                y_sample=y_sample,
                n_seen=n_seen,
                x=float(centroid.x),
                y=float(centroid.y),
            )

    return _spatial_sample_from_state(
        x_sample=x_sample,
        y_sample=y_sample,
        n_seen=n_seen,
        batches_read=n_batches,
    )
