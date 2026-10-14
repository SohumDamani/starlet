"""Filename-based spatial index for GeoParquet tiles.

Parquet tiles are named ``tile_XXXXXX__minx_miny_maxx_maxy.parquet``.  The
bounding box is parsed from the filename to enable fast MBR intersection
filtering without reading file metadata.

On-the-fly tile generation only needs the geometries that fall inside the
requested tile.  Two layers of pruning make that cheap:

  1. **Partition pruning** — the filename bbox selects which partitions can
     intersect a tile (``find_intersecting_files``).
  2. **Row-group + row pruning** — tiles written by the current tiling stage
     carry per-row bbox "covering" columns (``_bbox_*``) and are split into
     spatially-coherent row groups.  ``load_and_reproject`` then uses pyarrow
     predicate pushdown to read only the row groups and rows whose bbox
     overlaps the tile, decoding a handful of geometries instead of the whole
     partition.

Older tiles without the ``_bbox_*`` columns fall back to a cached full read +
in-memory bbox pre-filter, so the server stays correct on legacy datasets.
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import geopandas as gpd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely.ops
from pyproj import Transformer

from starlet._internal.tiling.crs import WGS84_CRS, WEB_MERCATOR_CRS, geoparquet_crs
from starlet._internal.internal_columns import BBOX_COLS, FEATURE_ID_COL, QUERY_INTERNAL_COLS, TILE_ID_COL

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]

# Per-row bbox covering columns written by the tiling stage (see writer_pool).
INTERNAL_COLS = QUERY_INTERNAL_COLS
_ALWAYS_DROP_COLS = (TILE_ID_COL, *BBOX_COLS)


def geometry_column(schema: pa.Schema) -> str:
    """Geometry column of a partition schema.

    ``geometry`` when present; otherwise the GeoParquet ``primary_column``
    from the ``geo`` metadata; otherwise the first binary column that is not
    one of starlet's internal columns; otherwise the last non-internal column.
    (The bbox covering columns are appended last, so "last column" alone is
    wrong for datasets whose geometry column has another name.)
    """
    names = list(schema.names)
    if "geometry" in names:
        return "geometry"
    raw = (schema.metadata or {}).get(b"geo")
    if raw:
        try:
            primary = json.loads(raw).get("primary_column")
        except Exception:
            primary = None
        if primary in names:
            return primary
    for name, typ in zip(names, schema.types):
        if name not in INTERNAL_COLS and (pa.types.is_binary(typ) or pa.types.is_large_binary(typ)):
            return name
    external = [n for n in names if n not in INTERNAL_COLS]
    return external[-1] if external else (names[-1] if names else "geometry")


def _coerce_geometry(geometry):
    from shapely.geometry import box, shape
    from shapely.geometry.base import BaseGeometry

    if isinstance(geometry, BaseGeometry):
        return geometry
    if isinstance(geometry, dict):
        return shape(geometry)
    if len(geometry) != 4:
        raise ValueError("Rectangle geometry must be (minx, miny, maxx, maxy)")
    return box(*[float(v) for v in geometry])


def _bounds_tuple(bounds) -> BBox:
    minx, miny, maxx, maxy = bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


def parse_parquet_bbox(fname: str) -> Optional[BBox]:
    """Parse ``(minx, miny, maxx, maxy)`` from a tile filename, or ``None``.

    Expected format: ``tile_XXXXXX__minx_miny_maxx_maxy.parquet`` where each
    coordinate is encoded as an ``int_decimal`` pair (e.g. ``-97_123`` →
    ``-97.123``).
    """
    try:
        coord = fname.replace(".parquet", "").split("__")[1].split("_")
    except IndexError:
        return None
    nums: List[float] = []
    pair: List[str] = []
    for p in coord:
        pair.append(p)
        if len(pair) == 2:
            try:
                nums.append(float(pair[0] + "." + pair[1]))
            except ValueError:
                return None
            pair = []
    if len(nums) != 4:
        return None
    return (nums[0], nums[1], nums[2], nums[3])


def bbox_intersects(a: BBox, b: BBox) -> bool:
    """Whether two ``(minx, miny, maxx, maxy)`` boxes overlap."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


class ParquetIndex:
    """Spatial index over GeoParquet tiles with read-time pruning.

    Filename bounding boxes are parsed once at construction.  For legacy tiles
    (no bbox columns) decoded partitions are kept in a bounded LRU cache
    (``partition_cache_size``) so panning/zooming in one region does not
    re-read the file.
    """

    def __init__(self, folder: Path, partition_cache_size: int = 4) -> None:
        self.folder = Path(folder)
        self._entries: List[Tuple[Path, BBox]] = []
        if self.folder.exists():
            for pf in sorted(self.folder.glob("*.parquet")):
                bbox = parse_parquet_bbox(pf.name)
                if bbox is not None:
                    self._entries.append((pf, bbox))
        self._partition_cache_size = partition_cache_size
        # key -> (native GeoDataFrame, cached per-geometry bounds DataFrame)
        self._partition_cache: "OrderedDict[tuple[str, bool], Tuple[gpd.GeoDataFrame, object]]" = OrderedDict()
        # key -> (column_names, geometry_column, has_bbox_columns, crs)
        self._schema_cache: dict = {}
        self._transformer_cache: dict = {}

    # Kept for backward compatibility with callers that used the static helper.
    parse_parquet_bbox = staticmethod(parse_parquet_bbox)

    def find_intersecting_files(self, bbox_4326: BBox) -> List[Path]:
        """Partitions whose filename bbox overlaps ``bbox_4326``.

        Partition filenames store bboxes in the partition's native CRS, so the
        requested WGS84 tile bounds are transformed before comparison.
        """
        matches = []
        for pf, pbbox in self._entries:
            _, _, _, crs = self._schema_info(pf)
            query_bbox = self._transform_bbox(bbox_4326, WGS84_CRS, crs)
            if bbox_intersects(pbbox, query_bbox):
                matches.append(pf)
        return matches

    # ------------------------------------------------------------------ schema

    def _schema_info(self, path: Path):
        """Return ``(names, geometry_column, has_bbox)`` for a partition (cached)."""
        key = str(path)
        info = self._schema_cache.get(key)
        if info is not None:
            return info
        schema = pq.ParquetFile(path).schema_arrow
        names = list(schema.names)
        has_bbox = all(c in names for c in BBOX_COLS)
        geom_col = geometry_column(schema)
        crs = geoparquet_crs(schema, geom_col) or WGS84_CRS
        info = (names, geom_col, has_bbox, crs)
        self._schema_cache[key] = info
        return info

    def _transformer(self, src_crs, dst_crs):
        key = (str(src_crs or WGS84_CRS), str(dst_crs or WGS84_CRS))
        transformer = self._transformer_cache.get(key)
        if transformer is None:
            transformer = Transformer.from_crs(
                src_crs or WGS84_CRS,
                dst_crs or WGS84_CRS,
                always_xy=True,
            )
            self._transformer_cache[key] = transformer
        return transformer

    def _transform_bbox(self, bbox: BBox, src_crs, dst_crs) -> BBox:
        if str(src_crs or WGS84_CRS) == str(dst_crs or WGS84_CRS):
            return bbox
        transformer = self._transformer(src_crs, dst_crs)
        return tuple(transformer.transform_bounds(*bbox, densify_pts=21))

    def _transform_geometry(self, geom, src_crs, dst_crs):
        if str(src_crs or WGS84_CRS) == str(dst_crs or WGS84_CRS):
            return geom
        transformer = self._transformer(src_crs, dst_crs)
        return shapely.ops.transform(transformer.transform, geom)

    # ------------------------------------------------------------------ reads

    def _read_native(self, path: Path, *, include_feature_id: bool = False):
        """Read a partition in its native CRS (defaulting to EPSG:4326).

        Returns ``(gdf, bounds_df)`` with a per-geometry envelope for spatial
        pre-filtering.  Used for legacy tiles; results are LRU-cached.
        """
        _, geom_col, _, crs = self._schema_info(path)
        if self._partition_cache_size <= 0:
            gdf = self._read_wkb_table(path, geom_col, crs, include_feature_id=include_feature_id)
            return gdf, gdf.geometry.bounds

        key = (str(path), bool(include_feature_id))
        cached = self._partition_cache.get(key)
        if cached is not None:
            self._partition_cache.move_to_end(key)
            return cached

        gdf = self._read_wkb_table(path, geom_col, crs, include_feature_id=include_feature_id)
        entry = (gdf, gdf.geometry.bounds)
        self._partition_cache[key] = entry
        self._partition_cache.move_to_end(key)
        while len(self._partition_cache) > self._partition_cache_size:
            self._partition_cache.popitem(last=False)
        return entry

    def _read_wkb_table(
        self,
        path: Path,
        geom_col: str,
        crs,
        *,
        include_feature_id: bool = False,
        attributes: frozenset[str] | None = None,
    ) -> gpd.GeoDataFrame:
        columns = self._projected_columns(path, geom_col, attributes)
        table = pq.read_table(path, columns=columns)
        return self._table_to_geodataframe(
            table,
            geom_col,
            crs,
            include_feature_id=include_feature_id,
        )

    def _table_to_geodataframe(
        self,
        table,
        geom_col: str,
        crs,
        *,
        target_crs=None,
        include_feature_id: bool = False,
    ) -> gpd.GeoDataFrame:
        internal_cols = _ALWAYS_DROP_COLS if include_feature_id else INTERNAL_COLS
        drop = [c for c in internal_cols if c in table.column_names]
        if drop:
            table = table.drop(drop)
        df = table.to_pandas()
        if geom_col not in df.columns:
            return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=target_crs or crs))
        geom = gpd.GeoSeries.from_wkb(df[geom_col].to_numpy(), crs=crs)
        gdf = gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=geom, crs=crs)
        if str(target_crs or crs) == str(crs):
            return gdf
        return gdf.to_crs(target_crs)

    def _pushdown_read(
        self,
        path: Path,
        geom_col: str,
        bbox_native: BBox,
        crs,
        target_crs=WEB_MERCATOR_CRS,
        include_feature_id: bool = False,
        attributes: frozenset[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """Read only rows whose bbox overlaps ``bbox_native``, reprojected if requested.

        Uses pyarrow predicate pushdown on the ``_bbox_*`` columns; row groups
        whose statistics miss the tile are skipped entirely.
        """
        minx, miny, maxx, maxy = bbox_native
        flt = (
            (pc.field("_bbox_xmax") >= minx)
            & (pc.field("_bbox_xmin") <= maxx)
            & (pc.field("_bbox_ymax") >= miny)
            & (pc.field("_bbox_ymin") <= maxy)
        )
        columns = self._projected_columns(path, geom_col, attributes, include_bbox=True)
        table = pq.read_table(path, filters=flt, columns=columns)
        return self._table_to_geodataframe(
            table,
            geom_col,
            crs,
            target_crs=target_crs,
            include_feature_id=include_feature_id,
        )

    def load_partition(
        self,
        path: Path,
        bbox_4326: Optional[BBox] = None,
        *,
        target_crs=WEB_MERCATOR_CRS,
        include_feature_id: bool = False,
        attributes: frozenset[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """Load a partition in ``target_crs``, pruned to ``bbox_4326`` when given.

        When the tile carries ``_bbox_*`` columns this uses pyarrow row-group +
        row pushdown (cost ~ geometries in the tile).  Otherwise it falls back
        to a cached full read plus an in-memory bbox pre-filter.  Both paths
        produce identical geometry sets — the downstream clip is exact.
        """
        if bbox_4326 is not None:
            try:
                _, geom_col, has_bbox, crs = self._schema_info(path)
                bbox_native = self._transform_bbox(bbox_4326, WGS84_CRS, crs)
            except Exception:
                has_bbox = False
                geom_col = "geometry"
                crs = WGS84_CRS
                bbox_native = bbox_4326
            if has_bbox:
                return self._pushdown_read(
                    path,
                    geom_col,
                    bbox_native,
                    crs,
                    target_crs=target_crs,
                    include_feature_id=include_feature_id,
                    attributes=attributes,
                )

        if attributes is None:
            gdf, bounds = self._read_native(path, include_feature_id=include_feature_id)
        else:
            _, geom_col, _, crs = self._schema_info(path)
            gdf = self._read_wkb_table(
                path,
                geom_col,
                crs,
                include_feature_id=include_feature_id,
                attributes=attributes,
            )
            bounds = gdf.geometry.bounds
        if bbox_4326 is not None and len(gdf):
            bbox_native = self._transform_bbox(bbox_4326, WGS84_CRS, gdf.crs or WGS84_CRS)
            minx, miny, maxx, maxy = bbox_native
            mask = ~(
                (bounds["maxx"] < minx)
                | (bounds["minx"] > maxx)
                | (bounds["maxy"] < miny)
                | (bounds["miny"] > maxy)
            )
            gdf = gdf.loc[mask]
        drop_candidates = BBOX_COLS if include_feature_id else (FEATURE_ID_COL, *BBOX_COLS)
        drop = [c for c in drop_candidates if c in gdf.columns]
        if drop:
            gdf = gdf.drop(columns=drop)
        if len(gdf) and gdf.crs is not None and str(target_crs or gdf.crs) != str(gdf.crs):
            gdf = gdf.to_crs(target_crs)
        return gdf

    def load_and_reproject(self, path: Path, bbox_4326: Optional[BBox] = None) -> gpd.GeoDataFrame:
        """Backward-compatible wrapper around :meth:`load_partition`."""
        return self.load_partition(path, bbox_4326, target_crs=WEB_MERCATOR_CRS)

    def _projected_columns(
        self,
        path: Path,
        geom_col: str,
        attributes: frozenset[str] | None,
        *,
        include_bbox: bool = False,
    ) -> list[str] | None:
        if attributes is None:
            return None
        names, _, _, _ = self._schema_info(path)
        required = {geom_col}
        if include_bbox:
            required.update(BBOX_COLS)
        return [name for name in names if name in required or name in attributes]

    def iter_query_batches(
        self,
        geometry: Sequence[float] | dict[str, Any] | Any,
        *,
        geometry_crs: str = WGS84_CRS,
        target_crs: str = WGS84_CRS,
        batch_size: int | None = None,
        include_feature_id: bool = False,
        attributes: frozenset[str] | None = None,
    ) -> Iterator[gpd.GeoDataFrame]:
        """Yield batches of records whose geometries intersect ``geometry``."""
        query_geom = _coerce_geometry(geometry)
        query_geom_4326 = self._transform_geometry(query_geom, geometry_crs, WGS84_CRS)
        query_geom_target = self._transform_geometry(query_geom_4326, WGS84_CRS, target_crs)
        query_bounds_4326 = _bounds_tuple(query_geom_4326.bounds)

        for path in self.find_intersecting_files(query_bounds_4326):
            gdf = self.load_partition(
                path,
                query_bounds_4326,
                target_crs=target_crs,
                include_feature_id=include_feature_id,
                attributes=attributes,
            )
            if gdf.empty:
                continue
            bounds = gdf.geometry.bounds
            minx, miny, maxx, maxy = query_geom_target.bounds
            bbox_mask = ~(
                (bounds["maxx"] < minx)
                | (bounds["minx"] > maxx)
                | (bounds["maxy"] < miny)
                | (bounds["miny"] > maxy)
            )
            candidates = gdf.loc[bbox_mask]
            if candidates.empty:
                continue
            matches = candidates.loc[candidates.geometry.intersects(query_geom_target)]
            if matches.empty:
                continue
            if batch_size is None or batch_size <= 0 or len(matches) <= batch_size:
                yield matches
                continue
            for offset in range(0, len(matches), batch_size):
                yield matches.iloc[offset:offset + batch_size]
