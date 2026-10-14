"""Vectorised pure-Python single-tile generation.

The same algorithm as the Rust core (``rust/starlet-core/src/tiler.rs``)
expressed with numpy and shapely array functions, so that a tile costs a
handful of array operations instead of a Python loop per feature:

1. partitions pruned by filename bbox, rows by the ``_bbox_*`` columns
   (Parquet predicate push-down) — or, for legacy datasets without them, by
   the bounds of the decoded geometries;
2. **raster-consistent selection**: priority ``(size, crc32(source WKB))``;
   the ``feature_capacity`` best features larger than a display pixel kept
   in full, every other feature reduced to one dot per pixel cell — all as
   array sorts / group-bys;
3. winners only are decoded: projected to y-down tile units, simplified
   (Douglas-Peucker), clipped, sub-pixel slivers dropped;
4. encoded with :mod:`starlet._internal.mvt.mvt_encoder` (numpy varints),
   dots packed into one MultiPolygon / MultiLineString feature.

Output is byte-compatible with the Rust engine for EPSG:4326 datasets.
"""
from __future__ import annotations

import threading
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely

from starlet._internal.config import config_value
from starlet._internal.internal_columns import MVT_EXCLUDED_ATTRIBUTE_COLS
from starlet._internal.mvt.helpers import mercator_tile_bounds
from starlet._internal.mvt.intermediate_tile import (
    PIXEL_GRID,
    _mvt_properties,
    normalize_simplify_tolerance,
    normalize_tile_attributes,
)
from starlet._internal.mvt.mvt_encoder import (
    GEOM_LINE,
    GEOM_POLYGON,
    LayerEncoder,
    geometry_commands,
    packed_segment_dots,
    packed_square_dots,
)
from starlet._internal.server.tiler.parquet_index import ParquetIndex

WORLD = 20037508.342789244
LAT_MAX = 85.05112878
SIMPLIFY_MIN_COORDS = 10
MIN_RING_AREA = 0.5
BBOX_COLS = ("_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax")


def lonlat_to_merc(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip(lon, -180.0, 180.0) * (WORLD / 180.0)
    la = np.clip(lat, -LAT_MAX, LAT_MAX)
    y = np.log(np.tan((90.0 + la) * (np.pi / 360.0))) / np.pi * WORLD
    return x, y


def merc_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = x / WORLD * 180.0
    lat = (2.0 * np.arctan(np.exp(y / WORLD * np.pi)) - np.pi / 2.0) * 180.0 / np.pi
    return float(lon), float(lat)


class TileFrame:
    """Tile geometry: Mercator bounds, buffered query box, and the affine map
    to y-down tile units (identical to the Rust ``TileTransform``)."""

    def __init__(self, z: int, x: int, y: int, extent: int, buffer: int) -> None:
        minx, miny, maxx, maxy = mercator_tile_bounds(int(z), int(x), int(y))
        self.z, self.x, self.y = int(z), int(x), int(y)
        self.extent = int(extent)
        self.buffer = int(buffer)
        self.minx = minx
        self.maxy = maxy
        self.scale = self.extent / (maxx - minx)
        self.cell = self.extent / PIXEL_GRID
        pad = self.buffer / self.scale
        self.query_merc = (minx - pad, miny - pad, maxx + pad, maxy + pad)
        x0, y0 = merc_to_lonlat(self.query_merc[0], self.query_merc[1])
        x1, y1 = merc_to_lonlat(self.query_merc[2], self.query_merc[3])
        self.query_lonlat = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def to_tile(self, xm: np.ndarray, ym: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (xm - self.minx) * self.scale, (self.maxy - ym) * self.scale

    def lonlat_to_tile_coords(self, coords: np.ndarray) -> np.ndarray:
        xm, ym = lonlat_to_merc(coords[:, 0], coords[:, 1])
        tx, ty = self.to_tile(xm, ym)
        return np.column_stack((tx, ty))


def _crc32_array(wkb_list: list) -> np.ndarray:
    crc = zlib.crc32
    return np.fromiter((crc(w) if w is not None else 0 for w in wkb_list), dtype=np.uint32, count=len(wkb_list))


def classify(
    frame: TileFrame,
    lon0: np.ndarray,
    lat0: np.ndarray,
    lon1: np.ndarray,
    lat1: np.ndarray,
    crc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-candidate ``(priority, pixel-cell key, is sub-pixel, centre x, centre y)``
    for EPSG:4326 bboxes against a tile (priority = size, then crc32)."""
    ax, ay = lonlat_to_merc(lon0, lat0)
    bx, by = lonlat_to_merc(lon1, lat1)
    x0, y0 = frame.to_tile(ax, ay)
    x1, y1 = frame.to_tile(bx, by)
    w = np.abs(x1 - x0)
    h = np.abs(y1 - y0)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    cell = frame.cell
    small = (w <= cell) & (h <= cell)
    size16 = np.minimum(np.maximum(w, h) * 16.0, 2.0**31 - 1).astype(np.int64)
    prio = (size16 << 32) | crc.astype(np.int64)
    cellx = np.floor(cx / cell).astype(np.int64)
    celly = np.floor(cy / cell).astype(np.int64)
    cellkey = ((celly + (1 << 20)) << 21) + (cellx + (1 << 20))
    return prio, cellkey, small, cx, cy


def select_from(prio: np.ndarray, cellkey: np.ndarray, small: np.ndarray, feature_capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """Raster-consistent selection: the ``feature_capacity`` best non-small
    candidates in full, every other candidate one-per-pixel-cell as a dot.
    Returns ``(full_idx, dot_idx)`` (indices into the inputs; ``full_idx``
    sorted, i.e. offer order). Also correct for merging partial selections:
    treat a partial's dots as ``small`` and its fulls as not small."""
    k = max(1, int(feature_capacity))
    large = np.flatnonzero(~small)
    if len(large) > k:
        order = np.argsort(-prio[large], kind="stable")
        full = large[order[:k]]
        demoted = large[order[k:]]
    else:
        full = large
        demoted = np.zeros(0, dtype=np.int64)
    dot_cand = np.concatenate((np.flatnonzero(small), demoted))
    if len(dot_cand):
        keys = cellkey[dot_cand]
        order = np.lexsort((dot_cand, -prio[dot_cand], keys))  # best priority per cell; ties -> earliest
        skeys = keys[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = skeys[1:] != skeys[:-1]
        dots = dot_cand[order[first]]
    else:
        dots = np.zeros(0, dtype=np.int64)
    return np.sort(full), dots


def select(
    frame: TileFrame,
    lon0: np.ndarray,
    lat0: np.ndarray,
    lon1: np.ndarray,
    lat1: np.ndarray,
    crc: np.ndarray,
    feature_capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(full_idx, dot_idx, dot_cx, dot_cy)`` — see :func:`classify` / :func:`select_from`."""
    prio, cellkey, small, cx, cy = classify(frame, lon0, lat0, lon1, lat1, crc)
    full, dots = select_from(prio, cellkey, small, feature_capacity)
    return full, dots, cx[dots], cy[dots]


def tile_geometries(frame: TileFrame, geoms: np.ndarray, tolerance: float) -> np.ndarray:
    """Project lon/lat geometries to tile units, simplify, clip, drop slivers.
    Returns an object array (None where the feature vanished)."""
    if len(geoms) == 0:
        return geoms
    g = shapely.transform(geoms, frame.lonlat_to_tile_coords)
    tid = shapely.get_type_id(g)
    is_point = (tid == 0) | (tid == 4)
    if tolerance > 0:
        many = shapely.get_num_coordinates(g) > SIMPLIFY_MIN_COORDS
        m = many & ~is_point
        if m.any():
            g[m] = shapely.simplify(g[m], tolerance, preserve_topology=False)
    b = frame.buffer
    e = frame.extent
    clip_mask = ~is_point
    if clip_mask.any():
        g[clip_mask] = shapely.clip_by_rect(g[clip_mask], -b, -b, e + b, e + b)
    # sub-pixel slivers (clipping artefacts) are dropped like the Rust core
    tid = shapely.get_type_id(g)
    poly = (tid == 3) | (tid == 6)
    if poly.any():
        idx = np.flatnonzero(poly)
        parts, part_of = shapely.get_parts(g[idx], return_index=True)
        keep = shapely.area(parts) >= MIN_RING_AREA
        if not keep.all():
            parts = parts[keep]
            part_of = part_of[keep]
            rebuilt = np.full(len(idx), None, dtype=object)
            if len(parts):
                uniq, inv = np.unique(part_of, return_inverse=True)
                rebuilt[uniq] = shapely.multipolygons(parts, indices=inv)
            g[idx] = rebuilt
    # GeometryCollections (mixed clip results): keep the polygon / line parts
    tid = shapely.get_type_id(g)
    coll = tid == 7
    if coll.any():
        for i in np.flatnonzero(coll):
            parts = shapely.get_parts(g[i])
            ptid = shapely.get_type_id(parts)
            polys = parts[(ptid == 3) | (ptid == 6)]
            lines = parts[(ptid == 1) | (ptid == 5)]
            if len(polys):
                g[i] = shapely.multipolygons(shapely.get_parts(polys)) if len(polys) > 1 else polys[0]
            elif len(lines):
                g[i] = shapely.multilinestrings(shapely.get_parts(lines)) if len(lines) > 1 else lines[0]
            else:
                g[i] = None
    empty = shapely.is_empty(g) | shapely.is_missing(g)
    g[empty] = None
    return g


class RowGroupCache:
    """Decoded row groups of one dataset, shared across tiles (the Python
    counterpart of the Rust core's LRU): each row group is read once —
    geometry, bbox and attribute columns — together with its per-row bbox
    arrays and crc32 priorities. Row groups are pruned by the Parquet
    statistics of the ``_bbox_*`` columns before being touched at all."""

    def __init__(self, index: ParquetIndex, capacity: int = 64) -> None:
        self.index = index
        self.capacity = max(1, int(capacity))
        self._files: dict[Path, tuple] = {}
        self._rgs: "OrderedDict[tuple[Path, int], tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def file_info(self, path: Path):
        info = self._files.get(path)
        if info is None:
            pf = pq.ParquetFile(path)
            names, geom_col, has_bbox, crs = self.index._schema_info(path)
            attrs = [n for n in names if n != geom_col and n not in MVT_EXCLUDED_ATTRIBUTE_COLS]
            stats = None
            if has_bbox:
                md = pf.metadata
                # Row-group statistics are indexed by Parquet leaf column, which differs
                # from the Arrow field index once a nested (struct/map/list) column exists.
                leaf_index = {md.schema.column(i).path: i for i in range(md.num_columns)}
                cols = [leaf_index[c] for c in BBOX_COLS]
                mins = np.full((md.num_row_groups, 2), -np.inf)
                maxs = np.full((md.num_row_groups, 2), np.inf)
                for rg in range(md.num_row_groups):
                    r = md.row_group(rg)
                    st = [r.column(c).statistics for c in cols]
                    if all(x is not None and x.has_min_max for x in st):
                        mins[rg] = (st[0].min, st[1].min)
                        maxs[rg] = (st[2].max, st[3].max)
                stats = (mins, maxs)
            info = (pf, geom_col, has_bbox, crs, attrs, stats)
            self._files[path] = info
        return info

    def candidate_row_groups(self, path: Path, q) -> list[int]:
        pf, _, has_bbox, _, _, stats = self.file_info(path)
        n = pf.metadata.num_row_groups
        if not has_bbox or stats is None:
            return list(range(n))
        mins, maxs = stats
        ok = (maxs[:, 0] >= q[0]) & (mins[:, 0] <= q[2]) & (maxs[:, 1] >= q[1]) & (mins[:, 1] <= q[3])
        return np.flatnonzero(ok).tolist()

    def row_group(self, path: Path, rg: int):
        """``(table, lon0, lat0, lon1, lat1, crc)`` for one row group."""
        key = (path, rg)
        with self._lock:
            hit = self._rgs.get(key)
            if hit is not None:
                self._rgs.move_to_end(key)
                return hit
        pf, geom_col, has_bbox, _, attrs, _ = self.file_info(path)
        columns = [geom_col] + (list(BBOX_COLS) if has_bbox else []) + attrs
        table = pf.read_row_group(rg, columns=columns)
        wkb = table[geom_col].to_pylist()
        if has_bbox:
            bb = [table[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in BBOX_COLS]
        else:
            bounds = shapely.bounds(shapely.from_wkb(wkb, on_invalid="ignore"))
            bb = [bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]]
        entry = (table, bb[0], bb[1], bb[2], bb[3], _crc32_array(wkb))
        with self._lock:
            self._rgs[key] = entry
            self._rgs.move_to_end(key)
            while len(self._rgs) > self.capacity:
                self._rgs.popitem(last=False)
        return entry


_CACHES: dict[str, RowGroupCache] = {}
_CACHES_LOCK = threading.Lock()


def row_group_cache(parquet_dir: Path, index: ParquetIndex, capacity: int | None = None) -> RowGroupCache:
    key = str(Path(parquet_dir).resolve())
    with _CACHES_LOCK:
        cache = _CACHES.get(key)
        if cache is None:
            if capacity is None:
                capacity = int(config_value("mvt", "row_group_cache"))
            cache = RowGroupCache(index, capacity)
            _CACHES[key] = cache
        return cache


def invalidate_caches() -> None:
    with _CACHES_LOCK:
        _CACHES.clear()


def generate_tile(
    dataset_path: str | Path,
    tile_id: tuple[int, int, int],
    *,
    feature_capacity: int,
    extent: int,
    buffer: int,
    tile_attributes: Any = None,
    simplify_tolerance: Any = None,
    layer_name: str = "layer0",
    index: ParquetIndex | None = None,
) -> bytes:
    """One MVT tile from a starlet dataset directory (EPSG:4326 partitions)."""
    return generate_tile_ex(
        dataset_path, tile_id, feature_capacity=feature_capacity, extent=extent, buffer=buffer,
        tile_attributes=tile_attributes, simplify_tolerance=simplify_tolerance, layer_name=layer_name,
        index=index, row_group_cache_size=None,
    )[0]


def generate_tile_ex(
    dataset_path: str | Path,
    tile_id: tuple[int, int, int],
    *,
    feature_capacity: int,
    extent: int,
    buffer: int,
    tile_attributes: Any = None,
    simplify_tolerance: Any = None,
    layer_name: str = "layer0",
    index: ParquetIndex | None = None,
    row_group_cache_size: int | None = None,
) -> tuple[bytes, int]:
    """``(tile bytes, number of features)``."""
    z, x, y = tile_id
    frame = TileFrame(z, x, y, extent, buffer)
    attrs_policy = normalize_tile_attributes(tile_attributes)
    tolerance = normalize_simplify_tolerance(simplify_tolerance, frame.cell)
    parquet_dir = Path(dataset_path) / "parquet_tiles"
    if index is None:
        index = ParquetIndex(parquet_dir)
    cache = row_group_cache(parquet_dir, index, row_group_cache_size)

    # candidate rows, gathered per (partition, row group) in file order
    tables: list[pa.Table] = []
    geom_cols: list[str] = []
    attr_cols: list[list[str]] = []
    lon0s, lat0s, lon1s, lat1s, crcs, part_of, row_of = [], [], [], [], [], [], []
    qx0, qy0, qx1, qy1 = frame.query_lonlat
    for path in index.find_intersecting_files(frame.query_lonlat):
        _pf, geom_col, _has_bbox, _crs, attrs, _stats = cache.file_info(path)
        if attrs_policy is not None:
            attrs = [n for n in attrs if n in attrs_policy]
        for rg in cache.candidate_row_groups(path, frame.query_lonlat):
            table, a, b, c, d, crc = cache.row_group(path, rg)
            hit = (c >= qx0) & (a <= qx1) & (d >= qy0) & (b <= qy1)
            if not hit.any():
                continue
            rows = np.flatnonzero(hit)
            tables.append(table)
            geom_cols.append(geom_col)
            attr_cols.append(attrs)
            lon0s.append(a[rows])
            lat0s.append(b[rows])
            lon1s.append(c[rows])
            lat1s.append(d[rows])
            crcs.append(crc[rows])
            part_of.append(np.full(len(rows), len(tables) - 1, dtype=np.int64))
            row_of.append(rows)

    layer = LayerEncoder(layer_name, extent)
    if not tables:
        return layer.encode_tile(), 0

    lon0 = np.concatenate(lon0s)
    lat0 = np.concatenate(lat0s)
    lon1 = np.concatenate(lon1s)
    lat1 = np.concatenate(lat1s)
    crc = np.concatenate(crcs)
    part_of_arr = np.concatenate(part_of)
    row_of_arr = np.concatenate(row_of)
    full, dots, dot_cx, dot_cy = select(frame, lon0, lat0, lon1, lat1, crc, feature_capacity)
    strip_point_attrs = len(dots) > max(1, int(feature_capacity))

    # ---- decode winners --------------------------------------------------
    def wkb_of(idx: np.ndarray) -> list:
        out = []
        for pi in np.unique(part_of_arr[idx]):
            m = part_of_arr[idx] == pi
            rows = row_of_arr[idx[m]]
            col = tables[pi][geom_cols[pi]].take(pa.array(rows)).to_pylist()
            out.append((idx[m], col))
        # restore idx order
        merged = [None] * len(idx)
        pos = {int(i): j for j, i in enumerate(idx)}
        for sub_idx, col in out:
            for i, w in zip(sub_idx.tolist(), col):
                merged[pos[i]] = w
        return merged

    def props_of(idx: np.ndarray) -> list:
        out = [None] * len(idx)
        pos = {int(i): j for j, i in enumerate(idx)}
        for pi in np.unique(part_of_arr[idx]):
            m = part_of_arr[idx] == pi
            sub = idx[m]
            rows = pa.array(row_of_arr[sub])
            cols = {name: tables[pi][name].take(rows).to_pylist() for name in attr_cols[pi]}
            for j, i in enumerate(sub.tolist()):
                out[pos[i]] = list(_mvt_properties({name: cols[name][j] for name in attr_cols[pi]}).items())
        return out

    # dots: kind from the geometry type, centre from the bbox
    dot_geoms = shapely.from_wkb(wkb_of(dots), on_invalid="ignore") if len(dots) else np.zeros(0, dtype=object)
    dot_tid = shapely.get_type_id(dot_geoms) if len(dots) else np.zeros(0, dtype=np.int64)
    lo, hi = -frame.buffer, frame.extent + frame.buffer
    inside = (dot_cx >= lo) & (dot_cy >= lo) & (dot_cx <= hi) & (dot_cy <= hi)
    poly_dots = ((dot_tid == 3) | (dot_tid == 6) | (dot_tid == 7)) & inside
    line_dots = ((dot_tid == 1) | (dot_tid == 5) | (dot_tid == 2)) & inside
    point_dots = ((dot_tid == 0) | (dot_tid == 4)) & inside

    # features encoded individually, in offer (file) order: full winners and native points
    indiv = np.sort(np.concatenate((full, dots[point_dots])))
    is_full = np.isin(indiv, full)
    geoms = shapely.from_wkb(wkb_of(indiv), on_invalid="ignore") if len(indiv) else np.zeros(0, dtype=object)
    tgeoms = tile_geometries(frame, geoms, tolerance) if len(indiv) else geoms
    # native points that turned out sub-pixel keep their point form (no simplification / clipping)
    cmds, types = geometry_commands(tgeoms)
    need_props = np.ones(len(indiv), dtype=bool)
    if strip_point_attrs:
        need_props &= is_full | (types != 1)
    props = props_of(indiv[need_props]) if need_props.any() else []
    pj = 0
    for j in range(len(indiv)):
        if cmds[j] is None:
            if need_props[j]:
                pj += 1
            continue
        p = None
        if need_props[j]:
            p = props[pj]
            pj += 1
        layer.add(cmds[j], int(types[j]), p)

    half = frame.cell * 0.5
    sq = packed_square_dots(np.column_stack((dot_cx[poly_dots], dot_cy[poly_dots])), half)
    if sq:
        layer.add(sq, GEOM_POLYGON, None)
    seg = packed_segment_dots(np.column_stack((dot_cx[line_dots], dot_cy[line_dots])), half)
    if seg:
        layer.add(seg, GEOM_LINE, None)
    return layer.encode_tile(), layer.feature_count
