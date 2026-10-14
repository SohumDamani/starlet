"""Intermediate vector tile used by the map/reduce MVT pipeline.

Provides a small tile object that collects Web Mercator geometries,
retains a bounded uniform sample of them, merges with partial tiles from
other mappers, and simplifies the retained features into tile pixel
coordinates only when encoding MVT bytes.

Sampling is **raster-consistent and priority-based**. Every feature gets a
priority of ``(size in tile units, hash)`` — bigger first, the hash (by
default ``crc32`` of its WKB — geometry-intrinsic and deterministic; the
batch pipeline passes the crc32 of the *source* WKB bytes computed before
decode) as the tie-break. The ``feature_capacity`` highest-priority
features larger than one display pixel (``extent / PIXEL_GRID`` tile units
per side) are kept in full. *Every other* feature — sub-pixel ones and the
larger ones that did not make the cut — contributes a one-pixel *dot* to
the pixel cell of its bbox centre, one dot per cell (highest priority), so
the tile shows every occupied pixel, like a rasterised plot, and its biggest
shapes in detail, with a bounded number of features. Because the same
geometry has the same priority in every tile it touches, adjacent tiles
make consistent decisions — no seam popping — and merging partial tiles
from different mappers is a deterministic union.

Dots are encoded as a one-pixel square / segment of their own geometry type
(so they are styled like their full-size siblings) and carry no attributes —
they are looked up on demand.
"""
from __future__ import annotations

import heapq
import json
import math
import random
import struct
import zlib
from datetime import date, datetime, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mapbox_vector_tile
import numpy as np
import pyarrow as pa
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Point, Polygon

from starlet._internal.mvt.pyramid_partitioner import PyramidPartitioner

from .helpers import EXTENT, explode_geom, mercator_tile_bounds


DEFAULT_FEATURE_CAPACITY = 10_000
_FEATURES_SEEN_HEADER = struct.Struct("<Q")
_FEATURES_SEEN_PADDING = 0

# Display pixels per tile side: a feature whose transformed bbox fits inside
# ``extent / PIXEL_GRID`` tile units in BOTH dimensions is a sub-pixel "dot".
# Judged per-dimension, not by bbox area: a long straight line has bbox area
# 0 but is not a dot.
PIXEL_GRID = 256


def feature_priority(wkb_bytes: bytes) -> int:
    """Deterministic, geometry-intrinsic sampling priority for a feature."""
    return zlib.crc32(wkb_bytes)


@dataclass(frozen=True)
class _TileFeature:
    geometry: Any
    properties: dict[str, Any]


class IntermediateVectorTile:
    """Collect sampled Web Mercator geometries before final MVT encoding."""

    def __init__(
        self,
        z: int,
        x: int,
        y: int,
        *,
        feature_capacity: int = DEFAULT_FEATURE_CAPACITY,
        extent: int = EXTENT,
        buffer: int = 256,
        rng: random.Random | None = None,
        tile_attributes: Any = None,
        simplify_tolerance: Any = None,
    ) -> None:
        self.z = int(z)
        # None = every attribute; a list = only those columns ([] = none)
        self.tile_attributes = normalize_tile_attributes(tile_attributes)
        self._simplify_tolerance = simplify_tolerance
        self.x = int(x)
        self.y = int(y)
        self.feature_capacity = max(1, int(feature_capacity))
        self.extent = int(extent)
        self.buffer = int(buffer)
        # Accepted for backward compatibility; sampling is deterministic
        # (priority top-k) and no longer draws from an RNG.
        self.rng = rng or random.Random()

        minx, miny, maxx, maxy = mercator_tile_bounds(self.z, self.x, self.y)
        width = maxx - minx
        height = maxy - miny
        x_scale = self.extent / width if width != 0 else 0.0
        y_scale = self.extent / height if height != 0 else 0.0
        self.affine_params = (
            x_scale,
            0.0,
            0.0,
            y_scale,
            -minx * x_scale,
            -miny * y_scale,
        )

        # Min-heap of (priority, seq, _TileFeature); holds the top-k larger-
        # than-a-pixel features by priority. seq is an insertion tiebreaker so
        # heap comparisons never fall through to comparing feature objects.
        self._heap: list[tuple[int, int, _TileFeature]] = []
        # Dots: best (priority, seq, feature) per pixel cell — sub-pixel
        # features and larger ones that did not make the top-k.
        self._cells: dict[tuple[int, int], tuple[int, int, _TileFeature]] = {}
        self.cell = self.extent / PIXEL_GRID
        # Douglas-Peucker tolerance in tile units; "auto"/None = a quarter of a display pixel
        self.simplify_tolerance = normalize_simplify_tolerance(self._simplify_tolerance, self.cell)
        self._seq = 0
        self._features_seen = 0

    def _tile_bbox(self, bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Web Mercator bounds -> tile-unit bounds."""
        minx, miny, maxx, maxy = bounds
        x_scale, _, _, y_scale, xoff, yoff = self.affine_params
        return (minx * x_scale + xoff, miny * y_scale + yoff, maxx * x_scale + xoff, maxy * y_scale + yoff)

    def place(self, bounds: tuple[float, float, float, float]) -> tuple[tuple[int, int], bool, int]:
        """``(pixel cell of the bbox centre, fits in one pixel, size16)`` for
        Web Mercator ``bounds``; ``size16`` is max(width, height) in 1/16 tile units."""
        x0, y0, x1, y1 = self._tile_bbox(bounds)
        w, h = abs(x1 - x0), abs(y1 - y0)
        cell = (math.floor((x0 + x1) * 0.5 / self.cell), math.floor((y0 + y1) * 0.5 / self.cell))
        return cell, (w <= self.cell and h <= self.cell), min(int(max(w, h) * 16.0), 2**31 - 1)

    def dot_cell(self, bounds: tuple[float, float, float, float]) -> tuple[int, int] | None:
        """Pixel cell of a sub-pixel feature (Web Mercator ``bounds``), else None."""
        cell, small, _ = self.place(bounds)
        return cell if small else None

    @staticmethod
    def combine_priority(size16: int, hash_priority: int) -> int:
        """Selection priority: size first, the (crc32) hash as the tie-break."""
        return (int(size16) << 32) | (int(hash_priority) & 0xFFFFFFFF)

    def _offer_cell(self, cell: tuple[int, int], entry: tuple[int, int, _TileFeature]) -> bool:
        current = self._cells.get(cell)
        if current is not None and entry[0] <= current[0]:
            return False
        self._cells[cell] = entry
        return True

    @property
    def tile_id(self) -> int:
        """Unique tile ID for this z/x/y."""
        return PyramidPartitioner.encode_tile_id(self.z, self.x, self.y)

    @property
    def feature_count(self) -> int:
        """Number of retained raw features."""
        return len(self._heap) + len(self._cells)

    def _entries(self) -> list[tuple[int, int, _TileFeature, bool]]:
        """All retained ``(priority, seq, feature, as_dot)`` entries in offer order."""
        entries = [(p, q, f, False) for (p, q, f) in self._heap]
        entries.extend((p, q, f, True) for (p, q, f) in self._cells.values())
        entries.sort(key=lambda e: e[1])
        return entries

    @property
    def _features(self) -> list[_TileFeature]:
        """Retained features (offer order); kept for introspection."""
        return [entry[2] for entry in self._entries()]

    @property
    def full_features(self) -> list[_TileFeature]:
        """Features kept in full (the top-k larger than a pixel)."""
        return [entry[2] for entry in sorted(self._heap, key=lambda e: e[1])]

    @property
    def dot_features(self) -> list[_TileFeature]:
        """Features kept as one-pixel dots."""
        return [entry[2] for entry in sorted(self._cells.values(), key=lambda e: e[1])]

    def add_feature(
        self,
        geometry: Any,
        properties: dict[str, Any] | None = None,
        priority: int | None = None,
    ) -> bool:
        """Offer a Web Mercator geometry; keep it if it ranks in the top-k.

        ``priority`` should be :func:`feature_priority` of the feature's
        canonical (source) WKB bytes so that every tile the feature touches
        ranks it identically. When omitted it is derived from the current
        geometry's WKB.
        """
        if geometry is None or geometry.is_empty:
            return False

        self._features_seen += 1

        if priority is None:
            priority = feature_priority(shapely.to_wkb(geometry))
        priority = int(priority)

        cell, small, size16 = self.place(geometry.bounds)
        priority = self.combine_priority(size16, priority)

        clean_properties = {
            key: value
            for key, value in (properties or {}).items()
            if value is not None
        }
        entry = (priority, self._seq, _TileFeature(geometry, clean_properties))
        self._seq += 1

        if small:
            return self._offer_cell(cell, entry)
        if len(self._heap) < self.feature_capacity:
            heapq.heappush(self._heap, entry)
            return True
        if priority <= self._heap[0][0]:
            # did not make the cut: a dot instead
            return self._offer_cell(cell, entry)
        evicted = heapq.heapreplace(self._heap, entry)
        self._offer_cell(self.place(evicted[2].geometry.bounds)[0], evicted)
        return True

    def simplify_geometry(self, geometry: Any) -> list[Any]:
        """Return simplified tile-pixel geometries ready for MVT encoding."""
        return self._tile_geometry(geometry)[0]

    def _tile_geometry(self, geometry: Any, as_dot: bool = False) -> tuple[list[Any], bool]:
        """Tile-pixel geometries plus whether the feature became a *dot*.
        ``as_dot`` forces the dot form (a feature that did not make the top-k)."""
        geometry = affine_transform(
            geometry,
            (
                self.affine_params[0],
                0.0,
                0.0,
                self.affine_params[3],
                self.affine_params[4],
                self.affine_params[5],
            ),
        )

        minx, miny, maxx, maxy = geometry.bounds
        if as_dot or ((maxx - minx) <= self.cell and (maxy - miny) <= self.cell):
            # A sub-pixel (or demoted) feature becomes a one-pixel shape of its own kind.
            cx, cy = (minx + maxx) * 0.5, (miny + maxy) * 0.5
            lo, hi = -self.buffer, self.extent + self.buffer
            if cx < lo or cy < lo or cx > hi or cy > hi:
                return [], True
            kind = _base_kind(geometry)
            h = self.cell * 0.5
            if kind == "Polygon":
                # same vertex order as the Rust engine (byte-identical tiles)
                # (tile units here are y-up; the Rust engine's are y-down, hence the mirrored order)
                square = Polygon([(cx - h, cy + h), (cx + h, cy + h), (cx + h, cy - h), (cx - h, cy - h), (cx - h, cy + h)])
                return [square], True
            if kind == "LineString":
                return [LineString([(cx - h, cy), (cx + h, cy)])], True
            return [Point(cx, cy)], False

        # Simplify the geometry to reduce the number of coordinates.
        if shapely.count_coordinates(geometry) > 10:
            geometry = shapely.simplify(geometry, self.simplify_tolerance, preserve_topology=False)
        if geometry.geom_type not in {"Point", "MultiPoint"}:
            geometry = shapely.clip_by_rect(
                geometry,
                -self.buffer, -self.buffer, self.extent + self.buffer, self.extent + self.buffer,
            )

        out = []
        for part in explode_geom(geometry):
            if not part.is_empty:
                out.append(part)
        return out, False

    def merge(self, other: "IntermediateVectorTile") -> None:
        """Merge another partial tile for the same z/x/y: top-k union.

        Every entry keeps the priority it was offered with, so the merged
        result is exactly the top ``feature_capacity`` features by priority
        across both partials — deterministic and independent of merge order.
        """
        if (self.z, self.x, self.y) != (other.z, other.x, other.y):
            raise ValueError("Cannot merge intermediate tiles with different tile IDs")

        combined = [(p, s) for (p, _, s) in self._heap]
        combined.extend((p, s) for (p, _, s) in other._heap)
        # Sort by priority (stable: self's entries win ties deterministically).
        combined.sort(key=lambda item: item[0], reverse=True)
        kept = combined[: self.feature_capacity]
        demoted = combined[self.feature_capacity:]

        self._heap = []
        self._seq = 0
        for priority, feature in kept:
            heapq.heappush(self._heap, (priority, self._seq, feature))
            self._seq += 1
        # Pixel cells: the higher priority wins; self keeps ties. Features
        # that lost their place in the top-k become dots.
        for cell, (priority, _, feature) in other._cells.items():
            self._offer_cell(cell, (priority, self._seq, feature))
            self._seq += 1
        for priority, feature in demoted:
            self._offer_cell(self.place(feature.geometry.bounds)[0], (priority, self._seq, feature))
            self._seq += 1
        self._features_seen += other._features_seen

    def write_features(self, path) -> None:
        """Write retained features, priorities, and seen count to disk."""
        entries = [e[:3] for e in self._entries()]
        table = pa.table(
            {
                "geometry": pa.array(
                    [entry[2].geometry.wkb for entry in entries],
                    type=pa.binary(),
                ),
                "properties": pa.array(
                    [
                        json.dumps(entry[2].properties, separators=(",", ":"), default=_json_safe_value)
                        for entry in entries
                    ],
                    type=pa.string(),
                ),
                "priority": pa.array(
                    [entry[0] for entry in entries],
                    type=pa.uint64(),
                ),
            }
        )
        payload_sink = pa.BufferOutputStream()
        with pa.ipc.new_file(payload_sink, table.schema) as writer:
            writer.write_table(table)
        payload = payload_sink.getvalue().to_pybytes()
        with pa.OSFile(str(path), "wb") as sink:
            sink.write(_FEATURES_SEEN_HEADER.pack(self._features_seen))
            sink.write(b"\x00" * _FEATURES_SEEN_PADDING)
            sink.write(payload)

    def load_features(self, path) -> None:
        """Load retained features (with their priorities) from disk."""
        data = Path(path).read_bytes()
        self._features_seen = _FEATURES_SEEN_HEADER.unpack(data[:_FEATURES_SEEN_HEADER.size])[0]
        payload_offset = _FEATURES_SEEN_HEADER.size + _FEATURES_SEEN_PADDING
        table = pa.ipc.open_file(pa.BufferReader(data[payload_offset:])).read_all()

        geometries = table["geometry"].to_pylist()
        properties = table["properties"].to_pylist()
        if "priority" in table.column_names:
            priorities = table["priority"].to_pylist()
        else:
            # Files written before the priority column: recompute from WKB.
            priorities = [feature_priority(geometry_bytes) for geometry_bytes in geometries]

        seen = self._features_seen
        for geometry_bytes, property_json, priority in zip(geometries, properties, priorities):
            # Re-offer: classification (pixel cell vs top-k) is recomputed.
            self.add_feature(shapely.from_wkb(geometry_bytes), json.loads(property_json), priority=int(priority))
        self._features_seen = seen

    def encode(self, layer_name: str = "layer0") -> bytes:
        """Encode the retained features as an MVT binary payload."""
        layer = {
            "name": layer_name,
            "features": self._mvt_features(),
            "extent": self.extent,
        }
        result = mapbox_vector_tile.encode(
            [layer],
            default_options={"extents": self.extent},
        )
        return result

    def _mvt_features(self) -> list[dict[str, Any]]:
        # Sub-pixel *points* keep their attributes only while the tile has at
        # most ``feature_capacity`` sub-pixel features; denser tiles carry
        # them as bare dots (attributes stay reachable through the lookup).
        strip_point_attrs = len(self._cells) > self.feature_capacity
        out = []
        poly_dots: list[tuple[float, float]] = []
        line_dots: list[tuple[float, float]] = []
        for _, _, feature, as_dot in self._entries():
            geometries, is_dot = self._tile_geometry(feature.geometry, as_dot)
            for geometry in geometries:
                if is_dot:
                    # packed below into one feature per kind
                    c = geometry.centroid
                    (poly_dots if geometry.geom_type == "Polygon" else line_dots).append((c.x, c.y))
                    continue
                bare = strip_point_attrs and geometry.geom_type == "Point"
                out.append(
                    {
                        "geometry": geometry,
                        "properties": {} if bare else _mvt_properties(self._select_properties(feature.properties)),
                    }
                )
        out.extend(self._packed_dots(poly_dots, line_dots))
        return out

    def _packed_dots(self, poly_dots, line_dots) -> list[dict[str, Any]]:
        """Polygon / line dots as one MultiPolygon / MultiLineString feature
        each, sorted by position (y-down, then x — the Rust engine's order) so
        delta-encoded coordinates stay small."""
        from shapely.geometry import MultiLineString, MultiPolygon

        h = self.cell * 0.5
        key = lambda c: (round(self.extent - c[1]), round(c[0]))  # noqa: E731
        out = []
        if poly_dots:
            poly_dots.sort(key=key)
            squares = [
                Polygon([(cx - h, cy + h), (cx + h, cy + h), (cx + h, cy - h), (cx - h, cy - h), (cx - h, cy + h)])
                for cx, cy in poly_dots
            ]
            out.append({"geometry": MultiPolygon(squares), "properties": {}})
        if line_dots:
            line_dots.sort(key=key)
            out.append({
                "geometry": MultiLineString([[(cx - h, cy), (cx + h, cy)] for cx, cy in line_dots]),
                "properties": {},
            })
        return out

    def _select_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        if self.tile_attributes is None:
            return dict(properties)
        return {k: v for k, v in properties.items() if k in self.tile_attributes}


def normalize_simplify_tolerance(value: Any, cell: float) -> float:
    """``"auto"``/``None`` -> a quarter of a display pixel; else a float in tile units."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "auto")):
        return cell * 0.25
    return max(0.0, float(value))


def normalize_tile_attributes(value: Any) -> list[str] | None:
    """Tile-attribute policy -> ``None`` (all columns) or a list of columns.

    Accepts ``None``/``"all"`` (everything), ``"none"``/``""``/``[]`` (no
    attributes in tiles — fetch them with the record lookup), a comma-separated
    string, or a sequence of column names.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("", "all", "*"):
            return None
        if text.lower() == "none":
            return []
        return [part.strip() for part in text.split(",") if part.strip()]
    return [str(v) for v in value]


def _base_kind(geometry: Any) -> str:
    """'Polygon' | 'LineString' | 'Point' for a (possibly multi/collection) geometry."""
    kinds = {part.geom_type for part in explode_geom(geometry) if not part.is_empty}
    if "Polygon" in kinds:
        return "Polygon"
    if "LineString" in kinds or "LinearRing" in kinds:
        return "LineString"
    return "Point"

def _json_safe_value(value: Any) -> Any:
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _mvt_properties(properties: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in properties.items():
        if value is not None:
            for key, flattened_value in _flatten_property_items(name, value):
                out[key] = _mvt_property_value(flattened_value)
    return out


def _flatten_property_items(name: str, value: Any):
    if isinstance(value, dict):
        for key, item_value in value.items():
            if item_value is not None:
                yield from _flatten_property_items(f"{name}.{key}", item_value)
        return

    if _is_map_entries(value):
        for key, item_value in value:
            if item_value is not None:
                yield from _flatten_property_items(f"{name}.{key}", item_value)
        return

    yield name, value


def _is_map_entries(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)
        for item in value
    )


def _mvt_property_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
