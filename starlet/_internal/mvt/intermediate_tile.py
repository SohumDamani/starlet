"""Intermediate vector tile used by the map/reduce MVT pipeline.

Provides a small tile object that collects Web Mercator geometries,
retains a bounded uniform sample of them, merges with partial tiles from
other mappers, and simplifies the retained features into tile pixel
coordinates only when encoding MVT bytes.

Sampling is **priority-based top-k** rather than an independent random
reservoir: every feature carries a priority (by default ``crc32`` of its
WKB — geometry-intrinsic and deterministic; the batch pipeline passes the
crc32 of the *source* WKB bytes computed before decode), and each tile
keeps the ``feature_capacity`` features with the highest priority. Because
the same geometry has the same priority in every tile (and zoom level) it
touches, adjacent tiles make consistent keep/drop decisions — no seam
popping — and merging partial tiles from different mappers is a
deterministic top-k union instead of a statistical resample. Hash
priorities are uniformly distributed, so the retained set is still a
uniform sample of everything seen.
"""
from __future__ import annotations

import heapq
import json
import math
import random
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mapbox_vector_tile
import pyarrow as pa
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import Point

from starlet._internal.mvt.pyramid_partitioner import PyramidPartitioner

from .helpers import EXTENT, explode_geom, mercator_tile_bounds


DEFAULT_FEATURE_CAPACITY = 2_000
_FEATURES_SEEN_HEADER = struct.Struct("<Q")
_FEATURES_SEEN_PADDING = 0

# Features whose transformed bbox fits inside this many tile pixels in BOTH
# dimensions collapse to their centroid Point (sub-pixel clutter). Judged
# per-dimension, not by bbox area: a long straight line has bbox area 0 but
# must keep its type.
_SMALL_GEOMETRY_EXTENT_PX = 5.5


def feature_priority(wkb_bytes: bytes) -> int:
    """Deterministic, geometry-intrinsic sampling priority for a feature."""
    return zlib.crc32(wkb_bytes)


def _prefix_conditional_entropy(values: list[str], prefix_len: int) -> float:
    """Conditional entropy H(value | value[:prefix_len]) in bits.

    Zero when every prefix uniquely identifies its source value — no information
    lost by truncation. Grows as more distinct values collapse into the same
    prefix bucket.
    """
    if not values:
        return 0.0
    buckets: dict[str, dict[str, int]] = {}
    for v in values:
        t = v[:prefix_len]
        if t not in buckets:
            buckets[t] = {}
        buckets[t][v] = buckets[t].get(v, 0) + 1
    n = len(values)
    h = 0.0
    for inner in buckets.values():
        bucket_n = sum(inner.values())
        q_t = bucket_n / n
        h_given_t = sum(-c / bucket_n * math.log2(c / bucket_n) for c in inner.values())
        h += q_t * h_given_t
    return h


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
    ) -> None:
        self.z = int(z)
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

        # Min-heap of (priority, seq, _TileFeature); holds the top-k by
        # priority. seq is an insertion tiebreaker so heap comparisons never
        # fall through to comparing feature objects.
        self._heap: list[tuple[int, int, _TileFeature]] = []
        self._seq = 0
        self._features_seen = 0

    @property
    def tile_id(self) -> int:
        """Unique tile ID for this z/x/y."""
        return PyramidPartitioner.encode_tile_id(self.z, self.x, self.y)

    @property
    def feature_count(self) -> int:
        """Number of retained raw features."""
        return len(self._heap)

    @property
    def _features(self) -> list[_TileFeature]:
        """Retained features (arbitrary order); kept for introspection."""
        return [entry[2] for entry in self._heap]

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

        if len(self._heap) >= self.feature_capacity and priority <= self._heap[0][0]:
            return False

        clean_properties = {
            key: value
            for key, value in (properties or {}).items()
            if value is not None
        }
        entry = (priority, self._seq, _TileFeature(geometry, clean_properties))
        self._seq += 1

        if len(self._heap) < self.feature_capacity:
            heapq.heappush(self._heap, entry)
        else:
            heapq.heapreplace(self._heap, entry)
        return True

    def simplify_geometry(self, geometry: Any) -> list[Any]:
        """Return simplified tile-pixel geometries ready for MVT encoding."""
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
        threshold = _SMALL_GEOMETRY_EXTENT_PX * (self.extent / EXTENT)
        if (maxx - minx) <= threshold and (maxy - miny) <= threshold:
            centroid = geometry.centroid
            geometry = Point(centroid.x, centroid.y)

        # Simplify the geometry to reduce the number of coordinates. Use tolerance of one pixel.
        if shapely.count_coordinates(geometry) > 10:
            geometry = shapely.simplify(geometry, 1.0, preserve_topology=False)
        if geometry.geom_type not in {"Point", "MultiPoint"}:
            geometry = shapely.clip_by_rect(
                geometry,
                -self.buffer, -self.buffer, self.extent + self.buffer, self.extent + self.buffer,
            )

        out = []
        for part in explode_geom(geometry):
            if not part.is_empty:
                out.append(part)
        return out

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

        self._heap = []
        self._seq = 0
        for priority, feature in kept:
            heapq.heappush(self._heap, (priority, self._seq, feature))
            self._seq += 1
        self._features_seen += other._features_seen

    def write_features(self, path) -> None:
        """Write retained features, priorities, and seen count to disk."""
        entries = list(self._heap)
        table = pa.table(
            {
                "geometry": pa.array(
                    [entry[2].geometry.wkb for entry in entries],
                    type=pa.binary(),
                ),
                "properties": pa.array(
                    [
                        json.dumps(entry[2].properties, separators=(",", ":"))
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

        for geometry_bytes, property_json, priority in zip(geometries, properties, priorities):
            geometry = shapely.from_wkb(geometry_bytes)
            entry = (int(priority), self._seq, _TileFeature(geometry, json.loads(property_json)))
            self._seq += 1
            heapq.heappush(self._heap, entry)

    def triage(self, n_bins: int = 10, kld_threshold: float = 0.1) -> None:
        """Apply numeric quantization and string prefix triage (HiFIVE §6.2).

        Numeric (§6.2.1): for each int/float property, divides the value range
        into n_bins equal-width bins and replaces each value with its bin midpoint.

        String (§6.2.2): for each string property, finds the shortest prefix
        length L such that the conditional entropy H(value | value[:L]) is at or
        below kld_threshold bits. Replaces all values with their length-L prefix.
        kld_threshold=0 requires that every prefix still uniquely identifies its
        original value (zero collisions); higher values allow more merging.

        Both steps reduce distinct values in the MVT property dictionary without
        dropping features or changing geometries. Call after add_feature() and
        before encode().
        """
        if not self._heap:
            return

        # ------------------------------------------------------------------
        # Step 1: collect column values for numeric and string columns
        # ------------------------------------------------------------------
        numeric_cols: dict[str, list[float]] = {}
        string_cols: dict[str, list[str]] = {}
        for _, _, feature in self._heap:
            for key, value in feature.properties.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_cols.setdefault(key, []).append(float(value))
                elif isinstance(value, str):
                    string_cols.setdefault(key, []).append(value)

        # ------------------------------------------------------------------
        # Step 2: numeric — compute equal-width bin midpoints per column
        # ------------------------------------------------------------------
        bin_midpoints: dict[str, list[float]] = {}
        bin_mins: dict[str, float] = {}
        bin_steps: dict[str, float] = {}
        for col, values in numeric_cols.items():
            lo, hi = min(values), max(values)
            if lo == hi:
                bin_midpoints[col] = [lo]
                bin_mins[col] = lo
                bin_steps[col] = 1.0
            else:
                step = (hi - lo) / n_bins
                bin_midpoints[col] = [lo + (i + 0.5) * step for i in range(n_bins)]
                bin_mins[col] = lo
                bin_steps[col] = step

        # ------------------------------------------------------------------
        # Step 3: string — find shortest prefix length with entropy <= threshold
        # ------------------------------------------------------------------
        # string_replacements[col] = {original_value: truncated_value}
        string_replacements: dict[str, dict[str, str]] = {}
        for col, values in string_cols.items():
            unique_vals = set(values)
            if len(unique_vals) <= 1:
                continue
            max_len = max(len(v) for v in unique_vals)
            for prefix_len in range(1, max_len + 1):
                if _prefix_conditional_entropy(values, prefix_len) <= kld_threshold:
                    if prefix_len < max_len:
                        # At least one value is shortened — worth applying
                        string_replacements[col] = {v: v[:prefix_len] for v in unique_vals}
                    break

        if not bin_midpoints and not string_replacements:
            return

        # ------------------------------------------------------------------
        # Step 4: rebuild heap with quantized / truncated property values
        # _TileFeature is frozen so new instances must be created
        # ------------------------------------------------------------------
        new_heap: list[tuple[int, int, _TileFeature]] = []
        for priority, seq, feature in self._heap:
            new_props = dict(feature.properties)

            for col, midpoints in bin_midpoints.items():
                if col not in new_props:
                    continue
                value = float(new_props[col])
                lo = bin_mins[col]
                step = bin_steps[col]
                bin_idx = int((value - lo) / step)
                bin_idx = max(0, min(len(midpoints) - 1, bin_idx))
                new_props[col] = midpoints[bin_idx]

            for col, replacements in string_replacements.items():
                if col in new_props and isinstance(new_props[col], str):
                    new_props[col] = replacements.get(new_props[col], new_props[col])

            new_heap.append((priority, seq, _TileFeature(feature.geometry, new_props)))

        self._heap = new_heap

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
        out = []
        for _, _, feature in self._heap:
            for geometry in self.simplify_geometry(feature.geometry):
                out.append(
                    {
                        "geometry": geometry,
                        "properties": dict(feature.properties),
                    }
                )
        return out
