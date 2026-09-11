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
import time
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
# HiFIVE paper's own default tile-size budget B (Table 6).
DEFAULT_SPARSIFY_BUDGET_BYTES = 256_000
# CP-SAT requires integer objective coefficients; utilities are floats in
# [0,1], so they're scaled by this factor and rounded before solving, then
# the achieved objective is divided back down by it -- see
# _solve_sparsify_problem_cpsat. 10,000x preserves ~4 decimal digits of
# utility precision, comfortably more than these utility scores carry
# meaningfully in the first place.
_CPSAT_UTILITY_SCALE = 10_000
_FEATURES_SEEN_HEADER = struct.Struct("<Q")
_FEATURES_SEEN_PADDING = 0

# Rough byte-cost model for the HiFIVE §5 linear size constraint (Eq. 9).
# MVT/protobuf encodes geometries as delta-encoded varint coordinate pairs and
# stores attribute values once in a shared per-layer dictionary, referenced by
# small varint (key_idx, value_idx) tag pairs per cell. These constants are
# engineering approximations of that encoding, not measured from the actual
# encoder -- good enough to rank/budget features, not a byte-exact model.
#
# _BYTES_PER_VERTEX was data-fit (not guessed) against real polygon tiles
# from benchmark_data/asia_postal_codes.parquet: pooled real geometry bytes
# / pooled vertex count across 8 dense real tiles gave ~0.43 bytes/vertex
# (delta+zigzag-varint encoding makes most polygon-boundary deltas tiny --
# an initial guess of 4 overestimated real geometry cost by ~8-9x). Rounded
# up slightly to 0.5 as a deliberate safety margin: for a budget constraint,
# a mild overestimate is safer than an underestimate that lets the solver
# exceed the real budget.
_BYTES_PER_VERTEX = 0.5     # estimated varint cost per (dx, dy) coordinate pair
_BYTES_PER_CELL_PTR = 2     # estimated varint cost of one (key_idx, value_idx) tag pair

# Features whose transformed bbox fits inside this many tile pixels in BOTH
# dimensions collapse to their centroid Point (sub-pixel clutter). Judged
# per-dimension, not by bbox area: a long straight line has bbox area 0 but
# must keep its type.
_SMALL_GEOMETRY_EXTENT_PX = 5.5


def feature_priority(wkb_bytes: bytes) -> int:
    """Deterministic, geometry-intrinsic sampling priority for a feature."""
    return zlib.crc32(wkb_bytes)


def _conditional_entropy(values: list, bucket_keys: list) -> float:
    """Conditional entropy H(value | bucket) in bits, given parallel
    value/bucket-key sequences.

    Zero when every bucket uniquely identifies its source value -- no
    information lost by whatever mapping produced the buckets. Grows as more
    distinct values collapse into the same bucket. This is the shared loss
    metric behind both string-prefix triage (bucket = value[:prefix_len]) and
    numeric quantization (bucket = bin index) -- same formula, different
    bucketing, so their losses are directly comparable when ranking triage
    candidates by "least damaging first" (HiFIVE Sec 6.2.3).
    """
    if not values:
        return 0.0
    buckets: dict[Any, dict[Any, int]] = {}
    for v, b in zip(values, bucket_keys):
        inner = buckets.setdefault(b, {})
        inner[v] = inner.get(v, 0) + 1
    n = len(values)
    h = 0.0
    for inner in buckets.values():
        bucket_n = sum(inner.values())
        q_b = bucket_n / n
        h_given_b = sum(-c / bucket_n * math.log2(c / bucket_n) for c in inner.values())
        h += q_b * h_given_b
    return h


def _prefix_conditional_entropy(values: list[str], prefix_len: int) -> float:
    """Conditional entropy H(value | value[:prefix_len]) in bits.

    Zero when every prefix uniquely identifies its source value — no information
    lost by truncation. Grows as more distinct values collapse into the same
    prefix bucket.
    """
    return _conditional_entropy(values, [v[:prefix_len] for v in values])


def _is_json_column(values: list[str]) -> bool:
    """True if a string column's values are JSON objects/arrays rather than
    plain text. Prefix truncation chops raw characters with no regard for
    structure, which silently corrupts JSON (cutting mid-key/mid-value
    produces unparseable text) -- such columns must be excluded from string
    triage rather than trimmed like an ordinary text field.
    """
    if not values:
        return False
    sample = values if len(values) <= 200 else values[:200]
    json_count = 0
    for v in sample:
        try:
            parsed = json.loads(v)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, (dict, list)):
            json_count += 1
    return json_count / len(sample) >= 0.9


@dataclass(frozen=True)
class _TileFeature:
    geometry: Any
    properties: dict[str, Any]


@dataclass
class _SparsifyProblem:
    """The CellSparsifyMILP problem (Algorithm 1, Eq. 7-10), assembled into
    the plain arrays scipy.optimize.milp expects. Built by
    ``_build_sparsify_problem``; not solved until Step 6.
    """
    entries: list
    columns: list[str]
    c: Any                              # objective coefficients (positive utility; a maximization problem)
    A_ub: Any                           # scipy.sparse constraint matrix
    b_ub: Any                           # constraint right-hand sides
    integrality: Any                    # 1 per variable (all binary here)
    bounds: Any                         # scipy.optimize.Bounds(0, 1)
    y_index: list[int]                  # var index for each record i
    u_index: dict[str, int]             # var index for each column
    x_index: dict[tuple[int, str], int]  # var index for each non-null cell (i, col)
    budget_bytes: float
    record_utilities: list[float]
    cell_utilities: dict[tuple[int, str], float]


@dataclass
class SolverResult:
    """Solver-agnostic answer to a _SparsifyProblem.

    Every MILP library we compare (scipy/HiGHS, OR-Tools CP-SAT, standalone
    HiGHS) has its own native result shape with different field names and
    different ways of reporting the same information. Each solver adapter's
    job is to translate its library's answer into this one shared shape, so
    everything downstream -- sparsify()'s apply logic, and the multi-solver
    benchmark comparing them -- only ever has to know about this shape, never
    about any one library's own conventions.
    """
    x: list[float]           # decision values, one per variable, same order/indexing as the problem's y/u/x_index maps
    objective_value: float   # the achieved objective (Eq. 10)
    wall_time_seconds: float
    status: str              # "optimal" / "feasible" / "infeasible" / etc. -- normalized per adapter
    num_variables: int
    num_constraints: int


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

    # ------------------------------------------------------------------
    # HiFIVE §5 sparsification helpers (Algorithm 1: CellSparsifyMILP)
    # ------------------------------------------------------------------

    def _pixel_footprint(self, geometry: Any) -> float:
        """Estimate the rendered pixel footprint pc_i of a geometry (§4/§5).

        Transforms the geometry into tile-pixel space (same affine transform
        used by encode()) and measures polygon area / line length (1px stroke
        assumed) / point count. A continuous stand-in for the paper's R x R
        rasterized footprint (Appendix A) -- cheap to compute and sufficient
        to rank features by visual salience.
        """
        transformed = affine_transform(
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
        geom_type = transformed.geom_type
        if geom_type in ("Polygon", "MultiPolygon"):
            return max(transformed.area, 1.0)
        if geom_type in ("LineString", "MultiLineString"):
            return max(transformed.length, 1.0)
        # Point, MultiPoint, GeometryCollection: count vertices as pixels.
        return float(max(1, shapely.count_coordinates(transformed)))

    def _record_utilities(
        self,
        footprints: list[float],
        lambda_rec: float = 1.0,
        p: float = 1.0,
    ) -> list[float]:
        """Record utility U_rec_i from normalized pixel footprints (Eq. 11-12).

        Normalizes footprints by the tile's maximum, then applies
        U_rec_i = lambda_rec * (pc_i / max_pc) ** p. Larger p sharpens the
        contrast between visually large and small records.
        """
        if not footprints:
            return []
        max_footprint = max(footprints)
        if max_footprint <= 0:
            return [0.0 for _ in footprints]
        return [
            lambda_rec * (footprint / max_footprint) ** p
            for footprint in footprints
        ]

    def _geometry_bytes(self, geometry: Any) -> float:
        """Estimate b_geom_i: encoded geometry byte cost from vertex count (§5).

        Delta-encoded MVT coordinates cost roughly a fixed number of varint
        bytes per (dx, dy) pair, so this scales linearly with vertex count.
        """
        return float(shapely.count_coordinates(geometry)) * _BYTES_PER_VERTEX

    @staticmethod
    def _dict_entry_bytes(value: Any) -> float:
        """Estimate the encoded byte cost of one distinct value in the shared
        MVT value dictionary (feeds b_dict_j in Eq. 9)."""
        if isinstance(value, bool):
            return 1.0
        if isinstance(value, str):
            return len(value.encode("utf-8")) + 2.0
        if isinstance(value, (int, float)):
            return 9.0
        return len(str(value).encode("utf-8")) + 2.0

    def _column_stats(
        self, entries: list[tuple[int, int, "_TileFeature"]]
    ) -> dict[str, dict[str, float]]:
        """Per-column stats needed for the Eq. 9 size constraint.

        For each non-geometry attribute column j seen across ``entries``,
        returns ``{"non_null_count": n_j, "dict_bytes": b_dict_j}`` --- the
        number of non-null cells and the estimated bytes to encode all
        distinct values in that column's shared dictionary.
        """
        columns: dict[str, dict[str, Any]] = {}
        for _, _, feature in entries:
            for key, value in feature.properties.items():
                col = columns.setdefault(key, {"non_null_count": 0, "_seen_values": set()})
                col["non_null_count"] += 1
                col["_seen_values"].add(value)

        for col in columns.values():
            col["dict_bytes"] = sum(self._dict_entry_bytes(v) for v in col["_seen_values"])
            del col["_seen_values"]
        return columns

    def _estimate_size_bytes(
        self, entries: list[tuple[int, int, "_TileFeature"]] | None = None
    ) -> float:
        """Estimate the tile's current encoded size in bytes.

        Same cost model as the Eq. 9 size constraint (geometry bytes + each
        column's amortized dictionary + pointer cost), just evaluated on the
        tile as it stands right now rather than as a MILP constraint. Shared
        by sparsify()'s early-exit check and triage()'s budget-aware loop so
        both agree on what "the tile's size" means.
        """
        if entries is None:
            entries = list(self._heap)
        geometry_total = sum(self._geometry_bytes(feature.geometry) for _, _, feature in entries)
        stats = self._column_stats(entries)
        dict_total = sum(col["dict_bytes"] for col in stats.values())
        cell_ptr_total = sum(col["non_null_count"] for col in stats.values()) * _BYTES_PER_CELL_PTR
        return geometry_total + dict_total + cell_ptr_total

    @staticmethod
    def _xlogx_ratio(num: float, den: float) -> float:
        """num * log2(num/den), defined as 0 when num <= 0 (standard x*log(x)
        convention: the limit of x*log(x) as x -> 0 is 0)."""
        if num <= 0:
            return 0.0
        return num * math.log2(num / den)

    @classmethod
    def _cell_divergence(cls, a: float, b: float, w: float) -> float:
        """Closed-form JSD (Eq. 1) between a column's pixel-weighted value
        distribution and the same distribution with one cell nulled.

        ``a`` = current probability mass in the cell's own value bucket,
        ``b`` = current probability mass in the null (⊥) bucket, ``w`` =
        this cell's own normalized pixel weight (the mass that moves from
        bucket ``a`` to bucket ``b`` when the cell is nulled). Only these
        two buckets change when nulling a single cell, so this is O(1)
        instead of recomputing the full column distribution per cell.
        """
        m_value = a - w / 2
        m_null = b + w / 2
        kld_p = cls._xlogx_ratio(a, m_value) + cls._xlogx_ratio(b, m_null)
        kld_q = cls._xlogx_ratio(a - w, m_value) + cls._xlogx_ratio(b + w, m_null)
        return 0.5 * kld_p + 0.5 * kld_q

    def _cell_utilities(
        self,
        entries: list[tuple[int, int, "_TileFeature"]],
        footprints: list[float],
        columns: set[str] | None = None,
    ) -> dict[tuple[int, str], float]:
        """Cell utility U_cell_i,j via normalized divergence (Eq. 13-15).

        For every non-null cell (record index i, column name), estimates the
        visual harm of nulling that one cell using the closed-form JSD
        shortcut in ``_cell_divergence``. Returns ``{(i, col): U_cell}``.
        """
        if columns is None:
            columns = set()
            for _, _, feature in entries:
                columns.update(feature.properties.keys())

        total_weight = sum(footprints)
        cell_utility: dict[tuple[int, str], float] = {}
        if total_weight <= 0:
            return cell_utility

        for col in columns:
            value_weights: dict[Any, float] = {}
            null_weight = total_weight
            for (_, _, feature), w in zip(entries, footprints):
                if col in feature.properties:
                    v = feature.properties[col]
                    value_weights[v] = value_weights.get(v, 0.0) + w
                    null_weight -= w

            b = null_weight / total_weight
            divergences: dict[int, float] = {}
            for i, ((_, _, feature), w) in enumerate(zip(entries, footprints)):
                if col not in feature.properties:
                    continue
                v = feature.properties[col]
                a = value_weights[v] / total_weight
                wn = w / total_weight
                divergences[i] = self._cell_divergence(a, b, wn)

            max_d = max(divergences.values()) if divergences else 0.0
            for i, d in divergences.items():
                normalized = d / max_d if max_d > 0 else 0.0
                cell_utility[(i, col)] = 1.0 - normalized  # KB_i,j = U_cell_i,j

        return cell_utility

    def _build_sparsify_problem(
        self,
        budget_bytes: float,
        alpha: float = 0.8,
        lambda_rec: float = 1.0,
        p: float = 1.0,
    ) -> "_SparsifyProblem":
        """Assemble CellSparsifyMILP (Algorithm 1, Eq. 7-10) as plain arrays.

        Lays out one flat variable vector [y_0..y_{N-1}, u_0..u_{d-1},
        x_0..x_{M-1}] (records, columns, non-null cells), builds the negated
        objective (Eq. 10) and the structural + size constraints (Eq. 7-9) as
        a sparse matrix. Does NOT call the solver -- see ``sparsify()``.
        """
        from scipy import sparse
        from scipy.optimize import Bounds
        import numpy as np

        entries = list(self._heap)
        n = len(entries)

        columns = sorted({key for _, _, feature in entries for key in feature.properties})
        column_index = {col: j for j, col in enumerate(columns)}
        d2 = len(columns)

        cells: list[tuple[int, str]] = [
            (i, col)
            for i, (_, _, feature) in enumerate(entries)
            for col in feature.properties
        ]
        m = len(cells)

        y_index = list(range(n))
        u_index = {col: n + j for col, j in column_index.items()}
        x_index = {cell: n + d2 + k for k, cell in enumerate(cells)}
        total_vars = n + d2 + m

        # --- Scores from Steps 2 and 4 ---
        footprints = [self._pixel_footprint(feature.geometry) for _, _, feature in entries]
        record_utilities = self._record_utilities(footprints, lambda_rec=lambda_rec, p=p)
        cell_utilities = self._cell_utilities(entries, footprints, columns=set(columns))

        # --- Costs from Step 3 ---
        geometry_bytes = [self._geometry_bytes(feature.geometry) for _, _, feature in entries]
        column_stats = self._column_stats(entries)

        # --- Objective (Eq. 10): the true, positive utility per variable.
        # This is a maximization problem; each solver adapter handles that
        # in whatever way its own library expects (e.g. scipy only minimizes,
        # so _solve_sparsify_problem_scipy negates its own local copy right
        # before solving, and un-negates the result right after -- that's a
        # scipy-specific detail, not something every adapter should have to
        # know about via this shared array). ---
        c = np.zeros(total_vars)
        for i in y_index:
            c[i] = alpha * record_utilities[i]
        for cell, k in x_index.items():
            i, col = cell
            c[k] = (1.0 - alpha) * cell_utilities.get((i, col), 0.0)
        # u_j coefficients stay 0.0: u only appears in the structural constraint (Eq. 8).

        # --- Structural constraints (Eq. 7, 8): x_k - y_i <= 0, x_k - u_j <= 0 ---
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        row = 0
        for cell, k in x_index.items():
            i, col = cell
            rows += [row, row]
            cols += [k, y_index[i]]
            data += [1.0, -1.0]
            row += 1

            rows += [row, row]
            cols += [k, u_index[col]]
            data += [1.0, -1.0]
            row += 1

        # --- Size constraint (Eq. 9): one more row ---
        for i in y_index:
            rows.append(row)
            cols.append(i)
            data.append(geometry_bytes[i])
        for cell, k in x_index.items():
            _, col = cell
            stats = column_stats[col]
            per_cell_cost = _BYTES_PER_CELL_PTR + (
                stats["dict_bytes"] / stats["non_null_count"] if stats["non_null_count"] else 0.0
            )
            rows.append(row)
            cols.append(k)
            data.append(per_cell_cost)
        size_row = row
        row += 1

        total_rows = row
        A_ub = sparse.csr_matrix((data, (rows, cols)), shape=(total_rows, total_vars))
        b_ub = np.zeros(total_rows)
        b_ub[size_row] = budget_bytes

        return _SparsifyProblem(
            entries=entries,
            columns=columns,
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            integrality=np.ones(total_vars),
            bounds=Bounds(0, 1),
            y_index=y_index,
            u_index=u_index,
            x_index=x_index,
            budget_bytes=budget_bytes,
            record_utilities=record_utilities,
            cell_utilities=cell_utilities,
        )

    @staticmethod
    def _solve_sparsify_problem_scipy(problem: "_SparsifyProblem") -> SolverResult:
        """Solve the assembled MILP (Step 5) with scipy.optimize.milp (HiGHS).

        The scipy adapter -- see SolverResult for why every solver adapter
        returns this same shape instead of its library's native result.
        Raises if the solver fails -- this problem is always feasible
        (dropping everything gives size 0), so failure means a real
        solver/setup error, not an infeasible budget.
        """
        from scipy.optimize import LinearConstraint, milp

        constraint = LinearConstraint(problem.A_ub, -math.inf, problem.b_ub)
        start = time.perf_counter()
        result = milp(
            # scipy.optimize.milp only minimizes, but Eq. 10 is a
            # maximization -- negate a LOCAL copy of the (positive,
            # solver-agnostic) objective just for this call, rather than
            # storing it negated on the shared problem for every adapter.
            c=-problem.c,
            constraints=constraint,
            integrality=problem.integrality,
            bounds=problem.bounds,
        )
        wall_time_seconds = time.perf_counter() - start
        if not result.success:
            raise RuntimeError(f"Sparsification MILP failed to solve: {result.message}")

        return SolverResult(
            x=list(result.x),
            # Un-negate to match: result.fun is scipy's minimized (negative)
            # value, so flip it back to the real, positive utility achieved.
            objective_value=-result.fun,
            wall_time_seconds=wall_time_seconds,
            status="optimal",
            num_variables=len(problem.c),
            num_constraints=problem.A_ub.shape[0],
        )

    @staticmethod
    def _solve_sparsify_problem_cpsat(problem: "_SparsifyProblem") -> SolverResult:
        """Solve the assembled MILP (Step 5) with OR-Tools CP-SAT.

        The CP-SAT adapter -- see SolverResult for why every adapter returns
        this same shape. CP-SAT is an integer solver throughout: every
        coefficient, in constraints as well as the objective, must be an
        integer. This reuses the exact (c, A_ub, b_ub) arrays already built
        for scipy, translated generically rather than re-deriving the
        problem's structure a second time:
          - A_ub/b_ub rows are either exactly 1/-1 (structural constraints)
            or byte-count estimates (the one size constraint) -- both round
            to the nearest integer with no meaningful precision loss.
          - c (utilities in [0,1], positive -- see _build_sparsify_problem)
            is scaled by _CPSAT_UTILITY_SCALE before rounding, since 0.73
            and 0.68 would otherwise both collapse to 1 and become
            indistinguishable.
        """
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        n_vars = len(problem.c)
        all_vars = [model.NewBoolVar(f"x{i}") for i in range(n_vars)]

        A_ub_csr = problem.A_ub.tocsr()
        for row in range(A_ub_csr.shape[0]):
            start, end = A_ub_csr.indptr[row], A_ub_csr.indptr[row + 1]
            cols = A_ub_csr.indices[start:end]
            vals = A_ub_csr.data[start:end]
            expr = sum(int(round(v)) * all_vars[c] for v, c in zip(vals, cols))
            model.Add(expr <= int(round(problem.b_ub[row])))

        objective_terms = [
            round(problem.c[i] * _CPSAT_UTILITY_SCALE) * all_vars[i] for i in range(n_vars)
        ]
        model.Maximize(sum(objective_terms))

        solver = cp_model.CpSolver()
        status_code = solver.Solve(model)

        status_name = {
            cp_model.OPTIMAL: "optimal",
            cp_model.FEASIBLE: "feasible",
            cp_model.INFEASIBLE: "infeasible",
            cp_model.MODEL_INVALID: "invalid",
            cp_model.UNKNOWN: "unknown",
        }.get(status_code, "unknown")

        if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"Sparsification MILP failed to solve (CP-SAT): status={status_name}"
            )

        return SolverResult(
            x=[float(solver.Value(v)) for v in all_vars],
            objective_value=solver.ObjectiveValue() / _CPSAT_UTILITY_SCALE,
            wall_time_seconds=solver.WallTime(),
            status=status_name,
            num_variables=n_vars,
            num_constraints=A_ub_csr.shape[0],
        )

    @staticmethod
    def _solve_sparsify_problem_highspy(problem: "_SparsifyProblem") -> SolverResult:
        """Solve the assembled MILP (Step 5) with highspy (native HiGHS).

        The highspy adapter -- see SolverResult for why every adapter
        returns this same shape. Unlike CP-SAT, HiGHS is not restricted to
        integers: c/A_ub/b_ub are handed over as real (float) numbers, with
        no scaling or rounding needed. HiGHS also maximizes natively, so
        (unlike the scipy adapter) problem.c is used as-is, no negate/
        un-negate trick required. This exists to isolate whether
        scipy.optimize.milp's wrapper costs anything over calling the same
        underlying HiGHS solver directly.
        """
        import highspy
        import numpy as np

        n_vars = len(problem.c)
        A_ub_csr = problem.A_ub.tocsr()
        n_rows = A_ub_csr.shape[0]

        lp = highspy.HighsLp()
        lp.num_col_ = n_vars
        lp.num_row_ = n_rows
        lp.col_cost_ = problem.c
        lp.col_lower_ = np.zeros(n_vars)
        lp.col_upper_ = np.ones(n_vars)
        lp.integrality_ = [highspy.HighsVarType.kInteger] * n_vars
        # A_ub x <= b_ub is one-sided, so the row has no lower bound.
        lp.row_lower_ = np.full(n_rows, -highspy.kHighsInf)
        lp.row_upper_ = problem.b_ub
        lp.sense_ = highspy.ObjSense.kMaximize

        matrix = highspy.HighsSparseMatrix()
        matrix.format_ = highspy.MatrixFormat.kRowwise
        matrix.num_col_ = n_vars
        matrix.num_row_ = n_rows
        matrix.start_ = A_ub_csr.indptr
        matrix.index_ = A_ub_csr.indices
        matrix.value_ = A_ub_csr.data
        lp.a_matrix_ = matrix

        solver = highspy.Highs()
        solver.silent()
        solver.passModel(lp)
        start = time.perf_counter()
        solver.run()
        wall_time_seconds = time.perf_counter() - start

        model_status = solver.getModelStatus()
        status_name = {
            highspy.HighsModelStatus.kOptimal: "optimal",
            highspy.HighsModelStatus.kTimeLimit: "feasible",
            highspy.HighsModelStatus.kInfeasible: "infeasible",
        }.get(model_status, "unknown")

        if status_name not in ("optimal", "feasible"):
            raise RuntimeError(
                f"Sparsification MILP failed to solve (highspy): status={status_name}"
            )

        return SolverResult(
            x=list(solver.getSolution().col_value),
            objective_value=solver.getObjectiveValue(),
            wall_time_seconds=wall_time_seconds,
            status=status_name,
            num_variables=n_vars,
            num_constraints=n_rows,
        )

    def sparsify(
        self,
        budget_bytes: float = DEFAULT_SPARSIFY_BUDGET_BYTES,
        alpha: float = 0.8,
        lambda_rec: float = 1.0,
        p: float = 1.0,
    ) -> None:
        """MILP-based sparsification (HiFIVE §5, Algorithm 1: CellSparsifyMILP).

        Builds and solves the MILP from Steps 5-6, then applies its decisions:
        records with y_i=0 are dropped entirely, columns with u_j=0 are
        removed from every feature, and individual cells with x_i,j=0 are
        removed from just that one feature. Call after triage() and before
        encode().

        Deviation from the paper: Eq. 3's problem statement keeps the row
        count fixed (|Tout| = |Tin|), nulling a dropped record's geometry
        rather than removing the row. MVT has no concept of a null geometry,
        so a y_i=0 record is dropped from the tile outright instead.
        """
        if not self._heap:
            return

        if self._estimate_size_bytes() <= budget_bytes:
            # Already fits -- the MILP's own optimum would just be "keep
            # everything," so solving it is a real solve wasted for zero
            # benefit. Skip it.
            return

        problem = self._build_sparsify_problem(
            budget_bytes, alpha=alpha, lambda_rec=lambda_rec, p=p
        )
        result = self._solve_sparsify_problem_scipy(problem)
        x = [round(value) for value in result.x]

        new_heap: list[tuple[int, int, _TileFeature]] = []
        for i, (priority, seq, feature) in enumerate(problem.entries):
            if x[problem.y_index[i]] == 0:
                continue  # y_i = 0: drop this record entirely

            new_props: dict[str, Any] = {}
            for key, value in feature.properties.items():
                if x[problem.u_index[key]] == 0:
                    continue  # u_j = 0: this column is dropped tile-wide
                cell_index = problem.x_index.get((i, key))
                if cell_index is not None and x[cell_index] == 0:
                    continue  # x_i,j = 0: this specific cell is nulled
                new_props[key] = value

            new_heap.append((priority, seq, _TileFeature(feature.geometry, new_props)))

        heapq.heapify(new_heap)
        self._heap = new_heap

    def triage(
        self,
        n_bins: int = 10,
        kld_threshold: float = 0.1,
        numeric_only: bool = False,
        string_only: bool = False,
        budget_bytes: float | None = None,
    ) -> None:
        """Apply numeric quantization and string prefix triage (HiFIVE §6.2).

        Numeric (§6.2.1): for each int/float property, divides the value range
        into n_bins equal-width bins and replaces each value with its bin midpoint.

        String (§6.2.2): for each string property, finds the shortest prefix
        length L such that the conditional entropy H(value | value[:L]) is at or
        below kld_threshold bits. Replaces all values with their length-L prefix.
        kld_threshold=0 requires that every prefix still uniquely identifies its
        original value (zero collisions); higher values allow more merging.
        Columns whose values are JSON objects/arrays (e.g. an OSM tagsMap-style
        column) are skipped entirely -- prefix truncation cuts raw characters
        with no regard for structure and would corrupt the JSON.

        budget_bytes (HiFIVE §6.2.3, "Prioritized Column Reduction"): if given,
        triage becomes budget-aware instead of unconditional. It does nothing
        at all if the tile already fits, and otherwise ranks every candidate
        reduction (each numeric column's quantization, each string column's
        truncation) by estimated information loss -- using the same
        conditional-entropy measure for both, so they're directly comparable
        -- and applies them least-damaging-first, stopping as soon as the
        running size estimate drops to or below budget_bytes. If left None
        (default), behavior is unchanged from before: every candidate is
        applied unconditionally, regardless of need.

        Both steps reduce distinct values in the MVT property dictionary without
        dropping features or changing geometries. Call after add_feature() and
        before encode().

        numeric_only: skip string prefix triage (measure numeric contribution only).
        string_only:  skip numeric quantization (measure string contribution only).
        """
        if not self._heap:
            return

        if budget_bytes is not None and self._estimate_size_bytes() <= budget_bytes:
            return  # already fits; no reduction needed

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

        original_stats = self._column_stats(self._heap)

        # ------------------------------------------------------------------
        # Step 2: numeric candidates -- equal-width bin midpoints per column,
        # each tagged with its conditional-entropy loss (§6.2.3 ranking input)
        # and the dict_bytes the column would have if this candidate is applied.
        # ------------------------------------------------------------------
        numeric_candidates: dict[str, dict[str, Any]] = {}
        if not string_only:
            for col, values in numeric_cols.items():
                lo, hi = min(values), max(values)
                if lo == hi:
                    midpoints = [lo]
                    step = 1.0
                else:
                    step = (hi - lo) / n_bins
                    # Rounded to 2dp: the exact midpoint is rarely meaningful past
                    # that (it's a styling stand-in, not a preserved measurement),
                    # and unrounded values pick up float noise like 6104591.199999999.
                    midpoints = [round(lo + (i + 0.5) * step, 2) for i in range(n_bins)]

                bin_indices = [
                    max(0, min(len(midpoints) - 1, int((v - lo) / step))) if lo != hi else 0
                    for v in values
                ]
                used_midpoints = {midpoints[i] for i in bin_indices}
                numeric_candidates[col] = {
                    "midpoints": midpoints,
                    "lo": lo,
                    "step": step,
                    "loss": _conditional_entropy(values, bin_indices),
                    "new_dict_bytes": sum(self._dict_entry_bytes(v) for v in used_midpoints),
                }

        # ------------------------------------------------------------------
        # Step 3: string candidates -- shortest prefix length with entropy
        # <= kld_threshold per column, same loss/new_dict_bytes tagging.
        # ------------------------------------------------------------------
        string_candidates: dict[str, dict[str, Any]] = {}
        if not numeric_only:
            for col, values in string_cols.items():
                if _is_json_column(values):
                    # Prefix truncation would corrupt JSON structure -- skip
                    # this column entirely rather than shorten it blindly.
                    continue
                unique_vals = set(values)
                if len(unique_vals) <= 1:
                    continue
                max_len = max(len(v) for v in unique_vals)
                for prefix_len in range(1, max_len + 1):
                    loss = _prefix_conditional_entropy(values, prefix_len)
                    if loss <= kld_threshold:
                        if prefix_len < max_len:
                            # At least one value is shortened — worth applying
                            replacements = {v: v[:prefix_len] for v in unique_vals}
                            string_candidates[col] = {
                                "replacements": replacements,
                                "loss": loss,
                                "new_dict_bytes": sum(
                                    self._dict_entry_bytes(v) for v in set(replacements.values())
                                ),
                            }
                        break

        if not numeric_candidates and not string_candidates:
            return

        # ------------------------------------------------------------------
        # Step 3.5: select which candidates to actually apply.
        # No budget: apply everything (unconditional, backward-compatible).
        # With a budget: rank by loss ascending, apply least-damaging-first,
        # stop the moment the running size estimate is under budget.
        # ------------------------------------------------------------------
        if budget_bytes is None:
            selected_numeric = numeric_candidates
            selected_string = string_candidates
        else:
            geometry_total = sum(
                self._geometry_bytes(feature.geometry) for _, _, feature in self._heap
            )
            cell_ptr_total = (
                sum(col["non_null_count"] for col in original_stats.values()) * _BYTES_PER_CELL_PTR
            )
            dict_total = sum(col["dict_bytes"] for col in original_stats.values())
            running_size = geometry_total + cell_ptr_total + dict_total

            ranked = sorted(
                [("numeric", col, cand) for col, cand in numeric_candidates.items()]
                + [("string", col, cand) for col, cand in string_candidates.items()],
                key=lambda item: item[2]["loss"],
            )

            selected_numeric = {}
            selected_string = {}
            for kind, col, cand in ranked:
                running_size += cand["new_dict_bytes"] - original_stats[col]["dict_bytes"]
                if kind == "numeric":
                    selected_numeric[col] = cand
                else:
                    selected_string[col] = cand
                if running_size <= budget_bytes:
                    break

        # ------------------------------------------------------------------
        # Step 4: rebuild heap with only the selected quantized / truncated
        # property values applied. _TileFeature is frozen so new instances
        # must be created.
        # ------------------------------------------------------------------
        new_heap: list[tuple[int, int, _TileFeature]] = []
        for priority, seq, feature in self._heap:
            new_props = dict(feature.properties)

            for col, cand in selected_numeric.items():
                if col not in new_props:
                    continue
                value = float(new_props[col])
                midpoints = cand["midpoints"]
                lo, step = cand["lo"], cand["step"]
                bin_idx = int((value - lo) / step) if step else 0
                bin_idx = max(0, min(len(midpoints) - 1, bin_idx))
                new_props[col] = midpoints[bin_idx]

            for col, cand in selected_string.items():
                if col in new_props and isinstance(new_props[col], str):
                    new_props[col] = cand["replacements"].get(new_props[col], new_props[col])

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
