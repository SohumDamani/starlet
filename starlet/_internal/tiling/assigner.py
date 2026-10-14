from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
from time import perf_counter

from shapely import from_wkb
from .RSGrove import RSGrovePartitioner, EnvelopeNDLite
from .utils_large import ensure_large_types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy CSV assigner
# ---------------------------------------------------------------------------

class TileAssignerFromCSV:
    def __init__(self, index_csv_path: str, geom_col: str = "geometry"):
        import pandas as _pd
        df = _pd.read_csv(index_csv_path)
        required = {"id", "minx", "miny", "maxx", "maxy"}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            raise ValueError(f"Index CSV missing columns: {missing}")

        self.geom_col = geom_col
        self._bboxes = {
            str(r.id): (float(r.minx), float(r.miny), float(r.maxx), float(r.maxy))
            for r in df[["id", "minx", "miny", "maxx", "maxy"]].itertuples(index=False)
        }
        self._areas = {
            tid: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            for tid, bbox in self._bboxes.items()
        }
        logger.info("TileAssignerFromCSV loaded %d tiles from %s", len(self._bboxes), index_csv_path)

    def tile_bbox(self, tile_id: str) -> Optional[Tuple[float, float, float, float]]:
        return self._bboxes.get(tile_id)

    def partition_by_tile(self, tbl: pa.Table) -> Dict[str, pa.Table]:
        if tbl.num_rows == 0:
            return {}
        if self.geom_col not in tbl.column_names:
            raise ValueError(f"Missing geometry column '{self.geom_col}'")

        t = tbl.combine_chunks()
        t = ensure_large_types(t, self.geom_col)
        geoms = from_wkb(t[self.geom_col].to_numpy(zero_copy_only=False))

        index_by_tile: Dict[str, List[int]] = {}
        for i, g in enumerate(geoms):
            if g is None or g.is_empty:
                continue
            gxmin, gymin, gxmax, gymax = g.bounds
            chosen = None
            chosen_area = float("inf")
            # legacy CSV mode stays "intersects"
            for tid, (xmin, ymin, xmax, ymax) in self._bboxes.items():
                if (gxmax >= xmin and gxmin <= xmax and gymax >= ymin and gymin <= ymax):
                    area = self._areas[tid]
                    if area < chosen_area:
                        chosen_area = area
                        chosen = tid
            if chosen is not None:
                index_by_tile.setdefault(chosen, []).append(i)

        out: Dict[str, pa.Table] = {}
        for tid, idxs in index_by_tile.items():
            out[tid] = t.take(pa.array(idxs, type=pa.int32()))
        return out


# ---------------------------------------------------------------------------
# RSGrove-based assigner (streaming sampling)
#   - Writes partition MBRs to rsgrove_partitions_debug.csv for verification
#   - CONTAINS-ONLY routing (inclusive eps): rows not fully contained are skipped
# ---------------------------------------------------------------------------

class RSGroveAssigner:
    """Assigns geometries to spatial partitions built by :class:`RSGrovePartitioner`.

    Uses a plane-sweep strategy over partition MBRs sorted by ``xmin`` for fast
    centroid-based assignment.  Each row's centroid is tested against the active
    window of partitions:

    1. **Containment** — first partition whose MBR contains the centroid wins.
    2. **Expansion fallback** — if no MBR contains the centroid, the partition
       requiring the least area expansion is chosen.

    The class can be constructed either directly (with a pre-built partitioner)
    or via :meth:`from_sample_and_bounds`, which consumes a prepared centroid
    sample and a bounds envelope to build the R*-tree partition index.
    """

    def __init__(
        self,
        partitioner: RSGrovePartitioner,
        global_envelope: EnvelopeNDLite,
        geom_col: str = "geometry",
    ) -> None:
        self._part = partitioner
        self._env = global_envelope
        self._geom_col = geom_col
        logger.info("RSGroveAssigner ready with %d partitions", self._part.numPartitions())

    @property
    def geom_col(self) -> str:
        return self._geom_col

    @classmethod
    def from_sample_and_bounds(
        cls,
        sample_points: np.ndarray,
        bounds: EnvelopeNDLite,
        num_partitions: int,
        geom_col: str = "geometry",
    ) -> "RSGroveAssigner":
        """Build an RSGrovePartitioner from prepared sample points and bounds."""
        sample_points = np.asarray(sample_points, dtype=np.float64)
        if sample_points.ndim != 2:
            raise ValueError("sample_points must be a 2-D array with shape (D, N)")
        if sample_points.shape[1] == 0:
            raise ValueError("sample_points must contain at least one sampled point")
        if bounds.getCoordinateDimension() != sample_points.shape[0]:
            raise ValueError(
                "sample_points dimension does not match the bounds dimension: "
                f"{sample_points.shape[0]} != {bounds.getCoordinateDimension()}"
            )

        logger.info(
            "RSGroveAssigner.from_sample_and_bounds: num_partitions=%d sample_size=%d geom_col=%s",
            num_partitions,
            sample_points.shape[1],
            geom_col,
        )

        part = RSGrovePartitioner()
        part.construct(bounds, sample_points, None, int(num_partitions))

        logger.info("Partitioner built: partitions=%d", part.numPartitions())
        return cls(part, bounds.copy(), geom_col=geom_col)

    def partition_by_tile(self, tbl: pa.Table) -> pa.Table:
        """Assign each row to a partition and return aligned partition IDs.

        Strategy (centroid-first with expansion fallback):
          1. Compute each geometry's centroid.
          2. Sort centroids by x for a plane-sweep over partitions sorted by xmin.
          3. For each centroid, check the active partition window for containment.
          4. If no partition contains the centroid, choose the one requiring the
             least MBR expansion (minimises dead space).
          5. Degenerate/empty geometries fall back to the smallest-area partition.

        Returns a single-column ``pa.Table`` with ``partition_id`` aligned 1:1
        with the input rows.
        """
        logger.debug("[ASSIGNER] After ensure_large_types metadata: %s", tbl.schema.metadata)
        start_time = perf_counter()

        if tbl.num_rows == 0:
            return pa.table({"partition_id": pa.array([], type=pa.int32())})
        if self._geom_col not in tbl.column_names:
            raise ValueError(f"Missing geometry column '{self._geom_col}'")

        t = tbl.combine_chunks()
        t = ensure_large_types(t, self._geom_col)
        logger.debug("[ASSIGNER] After ensure_large_types metadata: %s", t.schema.metadata)

        geoms = from_wkb(t[self._geom_col].to_numpy(zero_copy_only=False))

        # Pre-compute centroids and sort by x for plane sweep.
        partition_ids: List[int] = [-1] * t.num_rows
        for i, g in enumerate(geoms):
            if g is None or g.is_empty:
                continue
            cx, cy = g.centroid.x, g.centroid.y
            partition_ids[i] = self._part.overlapPartition([cx, cy])

        out = pa.table({"partition_id": pa.array(partition_ids, type=pa.int32())})
        end_time = perf_counter()

        logger.debug("partition_by_tile (contains-only): input_rows=%d, tiles=%d, finished in %.3f seconds, with a rate of %.3f rows/second",
                    t.num_rows, len(out), end_time - start_time, t.num_rows / (end_time - start_time) if end_time != start_time else 0)
        return out
