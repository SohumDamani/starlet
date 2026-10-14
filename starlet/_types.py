"""Public result types and dataset introspection for starlet."""
from __future__ import annotations

import collections
import json
import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple

from starlet._internal.histogram.io import load_histogram, resolve_histogram_path
from starlet._internal.pmtiles.paths import discover_pmtiles_path

# numpy / pyarrow / geopandas-adjacent imports are deliberately deferred to
# the methods that need them: `import starlet` must stay cheap (repo
# convention) and must not drag the geo stack in for CLI --help.


@dataclasses.dataclass(frozen=True)
class TileResult:
    """Result returned by :func:`starlet.tile`."""
    outdir: str
    num_files: int
    total_rows: int
    bbox: Tuple[float, float, float, float]
    histogram_path: str


@dataclasses.dataclass(frozen=True)
class MVTResult:
    """Result returned by :func:`starlet.generate_mvt`.

    New fields are appended after the 0.3.x fields (with defaults) so
    positional construction of the historical ``(outdir, zoom_levels,
    tile_count)`` prefix keeps working.
    """
    outdir: str
    zoom_levels: List[int]
    tile_count: int
    tile_counts_by_zoom: List[int] = dataclasses.field(default_factory=list)
    pmtiles_path: Optional[str] = None


class Dataset:
    """Read-only introspection object for a starlet dataset directory.

    A dataset directory is expected to contain at least ``parquet_tiles/``
    and optionally ``histograms/``, ``mvt/``, and ``stats/``.

    Parameters
    ----------
    path : str
        Path to the dataset root directory.
    """

    def __init__(self, path: str) -> None:
        self._root = Path(path)
        if not self._root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {path}")
        self._parquet_info: tuple[bool, str | None] | None = None

    @property
    def path(self) -> str:
        return str(self._root)

    @property
    def num_tiles(self) -> int:
        tiles_dir = self._root / "parquet_tiles"
        if not tiles_dir.exists():
            return 0
        return len(list(tiles_dir.glob("*.parquet")))

    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        stats_path = self._root / "stats" / "attributes.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                for attr in stats.get("attributes", []):
                    if attr["name"] == "geometry":
                        mbr = attr["stats"].get("mbr")
                        if mbr and len(mbr) == 4:
                            return tuple(mbr)
            except Exception:
                pass
        return None

    @property
    def zoom_levels(self) -> List[int]:
        counts = self.tile_counts_by_zoom
        if not counts:
            return []
        return [z for z, count in enumerate(counts) if count > 0]

    @property
    def tile_counts_by_zoom(self) -> List[int]:
        counts = self._tile_counts_by_zoom_mapping()
        if not counts:
            return []
        max_zoom = max(counts)
        return [counts.get(z, 0) for z in range(max_zoom + 1)]

    @property
    def mvt_tile_count(self) -> int:
        return sum(self.tile_counts_by_zoom)

    @property
    def has_histograms(self) -> bool:
        try:
            resolve_histogram_path(self._root / "histograms" / "global")
            return True
        except FileNotFoundError:
            return False

    @property
    def has_mvt(self) -> bool:
        return (self._root / "mvt").is_dir()

    @property
    def pmtiles_path(self) -> str | None:
        path = discover_pmtiles_path(self._root)
        if path.exists():
            return str(path)
        return None

    @property
    def has_pmtiles(self) -> bool:
        return self.pmtiles_path is not None

    @property
    def has_stats(self) -> bool:
        return (self._root / "stats" / "attributes.json").exists()

    @property
    def parquet_has_bbox(self) -> bool:
        has_bbox, _ = self._get_parquet_info()
        return has_bbox

    @property
    def parquet_crs(self) -> str | None:
        _, crs = self._get_parquet_info()
        return crs

    @property
    def histogram_resolution(self) -> int | None:
        hist_dir = self._root / "histograms"
        metadata_path = hist_dir / "global.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as handle:
                    metadata = json.load(handle)
                grid_size = metadata.get("grid_size")
                if grid_size is not None:
                    return int(grid_size)
            except Exception:
                pass

        try:
            arr = load_histogram(hist_dir)
            if arr.ndim >= 2:
                return int(arr.shape[0])
        except Exception:
            pass
        return None

    def _tile_counts_by_zoom_mapping(self) -> dict[int, int]:
        mvt_counts = self._mvt_tile_counts()
        if mvt_counts:
            return mvt_counts
        return self._pmtiles_tile_counts()

    def _mvt_tile_counts(self) -> dict[int, int]:
        mvt_dir = self._root / "mvt"
        if not mvt_dir.exists():
            return {}
        counts: dict[int, int] = {}
        for child in mvt_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                zoom = int(child.name)
            except ValueError:
                continue
            counts[zoom] = len(list(child.rglob("*.mvt")))
        return counts

    def _pmtiles_tile_counts(self) -> dict[int, int]:
        pmtiles_path = discover_pmtiles_path(self._root)
        if not pmtiles_path.exists():
            return {}
        try:
            from pmtiles.reader import MmapSource, Reader, all_tiles
        except Exception:
            return {}

        counts: collections.Counter[int] = collections.Counter()
        with open(pmtiles_path, "rb") as handle:
            get_bytes = MmapSource(handle)
            header = Reader(get_bytes).header()
            min_zoom = int(header.get("min_zoom", 0))
            max_zoom = int(header.get("max_zoom", min_zoom))
            for z in range(min_zoom, max_zoom + 1):
                counts.setdefault(z, 0)
            for (z, _x, _y), _tile_bytes in all_tiles(get_bytes):
                counts[z] += 1
        return dict(counts)

    def _get_parquet_info(self) -> tuple[bool, str | None]:
        if self._parquet_info is not None:
            return self._parquet_info

        import pyarrow.parquet as pq

        from starlet._internal.server.tiler.parquet_index import BBOX_COLS
        from starlet._internal.tiling.crs import geoparquet_crs

        tiles_dir = self._root / "parquet_tiles"
        parquet_files = sorted(tiles_dir.glob("*.parquet"))
        if not parquet_files:
            self._parquet_info = (False, None)
            return self._parquet_info

        has_bbox = True
        crs: str | None = None
        for parquet_file in parquet_files:
            schema = pq.ParquetFile(parquet_file).schema_arrow
            names = list(schema.names)
            has_bbox = has_bbox and all(column in names for column in BBOX_COLS)
            if crs is None:
                geom_col = "geometry"
                if geom_col not in names and names:
                    geom_col = names[-1]
                raw_crs = geoparquet_crs(schema, geom_col)
                if raw_crs is not None:
                    crs = str(raw_crs)

        self._parquet_info = (has_bbox, crs)
        return self._parquet_info

    def __repr__(self) -> str:
        return f"Dataset({self._root!s}, tiles={self.num_tiles})"
