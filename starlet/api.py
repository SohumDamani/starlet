"""Public convenience APIs for library consumers."""
from __future__ import annotations

import json
import math
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from starlet._internal.config import config_value, ensure_config_loaded, get_loaded_config
from starlet._internal.histogram.io import load_prefix_histogram, resolve_histogram_path
from starlet._internal.pmtiles.paths import discover_pmtiles_path
from starlet._internal.internal_columns import QUERY_INTERNAL_COLS

if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry

BBox = tuple[float, float, float, float]
PHI_MAX = 2 * math.atan(math.exp(math.pi)) - math.pi / 2
LIM = 6378137.0 * math.log(math.tan(math.pi / 4 + PHI_MAX / 2))
GLOBAL_BBOX: BBox = (-LIM, -LIM, LIM, LIM)

_TILER_CACHE: dict[tuple[str, int, int, int, int], Any] = {}
_TILER_CACHE_LOCK = threading.Lock()
_INTERNAL_QUERY_COLUMNS = frozenset(QUERY_INTERNAL_COLS)


def get_config() -> dict[str, Any]:
    """Return the current process-wide Starlet configuration."""
    ensure_config_loaded()
    return get_loaded_config()


def _configured_tiler_cache_size() -> int:
    ensure_config_loaded()
    return int(config_value("serve", "cache_size"))


def _configured_tiler_extent() -> int:
    return int(config_value("mvt", "extent"))


def _configured_tiler_buffer() -> int:
    return int(config_value("mvt", "buffer"))


def _configured_tiler_feature_capacity() -> int:
    return int(config_value("mvt", "feature_capacity"))


def _configured_tiler_attributes():
    from starlet._internal.mvt.intermediate_tile import normalize_tile_attributes

    return normalize_tile_attributes(config_value("mvt", "tile_attributes"))


def _get_cached_vector_tiler(dataset_dir: str | Path) -> Any:
    from starlet._internal.server.tiler.tiler import VectorTiler

    dataset_path = str(Path(dataset_dir).absolute())
    cache_size = _configured_tiler_cache_size()
    extent = _configured_tiler_extent()
    buffer = _configured_tiler_buffer()
    feature_capacity = _configured_tiler_feature_capacity()
    tile_attributes = _configured_tiler_attributes()
    key = (dataset_path, cache_size, extent, buffer, feature_capacity,
           None if tile_attributes is None else tuple(tile_attributes))

    with _TILER_CACHE_LOCK:
        tiler = _TILER_CACHE.get(key)
        if tiler is None:
            tiler = VectorTiler(
                dataset_path,
                memory_cache_size=cache_size,
                extent=extent,
                buffer=buffer,
                feature_capacity=feature_capacity,
                tile_attributes=tile_attributes,
            )
            _TILER_CACHE[key] = tiler
        return tiler


def list_datasets(datasets_dir: str | Path) -> list[str]:
    """Return dataset directory names under ``datasets_dir``."""
    root = Path(datasets_dir)
    if not root.exists():
        return []
    if not root.is_dir():
        raise NotADirectoryError(f"Datasets path is not a directory: {root}")
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "parquet_tiles").is_dir()
    )


def get_tile(
    dataset_dir: str | Path,
    z: int,
    x: int,
    y: int,
    output: dict[str, Any] | None = None,
    attributes: Sequence[str] | None = None,
) -> bytes:
    """Return a pre-generated MVT tile or generate it on the fly.

    The generated tile is not written to disk. A short-lived in-memory cache is
    used internally only for the duration of this call.

    If ``output`` is provided, it is updated with details such as where the tile
    came from and how long serving took. On-the-fly generation also reports the
    number of encoded features.

    Pass ``attributes`` to limit non-geometry attributes when the tile is
    generated from GeoParquet. Geometry is always included. Pre-generated
    MVT/PMTiles are returned as-is.
    """
    return _get_cached_vector_tiler(dataset_dir).get_tile(z, x, y, output=output, attributes=attributes)


def get_dataset_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    """Return cheap metadata and availability information for a dataset."""
    root = Path(dataset_dir)
    from starlet._types import Dataset

    ds = Dataset(str(root)) if root.is_dir() else None
    stats = _load_dataset_stats(root)
    summary = get_dataset_summary(root)
    files = list(root.rglob("*")) if root.exists() and root.is_dir() else []
    regular_files = [path for path in files if path.is_file()]
    mvt_dir = root / "mvt"
    pmtiles_path = discover_pmtiles_path(root)
    zoom_levels = sorted(
        int(child.name)
        for child in mvt_dir.iterdir()
        if child.is_dir() and child.name.isdigit()
    ) if mvt_dir.is_dir() else []
    missing = []
    if not root.is_dir():
        missing.append("dataset_dir")
    if not (root / "parquet_tiles").is_dir():
        missing.append("parquet_tiles")
    try:
        resolve_histogram_path(root / "histograms" / "global")
    except FileNotFoundError:
        missing.append("histograms")
    if stats is None:
        missing.append("stats")

    return {
        "name": root.name,
        "path": str(root),
        "exists": root.is_dir(),
        "size_bytes": sum(path.stat().st_size for path in regular_files),
        "file_count": len(regular_files),
        "parquet_tile_count": len(list((root / "parquet_tiles").glob("*.parquet"))),
        "parquet_has_bbox": ds.parquet_has_bbox if ds is not None else False,
        "parquet_crs": ds.parquet_crs if ds is not None else None,
        "bbox": _summary_bbox(summary) or _stats_bbox(stats),
        "zoom_levels": zoom_levels,
        "mvt_tile_count": ds.mvt_tile_count if ds is not None else 0,
        "has_histograms": "histograms" not in missing,
        "histogram_resolution": ds.histogram_resolution if ds is not None else None,
        "has_mvt": mvt_dir.is_dir(),
        "has_pmtiles": pmtiles_path.exists(),
        "pmtiles_path": str(pmtiles_path) if pmtiles_path.exists() else None,
        "has_stats": stats is not None,
        "has_summary": summary is not None,
        "missing": missing,
    }


def get_dataset_summary(dataset_dir: str | Path) -> dict[str, Any] | None:
    """Return a stored or derived dataset summary, or ``None``.

    If ``summary.json`` exists at the dataset root or under ``stats/``, it is
    returned directly. Otherwise, a summary is derived from stored attribute
    stats when available.
    """
    root = Path(dataset_dir)
    for summary_path in (root / "summary.json", root / "stats" / "summary.json"):
        if summary_path.exists():
            with open(summary_path, "r") as f:
                return json.load(f)

    stats = _load_dataset_stats(root)
    if stats is None:
        return None
    return _build_summary(root.name, _normalize_stats(stats))


def estimate_range_count(
    dataset_dir: str | Path,
    rectangle: Sequence[float],
    *,
    rectangle_crs: str = "EPSG:4326",
) -> float:
    """Estimate data amount in a rectangle using the dataset histogram.

    ``rectangle`` is ``(minx, miny, maxx, maxy)``. By default it is interpreted
    as longitude/latitude and transformed to the histogram CRS (EPSG:3857).
    """
    root = Path(dataset_dir)
    hist_dir = root / "histograms"
    try:
        prefix = load_prefix_histogram(hist_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Histogram not found under {hist_dir}") from exc
    bbox_3857 = _rectangle_to_3857(rectangle, rectangle_crs)
    return float(_prefix_sum_rectangle(prefix, bbox_3857))


def query_dataset(
    dataset_dir: str | Path,
    geometry: Sequence[float] | dict[str, Any] | "BaseGeometry",
    *,
    geometry_crs: str = "EPSG:4326",
    batch_size: int | None = None,
) -> Iterator["gpd.GeoDataFrame"]:
    """Yield GeoDataFrame batches whose geometries intersect ``geometry``.

    ``geometry`` can be a bbox tuple ``(minx, miny, maxx, maxy)``, a GeoJSON
    geometry mapping, or a Shapely geometry. Batches are returned in EPSG:4326.
    """
    from starlet._internal.server.tiler.parquet_index import ParquetIndex

    index = ParquetIndex(Path(dataset_dir) / "parquet_tiles")
    for batch in index.iter_query_batches(
        geometry,
        geometry_crs=geometry_crs,
        target_crs="EPSG:4326",
        batch_size=batch_size,
    ):
        yield _drop_internal_query_columns(batch)


def query_dataset_count(
    dataset_dir: str | Path,
    geometry: Sequence[float] | dict[str, Any] | "BaseGeometry",
    *,
    geometry_crs: str = "EPSG:4326",
    geom_col: str = "geometry",
    batch_size: int | None = None,
) -> int:
    """Return the number of records that intersect ``geometry``."""
    return sum(
        len(batch)
        for batch in query_dataset(
            dataset_dir,
            geometry,
            geometry_crs=geometry_crs,
            batch_size=batch_size,
        )
    )


def query_dataset_size(
    dataset_dir: str | Path,
    geometry: Sequence[float] | dict[str, Any] | "BaseGeometry",
    *,
    geometry_crs: str = "EPSG:4326",
    geom_col: str = "geometry",
    batch_size: int | None = None,
) -> int:
    """Return a rough in-memory byte estimate for records matching ``geometry``."""
    total = 0
    for batch in query_dataset(
        dataset_dir,
        geometry,
        geometry_crs=geometry_crs,
        batch_size=batch_size,
    ):
        total += _estimate_batch_size(batch)
    return int(total)


def get_sample_record(
    dataset_dir: str | Path,
    geometry: Sequence[float] | dict[str, Any] | "BaseGeometry",
    *,
    geometry_crs: str = "EPSG:4326",
) -> dict[str, Any] | None:
    """Return the first matching record, or ``None``."""
    for batch in query_dataset(
        dataset_dir,
        geometry,
        geometry_crs=geometry_crs,
        batch_size=1,
    ):
        if not batch.empty:
            record = {
                key: _record_property_value(value)
                for key, value in batch.iloc[0].to_dict().items()
            }
            for column in _INTERNAL_QUERY_COLUMNS:
                record.pop(column, None)
            return record
    return None


def add_dataset(
    input_path: str | Path,
    datasets_dir: str | Path,
    *,
    name: str | None = None,
    overwrite: bool = False,
    **build_kwargs: Any,
):
    """Build a dataset under ``datasets_dir`` from a source file or directory."""
    import starlet

    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input path not found: {source}")
    root = Path(datasets_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset_name = name or source.stem
    outdir = _dataset_child_path(root, dataset_name)
    if outdir.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset already exists: {outdir}")
        shutil.rmtree(outdir)
    return starlet.build(input=str(source), outdir=str(outdir), **build_kwargs)


def delete_dataset(datasets_dir: str | Path, name: str, *, missing_ok: bool = False) -> bool:
    """Delete a dataset directory by name."""
    root = Path(datasets_dir)
    target = _dataset_child_path(root, name)
    if not target.exists():
        if missing_ok:
            return False
        raise FileNotFoundError(f"Dataset not found: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {target}")
    shutil.rmtree(target)
    return True


class AsyncDatasetHandle:
    """Handle returned by :func:`add_dataset_async`."""

    def __init__(
        self,
        *,
        input_path: str | Path,
        datasets_dir: str | Path,
        name: str | None,
        overwrite: bool,
        build_kwargs: dict[str, Any],
    ) -> None:
        source = Path(input_path)
        root = Path(datasets_dir)
        self.input_path = str(source)
        self.datasets_dir = str(root)
        self.dataset_name = name or source.stem
        self.dataset_dir = str(root / self.dataset_name)
        self.started_at: float | None = None
        self.finished_at: float | None = None

        self._name = name
        self._overwrite = overwrite
        self._build_kwargs = dict(build_kwargs)
        self._cancel_requested = threading.Event()
        self._lock = threading.Lock()
        self._status = "pending"
        self._result = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"starlet-add-dataset-{self.dataset_name}",
            daemon=True,
        )

    def start(self) -> "AsyncDatasetHandle":
        self._thread.start()
        return self

    @property
    def status(self) -> str:
        """Current status: pending, running, cancel_requested, cancelled, succeeded, or failed."""
        with self._lock:
            return self._status

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def done(self) -> bool:
        return not self._thread.is_alive() and self.status in {"cancelled", "succeeded", "failed"}

    def cancel(self) -> bool:
        """Request cancellation.

        Thread cancellation is cooperative. If the job has not started, it will
        be cancelled. If the Starlet build is already running, the request is
        recorded and the thread exits when the current build call returns.
        """
        self._cancel_requested.set()
        with self._lock:
            if self._status == "pending":
                self._status = "cancelled"
                return True
            if self._status == "running":
                self._status = "cancel_requested"
                return True
            return False

    def join(self, timeout: float | None = None) -> bool:
        """Wait for completion. Returns ``True`` if the job is done."""
        self._thread.join(timeout)
        return self.done()

    def result(self, timeout: float | None = None):
        """Wait for and return the build result, re-raising background errors."""
        if not self.join(timeout):
            raise TimeoutError("Dataset build is still running")
        with self._lock:
            if self._status == "cancelled":
                raise RuntimeError("Dataset build was cancelled")
            if self._error is not None:
                raise self._error
            return self._result

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible status snapshot."""
        with self._lock:
            error = self._error
            return {
                "status": self._status,
                "dataset": self.dataset_name,
                "dataset_dir": self.dataset_dir,
                "input_path": self.input_path,
                "cancel_requested": self._cancel_requested.is_set(),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": str(error) if error is not None else None,
            }

    def _run(self) -> None:
        with self._lock:
            if self._status == "cancelled" or self._cancel_requested.is_set():
                self._status = "cancelled"
                self.finished_at = time.time()
                return
            self._status = "running"
            self.started_at = time.time()

        try:
            result = add_dataset(
                self.input_path,
                self.datasets_dir,
                name=self._name,
                overwrite=self._overwrite,
                **self._build_kwargs,
            )
        except BaseException as exc:
            with self._lock:
                self._error = exc
                self._status = "failed"
                self.finished_at = time.time()
            return

        with self._lock:
            self._result = result
            self._status = "succeeded"
            self.finished_at = time.time()


def add_dataset_async(
    input_path: str | Path,
    datasets_dir: str | Path,
    *,
    name: str | None = None,
    overwrite: bool = False,
    **build_kwargs: Any,
) -> AsyncDatasetHandle:
    """Start building a dataset in a background thread and return a handle."""
    return AsyncDatasetHandle(
        input_path=input_path,
        datasets_dir=datasets_dir,
        name=name,
        overwrite=overwrite,
        build_kwargs=build_kwargs,
    ).start()


def _load_dataset_stats(dataset_dir: str | Path) -> dict[str, Any] | None:
    stats_path = Path(dataset_dir) / "stats" / "attributes.json"
    if not stats_path.exists():
        return None
    with open(stats_path, "r") as f:
        return json.load(f)


def _dataset_child_path(root: Path, name: str) -> Path:
    raw = Path(name)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise ValueError(f"Invalid dataset name: {name!r}")
    return root / raw


def _summary_bbox(summary: dict[str, Any] | None) -> list[float] | None:
    if not summary:
        return None
    for item in summary.get("geometry") or []:
        bbox = item.get("mbr")
        if bbox and len(bbox) == 4:
            return list(bbox)
    return None


def _stats_bbox(stats: dict[str, Any] | None) -> list[float] | None:
    if stats is None:
        return None
    normalized = _normalize_stats(stats)
    for attr in normalized.get("attributes") or []:
        if not isinstance(attr, dict):
            continue
        bbox = (attr.get("stats") or {}).get("mbr")
        if bbox and len(bbox) == 4:
            return list(bbox)
    return None


def _estimate_batch_size(batch: Any) -> int:
    total = int(batch.drop(columns=["geometry"], errors="ignore").memory_usage(index=False, deep=True).sum())
    if "geometry" in batch:
        for geom in batch.geometry:
            if geom is not None:
                total += len(getattr(geom, "wkb", b""))
    return total


def _drop_internal_query_columns(batch: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    columns = [column for column in _INTERNAL_QUERY_COLUMNS if column in batch.columns]
    if not columns:
        return batch
    return batch.drop(columns=columns)


def _record_property_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _record_property_value(item) for key, item in value.items()}
    if _is_map_entries(value):
        return {key: _record_property_value(item_value) for key, item_value in value}
    if isinstance(value, list):
        return [_record_property_value(item) for item in value]
    return value


def _is_map_entries(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)
        for item in value
    )


def _normalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    attrs = stats.get("attributes")
    if isinstance(attrs, list):
        return stats
    if not isinstance(attrs, dict):
        return stats

    normalized = []
    for name, value in attrs.items():
        if not isinstance(value, dict):
            continue
        stats_payload = value.get("stats")
        if not isinstance(stats_payload, dict):
            stats_payload = {k: v for k, v in value.items() if k != "type"}
        normalized.append({
            "name": name,
            "type": value.get("type"),
            "stats": stats_payload,
        })
    out = dict(stats)
    out["attributes"] = normalized
    return out


def _build_summary(dataset_name: str, stats: dict[str, Any]) -> dict[str, Any]:
    attrs = stats.get("attributes") or []
    geometry = []
    attributes = []
    for attr in attrs:
        if not isinstance(attr, dict):
            continue
        item = _attribute_summary(attr)
        if item["role"] == "geometry":
            geometry.append(item)
        else:
            attributes.append(item)
    return {
        "dataset": dataset_name,
        "description": None,
        "geometry": geometry,
        "attributes": attributes,
        "attribute_count": len(attributes),
        "geometry_attribute_count": len(geometry),
    }


def _attribute_summary(attr: dict[str, Any]) -> dict[str, Any]:
    name = attr.get("name")
    stats = attr.get("stats") or {}
    role = _attribute_role(stats)
    item: dict[str, Any] = {"name": name, "role": role}
    if role == "geometry":
        item["geom_types"] = stats.get("geom_types") or {}
        item["mbr"] = stats.get("mbr")
        item["total_points"] = stats.get("total_points")
        return item
    item["approx_distinct"] = stats.get("approx_distinct")
    item["non_null_count"] = stats.get("non_null_count")
    if role == "numeric":
        item["min"] = stats.get("min")
        item["max"] = stats.get("max")
        item["mean"] = stats.get("mean")
        item["stddev"] = stats.get("stddev")
    elif role in {"text", "categorical_text"}:
        item["avg_length"] = stats.get("avg_length")
        item["min_length"] = stats.get("min_length")
        item["max_length"] = stats.get("max_length")
    item["top_k"] = stats.get("top_k") or []
    return item


def _attribute_role(stats: dict[str, Any]) -> str:
    if "geom_types" in stats or "mbr" in stats:
        return "geometry"
    if any(k in stats for k in ("min", "max", "mean", "stddev")):
        return "numeric"
    if any(k in stats for k in ("avg_length", "min_length", "max_length")):
        approx_distinct = stats.get("approx_distinct")
        if isinstance(approx_distinct, int) and approx_distinct <= 50:
            return "categorical_text"
        return "text"
    return "categorical"


def _rectangle_to_3857(rectangle: Sequence[float], rectangle_crs: str) -> BBox:
    if len(rectangle) != 4:
        raise ValueError("rectangle must be (minx, miny, maxx, maxy)")
    minx, miny, maxx, maxy = [float(v) for v in rectangle]
    if rectangle_crs.upper() == "EPSG:3857":
        return (minx, miny, maxx, maxy)
    from pyproj import Transformer

    transformer = Transformer.from_crs(rectangle_crs, "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform([minx, minx, maxx, maxx], [miny, maxy, miny, maxy])
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def _prefix_sum_rectangle(prefix: Any, bbox_3857: BBox) -> float:
    minx, miny, maxx, maxy = bbox_3857
    gxmin, gymin, gxmax, gymax = GLOBAL_BBOX
    if maxx < gxmin or minx > gxmax or maxy < gymin or miny > gymax:
        return 0.0

    minx = max(minx, gxmin)
    maxx = min(maxx, gxmax)
    miny = max(miny, gymin)
    maxy = min(maxy, gymax)

    n_rows, n_cols = prefix.shape
    col0 = _coord_to_cell(minx, gxmin, gxmax, n_cols)
    col1 = _coord_to_cell(maxx, gxmin, gxmax, n_cols)
    y0 = _coord_to_cell(miny, gymin, gymax, n_rows)
    y1 = _coord_to_cell(maxy, gymin, gymax, n_rows)
    row0 = n_rows - 1 - y1
    row1 = n_rows - 1 - y0
    return _prefix_area(prefix, row0, col0, row1, col1)


def _coord_to_cell(value: float, lower: float, upper: float, cells: int) -> int:
    scaled = (value - lower) / (upper - lower)
    return min(max(int(scaled * cells), 0), cells - 1)


def _prefix_area(prefix: Any, row0: int, col0: int, row1: int, col1: int) -> float:
    total = prefix[row1, col1]
    above = prefix[row0 - 1, col1] if row0 > 0 else 0
    left = prefix[row1, col0 - 1] if col0 > 0 else 0
    corner = prefix[row0 - 1, col0 - 1] if row0 > 0 and col0 > 0 else 0
    return float(total - above - left + corner)
