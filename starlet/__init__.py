"""starlet — spatial tiling, MVT generation, and tile serving for geospatial data."""
from __future__ import annotations

from importlib.metadata import version
__version__ = version("starlet")

from starlet._internal.config import (
    command_parallelism,
    ensure_config_loaded,
    parse_size_value,
    resolve_command_value,
)
from starlet._internal.histogram.io import resolve_histogram_path
from starlet._types import TileResult, MVTResult, Dataset

ensure_config_loaded()

__all__ = [
    "tile",
    "generate_mvt",
    "build",
    "create_app",
    "export_pmtiles",
    "list_datasets",
    "get_tile",
    "get_dataset_metadata",
    "get_dataset_summary",
    "estimate_range_count",
    "query_dataset",
    "query_dataset_count",
    "query_dataset_size",
    "get_sample_record",
    "get_config",
    "add_dataset",
    "delete_dataset",
    "add_dataset_async",
    "AsyncDatasetHandle",
    "set_temp_dir",
    "get_temp_dir",
    "TileResult",
    "MVTResult",
    "Dataset",
]

_GEOJSON_DEFAULT_PARTITION_SIZE = 512 * 1024 * 1024
_GEOPARQUET_DEFAULT_PARTITION_SIZE = 128 * 1024 * 1024


def tile(
    input: str,
    outdir: str,
    *,
    parallelism: int | None = None,
    partition_size: int | None = None,
    sort: str | None = None,
    compression: str | None = None,
    intermediate_compression: str | None = None,
    sample_cap: int | None = None,
    seed: int | None = None,
    geom_col: str | None = None,
    sfc_bits: int | None = None,
    covering_bbox: bool | None = None,
    csv_x_col: str | None = None,
    csv_y_col: str | None = None,
    csv_wkt_col: str | None = None,
    csv_x_index: int | None = None,
    csv_y_index: int | None = None,
    csv_wkt_index: int | None = None,
    csv_split_size: int | str | None = None,
    src_crs: str | None = None,
    temp_dir: str | None = None,
    grid_size: int | None = None,
    histogram_dtype: str | None = None,
) -> TileResult:
    """Partition a supported geospatial source into spatially-tiled Parquet files.

    Parameters
    ----------
    input : str
        Path to a supported source file or directory. GeoLife PLT and GPX
        inputs may be one file or a directory containing matching files.
    outdir : str
        Output directory. Tiled files go into ``<outdir>/parquet_tiles/``
        and histograms into ``<outdir>/histograms/``.
    parallelism : int | None
        Shared worker count used for sampling, two-stage assignment, reducers,
        and histogram construction.
    partition_size : int | None
        Target partition size in bytes. When omitted, defaults to 512 MiB for
        GeoJSON and 128 MiB for GeoParquet. The number of partitions is
        derived from the input file size.
    sort : str
        Row sort order within each tile: ``"zorder"``, ``"hilbert"``,
        ``"columns"``, or ``"none"``.
    compression : str
        Parquet compression codec (default ``"zstd"``).
    intermediate_compression : str
        Compression for two-stage shuffle intermediate files written by
        mappers before the reduce stage merges them: ``"gzip"`` or
        ``"none"`` (default ``"gzip"``).
    sample_cap : int | None
        Reservoir sampling cap for centroid sampling.
    seed : int
        Random seed for RSGrove partitioner.
    geom_col : str
        Name of the geometry column.
    sfc_bits : int
        Bits per axis for Z-order / Hilbert key.
    covering_bbox : bool
        Read-time pruning support. If True, write four per-row bbox covering
        columns plus bounded, spatially-coherent row groups so the on-demand
        tile server can skip row groups/rows at read time. Enabled by default;
        disable when optimizing only for batch tiling speed and smaller files.
    csv_x_col, csv_y_col : str | None
        Column names containing x/y coordinates for CSV inputs. Provide both,
        or provide ``csv_wkt_col`` instead.
    csv_wkt_col : str | None
        Column name containing WKT geometry for CSV inputs.
    csv_x_index, csv_y_index : int | None
        Zero-based x/y column positions for headerless CSV inputs. Provide
        both, or provide ``csv_wkt_index`` instead.
    csv_wkt_index : int | None
        Zero-based WKT column position for a headerless CSV input.
    csv_split_size : int
        Target byte length for each CSV source split.
    src_crs : str
        CRS hint for CSV inputs and other sources without embedded CRS.
    temp_dir : str | None
        Parent directory for two-stage temporary shard files. Defaults to
        ``./tmp`` under the current working directory.
    grid_size : int
        Histogram grid size per axis.
    histogram_dtype : str
        Histogram data type.

    Returns
    -------
    TileResult
    """
    import logging
    import math
    from pathlib import Path

    from starlet._internal.tiling.datasource import read_spatial_sample, source_for_path
    from starlet._internal.tiling.geojson_source import is_geojson_path
    from starlet._internal.tiling.assigner import RSGroveAssigner
    from starlet._internal.tiling.RSGrove import EnvelopeNDLite
    from starlet._internal.tiling.two_stage_orchestrator import TwoStageOrchestrator
    from starlet._internal.tiling.writer_pool import SortMode
    from starlet._internal.histogram.hist_pyramid import build_histograms_for_dir

    logger = logging.getLogger("starlet.tile")
    parallelism = command_parallelism("tile", explicit=parallelism)
    partition_size = parse_size_value(resolve_command_value("tile", "partition_size", partition_size))
    sort = str(resolve_command_value("tile", "sort", sort))
    compression = str(resolve_command_value("tile", "compression", compression))
    intermediate_compression = str(
        resolve_command_value("tile", "intermediate_compression", intermediate_compression)
    )
    sample_cap = resolve_command_value("tile", "sample_cap", sample_cap)
    seed = int(seed if seed is not None else 42)
    geom_col = str(geom_col or "geometry")
    sfc_bits = int(resolve_command_value("tile", "sfc_bits", sfc_bits))
    covering_bbox = bool(True if covering_bbox is None else covering_bbox)
    csv_split_size = parse_size_value(resolve_command_value("tile", "csv_split_size", csv_split_size))
    src_crs = str(src_crs or "EPSG:4326")
    temp_dir = resolve_command_value("tile", "temp_dir", temp_dir)
    grid_size = int(resolve_command_value("tile", "grid_size", grid_size))
    histogram_dtype = str(resolve_command_value("tile", "dtype", histogram_dtype))

    # Parse sort mode
    _sort_map = {
        "none": SortMode.NONE,
        "columns": SortMode.COLUMNS,
        "zorder": SortMode.ZORDER,
        "hilbert": SortMode.HILBERT,
    }
    sort_mode = _sort_map.get(sort.strip().lower(), SortMode.ZORDER)

    # Build data source and choose a format-appropriate default size.
    source = source_for_path(
        input,
        geom_col=geom_col,
        csv_x_col=csv_x_col,
        csv_y_col=csv_y_col,
        csv_wkt_col=csv_wkt_col,
        csv_x_index=csv_x_index,
        csv_y_index=csv_y_index,
        csv_wkt_index=csv_wkt_index,
        csv_split_size=csv_split_size,
        src_crs=src_crs,
    )
    geom_col = getattr(source, "geom_col", geom_col)
    if partition_size is None:
        partition_size = (
            _GEOJSON_DEFAULT_PARTITION_SIZE
            if is_geojson_path(input)
            else _GEOPARQUET_DEFAULT_PARTITION_SIZE
        )

    # Determine partition count
    if partition_size <= 0:
        raise ValueError("partition_size must be greater than zero")
    input_size_bytes = source.input_size_bytes()
    target_partitions = max(1, math.ceil(input_size_bytes / partition_size))
    logger.info(
        "Target partitions: %d (input=%d bytes, target_partition_size=%d bytes)",
        target_partitions,
        input_size_bytes,
        partition_size,
    )

    # Build assigner
    spatial_sample = read_spatial_sample(
        input,
        geom_col=geom_col,
        seed=seed,
        sample_cap=sample_cap,
        csv_x_col=csv_x_col,
        csv_y_col=csv_y_col,
        csv_wkt_col=csv_wkt_col,
        csv_x_index=csv_x_index,
        csv_y_index=csv_y_index,
        csv_wkt_index=csv_wkt_index,
        csv_split_size=csv_split_size,
        src_crs=src_crs,
        geojson_workers=parallelism,
        geoparquet_workers=parallelism,
        source_workers=parallelism,
    )
    if spatial_sample.schema is not None:
        source.set_schema(spatial_sample.schema)
    sample_points = spatial_sample.sample_points
    sample_bounds = EnvelopeNDLite(
        sample_points.min(axis=1),
        sample_points.max(axis=1),
    )
    assigner = RSGroveAssigner.from_sample_and_bounds(
        sample_points=sample_points,
        bounds=sample_bounds,
        num_partitions=target_partitions,
        geom_col=geom_col,
    )

    tiles_dir = str(Path(outdir) / "parquet_tiles")
    hist_dir = str(Path(outdir) / "histograms")

    tiling_orchestrator = TwoStageOrchestrator(
        source=source,
        assigner=assigner,
        outdir=tiles_dir,
        geom_col=geom_col,
        compression=compression,
        intermediate_compression=intermediate_compression,
        sort_mode=sort_mode,
        sfc_bits=sfc_bits,
        parallelism=parallelism,
        temp_dir=temp_dir,
        covering_bbox=covering_bbox,
    )
    tiling_orchestrator.run()

    logger.info("Tiling complete. Building histograms.")
    build_histograms_for_dir(
        tiles_dir=tiles_dir,
        outdir=hist_dir,
        geom_col=geom_col,
        grid_size=grid_size,
        dtype=histogram_dtype,
        parallelism=parallelism,
    )

    # Gather result metadata
    tile_files = list(Path(tiles_dir).glob("*.parquet"))
    total_rows = 0
    for tf in tile_files:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(str(tf))
        total_rows += meta.num_rows

    ds = Dataset(outdir)
    result_bbox = ds.bbox or (0.0, 0.0, 0.0, 0.0)

    return TileResult(
        outdir=outdir,
        num_files=len(tile_files),
        total_rows=total_rows,
        bbox=result_bbox,
        histogram_path=str(resolve_histogram_path(Path(hist_dir) / "global")),
    )


def generate_mvt(
    tile_dir: str,
    *,
    zoom: int | None = None,
    threshold: float | None = None,
    pmtiles: bool | None = None,
    pmtiles_compression: str | None = None,
    outdir: str | None = None,
    parallelism: int | None = None,
    temp_dir: str | None = None,
    feature_capacity: int | None = None,
    mvt_memory_budget: str | int | None = None,
    extent: int | None = None,
    buffer: int | None = None,
    tile_attributes: str | list[str] | None = None,
) -> MVTResult:
    """Generate Mapbox Vector Tiles from a tiled dataset.

    Parameters
    ----------
    tile_dir : str
        Dataset directory containing ``parquet_tiles/`` and ``histograms/``.
    zoom : int
        Maximum zoom level.
    threshold : float
        Minimum feature count per tile.
    pmtiles : bool
        If True, also export the generated tiles to a PMTiles archive.
    pmtiles_compression : str
        Compression for PMTiles export: "gzip", "brotli", "zstd", "none".
    outdir : str | None
        MVT output directory. Defaults to ``<tile_dir>/mvt/``.
    temp_dir : str | None
        Parent directory for temporary MVT map/reduce files. If omitted, uses
        the process-wide Starlet temp directory when configured, otherwise
        ``<tile_dir>/tmp``.
    feature_capacity : int
        Maximum features larger than a display pixel kept per tile.
    tile_attributes : str or list of str, optional
        Attribute columns written into tiles: ``"all"`` (default), ``"none"``
        (geometry only; attributes come from the record lookup) or column names.
        Maximum retained features per intermediate tile reservoir.
    mvt_memory_budget : str | int
        Total memory target for all concurrent MVT mappers, e.g. ``"24gb"``.
        The default ``"auto"`` uses a fraction of available system memory and
        lets each mapper spill partial tiles when its RSS crosses its share.
    extent : int
        Vector tile extent.
    buffer : int
        Vector tile buffer in extent units.

    Returns
    -------
    MVTResult
    """
    from pathlib import Path

    zoom = int(resolve_command_value("mvt", "zoom", zoom))
    threshold = float(resolve_command_value("mvt", "threshold", threshold))
    pmtiles = bool(resolve_command_value("mvt", "pmtiles", pmtiles))
    pmtiles_compression = str(resolve_command_value("mvt", "pmtiles_compression", pmtiles_compression))
    parallelism = command_parallelism("mvt", explicit=parallelism)
    temp_dir = resolve_command_value("mvt", "temp_dir", temp_dir)
    feature_capacity = int(resolve_command_value("mvt", "feature_capacity", feature_capacity))
    mvt_memory_budget = resolve_command_value("mvt", "mvt_memory_budget", mvt_memory_budget)
    extent = int(resolve_command_value("mvt", "extent", extent))
    buffer = int(resolve_command_value("mvt", "buffer", buffer))

    if extent <= 0:
        raise ValueError("extent must be positive")
    dataset_path = Path(tile_dir)
    mvt_outdir = outdir or str(dataset_path / "mvt")
    pmtiles_path = str(dataset_path / "tiles.pmtiles") if pmtiles else None

    from starlet._internal.mvt.mvt_generator import DatasetMVTGenerator

    result = DatasetMVTGenerator(
        tile_dir,
        num_zoom_levels=zoom + 1,
        threshold=threshold,
        output_format="pmtiles" if pmtiles else "mvt",
        outdir=mvt_outdir,
        pmtiles_path=pmtiles_path,
        pmtiles_compression=pmtiles_compression,
        workers=parallelism,
        temp_dir=temp_dir,
        feature_capacity=feature_capacity,
        mvt_memory_budget=mvt_memory_budget,
        extent=extent,
        buffer=buffer,
        tile_attributes=tile_attributes,
    ).run()

    generated_pmtiles_path = getattr(result, "pmtiles_path", None)
    tile_counts_by_zoom = list(getattr(result, "tile_counts_by_zoom", []))
    tile_count = int(getattr(result, "tile_count", sum(tile_counts_by_zoom)))
    zoom_levels = list(getattr(result, "zoom_levels", [z for z, count in enumerate(tile_counts_by_zoom) if count > 0]))

    return MVTResult(
        outdir=mvt_outdir,
        zoom_levels=zoom_levels,
        tile_counts_by_zoom=tile_counts_by_zoom,
        tile_count=tile_count,
        pmtiles_path=generated_pmtiles_path,
    )


def build(
    input: str,
    outdir: str,
    *,
    zoom: int | None = None,
    parallelism: int | None = None,
    partition_size: int | None = None,
    threshold: float | None = None,
    pmtiles: bool | None = None,
    pmtiles_compression: str | None = None,
    temp_dir: str | None = None,
    feature_capacity: int | None = None,
    mvt_memory_budget: str | int | None = None,
    extent: int | None = None,
    buffer: int | None = None,
    tile_attributes: str | list[str] | None = None,
    **tile_kwargs,
) -> tuple[TileResult, MVTResult, str | None]:
    """Run the full pipeline: tile then generate MVTs.

    Parameters
    ----------
    input : str
        Path to a supported source file or directory.
    outdir : str
        Output dataset directory.
    zoom : int
        Maximum zoom level for MVT generation.
    parallelism : int | None
        Shared worker count used across tiling and MVT generation.
    partition_size : int | None
        Target partition size in bytes (forwarded to :func:`tile`). When
        omitted, a format-appropriate default is used.
    threshold : float
        Minimum feature count per MVT tile.
    pmtiles : bool
        If True, export MVT tiles to a PMTiles archive after generation.
        Default False.
    pmtiles_compression : str
        Compression for PMTiles export: "gzip", "brotli", "zstd", "none".
        Default "gzip". Only used if pmtiles=True.
    temp_dir : str | None
        Parent directory for temporary files used by all build steps. Explicit
        values override the process-wide Starlet temp directory.
    feature_capacity : int
        Maximum features larger than a display pixel kept per tile.
    tile_attributes : str or list of str, optional
        Attribute columns written into tiles: ``"all"`` (default), ``"none"``
        (geometry only; attributes come from the record lookup) or column names.
        Maximum retained features per intermediate tile reservoir.
    mvt_memory_budget : str | int
        Total memory target for all concurrent MVT mappers, e.g. ``"24gb"``.
        The default ``"auto"`` uses a fraction of available system memory.
    extent : int
        Vector tile extent.
    buffer : int
        Vector tile buffer in extent units.
    **tile_kwargs
        Additional keyword arguments forwarded to :func:`tile`
        (e.g. ``covering_bbox=False``).

    Returns
    -------
    tuple[TileResult, MVTResult, str | None]
        Returns (tile_result, mvt_result, pmtiles_path).
        pmtiles_path is None if pmtiles=False.
    """
    if "temp_dir" in tile_kwargs:
        if temp_dir is not None:
            raise ValueError("temp_dir was provided twice")
        temp_dir = tile_kwargs.pop("temp_dir")

    parallelism = command_parallelism("build", explicit=parallelism, fallback_sections=("tile", "mvt"))
    partition_size = parse_size_value(
        resolve_command_value("build", "partition_size", partition_size, fallback_sections=("tile",))
    )
    zoom = int(resolve_command_value("build", "zoom", zoom, fallback_sections=("mvt",)))
    threshold = float(resolve_command_value("build", "threshold", threshold, fallback_sections=("mvt",)))
    pmtiles = bool(resolve_command_value("build", "pmtiles", pmtiles, fallback_sections=("mvt",)))
    pmtiles_compression = str(
        resolve_command_value("build", "pmtiles_compression", pmtiles_compression, fallback_sections=("mvt",))
    )
    temp_dir = resolve_command_value("build", "temp_dir", temp_dir)
    feature_capacity = int(
        resolve_command_value("build", "feature_capacity", feature_capacity, fallback_sections=("mvt",))
    )
    mvt_memory_budget = resolve_command_value(
        "build",
        "mvt_memory_budget",
        mvt_memory_budget,
        fallback_sections=("mvt",),
    )
    extent = int(resolve_command_value("build", "extent", extent, fallback_sections=("mvt",)))
    buffer = int(resolve_command_value("build", "buffer", buffer, fallback_sections=("mvt",)))
    tile_kwargs.setdefault(
        "sort",
        str(resolve_command_value("build", "sort", tile_kwargs.get("sort"), fallback_sections=("tile",))),
    )
    tile_kwargs.setdefault(
        "compression",
        str(
            resolve_command_value(
                "build",
                "compression",
                tile_kwargs.get("compression"),
                fallback_sections=("tile",),
            )
        ),
    )
    tile_kwargs.setdefault(
        "intermediate_compression",
        str(
            resolve_command_value(
                "build",
                "intermediate_compression",
                tile_kwargs.get("intermediate_compression"),
                fallback_sections=("tile",),
            )
        ),
    )
    tile_kwargs.setdefault(
        "sample_cap",
        resolve_command_value("build", "sample_cap", tile_kwargs.get("sample_cap"), fallback_sections=("tile",)),
    )
    tile_kwargs.setdefault(
        "csv_split_size",
        parse_size_value(
            resolve_command_value(
                "build",
                "csv_split_size",
                tile_kwargs.get("csv_split_size"),
                fallback_sections=("tile",),
            )
        ),
    )
    tile_kwargs.setdefault(
        "sfc_bits",
        int(resolve_command_value("build", "sfc_bits", tile_kwargs.get("sfc_bits"), fallback_sections=("tile",))),
    )
    tile_kwargs.setdefault(
        "grid_size",
        int(resolve_command_value("build", "grid_size", tile_kwargs.get("grid_size"), fallback_sections=("tile",))),
    )
    tile_kwargs.setdefault(
        "histogram_dtype",
        str(resolve_command_value("build", "dtype", tile_kwargs.get("histogram_dtype"), fallback_sections=("tile",))),
    )

    tile_result = tile(
        input=input,
        outdir=outdir,
        parallelism=parallelism,
        partition_size=partition_size,
        temp_dir=temp_dir,
        **tile_kwargs,
    )
    mvt_result = generate_mvt(
        tile_dir=outdir,
        zoom=zoom,
        threshold=threshold,
        pmtiles=pmtiles,
        pmtiles_compression=pmtiles_compression,
        temp_dir=temp_dir,
        parallelism=parallelism,
        feature_capacity=feature_capacity,
        mvt_memory_budget=mvt_memory_budget,
        extent=extent,
        buffer=buffer,
        tile_attributes=tile_attributes,
    )

    return tile_result, mvt_result, mvt_result.pmtiles_path


def export_pmtiles(
    mvt_dir: str,
    output_path: str,
    tile_type: str | None = None,
    compression: str | None = None,
) -> str:
    """Export MVT tiles to PMTiles archive format.

    Parameters
    ----------
    mvt_dir : str
        Directory containing MVT tiles in z/x/y.mvt structure.
        Typically ``<dataset>/mvt/``.
    output_path : str
        Path to output .pmtiles file.
    tile_type : str
        Tile type: "mvt" (vector), "png", "jpg", "webp" (raster).
        Default "mvt".
    compression : str
        Compression: "gzip", "none", "brotli", "zstd".
        Default "gzip".

    Returns
    -------
    str
        Path to created PMTiles file.

    Examples
    --------
    >>> # After running build/generate_mvt
    >>> export_pmtiles(
    ...     mvt_dir="datasets/mydata/mvt",
    ...     output_path="datasets/mydata/tiles.pmtiles"
    ... )
    """
    from starlet._internal.pmtiles.exporter import export_to_pmtiles

    tile_type = tile_type or "mvt"
    compression = compression or str(resolve_command_value("mvt", "pmtiles_compression", None))
    return export_to_pmtiles(mvt_dir, output_path, tile_type, compression)


def create_app(
    data_dir: str,
    cache_size: int | None = None,
    extent: int | None = None,
    buffer: int | None = None,
    tile_attributes: str | list[str] | None = None,
):
    """Create a Flask tile server application.

    Parameters
    ----------
    data_dir : str
        Root directory containing dataset subdirectories.
    cache_size : int, optional
        Number of tiles in the in-memory LRU cache. When omitted, Starlet uses
        the configured ``serve.cache_size`` value, or the built-in default.
    tile_attributes : str or list of str, optional
        Which attribute columns on-the-fly tiles carry: ``"all"`` (default),
        ``"none"`` (geometry only; use the record lookup for attributes) or a
        list / comma-separated string of column names.
    Returns
    -------
    Flask
        Configured Flask application.
    """
    from starlet._internal.server.app import create_app as _create_app
    return _create_app(
        data_dir=data_dir,
        cache_size=cache_size,
        extent=extent,
        buffer=buffer,
        tile_attributes=tile_attributes,
    )


def set_temp_dir(path: str | None):
    """Set the process-wide parent directory for Starlet temporary files."""
    from starlet._internal.config import set_temp_dir as _set_temp_dir
    return _set_temp_dir(path)


def get_temp_dir():
    """Return the configured process-wide Starlet temp directory, if any."""
    from starlet._internal.config import get_temp_dir as _get_temp_dir
    return _get_temp_dir()


from starlet.api import (  # noqa: E402
    AsyncDatasetHandle,
    add_dataset,
    add_dataset_async,
    delete_dataset,
    estimate_range_count,
    get_dataset_metadata,
    get_dataset_summary,
    get_config,
    get_tile,
    get_sample_record,
    list_datasets,
    query_dataset,
    query_dataset_count,
    query_dataset_size,
)
