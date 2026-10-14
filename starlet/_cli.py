"""Click CLI for starlet: spatial tiling, MVT generation, and tile serving."""
from __future__ import annotations

import logging
from pathlib import Path
import sys

import click

from starlet._internal.config import (
    command_parallelism,
    load_config,
    parse_size_value,
    resolve_command_value,
    set_loaded_config,
)


def _setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(relativeCreated).0fms] %(levelname)s %(name)s: %(message)s",
    )


def _resolved_log_level(command: str, explicit: str | None) -> str:
    return str(resolve_command_value(command, "log_level", explicit))


def _format_zoom_counts(result: object) -> str:
    counts = getattr(result, "tile_counts_by_zoom", None)
    if counts:
        counts_list = list(counts)
        return f"{counts_list} total={sum(counts_list)}"
    zoom_levels = getattr(result, "zoom_levels", None)
    if zoom_levels:
        return str(list(zoom_levels))
    return "(no MVTs)"


@click.group()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=str),
    help="Path to a Starlet TOML config file.",
)
@click.pass_context
@click.version_option(package_name="starlet")
def main(ctx: click.Context, config_path: str | None):
    """starlet — spatial tiling, MVT generation, and tile serving."""
    config = load_config(config_path)
    set_loaded_config(config, config_path)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@main.command()
@click.option("--input", "input_path", required=True, help="Path to a supported geospatial source.")
@click.option("--outdir", required=True, help="Output dataset directory.")
@click.option("--parallelism", type=int, default=None, help="Shared worker count used across tiling steps.")
@click.option("--temp-dir", default=None, help="Parent directory for temporary tiling files.")
@click.option("--partition-size", default=None, help="Target partition size for tiling (e.g. 128mb).")
@click.option("--sort", default=None, help="Row sort order within each tile.")
@click.option("--compression", default=None, help="Parquet compression codec.")
@click.option(
    "--tile-intermediate-compression",
    "intermediate_compression",
    default=None,
    help="Compression for two-stage shuffle intermediate files: none or gzip.",
)
@click.option("--sample-cap", type=int, default=None, help="Reservoir sampling cap for centroid sampling.")
@click.option("--csv-split-size", default=None, help="Target byte length for each CSV source split.")
@click.option("--grid-size", type=int, default=None, help="Histogram grid size per axis.")
@click.option("--dtype", "histogram_dtype", default=None, help="Histogram data type.")
@click.option("--sfc-bits", type=int, default=None, help="Bits per axis for Z-order / Hilbert key.")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for partitioner.")
@click.option("--geom-col", default="geometry", show_default=True, help="Geometry column name.")
@click.option("--csv-x-col", default=None, help="CSV x-coordinate column. Use with --csv-y-col.")
@click.option("--csv-y-col", default=None, help="CSV y-coordinate column. Use with --csv-x-col.")
@click.option("--csv-wkt-col", default=None, help="CSV WKT geometry column.")
@click.option("--csv-x-index", type=int, default=None, help="Headerless CSV x-coordinate column index. Use with --csv-y-index.")
@click.option("--csv-y-index", type=int, default=None, help="Headerless CSV y-coordinate column index. Use with --csv-x-index.")
@click.option("--csv-wkt-index", type=int, default=None, help="Headerless CSV WKT geometry column index.")
@click.option("--src-crs", default="EPSG:4326", show_default=True, help="Source CRS hint for CSV inputs.")
@click.option("--covering-bbox/--no-covering-bbox", default=True, show_default=True,
              help="Write per-row bbox covering columns + bounded row groups for "
                   "fast on-demand serving. Use --no-covering-bbox to disable.")
@click.option("--log-level", default=None, help="Logging level.")
def tile(
    input_path,
    outdir,
    parallelism,
    temp_dir,
    partition_size,
    sort,
    compression,
    intermediate_compression,
    sample_cap,
    csv_split_size,
    grid_size,
    histogram_dtype,
    sfc_bits,
    seed,
    geom_col,
    csv_x_col,
    csv_y_col,
    csv_wkt_col,
    csv_x_index,
    csv_y_index,
    csv_wkt_index,
    src_crs,
    covering_bbox,
    log_level,
):
    """Partition a geospatial dataset into spatially-tiled Parquet files."""
    _setup_logging(_resolved_log_level("tile", log_level))
    import starlet

    result = starlet.tile(
        input=input_path,
        outdir=outdir,
        parallelism=command_parallelism("tile", explicit=parallelism),
        partition_size=parse_size_value(resolve_command_value("tile", "partition_size", partition_size)),
        sort=str(resolve_command_value("tile", "sort", sort)),
        compression=str(resolve_command_value("tile", "compression", compression)),
        intermediate_compression=str(
            resolve_command_value("tile", "intermediate_compression", intermediate_compression)
        ),
        sample_cap=resolve_command_value("tile", "sample_cap", sample_cap),
        seed=seed,
        geom_col=geom_col,
        csv_x_col=csv_x_col,
        csv_y_col=csv_y_col,
        csv_wkt_col=csv_wkt_col,
        csv_x_index=csv_x_index,
        csv_y_index=csv_y_index,
        csv_wkt_index=csv_wkt_index,
        csv_split_size=parse_size_value(resolve_command_value("tile", "csv_split_size", csv_split_size)),
        src_crs=src_crs,
        sfc_bits=int(resolve_command_value("tile", "sfc_bits", sfc_bits)),
        covering_bbox=covering_bbox,
        temp_dir=resolve_command_value("tile", "temp_dir", temp_dir),
        grid_size=int(resolve_command_value("tile", "grid_size", grid_size)),
        histogram_dtype=str(resolve_command_value("tile", "dtype", histogram_dtype)),
    )
    click.echo(f"Tiling complete: {result.num_files} tiles, {result.total_rows} rows")
    click.echo(f"  Output: {result.outdir}")
    click.echo(f"  Histogram: {result.histogram_path}")


@main.command()
@click.option("--dir", "tile_dir", required=True, help="Dataset directory with parquet_tiles/ and histograms/.")
@click.option("--zoom", type=int, default=None, help="Maximum zoom level.")
@click.option("--outdir", default=None, help="MVT output directory (default: <dir>/mvt/).")
@click.option("--threshold", type=float, default=None, help="Minimum feature threshold.")
@click.option("--parallelism", type=int, default=None, help="Shared worker count used for MVT generation.")
@click.option("--temp-dir", default=None, help="Parent directory for temporary MVT files.")
@click.option("--feature-capacity", type=int, default=None, help="Maximum features larger than a display pixel kept per tile (default 10000).")
@click.option("--tile-attributes", default=None, help="Attribute columns written into tiles: all (default), none, or a comma-separated list.")
@click.option("--mvt-memory-budget", default=None, help="Total memory target for concurrent MVT mappers (e.g. auto, 24gb, none).")
@click.option("--extent", type=int, default=None, help="Vector tile extent.")
@click.option("--buffer", type=int, default=None, help="Vector tile buffer in extent units.")
@click.option("--pmtiles-compression", default=None, help="Compression for PMTiles export.")
@click.option("--pmtiles/--no-pmtiles", default=None, help="Export generated tiles to a PMTiles archive.")
@click.option("--log-level", default=None, help="Logging level.")
def mvt(tile_dir, zoom, outdir, threshold, parallelism, temp_dir, feature_capacity, tile_attributes, mvt_memory_budget, extent, buffer, pmtiles_compression, pmtiles, log_level):
    """Generate Mapbox Vector Tiles from a tiled dataset."""
    _setup_logging(_resolved_log_level("mvt", log_level))
    import starlet

    result = starlet.generate_mvt(
        tile_dir=tile_dir,
        zoom=int(resolve_command_value("mvt", "zoom", zoom)),
        threshold=float(resolve_command_value("mvt", "threshold", threshold)),
        pmtiles=bool(resolve_command_value("mvt", "pmtiles", pmtiles)),
        pmtiles_compression=str(resolve_command_value("mvt", "pmtiles_compression", pmtiles_compression)),
        outdir=outdir,
        temp_dir=resolve_command_value("mvt", "temp_dir", temp_dir),
        parallelism=command_parallelism("mvt", explicit=parallelism),
        feature_capacity=int(resolve_command_value("mvt", "feature_capacity", feature_capacity)),
        tile_attributes=resolve_command_value("mvt", "tile_attributes", tile_attributes),
        mvt_memory_budget=resolve_command_value("mvt", "mvt_memory_budget", mvt_memory_budget),
        extent=int(resolve_command_value("mvt", "extent", extent)),
        buffer=int(resolve_command_value("mvt", "buffer", buffer)),
    )
    click.echo(f"MVT generation complete: {result.tile_count} tiles")
    output_path = result.pmtiles_path if result.pmtiles_path and not Path(result.outdir).exists() else result.outdir
    click.echo(f"  Output: {output_path}")
    click.echo(f"  Zoom levels: {_format_zoom_counts(result)}")
    if result.pmtiles_path:
        click.echo(f"  PMTiles: {result.pmtiles_path}")


@main.command()
@click.option("--input", "input_path", required=True, help="Path to a supported geospatial source.")
@click.option("--outdir", required=True, help="Output dataset directory.")
@click.option("--zoom", type=int, default=None, help="Maximum zoom level.")
@click.option("--parallelism", type=int, default=None, help="Shared worker count used across build steps.")
@click.option("--temp-dir", default=None, help="Parent directory for temporary build files.")
@click.option("--partition-size", default=None, help="Target partition size for tiling (e.g. 128mb).")
@click.option("--sort", default=None, help="Row sort order within each tile.")
@click.option("--compression", default=None, help="Parquet compression codec.")
@click.option(
    "--tile-intermediate-compression",
    "intermediate_compression",
    default=None,
    help="Compression for two-stage shuffle intermediate files: none or gzip.",
)
@click.option("--sample-cap", type=int, default=None, help="Reservoir sampling cap for centroid sampling.")
@click.option("--csv-split-size", default=None, help="Target byte length for each CSV source split.")
@click.option("--grid-size", type=int, default=None, help="Histogram grid size per axis.")
@click.option("--dtype", "histogram_dtype", default=None, help="Histogram data type.")
@click.option("--sfc-bits", type=int, default=None, help="Bits per axis for Z-order / Hilbert key.")
@click.option("--threshold", type=float, default=None, help="Minimum feature threshold.")
@click.option("--feature-capacity", type=int, default=None, help="Maximum features larger than a display pixel kept per tile (default 10000).")
@click.option("--tile-attributes", default=None, help="Attribute columns written into tiles: all (default), none, or a comma-separated list.")
@click.option("--mvt-memory-budget", default=None, help="Total memory target for concurrent MVT mappers (e.g. auto, 24gb, none).")
@click.option("--extent", type=int, default=None, help="Vector tile extent.")
@click.option("--buffer", type=int, default=None, help="Vector tile buffer in extent units.")
@click.option("--pmtiles-compression", default=None, help="Compression for PMTiles export.")
@click.option("--covering-bbox/--no-covering-bbox", default=True, show_default=True,
              help="Write per-row bbox covering columns for fast on-demand serving. "
                   "Use --no-covering-bbox to disable.")
@click.option("--csv-x-col", default=None, help="CSV x-coordinate column. Use with --csv-y-col.")
@click.option("--csv-y-col", default=None, help="CSV y-coordinate column. Use with --csv-x-col.")
@click.option("--csv-wkt-col", default=None, help="CSV WKT geometry column.")
@click.option("--csv-x-index", type=int, default=None, help="Headerless CSV x-coordinate column index. Use with --csv-y-index.")
@click.option("--csv-y-index", type=int, default=None, help="Headerless CSV y-coordinate column index. Use with --csv-x-index.")
@click.option("--csv-wkt-index", type=int, default=None, help="Headerless CSV WKT geometry column index.")
@click.option("--src-crs", default="EPSG:4326", show_default=True, help="Source CRS hint for CSV inputs.")
@click.option("--pmtiles/--no-pmtiles", default=None, help="Export MVT tiles to a PMTiles archive after generation.")
@click.option("--log-level", default=None, help="Logging level.")
def build(
    input_path,
    outdir,
    zoom,
    parallelism,
    temp_dir,
    partition_size,
    sort,
    compression,
    intermediate_compression,
    sample_cap,
    csv_split_size,
    grid_size,
    histogram_dtype,
    sfc_bits,
    threshold,
    feature_capacity,
    tile_attributes,
    mvt_memory_budget,
    extent,
    buffer,
    pmtiles_compression,
    covering_bbox,
    csv_x_col,
    csv_y_col,
    csv_wkt_col,
    csv_x_index,
    csv_y_index,
    csv_wkt_index,
    src_crs,
    pmtiles,
    log_level,
):
    """Run the full pipeline: tile then generate MVTs."""
    _setup_logging(_resolved_log_level("build", log_level))
    import starlet

    parallelism = command_parallelism("build", explicit=parallelism, fallback_sections=("tile", "mvt"))
    tile_result, mvt_result, pmtiles_path = starlet.build(
        input=input_path,
        outdir=outdir,
        zoom=int(resolve_command_value("build", "zoom", zoom, fallback_sections=("mvt",))),
        partition_size=parse_size_value(resolve_command_value("build", "partition_size", partition_size, fallback_sections=("tile",))),
        threshold=float(resolve_command_value("build", "threshold", threshold, fallback_sections=("mvt",))),
        pmtiles=bool(resolve_command_value("build", "pmtiles", pmtiles, fallback_sections=("mvt",))),
        pmtiles_compression=str(resolve_command_value("build", "pmtiles_compression", pmtiles_compression, fallback_sections=("mvt",))),
        temp_dir=resolve_command_value("build", "temp_dir", temp_dir),
        parallelism=parallelism,
        feature_capacity=int(resolve_command_value("build", "feature_capacity", feature_capacity, fallback_sections=("mvt",))),
        tile_attributes=resolve_command_value("build", "tile_attributes", tile_attributes, fallback_sections=("mvt",)),
        mvt_memory_budget=resolve_command_value("build", "mvt_memory_budget", mvt_memory_budget, fallback_sections=("mvt",)),
        extent=int(resolve_command_value("build", "extent", extent, fallback_sections=("mvt",))),
        buffer=int(resolve_command_value("build", "buffer", buffer, fallback_sections=("mvt",))),
        sort=str(resolve_command_value("build", "sort", sort, fallback_sections=("tile",))),
        compression=str(resolve_command_value("build", "compression", compression, fallback_sections=("tile",))),
        intermediate_compression=str(
            resolve_command_value(
                "build", "intermediate_compression", intermediate_compression, fallback_sections=("tile",)
            )
        ),
        sample_cap=resolve_command_value("build", "sample_cap", sample_cap, fallback_sections=("tile",)),
        csv_x_col=csv_x_col,
        csv_y_col=csv_y_col,
        csv_wkt_col=csv_wkt_col,
        csv_x_index=csv_x_index,
        csv_y_index=csv_y_index,
        csv_wkt_index=csv_wkt_index,
        csv_split_size=parse_size_value(resolve_command_value("build", "csv_split_size", csv_split_size, fallback_sections=("tile",))),
        src_crs=src_crs,
        sfc_bits=int(resolve_command_value("build", "sfc_bits", sfc_bits, fallback_sections=("tile",))),
        covering_bbox=covering_bbox,
        grid_size=int(resolve_command_value("build", "grid_size", grid_size, fallback_sections=("tile",))),
        histogram_dtype=str(resolve_command_value("build", "dtype", histogram_dtype, fallback_sections=("tile",))),
    )
    click.echo("Build complete:")
    click.echo(f"  Tiles: {tile_result.num_files} files, {tile_result.total_rows} rows")
    click.echo(f"  MVTs: {mvt_result.tile_count} tiles across zoom levels {_format_zoom_counts(mvt_result)}")
    if pmtiles_path:
        click.echo(f"  PMTiles: {pmtiles_path}")


@main.command()
@click.option("--dir", "data_dir", required=True, help="Root directory containing dataset subdirectories.")
@click.option("--host", default=None, help="Server host.")
@click.option("--port", type=int, default=None, help="Server port.")
@click.option("--cache-size", type=int, default=None, help="Number of tiles to keep in the in-memory cache.")
@click.option("--tile-attributes", default=None, help="Attribute columns in on-the-fly tiles: all (default), none, or a comma-separated list.")
@click.option("--log-level", default=None, help="Logging level.")
def serve(data_dir, host, port, cache_size, tile_attributes, log_level):
    """Launch the tile server."""
    _setup_logging(_resolved_log_level("serve", log_level))
    import starlet

    host = str(resolve_command_value("serve", "host", host))
    port = int(resolve_command_value("serve", "port", port))
    cache_size = int(resolve_command_value("serve", "cache_size", cache_size))
    app = starlet.create_app(
        data_dir=data_dir,
        cache_size=cache_size,
        tile_attributes=resolve_command_value("serve", "tile_attributes", tile_attributes, fallback_sections=("mvt",)),
    )
    click.echo(f"Starting starlet server on {host}:{port}")
    click.echo(f"  Data root: {data_dir}")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


@main.command()
@click.option("--dir", "data_dir", required=True, help="Dataset directory to inspect.")
def info(data_dir):
    """Print dataset metadata summary."""
    import starlet

    try:
        ds = starlet.Dataset(data_dir)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Dataset: {Path(data_dir).name}")
    click.echo(f"  Path:        {ds.path}")
    click.echo(f"  Tiles:       {ds.num_tiles}")
    click.echo(f"  Parquet bbox:{'yes' if ds.parquet_has_bbox else 'no'}")
    click.echo(f"  Parquet CRS: {ds.parquet_crs or '(unknown)'}")
    click.echo(f"  BBox:        {ds.bbox}")
    tile_counts = ds.tile_counts_by_zoom
    if tile_counts:
        click.echo(f"  Zoom levels: {tile_counts} total={sum(tile_counts)}")
    else:
        click.echo("  Zoom levels: (no MVTs)")
    click.echo(f"  Histograms:  {'yes' if ds.has_histograms else 'no'}")
    if ds.has_histograms:
        click.echo(f"  Hist res:    {ds.histogram_resolution or '(unknown)'}")
    click.echo(f"  MVTs:        {'yes' if (ds.has_mvt or ds.has_pmtiles) else 'no'}")
    if tile_counts:
        click.echo(f"  MVT count:   {sum(tile_counts)}")
    click.echo(f"  Stats:       {'yes' if ds.has_stats else 'no'}")

    total_bytes = sum(f.stat().st_size for f in Path(data_dir).rglob("*") if f.is_file())
    if total_bytes < 1024 ** 2:
        size_str = f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024 ** 3:
        size_str = f"{total_bytes / 1024 ** 2:.1f} MB"
    else:
        size_str = f"{total_bytes / 1024 ** 3:.2f} GB"
    click.echo(f"  Total size:  {size_str}")
