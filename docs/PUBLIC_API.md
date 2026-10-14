# Starlet Public API Integration Guide

## What Starlet Does

Starlet builds and serves spatial datasets. A Starlet dataset directory usually
contains:

```text
<dataset>/
  parquet_tiles/        # spatially partitioned GeoParquet files
  histograms/           # global.npy and/or global_prefix.npy
  mvt/                  # optional pre-generated vector tiles in z/x/y.mvt layout
  stats/attributes.json # optional attribute and geometry statistics
```

Use a datasets root directory, for example `datasets/`, to hold multiple
dataset directories:

```text
datasets/
  postal_codes/
  counties/
  roads/
```

## Import

```python
import starlet
```

All functions below are available directly from the `starlet` package.

## Coordinate And Geometry Conventions

- Bounding boxes are tuples: `(minx, miny, maxx, maxy)`.
- Query geometries can be:
  - a bounding box tuple,
  - a GeoJSON geometry dictionary,
  - a Shapely geometry.
- Query geometry CRS defaults to `EPSG:4326`.
- Histogram rectangles default to `EPSG:4326`; pass `rectangle_crs="EPSG:3857"`
  for Web Mercator coordinates.
- Query results are GeoPandas `GeoDataFrame` batches in `EPSG:4326`.

## Configuration

```python
config: dict = starlet.get_config()
```

Returns the current process-wide Starlet configuration after applying built-in
defaults and any discovered or explicitly loaded config file values. The
returned dictionary is a copy, so changing it does not mutate Starlet's active
configuration.

## Dataset Discovery

```python
names: list[str] = starlet.list_datasets("datasets")
```

Returns sorted child directory names under the datasets root. If the root does
not exist, returns an empty list. If the path exists but is not a directory,
raises `NotADirectoryError`.

## Dataset Metadata

```python
metadata: dict = starlet.get_dataset_metadata("datasets/postal_codes")
```

Returns cheap JSON-compatible metadata. Use this instead of reading the raw
stats file directly.

Expected keys:

```python
{
    "name": "postal_codes",
    "path": "datasets/postal_codes",
    "exists": True,
    "size_bytes": 123456,
    "file_count": 42,
    "parquet_tile_count": 30,
    "bbox": [-180.0, -90.0, 180.0, 90.0],  # or None
    "zoom_levels": [0, 1, 2, 3],
    "has_histograms": True,
    "has_mvt": True,
    "has_stats": True,
    "has_summary": True,
    "missing": [],
}
```

The `missing` list can contain:

```python
["dataset_dir", "parquet_tiles", "histograms", "stats"]
```

## Dataset Summary

```python
summary: dict | None = starlet.get_dataset_summary("datasets/postal_codes")
```

Returns a JSON-compatible summary or `None`. Starlet checks:

1. `<dataset>/summary.json`
2. `<dataset>/stats/summary.json`
3. A summary derived from `<dataset>/stats/attributes.json`

The derived shape is:

```python
{
    "dataset": "postal_codes",
    "description": None,
    "geometry": [
        {
            "name": "geometry",
            "role": "geometry",
            "geom_types": {"Polygon": 123},
            "mbr": [-180.0, -90.0, 180.0, 90.0],
            "total_points": 123456,
        }
    ],
    "attributes": [
        {
            "name": "name",
            "role": "text",
            "approx_distinct": 1000,
            "non_null_count": 1000,
            "top_k": [],
        }
    ],
    "attribute_count": 1,
    "geometry_attribute_count": 1,
}
```

## Tiles

```python
tile_bytes: bytes = starlet.get_tile(
    "datasets/postal_codes",
    z=7,
    x=22,
    y=49,
)
```

Returns a Mapbox Vector Tile payload as `bytes`.

To limit attributes when Starlet generates a tile on the fly, pass an attribute
whitelist. Geometry is always included. If `attributes` is omitted, all
attributes are included by default. Pre-generated PMTiles and MVT files are
returned as-is even when `attributes` is provided.

```python
tile_bytes = starlet.get_tile(
    "datasets/postal_codes",
    z=7,
    x=22,
    y=49,
    attributes=["name", "population"],
)
```

Pass an output dictionary to receive details about how the tile was served:

```python
info = {}
tile_bytes = starlet.get_tile(
    "datasets/postal_codes",
    z=7,
    x=22,
    y=49,
    output=info,
)

# Example:
# {
#     "source": "disk",  # "disk", "generated", or "memory"
#     "generation": "read_from_disk",
#     "path": "datasets/postal_codes/mvt/7/22/49.mvt",
#     "elapsed_ms": 0.4,
# }
```

For on-the-fly tiles, Starlet also reports `feature_count`, using the features
already assembled during generation. Disk and memory-cache hits do not decode
the MVT payload for counts, so they only report cheap metadata such as source,
path, and elapsed time.

Lookup behavior:

1. Check pre-generated PMTiles and `<dataset>/mvt/<z>/<x>/<y>.mvt`.
2. If missing, read matching GeoParquet partitions and generate the MVT on the
   fly.
3. If `attributes` is provided for an on-the-fly tile, only those attributes
   are read from GeoParquet and encoded.
4. Generated tiles are not persisted to disk.

Typical web response:

```python
from flask import Response, request

@app.get("/tiles/<dataset>/<int:z>/<int:x>/<int:y>.mvt")
def tile(dataset: str, z: int, x: int, y: int):
    attrs = request.args.get("attributes")
    attributes = [a.strip() for a in attrs.split(",") if a.strip()] if attrs else None
    data = starlet.get_tile(f"datasets/{dataset}", z, x, y, attributes=attributes)
    return Response(data, mimetype="application/vnd.mapbox-vector-tile")
```

The built-in REST server accepts the same whitelist as a comma-separated
query parameter, for example
`/postal_codes/7/22/49.mvt?attributes=name,population`.

## Histogram Estimate

```python
estimate: float = starlet.estimate_range_count(
    "datasets/postal_codes",
    (-125.0, 24.0, -66.0, 50.0),
)
```

Returns a histogram-based estimate for the amount of data in a rectangle. This
is approximate and fast. It uses `histograms/global_prefix.npy` if present, or
`histograms/global.npy` as a fallback.

Use Web Mercator input like this:

```python
estimate = starlet.estimate_range_count(
    "datasets/postal_codes",
    (-20037508.34, -20037508.34, 20037508.34, 20037508.34),
    rectangle_crs="EPSG:3857",
)
```

## Streaming Query

```python
for batch in starlet.query_dataset(
    "datasets/postal_codes",
    (-125.0, 24.0, -66.0, 50.0),
    batch_size=1000,
):
    # batch is a GeoPandas GeoDataFrame
    process(batch)
```

`query_dataset()` yields GeoPandas `GeoDataFrame` batches whose geometries
intersect the query. It does not load all matching records into one large
dataframe.

Signature:

```python
starlet.query_dataset(
    dataset_dir: str | Path,
    geometry: tuple[float, float, float, float] | dict | shapely.Geometry,
    *,
    geometry_crs: str = "EPSG:4326",
    geom_col: str = "geometry",
    batch_size: int | None = None,
) -> Iterator[geopandas.GeoDataFrame]
```

GeoJSON polygon query:

```python
polygon = {
    "type": "Polygon",
    "coordinates": [[
        [-125.0, 24.0],
        [-66.0, 24.0],
        [-66.0, 50.0],
        [-125.0, 50.0],
        [-125.0, 24.0],
    ]],
}

for batch in starlet.query_dataset("datasets/postal_codes", polygon):
    process(batch)
```

Collect all results only when the expected result set is small:

```python
import pandas as pd
import geopandas as gpd

batches = list(starlet.query_dataset("datasets/postal_codes", bbox))
records = (
    gpd.GeoDataFrame(pd.concat(batches, ignore_index=True), crs=batches[0].crs)
    if batches
    else gpd.GeoDataFrame()
)
```

## Query Count

```python
count: int = starlet.query_dataset_count(
    "datasets/postal_codes",
    (-125.0, 24.0, -66.0, 50.0),
)
```

Returns the number of records intersecting the query. It uses the same
streaming path as `query_dataset()`.

## Query Download Size Estimate

```python
size_bytes: int = starlet.query_dataset_size(
    "datasets/postal_codes",
    (-125.0, 24.0, -66.0, 50.0),
)
```

Returns a rough estimate of matching record size in bytes. It streams matching
batches and sums approximate geometry WKB size plus attribute memory usage. Use
this for download planning, not exact serialized file sizes.

## First Matching Record

```python
record: dict | None = starlet.get_sample_record(
    "datasets/postal_codes",
    (-125.0, 24.0, -66.0, 50.0),
)
```

Returns the first matching record as a Python dictionary, or `None` if no
records match. This is useful for previews.

## Build Datasets

`starlet.tile()`, `starlet.build()`, and `starlet.add_dataset()` accept the
same source inputs and source-specific options.

Supported sources:

| Source | Accepted input path | Geometry configuration | Notes |
| --- | --- | --- | --- |
| GeoParquet | `.parquet`, `.geoparquet`, or a directory containing only GeoParquet files | `geom_col="geometry"` by default | Reads Parquet row groups as splits. Geometry-only sampling reads only the geometry column. |
| GeoJSON | `.geojson`, `.geojsonl`, `.json`, `.jsonl`, or a directory containing only GeoJSON files | Geometry comes from GeoJSON feature geometry | FeatureCollection inputs are byte-partitioned; GeoJSONL is streamed by feature records. |
| GeoLife PLT | A `.plt` file or a directory containing only `.plt` files, nested at any depth | Longitude/latitude records become WGS 84 points | Each file is a split. `trajectory_id` repeats the file ID on every point so trajectories can be regrouped. |
| GPX | A `.gpx` file or a directory containing only `.gpx` files, nested at any depth | GPX longitude/latitude points become WGS 84 points | Tracks, routes, and waypoints are flattened into point rows. File and GPX hierarchy metadata repeats on every point when present in the selected input. |
| Shapefile | `.shp`, `.zip` containing shapefile sidecars, or a directory containing `.shp` and/or `.zip` files | Geometry comes from the Shapefile geometry | Uses `pyogrio`. Feature-range splits are used when feature counts are available. Geometry-only sampling reads geometry without attributes. |
| CSV/TSV | `.csv`, `.tsv`, or a directory containing only CSV/TSV files | Use either `csv_x_col` + `csv_y_col`, `csv_wkt_col`, `csv_x_index` + `csv_y_index`, or `csv_wkt_index` | Files are read in row chunks. `src_crs` provides the CRS hint. Name-based options expect a header row; index-based options read the file as headerless. |
| File Geodatabase | `.gdb` directory, `.gdb.zip` archive, or a directory containing `.gdb` directories | Geometry comes from each GDB layer | Uses `pyogrio`. Multiple layers are read as separate splits. Zipped GDBs are extracted to a temp cache before reading. |

Directories must contain one supported source type. For example, a directory
containing both CSV/TSV and GeoJSON files is rejected to avoid ambiguous
ingestion.

Basic tiling:

```python
result = starlet.tile(
    input="data/roads.parquet",
    outdir="datasets/roads",
)
```

Returns a `TileResult` with output path, file count, row count, bounds, and
histogram path.

Run the full pipeline with vector-tile generation:

```python
tile_result, mvt_result, pmtiles_path = starlet.build(
    input="data/stops.csv",
    outdir="datasets/stops",
    csv_x_col="longitude",
    csv_y_col="latitude",
    zoom=10,
)
```

### GeoParquet Sources

```python
result = starlet.tile(
    input="data/buildings.geoparquet",
    outdir="datasets/buildings",
    geom_col="geometry",
)
```

GeoParquet inputs are split by row group. If a directory is passed, Starlet
recursively reads `.parquet` and `.geoparquet` files in that directory.

### GeoJSON Sources

```python
result = starlet.tile(
    input="data/places.geojson",
    outdir="datasets/places",
    parallelism=8,
)
```

GeoJSON FeatureCollections are partitioned by byte range while preserving
complete feature objects. GeoJSON Lines inputs are streamed by feature record.

### GeoLife PLT Sources

Pass either one `.plt` file or the directory above a collection of trajectory
files.

```python
result = starlet.tile(
    input="data/geolife/trajectory",
    outdir="datasets/trajectories",
)
```

Starlet recursively discovers `.plt` files and turns every trajectory record
into a WGS 84 point. It preserves `latitude`, `longitude`, `reserved`,
`altitude` (in feet), `date_days`, `date`, and `time`. The repeated
`trajectory_id` is just the basename for single-file input and the file path
relative to the input directory for directory input. `filename` always contains
the basename. Keeping both makes same-named files in different subdirectories
unambiguous without losing the convenient filename property.

### GPX Sources

Pass either one `.gpx` file or the directory above a collection of GPX files.

```python
result = starlet.tile(
    input="data/tracks",
    outdir="datasets/gpx-tracks",
)
```

Starlet recursively discovers `.gpx` files and flattens tracks (`trkpt`),
routes (`rtept`), and waypoints (`wpt`) into WGS 84 point rows. As with PLT,
`trajectory_id` is the basename for single-file input and the file path
relative to the input directory for directory input. `filename` always contains
the basename.

The first spatial scan also infers a compact GPX schema while computing the
sample and MBR. Fields that never appear in the selected input are omitted, so
a track-only file does not get all-null route columns, and a file without GPX
metadata does not get all-null metadata columns.

The flattened schema can repeat GPX hierarchy on each point: file metadata
(`gpx_version`, `gpx_creator`, `gpx_name`, `gpx_description`, `gpx_author`,
`gpx_time`, `gpx_keywords`, `gpx_bounds`, `gpx_metadata_xml`,
`gpx_extensions_xml`), point placement (`point_kind`, `track_index`,
`route_index`, `segment_index`, `point_index`), track/route fields
(`track_name`, `track_number`, `track_type`, `route_name`, `route_number`,
`route_type`, plus comments, descriptions, sources, links, and extensions),
and point fields (`latitude`, `longitude`, `elevation`, `point_time`,
`point_name`, `point_comment`, `point_description`, `point_source`,
`point_symbol`, `point_type`, `point_fix`, `point_satellites`, DOP/DGPS
fields, `point_links`, and `point_extensions_xml`).

### Shapefile Sources

Use a `.shp` file:

```python
result = starlet.tile(
    input="data/roads/roads.shp",
    outdir="datasets/roads",
)
```

Use a zipped Shapefile:

```python
result = starlet.tile(
    input="data/roads.zip",
    outdir="datasets/roads",
)
```

Use a directory containing many Shapefiles or zipped Shapefiles:

```python
result = starlet.tile(
    input="data/shapefiles",
    outdir="datasets/shapefiles",
)
```

Starlet uses `pyogrio` for Shapefile reads. When building the spatial sample,
it requests geometry only so attribute columns are not read unnecessarily.

### CSV/TSV Sources

CSV and TSV inputs need explicit geometry column configuration.

For headered files, configure geometry fields by column name:

- `csv_x_col` and `csv_y_col` must match the x/y header names, or
- `csv_wkt_col` must match the WKT header name.

For headerless files, configure geometry fields by zero-based column index:

- `csv_x_index` and `csv_y_index` for x/y coordinates, or
- `csv_wkt_index` for WKT geometry.

When you use index-based CSV geometry options, Starlet does not read a header
row from the file. Non-geometry columns are exposed with generated names such
as `column_0`, `column_1`, and so on.

For x/y coordinate columns in a headered file:

```python
result = starlet.tile(
    input="data/stops.csv",
    outdir="datasets/stops",
    csv_x_col="longitude",
    csv_y_col="latitude",
    src_crs="EPSG:4326",
)
```

For x/y coordinate columns in a headerless file:

```python
result = starlet.tile(
    input="data/stops-no-header.csv",
    outdir="datasets/stops",
    csv_x_index=0,
    csv_y_index=1,
    src_crs="EPSG:4326",
)
```

For WKT geometry in a headered file:

```python
result = starlet.tile(
    input="data/parcels.csv",
    outdir="datasets/parcels",
    csv_wkt_col="wkt",
    src_crs="EPSG:4326",
)
```

For WKT geometry in a headerless file:

```python
result = starlet.tile(
    input="data/parcels-no-header.csv",
    outdir="datasets/parcels",
    csv_wkt_index=2,
    src_crs="EPSG:4326",
)
```

UCR Star also supports CSV line segments through its CLI by converting four endpoint columns into temporary WKT `LINESTRING` values before calling Starlet. For a headered CSV:

```bash
ucrstar add-dataset data/roads.csv --name roads --csv-segment-cols x1,y1,x2,y2
```

For a headerless CSV:

```bash
ucrstar add-dataset data/roads-no-header.csv --name roads --csv-segment-indexes 0,1,2,3
```

Useful CSV options:

```python
result = starlet.tile(
    input="data/points.csv",
    outdir="datasets/points",
    csv_x_col="x",
    csv_y_col="y",
    csv_split_size=64 * 1024 * 1024,
)
```

`csv_split_size` controls the target byte length for each CSV source split. The
default is 32 MiB. CSV splits follow Hadoop-style line ownership: a split starts
at a byte offset, skips to the next newline if needed, and reads every complete
line whose starting byte falls inside `[offset, offset + length)`.

### File Geodatabase Sources

Use a `.gdb` directory directly:

```python
result = starlet.tile(
    input="data/city.gdb",
    outdir="datasets/city",
)
```

Use a zipped File Geodatabase:

```python
result = starlet.tile(
    input="data/city.gdb.zip",
    outdir="datasets/city",
)
```

Use a parent directory containing one or more `.gdb` directories:

```python
result = starlet.tile(
    input="data/geodatabases",
    outdir="datasets/geodatabases",
)
```

Starlet reads all layers discovered by `pyogrio`. Layers are handled as
separate source splits, and feature-range splits are used when feature counts
are available.

### CLI Source Examples

GeoParquet:

```bash
starlet tile --input data/buildings.parquet --outdir datasets/buildings
```

CSV with x/y columns:

```bash
starlet tile \
  --input data/stops.csv \
  --outdir datasets/stops \
  --csv-x-col longitude \
  --csv-y-col latitude
```

Headerless CSV with x/y columns:

```bash
starlet tile \
  --input data/stops-no-header.csv \
  --outdir datasets/stops \
  --csv-x-index 0 \
  --csv-y-index 1
```

CSV with WKT:

```bash
starlet tile \
  --input data/parcels.csv \
  --outdir datasets/parcels \
  --csv-wkt-col wkt
```

Headerless CSV with WKT:

```bash
starlet tile \
  --input data/parcels-no-header.csv \
  --outdir datasets/parcels \
  --csv-wkt-index 2
```

Shapefile:

```bash
starlet tile --input data/roads.zip --outdir datasets/roads
```

GeoLife PLT directory:

```bash
starlet tile --input data/geolife/trajectory --outdir datasets/trajectories
```

GPX file or directory:

```bash
starlet tile --input data/tracks.gpx --outdir datasets/gpx-tracks
```

File Geodatabase:

```bash
starlet tile --input data/city.gdb --outdir datasets/city
```

Zipped File Geodatabase:

```bash
starlet tile --input data/city.gdb.zip --outdir datasets/city
```

The `starlet build` command supports the same input options.

## Add A Dataset

```python
tile_result, mvt_result, pmtiles_path = starlet.add_dataset(
    "source/postal_codes.geojson",
    "datasets",
    name="postal_codes",
    overwrite=True,
    zoom=7,
    covering_bbox=True,
)
```

Builds a dataset under the datasets root using the same pipeline as
`starlet.build()`.

Parameters:

- `input_path`: supported source file or directory. See Build Datasets above.
- `datasets_dir`: root directory that contains all datasets.
- `name`: dataset directory name. Defaults to the input path stem.
- `overwrite`: if `True`, remove an existing dataset with the same name first.
- `**build_kwargs`: forwarded to `starlet.build()`.

Common build kwargs:

```python
{
    "zoom": 7,
    "parallelism": 8,
    "partition_size": None,
    "threshold": 100_000,
    "covering_bbox": True,
    "pmtiles": False,
}
```

`covering_bbox=True` is recommended for datasets that will be queried or served
on demand because it writes per-row bbox columns used for read-time pruning.

Return value:

```python
(tile_result, mvt_result, pmtiles_path)
```

## Add A Dataset Asynchronously

```python
handle = starlet.add_dataset_async(
    "source/postal_codes.geojson",
    "datasets",
    name="postal_codes",
    overwrite=True,
    zoom=7,
    covering_bbox=True,
)
```

Starts `add_dataset()` in a background thread and immediately returns an
`AsyncDatasetHandle`.

Handle API:

```python
handle.status                 # pending | running | cancel_requested | cancelled | succeeded | failed
handle.cancel_requested       # bool
handle.error                  # BaseException | None
handle.done()                 # bool
handle.join(timeout=None)     # bool: True if terminal
handle.result(timeout=None)   # returns build result or raises
handle.cancel()               # bool: cancellation request accepted
handle.as_dict()              # JSON-compatible status snapshot
```

Polling example:

```python
import time

handle = starlet.add_dataset_async("source/data.geojson", "datasets", name="data")

while not handle.done():
    print(handle.as_dict())
    time.sleep(1)

try:
    tile_result, mvt_result, pmtiles_path = handle.result()
except Exception as exc:
    handle_info = handle.as_dict()
    raise RuntimeError(f"Dataset build failed: {handle_info}") from exc
```

Timeout and cancellation example:

```python
handle = starlet.add_dataset_async("source/data.geojson", "datasets", name="data")

try:
    result = handle.result(timeout=60)
except TimeoutError:
    handle.cancel()
```

Cancellation is best-effort. Python threads cannot safely be killed in the
middle of the existing build pipeline. If the job has not started yet, it is
cancelled. If the build is already running, the handle records
`cancel_requested` and exits when the current build call returns.

## Delete A Dataset

```python
deleted: bool = starlet.delete_dataset("datasets", "postal_codes")
```

Deletes the named dataset directory under the datasets root and returns `True`.

```python
deleted = starlet.delete_dataset("datasets", "postal_codes", missing_ok=True)
```

With `missing_ok=True`, returns `False` when the dataset is not present.

Safety behavior:

- Rejects absolute dataset names.
- Rejects names containing `..`.
- Raises `FileNotFoundError` for missing datasets unless `missing_ok=True`.
- Raises `NotADirectoryError` if the target exists but is not a directory.

## Error Handling Cheatsheet

Typical exceptions to handle in application code:

```python
try:
    metadata = starlet.get_dataset_metadata(dataset_dir)
    for batch in starlet.query_dataset(dataset_dir, bbox, batch_size=1000):
        process(batch)
except FileNotFoundError:
    # Missing input, dataset, histogram, or other required artifact.
    ...
except NotADirectoryError:
    # Expected directory path is not a directory.
    ...
except ValueError:
    # Invalid bbox, CRS, dataset name, or geometry input.
    ...
```

## Minimal Flask Integration

```python
from flask import Flask, Response, jsonify, request
import starlet

app = Flask(__name__)
DATASETS_DIR = "datasets"

@app.get("/api/datasets")
def datasets():
    return jsonify({"datasets": starlet.list_datasets(DATASETS_DIR)})

@app.get("/api/datasets/<dataset>")
def dataset_metadata(dataset: str):
    return jsonify(starlet.get_dataset_metadata(f"{DATASETS_DIR}/{dataset}"))

@app.get("/api/datasets/<dataset>/count")
def dataset_count(dataset: str):
    bbox = tuple(float(v) for v in request.args["bbox"].split(","))
    count = starlet.query_dataset_count(f"{DATASETS_DIR}/{dataset}", bbox)
    return jsonify({"count": count})

@app.get("/tiles/<dataset>/<int:z>/<int:x>/<int:y>.mvt")
def tile(dataset: str, z: int, x: int, y: int):
    data = starlet.get_tile(f"{DATASETS_DIR}/{dataset}", z, x, y)
    return Response(data, mimetype="application/vnd.mapbox-vector-tile")
```

## Minimal FastAPI Integration

```python
from fastapi import FastAPI, Response
import starlet

app = FastAPI()
DATASETS_DIR = "datasets"

@app.get("/api/datasets")
def datasets():
    return {"datasets": starlet.list_datasets(DATASETS_DIR)}

@app.get("/api/datasets/{dataset}")
def dataset_metadata(dataset: str):
    return starlet.get_dataset_metadata(f"{DATASETS_DIR}/{dataset}")

@app.get("/tiles/{dataset}/{z}/{x}/{y}.mvt")
def tile(dataset: str, z: int, x: int, y: int):
    data = starlet.get_tile(f"{DATASETS_DIR}/{dataset}", z, x, y)
    return Response(content=data, media_type="application/vnd.mapbox-vector-tile")
```

## Recommended AI Assistant Rules

When using Starlet in another project:

1. Import only `starlet`; do not import from `starlet._internal`.
2. Use `get_dataset_metadata()` for readiness and artifact checks.
3. Use `query_dataset()` as an iterator; do not assume it returns one dataframe.
4. Use `query_dataset_count()` for counts instead of materializing records.
5. Use `query_dataset_size()` only as an estimate.
6. Use `get_sample_record()` for previews.
7. Use `add_dataset_async()` for UI/API-triggered builds.
8. Use `delete_dataset()` for removal; do not manually `shutil.rmtree()` dataset
   paths in application code.
