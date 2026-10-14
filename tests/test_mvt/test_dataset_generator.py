"""Tests for the two-stage dataset MVT generator helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import mapbox_vector_tile
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkb
from shapely.geometry import Point

from starlet._internal.mvt.intermediate_tile import IntermediateVectorTile
from starlet._internal.mvt.mvt_generator import (
    _MapperMemoryBudget,
    _MapperTileCache,
    _ReduceTileInput,
    _TableBatch,
    _bucket_tile_ids,
    _group_splits,
    _group_table_batches,
    _intermediate_tile_part_filename,
    _mvt_memory_budget_for_stage,
    _positive_bounds_tuple,
    _resolve_total_mvt_memory_budget,
    _reduce_tile_group,
    _sample_single_tile_records,
    _single_tile_index_cache,
    _single_tile_parquet_index,
    generate_single_mvt_tile,
)
from starlet._internal.mvt.intermediate_tile import IntermediateVectorTile
from starlet._internal.mvt import mvt_generator
from starlet._internal.server.tiler.parquet_index import ParquetIndex
from starlet._internal.tiling.geoparquet_source import GeoParquetSplit


def test_group_splits_round_robins_to_requested_groups():
    splits = [
        GeoParquetSplit(path=f"part-{index}.parquet", row_groups=(0,))
        for index in range(5)
    ]

    groups = _group_splits(splits, 2)

    assert groups == [
        [splits[0], splits[2], splits[4]],
        [splits[1], splits[3]],
    ]


def test_group_table_batches_splits_rows_across_requested_groups():
    table = pa.table({"value": list(range(10))})

    groups = _group_table_batches(table, 3)

    assert len(groups) == 3
    assert all(isinstance(group[0], _TableBatch) for group in groups)
    assert [group[0].table.num_rows for group in groups] == [4, 4, 2]


def test_bucket_tile_ids_uses_mod_hash():
    assert _bucket_tile_ids([1, 2, 3, 4, 5], 3) == [
        [3],
        [1, 4],
        [2, 5],
    ]


def test_positive_bounds_tuple_expands_zero_sized_bounds():
    minx, miny, maxx, maxy = _positive_bounds_tuple((1.0, 2.0, 1.0, 2.0))

    assert minx == 1.0
    assert miny == 2.0
    assert maxx > minx
    assert maxy > miny


def test_mvt_memory_budget_resolves_total_and_stage_share():
    total = _resolve_total_mvt_memory_budget("8gb")
    budget = _mvt_memory_budget_for_stage(
        total_budget_bytes=total,
        active_mappers=4,
    )

    assert total == 8 * 1024 ** 3
    assert budget is not None
    assert budget.high_watermark_bytes == 2 * 1024 ** 3
    assert budget.low_watermark_bytes == int(2 * 1024 ** 3 * 0.85)


def test_mvt_memory_budget_can_be_disabled():
    assert _resolve_total_mvt_memory_budget("none") is None
    assert _resolve_total_mvt_memory_budget(0) is None


def test_mapper_tile_cache_spills_when_memory_watermark_is_exceeded(tmp_path, monkeypatch):
    cache = _MapperTileCache(
        mapper_index=0,
        intermediate_dir=tmp_path / "mapper",
        feature_capacity=10,
        memory_budget=_MapperMemoryBudget(
            high_watermark_bytes=100 * 1024 ** 2,
            low_watermark_bytes=80 * 1024 ** 2,
            total_budget_bytes=100 * 1024 ** 2,
        ),
        extent=4096,
        buffer=256,
    )
    monkeypatch.setattr(mvt_generator, "_MAPPER_MEMORY_CHECK_FEATURES", 1)
    rss_values = iter([(120.0, 120.0), (70.0, 120.0)])
    monkeypatch.setattr(mvt_generator, "_process_rss_mib", lambda: next(rss_values))

    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    cache.put(tile.tile_id, tile)
    tile.add_feature(Point(0, 0), {"id": 1}, priority=1)
    cache.account_feature_change(tile.tile_id)

    assert tile.tile_id not in cache.tiles
    assert cache.live_features == 0
    assert cache.memory_spill_events == 1
    assert cache.memory_evictions == 1
    assert cache.tile_part_counts == {tile.tile_id: 1}


def test_reduce_merges_multiple_spills_for_same_tile(tmp_path):
    tile_id = IntermediateVectorTile(0, 0, 0, feature_capacity=1).tile_id
    mapper_dir = tmp_path / "mapper"
    mapper_dir.mkdir()
    left_path = mapper_dir / _intermediate_tile_part_filename(0, 0, 0, 0)
    right_path = mapper_dir / _intermediate_tile_part_filename(0, 0, 0, 1)
    left = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    right = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    left.add_feature(Point(0, 0), {"id": 1}, priority=1)
    right.add_feature(Point(0, 0), {"id": 2}, priority=9)
    left.write_features(left_path)
    right.write_features(right_path)

    outdir = tmp_path / "mvt"
    _reduce_tile_group(
        [_ReduceTileInput(tile_id, ((str(mapper_dir), 2),))],
        str(outdir),
        feature_capacity=1,
        extent=4096,
        buffer=256,
    )

    decoded = mapbox_vector_tile.decode((outdir / "0" / "0" / "0.mvt").read_bytes())
    assert decoded["layer0"]["features"][0]["properties"] == {"id": 2}


def test_generate_single_mvt_tile_uses_partition_and_row_bbox_pruning(tmp_path):
    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }
    table = pa.table(
        {
            "geometry": [
                wkb.dumps(Point(-100.0, 80.0)),
                wkb.dumps(Point(100.0, -80.0)),
            ],
            "_id": [10, 11],
            "id": [1, 2],
            "_bbox_xmin": [-100.0, 100.0],
            "_bbox_ymin": [80.0, -80.0],
            "_bbox_xmax": [-100.0, 100.0],
            "_bbox_ymax": [80.0, -80.0],
        }
    ).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
    pq.write_table(
        table,
        parquet_dir / "tile_000000__-100_0_-80_0_100_0_80_0.parquet",
    )

    tile_bytes = generate_single_mvt_tile(
        str(dataset_dir),
        (1, 0, 0),
        feature_capacity=10,
    )
    decoded = mapbox_vector_tile.decode(tile_bytes)

    features = decoded["layer0"]["features"]
    assert len(features) == 1
    assert features[0]["properties"] == {"_id": 10, "id": 1}


def test_generate_single_mvt_tile_projects_requested_attributes(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }
    table = pa.table(
        {
            "geometry": [wkb.dumps(Point(-100.0, 80.0))],
            "id": [1],
            "name": ["keep"],
            "unused": ["drop"],
            "_bbox_xmin": [-100.0],
            "_bbox_ymin": [80.0],
            "_bbox_xmax": [-100.0],
            "_bbox_ymax": [80.0],
        }
    ).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
    pq.write_table(
        table,
        parquet_dir / "tile_000000__-100_0_80_0_-100_0_80_0.parquet",
    )

    read_columns = []
    original_read_table = mvt_generator.pq.read_table

    def recording_read_table(*args, **kwargs):
        read_columns.append(kwargs.get("columns"))
        return original_read_table(*args, **kwargs)

    monkeypatch.setattr(mvt_generator.pq, "read_table", recording_read_table)
    # Column pushdown applies to the per-feature path; the vectorised EPSG:4326
    # path reads whole row groups into a cache shared by all requests.
    monkeypatch.setattr(mvt_generator, "_rust_supports_dataset", lambda _path: False)

    tile_bytes = generate_single_mvt_tile(
        str(dataset_dir),
        (1, 0, 0),
        feature_capacity=10,
        attributes=["name"],
    )
    decoded = mapbox_vector_tile.decode(tile_bytes)

    assert decoded["layer0"]["features"][0]["properties"] == {"name": "keep"}
    assert read_columns == [["geometry", "name", "_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax"]]


def test_generate_single_mvt_tile_flattens_nested_map_properties(tmp_path):
    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }
    table = pa.table(
        {
            "geometry": [wkb.dumps(Point(-100.0, 80.0))],
            "id": [1],
            "tagsMap": pa.array(
                [[("name", "Oak Hill"), ("landuse", "cemetery")]],
                type=pa.map_(pa.string(), pa.string()),
            ),
            "_bbox_xmin": [-100.0],
            "_bbox_ymin": [80.0],
            "_bbox_xmax": [-100.0],
            "_bbox_ymax": [80.0],
        }
    ).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
    pq.write_table(
        table,
        parquet_dir / "tile_000000__-100_0_80_0_-100_0_80_0.parquet",
    )

    tile_bytes = generate_single_mvt_tile(
        str(dataset_dir),
        (1, 0, 0),
        feature_capacity=10,
    )
    decoded = mapbox_vector_tile.decode(tile_bytes)

    assert decoded["layer0"]["features"][0]["properties"] == {
        "id": 1,
        "tagsMap.name": "Oak Hill",
        "tagsMap.landuse": "cemetery",
    }


def test_generate_single_mvt_tile_keeps_feature_id_without_bbox_columns(tmp_path):
    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }
    table = pa.table(
        {
            "geometry": [wkb.dumps(Point(-100.0, 80.0))],
            "_id": [42],
            "id": [1],
        }
    ).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
    pq.write_table(
        table,
        parquet_dir / "tile_000000__-100_0_80_0_-100_0_80_0.parquet",
    )

    # Warm the public-query cache first; on-demand MVT must still be able to
    # opt into _id even if a hidden-column read happened earlier.
    index = ParquetIndex(parquet_dir)
    assert "_id" not in next(index.iter_query_batches((-101.0, 79.0, -99.0, 81.0))).columns

    tile_bytes = generate_single_mvt_tile(
        str(dataset_dir),
        (1, 0, 0),
        feature_capacity=10,
    )
    decoded = mapbox_vector_tile.decode(tile_bytes)

    assert decoded["layer0"]["features"][0]["properties"] == {"_id": 42, "id": 1}


def test_generate_single_mvt_tile_queries_buffered_tile_bounds(tmp_path):
    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }
    table = pa.table(
        {
            "geometry": [wkb.dumps(Point(1.0, 45.0))],
            "id": [9],
            "_bbox_xmin": [1.0],
            "_bbox_ymin": [45.0],
            "_bbox_xmax": [1.0],
            "_bbox_ymax": [45.0],
        }
    ).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
    pq.write_table(
        table,
        parquet_dir / "tile_000000__-1_0_40_0_2_0_50_0.parquet",
    )

    tile_bytes = generate_single_mvt_tile(
        str(dataset_dir),
        (1, 0, 0),
        feature_capacity=10,
        extent=4096,
        buffer=256,
    )
    decoded = mapbox_vector_tile.decode(tile_bytes)

    features = decoded["layer0"]["features"]
    assert len(features) == 1
    assert features[0]["properties"] == {"id": 9}


def test_generate_single_mvt_tile_honors_custom_extent(tmp_path):
    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }
    table = pa.table(
        {
            "geometry": [wkb.dumps(Point(0.0, 0.0))],
            "id": [7],
            "_bbox_xmin": [0.0],
            "_bbox_ymin": [0.0],
            "_bbox_xmax": [0.0],
            "_bbox_ymax": [0.0],
        }
    ).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
    pq.write_table(
        table,
        parquet_dir / "tile_000000__0_0_0_0_0_0_0.parquet",
    )

    tile_bytes = generate_single_mvt_tile(
        str(dataset_dir),
        (0, 0, 0),
        feature_capacity=10,
        extent=256,
    )
    decoded = mapbox_vector_tile.decode(tile_bytes)

    assert decoded["layer0"]["extent"] == 256


def test_single_tile_sampling_decodes_only_retained_rows(monkeypatch, tmp_path):
    parquet_dir = tmp_path / "dataset" / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }
    rows = list(range(10))
    table = pa.table(
        {
            # ten points inside one display pixel at z0 (0.0001 deg apart,
            # away from the origin, which sits on a pixel-cell boundary)
            "geometry": [wkb.dumps(Point(0.01 + index * 1e-4, 0.01 + index * 1e-4)) for index in rows],
            "id": rows,
            "_bbox_xmin": [0.01 + index * 1e-4 for index in rows],
            "_bbox_ymin": [0.01 + index * 1e-4 for index in rows],
            "_bbox_xmax": [0.01 + index * 1e-4 for index in rows],
            "_bbox_ymax": [0.01 + index * 1e-4 for index in rows],
        }
    ).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
    pq.write_table(
        table,
        parquet_dir / "tile_000000__0_0_0_0_9_0_9_0.parquet",
    )
    decoded_batch_sizes = []

    def fake_from_wkb(values):
        decoded_batch_sizes.append(len(values))
        return [Point(0, 0) for _ in values]

    monkeypatch.setattr(mvt_generator, "from_wkb", fake_from_wkb)
    monkeypatch.setattr(
        mvt_generator,
        "reproject_geometries",
        lambda geometries, source_crs, target_crs: (geometries, target_crs),
    )

    features = _sample_single_tile_records(
        ParquetIndex(parquet_dir),
        (0.0, 0.0, 9.0, 9.0),
        3,
        IntermediateVectorTile(0, 0, 0, feature_capacity=3),
    )

    # one pixel cell -> one winner; only that row is decoded
    assert features is not None
    assert len(features) == 1
    assert decoded_batch_sizes == [1]


def test_single_tile_parquet_index_is_cached_by_path(tmp_path):
    parquet_dir = tmp_path / "dataset" / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    _single_tile_index_cache.clear()

    first = _single_tile_parquet_index(parquet_dir)
    second = _single_tile_parquet_index(parquet_dir)

    assert first is second


def test_generate_mvt_forwards_temp_dir(monkeypatch, tmp_path):
    import starlet

    captured = {}

    class FakeDatasetMVTGenerator:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def run(self):
            return None

    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.DatasetMVTGenerator",
        FakeDatasetMVTGenerator,
    )
    outdir = tmp_path / "mvt"
    outdir.mkdir()
    temp_dir = tmp_path / "scratch"

    starlet.generate_mvt(
        str(tmp_path / "dataset"),
        outdir=str(outdir),
        temp_dir=str(temp_dir),
    )

    assert captured["temp_dir"] == str(temp_dir)


def test_generate_mvt_forwards_pmtiles_settings(monkeypatch, tmp_path):
    import starlet

    captured = {}

    class FakeDatasetMVTGenerator:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def run(self):
            return SimpleNamespace(
                pmtiles_path=str(tmp_path / "dataset" / "tiles.pmtiles"),
                tile_count=3,
                tile_counts_by_zoom=[1, 2],
                zoom_levels=[0, 1],
            )

    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.DatasetMVTGenerator",
        FakeDatasetMVTGenerator,
    )
    outdir = tmp_path / "mvt"
    outdir.mkdir()

    starlet.generate_mvt(
        str(tmp_path / "dataset"),
        outdir=str(outdir),
        pmtiles=True,
        pmtiles_compression="brotli",
    )

    assert captured["output_format"] == "pmtiles"
    assert captured["pmtiles_compression"] == "brotli"


def test_dataset_generator_removes_mvt_dir_after_pmtiles_export(monkeypatch, tmp_path):
    from starlet._internal.mvt.mvt_generator import DatasetMVTGenerator, _MapStageResult

    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "parquet_tiles"
    hist_dir = dataset_dir / "histograms"
    parquet_dir.mkdir(parents=True)
    hist_dir.mkdir(parents=True)
    (hist_dir / "global.npy.gz").write_bytes(b"fake")

    generator = DatasetMVTGenerator(
        str(dataset_dir),
        num_zoom_levels=2,
        threshold=0,
        output_format="pmtiles",
    )
    tile_path = generator.outdir / "0" / "0"
    tile_path.mkdir(parents=True)
    (tile_path / "0.mvt").write_bytes(b"tile")

    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.GeoParquetSource",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator._create_map_groups",
        lambda source, workers: [[object()]],
    )
    monkeypatch.setattr(
        DatasetMVTGenerator,
        "_run_map_stage",
        lambda self, groups, source, temp_root: [_MapStageResult("tmp", ())],
    )
    monkeypatch.setattr(
        DatasetMVTGenerator,
        "_run_reduce_stage",
        lambda self, map_results: None,
    )
    exported = {}
    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.export_to_pmtiles",
        lambda **kwargs: exported.update(kwargs),
    )

    result = generator.run()

    assert result.tile_counts_by_zoom == [1]
    assert result.tile_count == 1
    assert result.pmtiles_path == str(dataset_dir / "tiles.pmtiles")
    assert exported["output_path"] == str(dataset_dir / "tiles.pmtiles")
    assert not generator.outdir.exists()


def test_dataset_generator_skips_pmtiles_export_when_threshold_filters_all_tiles(
    monkeypatch,
    tmp_path,
):
    from starlet._internal.mvt.mvt_generator import DatasetMVTGenerator, _MapStageResult

    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "parquet_tiles"
    hist_dir = dataset_dir / "histograms"
    parquet_dir.mkdir(parents=True)
    hist_dir.mkdir(parents=True)
    (hist_dir / "global.npy.gz").write_bytes(b"fake")

    generator = DatasetMVTGenerator(
        str(dataset_dir),
        num_zoom_levels=2,
        threshold=100_000,
        output_format="pmtiles",
    )

    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.GeoParquetSource",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator._create_map_groups",
        lambda source, workers: [[object()]],
    )
    monkeypatch.setattr(
        DatasetMVTGenerator,
        "_run_map_stage",
        lambda self, groups, source, temp_root: [_MapStageResult("tmp", ())],
    )

    def make_empty_output(self, map_results):
        self.outdir.mkdir(parents=True)

    monkeypatch.setattr(DatasetMVTGenerator, "_run_reduce_stage", make_empty_output)
    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.export_to_pmtiles",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("empty tiles exported")),
    )

    result = generator.run()

    assert result.tile_count == 0
    assert result.tile_counts_by_zoom == []
    assert result.zoom_levels == []
    assert result.pmtiles_path is None
    assert not generator.outdir.exists()
