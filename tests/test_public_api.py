import json
import shutil
import time
import gzip
import importlib

import starlet
from starlet.api import AsyncDatasetHandle
from starlet._internal.server.download_service import DatasetFeatureService
from starlet._internal.tiling.csv_source import CSVSource
from starlet._internal.tiling.geojson_source import GeoJSONSource


def test_list_datasets(sample_dataset_dir):
    assert starlet.list_datasets(sample_dataset_dir.parent) == ["test_dataset"]


def test_process_wide_temp_dir_can_be_configured(temp_dir):
    previous = starlet.get_temp_dir()
    custom_temp = temp_dir / "starlet_tmp"
    try:
        configured = starlet.set_temp_dir(str(custom_temp))

        assert configured == custom_temp
        assert starlet.get_temp_dir() == custom_temp
        assert custom_temp.is_dir()
    finally:
        starlet.set_temp_dir(str(previous) if previous is not None else None)


def test_geojson_tiling_preserves_nested_properties_in_geojson_export(tmp_path, monkeypatch):
    original_batches = GeoJSONSource._iter_feature_batches_for_split
    schema_scans = 0

    def tracked_batches(self, split):
        nonlocal schema_scans
        if split is None:
            schema_scans += 1
        yield from original_batches(self, split)

    monkeypatch.setattr(GeoJSONSource, "_iter_feature_batches_for_split", tracked_batches)

    source = tmp_path / "cemetery_tags.geojson"
    dataset = tmp_path / "cemetery_dataset"
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "id": 1,
                            "tagsMap": {
                                "landuse": "cemetery",
                                "type": "multipolygon",
                            },
                        },
                        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "id": 2,
                            "tagsMap": {"name": "Oak Hill"},
                        },
                        "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    starlet.tile(str(source), str(dataset), partition_size=1, parallelism=1)

    assert schema_scans == 0

    body = "".join(
        DatasetFeatureService(tmp_path).get_features_stream("cemetery_dataset", "geojson")
    )
    exported = json.loads(body)
    tags = [feature["properties"]["tagsMap"] for feature in exported["features"]]

    assert {"landuse": "cemetery", "type": "multipolygon"} in tags
    assert {"name": "Oak Hill"} in tags

    sample = DatasetFeatureService(tmp_path).get_sample_record(
        "cemetery_dataset",
        "-1,-1,2,2",
        include_geometry=True,
    )
    assert sample is not None
    assert isinstance(sample["properties"]["tagsMap"], dict)

    public_sample = starlet.get_sample_record(dataset, (-1, -1, 2, 2))
    assert public_sample is not None
    assert isinstance(public_sample["tagsMap"], dict)
    assert public_sample["tagsMap"] in tags


def test_csv_tiling_reuses_schema_from_sampling(tmp_path, monkeypatch):
    source = tmp_path / "mixed.csv"
    dataset = tmp_path / "mixed_dataset"
    source.write_text(
        "id,x,y,units,active\n"
        "1,0,0,10,true\n"
        "2,1,1,2.5,false\n"
    )

    original_inference = CSVSource.iter_tables_for_schema_inference
    full_schema_scans = 0

    def tracked_inference(self, split=None):
        nonlocal full_schema_scans
        if split is None:
            full_schema_scans += 1
        yield from original_inference(self, split)

    monkeypatch.setattr(CSVSource, "iter_tables_for_schema_inference", tracked_inference)

    starlet.tile(
        str(source),
        str(dataset),
        partition_size=1,
        parallelism=1,
        csv_x_col="x",
        csv_y_col="y",
    )

    assert full_schema_scans == 0


def test_import_auto_loads_config_once(tmp_path, monkeypatch):
    from starlet._internal.config import (
        _reset_loaded_config_for_tests,
        get_loaded_config_path,
        get_temp_dir,
    )

    _reset_loaded_config_for_tests()
    monkeypatch.chdir(tmp_path)
    temp_root = tmp_path / "configured_tmp"
    (tmp_path / "starlet.toml").write_text(
        f"""
[global]
temp_dir = "{temp_root}"
""".strip()
    )

    try:
        importlib.reload(starlet)

        assert get_loaded_config_path() == tmp_path / "starlet.toml"
        assert get_temp_dir() == temp_root
    finally:
        _reset_loaded_config_for_tests()


def test_get_config_returns_current_loaded_config_copy():
    from starlet._internal.config import _reset_loaded_config_for_tests, set_loaded_config

    _reset_loaded_config_for_tests()
    set_loaded_config({"global": {"parallelism": 3}, "serve": {"cache_size": 12}})

    try:
        config = starlet.get_config()

        assert config["global"]["parallelism"] == 3
        assert config["serve"]["cache_size"] == 12
        assert config["mvt"]["extent"] == 4096

        config["serve"]["cache_size"] = 99
        assert starlet.get_config()["serve"]["cache_size"] == 12
    finally:
        _reset_loaded_config_for_tests()


def test_build_forwards_configured_pmtiles_options(monkeypatch):
    from types import SimpleNamespace

    from starlet._internal.config import _reset_loaded_config_for_tests, set_loaded_config

    _reset_loaded_config_for_tests()
    set_loaded_config(
        {
            "global": {},
            "tile": {},
            "mvt": {
                "zoom": 4,
                "threshold": 3,
                "pmtiles": True,
                "pmtiles_compression": "brotli",
            },
            "build": {},
            "serve": {},
        }
    )
    captured = {}

    def fake_tile(**kwargs):
        return SimpleNamespace(outdir=kwargs["outdir"])

    def fake_generate_mvt(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(pmtiles_path="dataset/tiles.pmtiles")

    monkeypatch.setattr(starlet, "tile", fake_tile)
    monkeypatch.setattr(starlet, "generate_mvt", fake_generate_mvt)

    try:
        _, _, pmtiles_path = starlet.build("input.geojson", "dataset")

        assert captured["zoom"] == 4
        assert captured["threshold"] == 3.0
        assert captured["pmtiles"] is True
        assert captured["pmtiles_compression"] == "brotli"
        assert pmtiles_path == "dataset/tiles.pmtiles"
    finally:
        _reset_loaded_config_for_tests()


def test_get_dataset_metadata(sample_dataset_dir):
    metadata = starlet.get_dataset_metadata(sample_dataset_dir)
    assert metadata["name"] == "test_dataset"
    assert metadata["exists"] is True
    assert metadata["parquet_tile_count"] >= 0
    assert metadata["has_stats"] is True
    assert metadata["parquet_has_bbox"] is False
    assert metadata["parquet_crs"] is None
    assert metadata["mvt_tile_count"] == 0
    assert metadata["histogram_resolution"] == 64


def test_get_dataset_summary_derives_from_stats(sample_dataset_dir):
    summary = starlet.get_dataset_summary(sample_dataset_dir)
    assert summary is not None
    assert summary["dataset"] == "test_dataset"
    assert summary["attribute_count"] >= 0


def test_get_dataset_summary_prefers_stored_summary(sample_dataset_dir):
    expected = {"dataset": "test_dataset", "custom": True}
    with open(sample_dataset_dir / "summary.json", "w") as f:
        json.dump(expected, f)

    assert starlet.get_dataset_summary(sample_dataset_dir) == expected


def test_estimate_range_count_full_extent(sample_dataset_dir, web_mercator_bounds, sample_histogram):
    estimate = starlet.estimate_range_count(
        sample_dataset_dir,
        web_mercator_bounds,
        rectangle_crs="EPSG:3857",
    )
    assert estimate == float(sample_histogram.sum())


def test_get_tile_populates_cheap_output_for_disk_tile(temp_dir):
    dataset_dir = temp_dir / "tile_dataset"
    (dataset_dir / "parquet_tiles").mkdir(parents=True)
    tile_path = dataset_dir / "mvt" / "0" / "0" / "0.mvt"
    tile_path.parent.mkdir(parents=True)
    expected = b"disk-tile"
    tile_path.write_bytes(expected)

    output = {}
    data = starlet.get_tile(dataset_dir, 0, 0, 0, output=output)

    assert data == expected
    assert output["source"] == "disk"
    assert output["generation"] == "read_from_disk"
    assert output["path"] == str(tile_path)
    assert "record_count" not in output
    assert "feature_count" not in output
    assert output["elapsed_ms"] >= 0


def test_get_tile_prefers_pmtiles_over_disk(temp_dir):
    from pmtiles.writer import Writer
    from pmtiles.tile import Compression, TileType, zxy_to_tileid

    dataset_dir = temp_dir / "tile_dataset"
    (dataset_dir / "parquet_tiles").mkdir(parents=True)
    tile_path = dataset_dir / "mvt" / "0" / "0" / "0.mvt"
    tile_path.parent.mkdir(parents=True)
    disk_bytes = b"disk-tile"
    tile_path.write_bytes(disk_bytes)

    pmtiles_bytes = b"pmtiles-tile"
    pmtiles_path = dataset_dir / "tiles.pmtiles"
    with open(pmtiles_path, "wb") as handle:
        writer = Writer(handle)
        writer.write_tile(zxy_to_tileid(0, 0, 0), gzip.compress(pmtiles_bytes))
        writer.finalize(
            {
                "tile_compression": Compression.GZIP,
                "tile_type": TileType.MVT,
            },
            {},
        )

    output = {}
    data = starlet.get_tile(dataset_dir, 0, 0, 0, output=output)

    assert data == pmtiles_bytes
    assert output["source"] == "pmtiles"
    assert output["generation"] == "read_from_pmtiles"
    assert output["path"] == str(pmtiles_path)


def test_vector_tiler_only_caches_generated_tiles(temp_dir, monkeypatch):
    from starlet._internal.server.tiler.tiler import VectorTiler

    dataset_dir = temp_dir / "tile_dataset"
    (dataset_dir / "parquet_tiles").mkdir(parents=True)
    tile_path = dataset_dir / "mvt" / "0" / "0" / "0.mvt"
    tile_path.parent.mkdir(parents=True)
    disk_bytes = b"disk-tile"
    tile_path.write_bytes(disk_bytes)

    tiler = VectorTiler(str(dataset_dir), memory_cache_size=8)

    assert tiler.get_tile(0, 0, 0) == disk_bytes
    assert (0, 0, 0, None) not in tiler.cache.store

    generated_bytes = b"generated-tile"
    captured = {}

    def fake_generate_single_mvt_tile(dataset_root, tile_coords, **kwargs):
        captured.update(kwargs)
        return generated_bytes

    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.generate_single_mvt_tile",
        fake_generate_single_mvt_tile,
    )

    assert tiler.get_tile(1, 0, 0) == generated_bytes
    assert tiler.cache.store[(1, 0, 0, None)] == generated_bytes
    assert captured["attributes"] is None


def test_get_tile_returns_prebuilt_tile_when_attributes_are_requested(temp_dir, monkeypatch):
    dataset_dir = temp_dir / "tile_dataset"
    (dataset_dir / "parquet_tiles").mkdir(parents=True)
    tile_path = dataset_dir / "mvt" / "0" / "0" / "0.mvt"
    tile_path.parent.mkdir(parents=True)
    disk_bytes = b"disk-tile"
    tile_path.write_bytes(disk_bytes)

    def fail_generate_single_mvt_tile(*args, **kwargs):
        raise AssertionError("prebuilt tiles should be returned without generation")

    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.generate_single_mvt_tile",
        fail_generate_single_mvt_tile,
    )

    output = {}
    data = starlet.get_tile(dataset_dir, 0, 0, 0, output=output, attributes=["name", "id"])

    assert data == disk_bytes
    assert output["source"] == "disk"
    assert output["generation"] == "read_from_disk"
    assert "attributes" not in output


def test_get_tile_forwards_attribute_whitelist_on_generated_miss(temp_dir, monkeypatch):
    dataset_dir = temp_dir / "tile_dataset"
    (dataset_dir / "parquet_tiles").mkdir(parents=True)

    generated_bytes = b"generated-tile"
    captured = {}

    def fake_generate_single_mvt_tile(dataset_root, tile_coords, **kwargs):
        captured["dataset_root"] = dataset_root
        captured["tile_coords"] = tile_coords
        captured.update(kwargs)
        return generated_bytes

    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.generate_single_mvt_tile",
        fake_generate_single_mvt_tile,
    )

    output = {}
    data = starlet.get_tile(dataset_dir, 0, 0, 0, output=output, attributes=["name", "id"])

    assert data == generated_bytes
    assert output["source"] == "generated"
    assert output["attributes"] == ["id", "name"]
    assert captured["attributes"] == ("id", "name")


def test_query_dataset_uses_indexed_tile(temp_dir, sample_parquet_file):
    dataset_dir = temp_dir / "indexed_dataset"
    tiles_dir = dataset_dir / "parquet_tiles"
    tiles_dir.mkdir(parents=True)
    indexed_name = "tile_000000__0_0_0_0_100_0_100_0.parquet"
    shutil.copy(sample_parquet_file, tiles_dir / indexed_name)

    batches = list(starlet.query_dataset(dataset_dir, (0, 0, 15, 15)))
    result = batches[0]

    assert len(batches) == 1
    assert not result.empty
    assert set(result["id"]) == {0}


def test_query_dataset_hides_internal_tile_columns(temp_dir, sample_parquet_table):
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_dir = temp_dir / "indexed_dataset"
    tiles_dir = dataset_dir / "parquet_tiles"
    tiles_dir.mkdir(parents=True)
    tile_path = tiles_dir / "tile_000000__0_0_0_0_100_0_100_0.parquet"
    internal_table = sample_parquet_table.append_column(
        "_tile_id",
        pa.array([0] * sample_parquet_table.num_rows, type=pa.int64()),
    )
    internal_table = internal_table.append_column(
        "_id",
        pa.array(list(range(sample_parquet_table.num_rows)), type=pa.int64()),
    )
    for col in ("_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax"):
        internal_table = internal_table.append_column(
            col,
            pa.array([0.0] * sample_parquet_table.num_rows, type=pa.float64()),
        )
    pq.write_table(internal_table, tile_path)

    result = next(starlet.query_dataset(dataset_dir, (0, 0, 15, 15)))
    record = starlet.get_sample_record(dataset_dir, (0, 0, 15, 15))

    internal_columns = {"_id", "_tile_id", "_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax"}
    assert internal_columns.isdisjoint(result.columns)
    assert record is not None
    assert internal_columns.isdisjoint(record)


def test_query_dataset_count_size_and_sample(temp_dir, sample_parquet_file):
    dataset_dir = temp_dir / "indexed_dataset"
    tiles_dir = dataset_dir / "parquet_tiles"
    tiles_dir.mkdir(parents=True)
    indexed_name = "tile_000000__0_0_0_0_100_0_100_0.parquet"
    shutil.copy(sample_parquet_file, tiles_dir / indexed_name)

    assert starlet.query_dataset_count(dataset_dir, (0, 0, 100, 100), batch_size=2) == 5
    assert starlet.query_dataset_size(dataset_dir, (0, 0, 100, 100), batch_size=2) > 0
    record = starlet.get_sample_record(dataset_dir, (0, 0, 15, 15))
    assert record is not None
    assert record["id"] == 0


def test_missing_metadata_and_summary(temp_dir):
    dataset_dir = temp_dir / "dataset"
    dataset_dir.mkdir()

    metadata = starlet.get_dataset_metadata(dataset_dir)
    assert metadata["has_stats"] is False
    assert "parquet_tiles" in metadata["missing"]
    assert starlet.get_dataset_summary(dataset_dir) is None


def test_delete_dataset(temp_dir):
    dataset_dir = temp_dir / "delete_me"
    dataset_dir.mkdir()

    assert starlet.delete_dataset(temp_dir, "delete_me") is True
    assert not dataset_dir.exists()
    assert starlet.delete_dataset(temp_dir, "delete_me", missing_ok=True) is False


def test_add_dataset_async_returns_result(monkeypatch, temp_dir):
    expected = ("tile", "mvt", None)

    def fake_add_dataset(*args, **kwargs):
        return expected

    monkeypatch.setattr("starlet.api.add_dataset", fake_add_dataset)

    source = temp_dir / "source.geojson"
    source.write_text("{}")
    handle = starlet.add_dataset_async(source, temp_dir, name="async_dataset")

    assert handle.result(timeout=2) == expected
    assert handle.status == "succeeded"
    assert handle.as_dict()["dataset"] == "async_dataset"


def test_async_handle_cancel_before_start(temp_dir):
    source = temp_dir / "source.geojson"
    source.write_text("{}")
    handle = AsyncDatasetHandle(
        input_path=source,
        datasets_dir=temp_dir,
        name="cancelled_dataset",
        overwrite=False,
        build_kwargs={},
    )

    assert handle.cancel() is True
    handle.start()

    assert handle.join(timeout=2) is True
    assert handle.status == "cancelled"


def test_add_dataset_async_cancel_running(monkeypatch, temp_dir):
    def slow_add_dataset(*args, **kwargs):
        time.sleep(0.2)
        return ("tile", "mvt", None)

    monkeypatch.setattr("starlet.api.add_dataset", slow_add_dataset)

    source = temp_dir / "source.geojson"
    source.write_text("{}")
    handle = starlet.add_dataset_async(source, temp_dir)
    cancelled = handle.cancel()

    assert cancelled is True
    assert handle.result(timeout=2) == ("tile", "mvt", None)
    assert handle.status == "succeeded"
    assert handle.cancel_requested is True
