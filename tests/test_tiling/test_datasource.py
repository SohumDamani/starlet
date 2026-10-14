"""Unit tests for data source readers.

Tests cover:
- GeoParquetSource reading and iteration
- GeoJSONSource reading and iteration
- Column detection (geometry column)
- Error handling for missing files
- Schema validation
"""
import bz2
import json
import logging
import os
import struct
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely.geometry import Point
from shapely import wkb

from starlet._internal.tiling.datasource import (
    GeoJSONSource,
    GeoJSONSplit,
    GeoParquetSource,
    GeoParquetSplit,
    CSVSource,
    BZ2DecompressedBlock,
    GDBSource,
    ShapefileSource,
    _read_bz2_split_payload,
    _properties_dataframe_to_arrow_table,
    read_spatial_sample,
    source_for_path,
)
from starlet._internal.tiling.geojson_source import (
    _iter_bz2_geojson_batches,
    _iter_feature_collection_batches_from_bytes,
    _read_geojson_partition_spatial_sample,
    iter_geojson_xy,
)
from starlet._internal.tiling.csv_source import CSVSplit, _iter_bz2_csv_split_byte_batches
from starlet._internal.tiling.geoparquet_source import _read_geoparquet_split_spatial_sample
from starlet._internal.tiling.partition_reader import GeoJSONPartitionReader
from starlet._internal.tiling.vector_source import _zip_gdb_member_dirs


def _linestring_wkb(coords):
    data = bytearray(struct.pack("<BI", 1, 2))
    data.extend(struct.pack("<I", len(coords)))
    for x, y in coords:
        data.extend(struct.pack("<dd", x, y))
    return bytes(data)


def _multicurve_wkb(lines):
    data = bytearray(struct.pack("<BI", 1, 11))
    data.extend(struct.pack("<I", len(lines)))
    for coords in lines:
        data.extend(_linestring_wkb(coords))
    return bytes(data)


def _write_bz2(path: Path, data: bytes, *, compresslevel: int = 1) -> None:
    path.write_bytes(bz2.compress(data, compresslevel=compresslevel))


def _write_zip_from_dir(zip_path: Path, source_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        for member in sorted(source_dir.rglob("*")):
            if member.is_file():
                archive.write(member, arcname=member.relative_to(source_dir).as_posix())


class TestGeoParquetSource:
    """Test GeoParquet data source."""

    def test_init_with_file(self, sample_parquet_file):
        """Test initializing with a valid Parquet file."""
        source = GeoParquetSource(str(sample_parquet_file))
        # Source should be initialized successfully
        assert source is not None
        assert source.schema() is not None

    def test_init_with_missing_file(self, temp_dir):
        """Test that missing file raises appropriate error."""
        missing_file = temp_dir / "nonexistent.parquet"
        with pytest.raises(Exception):  # FileNotFoundError or similar
            GeoParquetSource(str(missing_file))

    def test_detect_geometry_column(self, sample_parquet_file):
        """Test that geometry column is in schema."""
        source = GeoParquetSource(str(sample_parquet_file))
        schema = source.schema()
        # Should have geometry column
        assert 'geometry' in schema.names

    def test_read_geometries(self, sample_parquet_file):
        """Test reading and decoding geometries."""
        source = GeoParquetSource(str(sample_parquet_file))

        # Use iter_tables() to read data
        for table in source.iter_tables():
            geoms_wkb = table['geometry'].to_pylist()
            # Decode first geometry
            geom = wkb.loads(geoms_wkb[0])
            assert geom is not None
            assert geom.geom_type == 'Polygon'
            break  # Only test first batch

    def test_read_all_columns(self, sample_parquet_file):
        """Test reading all columns including attributes."""
        source = GeoParquetSource(str(sample_parquet_file))

        for table in source.iter_tables():
            assert 'geometry' in table.column_names
            assert 'id' in table.column_names
            assert 'name' in table.column_names
            break  # Only test first batch

    def test_geometry_only_reads_only_geometry_column(self, sample_parquet_file):
        """Test geometry-only projection avoids reading attribute columns."""
        source = GeoParquetSource(str(sample_parquet_file), geometry_only=True)

        for table in source.iter_tables():
            assert table.column_names == ["geometry"]
            break

    def test_detects_geoparquet_primary_geometry_column(self, temp_dir):
        parquet_path = temp_dir / "shape_geom.parquet"
        geo = {
            "version": "1.1.0",
            "primary_column": "SHAPE",
            "columns": {"SHAPE": {"encoding": "WKB", "crs": "EPSG:4326"}},
        }
        table = pa.table({
            "id": [1],
            "SHAPE": [wkb.dumps(Point(1, 2))],
        }).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
        pq.write_table(table, str(parquet_path))

        source = GeoParquetSource(str(parquet_path), geometry_only=True)
        result = next(source.iter_tables())

        assert source.geom_col == "SHAPE"
        assert result.column_names == ["SHAPE"]
        assert wkb.loads(result["SHAPE"][0].as_py()).equals(Point(1, 2))

    def test_geoparquet_spatial_sample_uses_detected_geometry_column(self, temp_dir):
        parquet_path = temp_dir / "shape_sample.parquet"
        geo = {
            "version": "1.1.0",
            "primary_column": "SHAPE",
            "columns": {"SHAPE": {"encoding": "WKB", "crs": "EPSG:4326"}},
        }
        table = pa.table({
            "id": [1],
            "SHAPE": [wkb.dumps(Point(1, 2))],
        }).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
        pq.write_table(table, str(parquet_path))

        source = GeoParquetSource(str(parquet_path), geometry_only=True)
        sample = _read_geoparquet_split_spatial_sample(
            str(parquet_path),
            source.create_splits()[0],
            source.geom_col,
            sample_cap=None,
            seed=42,
        )

        assert sample.total_seen == 1
        assert sample.sample_points.tolist() == [[1.0], [2.0]]

    def test_geoparquet_source_preserves_native_crs(self, temp_dir):
        lon, lat = -118.25, 34.05
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(lon, lat)
        table = pa.table({
            "geometry": [wkb.dumps(Point(x, y))],
            "id": [1],
        })
        geo = {
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:3857"}},
        }
        table = table.replace_schema_metadata({
            b"geo": json.dumps(geo).encode("utf-8"),
        })
        parquet_path = temp_dir / "mercator.parquet"
        pq.write_table(table, str(parquet_path))

        source = GeoParquetSource(str(parquet_path))
        result = next(source.iter_tables())
        point = wkb.loads(result["geometry"][0].as_py())

        assert point.x == pytest.approx(x)
        assert point.y == pytest.approx(y)
        geo = json.loads(result.schema.metadata[b"geo"].decode("utf-8"))
        assert geo["columns"]["geometry"]["crs"] == "EPSG:3857"

    def test_multiple_row_groups(self, temp_dir, sample_polygons):
        """Test reading Parquet file with multiple row groups."""
        # Create file with multiple row groups
        geoms = [wkb.dumps(g) for g in sample_polygons]
        table = pa.table({
            'geometry': geoms,
            'id': list(range(len(geoms)))
        })

        file_path = temp_dir / "multi_rg.parquet"
        pq.write_table(table, str(file_path), row_group_size=2)

        source = GeoParquetSource(str(file_path))
        # Check that we can iterate through all row groups
        tables = list(source.iter_tables())
        assert len(tables) > 1

    def test_geoparquet_splits_read_independently(self, temp_dir, sample_polygons):
        """Test row-group splits can be read independently."""
        geoms = [wkb.dumps(g) for g in sample_polygons]
        table = pa.table({
            'geometry': geoms,
            'id': list(range(len(geoms)))
        })

        file_path = temp_dir / "split_rg.parquet"
        pq.write_table(table, str(file_path), row_group_size=2)

        source = GeoParquetSource(str(file_path))
        splits = source.create_splits()
        assert len(splits) > 1
        assert all(isinstance(split, GeoParquetSplit) for split in splits)

        row_count = sum(
            table.num_rows
            for split in splits
            for table in source.iter_tables(split)
        )
        assert row_count == len(geoms)

    def test_directory_source_splits_all_geoparquet_files(self, temp_dir):
        """Test a directory source includes row groups from every Parquet file."""
        data_dir = temp_dir / "parquet_parts"
        data_dir.mkdir()
        for part, start in enumerate((0, 2)):
            pq.write_table(
                pa.table({
                    "geometry": [wkb.dumps(Point(start, start))],
                    "id": [start],
                }),
                str(data_dir / f"part-{part}.parquet"),
            )

        source = GeoParquetSource(str(data_dir))
        splits = source.create_splits()

        assert {Path(split.path).name for split in splits} == {"part-0.parquet", "part-1.parquet"}
        assert sum(
            table.num_rows
            for split in splits
            for table in source.iter_tables(split)
        ) == 2


class TestGeoJSONSpatialSampleErrors:
    def test_bz2_split_payload_does_not_read_entire_file(self, temp_dir, monkeypatch):
        path = temp_dir / "guarded.geojson.bz2"
        payload = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": i, "blob": os.urandom(64).hex()},
                    "geometry": {"type": "Point", "coordinates": [float(i), float(i)]},
                }
                for i in range(4_000)
            ],
        }).encode("utf-8")
        _write_bz2(path, payload, compresslevel=1)
        original_open = open

        class NoReadAll:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __enter__(self):
                self._wrapped.__enter__()
                return self

            def __exit__(self, *args):
                return self._wrapped.__exit__(*args)

            def read(self, size=-1):
                if size is None or size < 0:
                    raise AssertionError("bz2 split reader should not read the whole file")
                return self._wrapped.read(size)

            def seek(self, *args):
                return self._wrapped.seek(*args)

            def tell(self):
                return self._wrapped.tell()

        def guarded_open(file, mode="r", *args, **kwargs):
            handle = original_open(file, mode, *args, **kwargs)
            if Path(file) == path and "b" in mode:
                return NoReadAll(handle)
            return handle

        monkeypatch.setattr("builtins.open", guarded_open)

        decoded = _read_bz2_split_payload(
            str(path),
            offset=0,
            length=max(1, path.stat().st_size // 2),
        )

        assert decoded is not None
        assert decoded.payload

    def test_bz2_csv_batches_yield_before_reading_next_block(self, monkeypatch):
        consumed: list[int] = []

        def fake_blocks(*args, **kwargs):
            consumed.append(1)
            yield BZ2DecompressedBlock(b"1,0,0\n", 4, True, True, 100)
            consumed.append(2)
            yield BZ2DecompressedBlock(b"2,0,0\n", 10, True, False, 100)

        monkeypatch.setattr(
            "starlet._internal.tiling.csv_source._iter_bz2_decompressed_blocks",
            fake_blocks,
        )

        batches = _iter_bz2_csv_split_byte_batches(
            CSVSplit("points.csv.bz2", 0, 100),
            has_header=False,
        )

        assert next(batches) == b"1,0,0\n"
        assert consumed == [1]

    def test_bz2_geojson_batches_yield_before_reading_next_block(self, monkeypatch):
        consumed: list[int] = []

        def fake_blocks(*args, **kwargs):
            consumed.append(1)
            yield BZ2DecompressedBlock(
                b'{"type":"Feature","properties":{"id":1},"geometry":null}\n',
                4,
                True,
                True,
                100,
            )
            consumed.append(2)
            yield BZ2DecompressedBlock(
                b'{"type":"Feature","properties":{"id":2},"geometry":null}\n',
                10,
                True,
                False,
                100,
            )

        monkeypatch.setattr(
            "starlet._internal.tiling.geojson_source._iter_bz2_decompressed_blocks",
            fake_blocks,
        )

        batches = _iter_bz2_geojson_batches(
            "features.geojsonl.bz2",
            0,
            100,
            batch_size=1,
        )

        assert next(batches) == ['{"type":"Feature","properties":{"id":1},"geometry":null}']
        assert consumed == [1]

    def test_bz2_feature_collection_batches_skip_partial_leading_feature(self):
        payload = (
            b'\t\t"type": "Feature",\n'
            b'  "geometry": {"type": "Point", "coordinates": [0, 0]}\n'
            b'},\n'
            b'{"type": "Feature", "properties": {"id": 1}, '
            b'"geometry": {"type": "Point", "coordinates": [1, 2]}},\n'
            b'{"type": "Feature", "properties": {"id": 2}, '
            b'"geometry": {"type": "Point", "coordinates": [3, 4]}}'
        )

        batches = list(
            _iter_feature_collection_batches_from_bytes(
                payload,
                batch_size=10,
                owned_output_len=len(payload),
            )
        )

        assert [json.loads(feature)["properties"]["id"] for feature in batches[0]] == [1, 2]

    def test_bz2_feature_collection_batches_read_concatenated_streams(self, temp_dir):
        path = temp_dir / "concat.geojson.bz2"
        first_member = (
            b'{"type": "FeatureCollection", "features": ['
            b'{"type": "Feature", "properties": {"id": 1}, '
            b'"geometry": {"type": "Point", "coordinates": [1, 2]}},'
        )
        second_member = (
            b'{"type": "Feature", "properties": {"id": 2}, '
            b'"geometry": {"type": "Point", "coordinates": [3, 4]}}'
            b']}'
        )
        path.write_bytes(bz2.compress(first_member) + bz2.compress(second_member))

        batches = list(
            GeoJSONSource(str(path), batch_rows=10)._iter_feature_batches_for_split(
                GeoJSONSplit(str(path), 0, path.stat().st_size)
            )
        )

        assert [[feature["properties"]["id"] for feature in batch] for batch in batches] == [[1, 2]]

    def test_partition_error_includes_split_context(self, monkeypatch):
        def fail_iter(*args, **kwargs):
            raise ValueError('Could not find the object containing a "type": "Feature" member')

        monkeypatch.setattr(
            "starlet._internal.tiling.geojson_source._iter_feature_json_batches",
            fail_iter,
        )

        with pytest.raises(ValueError, match=r"split_index=7 .*broken\.geojson\.bz2.*offset=1024.*length=2048"):
            _read_geojson_partition_spatial_sample(
                7,
                "/tmp/broken.geojson.bz2",
                1024,
                2048,
                100,
                42,
            )

    def test_schema_validation(self, sample_parquet_file):
        """Test that schema is accessible and valid."""
        source = GeoParquetSource(str(sample_parquet_file))
        schema = source.schema()

        assert 'geometry' in schema.names
        # Geometry should be binary type
        geom_field = schema.field('geometry')
        assert pa.types.is_binary(geom_field.type)


class TestGeoJSONSource:
    """Test GeoJSON data source.

    Note: GeoJSONSource implementation may vary. These are placeholder tests
    that should be adapted based on the actual implementation.
    """

    def test_read_geojson_feature_collection(self, temp_dir):
        """Test reading a GeoJSON FeatureCollection."""
        # Create a simple GeoJSON file
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0.0, 0.0]
                    },
                    "properties": {"id": 1}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [10.0, 10.0]
                    },
                    "properties": {"id": 2}
                }
            ]
        }

        json_path = temp_dir / "test.geojson"
        with open(json_path, 'w') as f:
            json.dump(geojson, f)

        source = GeoJSONSource(str(json_path))
        source.set_schema(source.infer_schema())
        tables = list(source.iter_tables(*source.create_splits()))
        ids = sorted(
            row_id
            for table in tables
            for row_id in table["id"].to_pylist()
        )

        assert sum(table.num_rows for table in tables) == 2
        assert ids == [1, 2]
        assert "geometry" in tables[0].column_names

    def test_read_empty_geojson(self, temp_dir):
        """Test reading empty GeoJSON file."""
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        json_path = temp_dir / "empty.geojson"
        with open(json_path, 'w') as f:
            json.dump(geojson, f)

        source = GeoJSONSource(str(json_path))
        source.set_schema(source.infer_schema())
        assert list(source.iter_tables(*source.create_splits())) == []
        assert "geometry" in source.schema().names

    def test_partition_reader_returns_each_feature_once(self, temp_dir):
        """Test byte partitions align to complete Feature objects."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": i, "name": f"feature-{i}"},
                    "geometry": {"type": "Point", "coordinates": [float(i), float(i * 2)]},
                }
                for i in range(12)
            ],
        }

        json_path = temp_dir / "partitioned.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        file_size = json_path.stat().st_size
        payload = json_path.read_bytes()
        partition_size = max(1, file_size // 4)
        decoded = []

        for offset in range(0, file_size, partition_size):
            split_end = min(offset + partition_size, file_size)
            reader = GeoJSONPartitionReader(
                [
                    payload[offset:split_end],
                    b"\0",
                    payload[split_end:],
                ]
            )
            decoded.extend(json.loads(feature) for feature in reader)

        ids = sorted(feature["properties"]["id"] for feature in decoded)
        assert ids == list(range(12))

    def test_geojson_source_reads_feature_collection_in_parallel(self, temp_dir):
        """Test GeoJSONSource uses partitioned FeatureCollection reads."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": i},
                    "geometry": {"type": "Point", "coordinates": [float(i), float(i)]},
                }
                for i in range(20)
            ],
        }

        json_path = temp_dir / "parallel.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        source = GeoJSONSource(str(json_path), batch_rows=3)
        source.set_schema(source.infer_schema())
        tables = list(source.iter_tables(*source.create_splits()))
        ids = sorted(
            row_id
            for table in tables
            for row_id in table["id"].to_pylist()
        )

        assert ids == list(range(20))
        assert sum(table.num_rows for table in tables) == 20

    def test_geojson_source_extracts_crs_hint_from_file_header(self, temp_dir):
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "EPSG:3857"},
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1},
                    "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                }
            ],
        }

        json_path = temp_dir / "crs_hint.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        source = GeoJSONSource(str(json_path))

        assert source._crs_hint == "EPSG:3857"
        assert source.infer_schema().metadata[b"geo"].decode("utf-8")

    def test_geojson_nested_properties_have_stable_schema_across_batches(self, temp_dir):
        """Test dynamic JSON object properties do not infer different struct schemas."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1, "tagsMap": {"a": "1"}},
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                },
                {
                    "type": "Feature",
                    "properties": {"id": 2, "tagsMap": {"b": "2"}},
                    "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                },
            ],
        }

        json_path = temp_dir / "dynamic_tags.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        source = GeoJSONSource(str(json_path), batch_rows=1)
        source.set_schema(source.infer_schema())
        tables = list(source.iter_tables(*source.create_splits()))

        _tags_types = {table.schema.field("tagsMap").type for table in tables}
        assert _tags_types == {pa.map_(pa.string(), pa.string())}
        assert [dict(value) for table in tables for value in table["tagsMap"].to_pylist()] == [
            {"a": "1"},
            {"b": "2"},
        ]

    def test_geojson_null_first_batch_promotes_later_string_column(self, temp_dir):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"OLD_BLD_ID": None},
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                },
                {
                    "type": "Feature",
                    "properties": {"OLD_BLD_ID": "B123"},
                    "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                },
            ],
        }

        json_path = temp_dir / "null_then_string.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f)

        source = GeoJSONSource(str(json_path), batch_rows=1)
        source.set_schema(source.infer_schema())
        tables = list(source.iter_tables(*source.create_splits()))

        promoted_type = tables[-1].schema.field("OLD_BLD_ID").type
        assert pa.types.is_large_string(promoted_type) or pa.types.is_string(promoted_type)
        assert tables[-1]["OLD_BLD_ID"].to_pylist() == ["B123"]

    def test_geojson_mixed_scalar_property_values_promote_to_string(self, temp_dir):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"parcel_id": "A123"},
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                },
                {
                    "type": "Feature",
                    "properties": {"parcel_id": 123.5},
                    "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                },
            ],
        }

        json_path = temp_dir / "mixed_scalar_property.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f)

        source = GeoJSONSource(str(json_path), batch_rows=1)
        schema = source.infer_schema()
        source.set_schema(schema)
        tables = list(source.iter_tables(*source.create_splits()))

        assert pa.types.is_large_string(schema.field("parcel_id").type)
        assert {table.schema.field("parcel_id").type for table in tables} == {
            pa.large_string()
        }
        assert [value for table in tables for value in table["parcel_id"].to_pylist()] == [
            "A123",
            "123.5",
        ]

    def test_geojson_numeric_property_type_is_stable_across_batches(self, temp_dir):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"TOTAL_UNITS": 10},
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                },
                {
                    "type": "Feature",
                    "properties": {"TOTAL_UNITS": 2.5, "STATUS": "active"},
                    "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                },
            ],
        }

        json_path = temp_dir / "numeric_promotion.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f)

        source = GeoJSONSource(str(json_path), batch_rows=1)
        schema = source.infer_schema()
        source.set_schema(schema)
        tables = list(source.iter_tables(*source.create_splits()))

        assert schema.field("TOTAL_UNITS").type == pa.float64()
        assert schema.field("STATUS").type == pa.string()
        assert all(table.schema.equals(schema) for table in tables)
        assert [value for table in tables for value in table["TOTAL_UNITS"].to_pylist()] == [
            10.0,
            2.5,
        ]
        assert [value for table in tables for value in table["STATUS"].to_pylist()] == [
            None,
            "active",
        ]

    def test_geojson_arrow_inference_failure_promotes_property_to_string(self):
        table = _properties_dataframe_to_arrow_table(
            pd.DataFrame({"parcel_id": [b"A123", 123.5]})
        )

        assert pa.types.is_large_string(table.schema.field("parcel_id").type)
        assert table["parcel_id"].to_pylist() == ["b'A123'", "123.5"]

    def test_geojson_splits_read_independently_from_threads(self, temp_dir):
        """Test GeoJSON byte splits can be read independently by threads."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": i},
                    "geometry": {"type": "Point", "coordinates": [float(i), float(i)]},
                }
                for i in range(24)
            ],
        }

        json_path = temp_dir / "threaded_splits.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        source = GeoJSONSource(str(json_path), batch_rows=4)
        source.set_schema(source.infer_schema())
        file_size = json_path.stat().st_size
        split_size = max(1, (file_size + 3) // 4)
        splits = [
            GeoJSONSplit(
                path=str(json_path),
                offset=offset,
                length=min(split_size, file_size - offset),
            )
            for offset in range(0, file_size, split_size)
        ]
        assert len(splits) == 4
        assert all(isinstance(split, GeoJSONSplit) for split in splits)

        def read_split(split):
            return [
                row_id
                for table in source.iter_tables(split)
                for row_id in table["id"].to_pylist()
            ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            ids = sorted(
                row_id
                for part in executor.map(read_split, splits)
                for row_id in part
            )

        assert ids == list(range(24))

    def test_directory_source_splits_all_geojson_files(self, temp_dir):
        """Test a directory source includes splits from every GeoJSON file."""
        data_dir = temp_dir / "geojson_parts"
        data_dir.mkdir()
        for part, feature_id in enumerate((1, 2)):
            (data_dir / f"part-{part}.geojson").write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {"id": feature_id},
                    "geometry": {"type": "Point", "coordinates": [feature_id, feature_id]},
                }],
            }))

        source = GeoJSONSource(str(data_dir))
        source.set_schema(source.infer_schema())
        splits = source.create_splits()
        ids = sorted(
            row_id
            for split in splits
            for table in source.iter_tables(split)
            for row_id in table["id"].to_pylist()
        )

        assert {Path(split.path).name for split in splits} == {"part-0.geojson", "part-1.geojson"}
        assert ids == [1, 2]

    def test_bzip2_feature_collection_splits_read_features_once(self, temp_dir):
        features = [
            {
                "type": "Feature",
                "properties": {
                    "id": i,
                    "blob": os.urandom(120).hex(),
                },
                "geometry": {"type": "Point", "coordinates": [float(i), float(i)]},
            }
            for i in range(8_000)
        ]
        payload = json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")
        json_path = temp_dir / "features.geojson.bz2"
        _write_bz2(json_path, payload, compresslevel=1)

        source = GeoJSONSource(str(json_path), batch_rows=256)
        source.set_schema(source.infer_schema())
        splits = source.create_splits(num_splits=4)
        ids = [
            row_id
            for split in splits
            for table in source.iter_tables(split)
            for row_id in table["id"].to_pylist()
        ]

        assert len(splits) == 4
        assert ids == list(range(8_000))

    def test_read_spatial_sample_returns_sample_and_schema(self, temp_dir):
        """Test standalone sampling reads centroids and schema from a file."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1},
                    "geometry": {"type": "Point", "coordinates": [0.0, 2.0]},
                },
                {
                    "type": "Feature",
                    "properties": {"id": 2},
                    "geometry": {"type": "Point", "coordinates": [10.0, 12.0]},
                },
            ],
        }

        json_path = temp_dir / "sample.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f)

        spatial_sample = read_spatial_sample(
            str(json_path),
            sample_cap=None,
            seed=42,
            geojson_workers=1,
        )

        assert spatial_sample.total_seen == 2
        assert spatial_sample.total_sampled == 2
        assert spatial_sample.sample_points.shape == (2, 2)
        assert spatial_sample.schema is not None
        assert spatial_sample.schema.field("id").type == pa.int64()
        assert spatial_sample.schema.field("geometry").type == pa.binary()

    def test_geojson_sampling_schema_can_be_reused_for_tiling(self, temp_dir, monkeypatch):
        data_dir = temp_dir / "sample_schema"
        data_dir.mkdir()
        for index, total_units in enumerate((10, 2.5)):
            (data_dir / f"part-{index}.geojson").write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {"TOTAL_UNITS": total_units},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(index), float(index)],
                    },
                }],
            }))

        spatial_sample = read_spatial_sample(
            str(data_dir),
            sample_cap=None,
            seed=42,
            geojson_workers=1,
        )
        source = GeoJSONSource(str(data_dir), batch_rows=1)
        source.set_schema(spatial_sample.schema)
        monkeypatch.setattr(
            source,
            "_iter_feature_batches_for_split",
            lambda split: (_ for _ in ()).throw(AssertionError("schema rescanned input")),
        )

        assert source.schema().field("TOTAL_UNITS").type == pa.float64()

    def test_read_spatial_sample_splits_geojson_sample_cap(self, temp_dir):
        """Test GeoJSON partition sampling respects the total requested cap."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": i},
                    "geometry": {"type": "Point", "coordinates": [float(i), float(i * 2)]},
                }
                for i in range(24)
            ],
        }

        json_path = temp_dir / "capped.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        spatial_sample = read_spatial_sample(
            str(json_path),
            sample_cap=4,
            seed=42,
            geojson_workers=4,
        )

        assert spatial_sample.total_seen == 24
        assert spatial_sample.total_sampled <= 4

    def test_read_spatial_sample_geoparquet_uses_parallel_splits(self, temp_dir):
        """Test GeoParquet sampling merges row-group splits under one cap."""
        points = [Point(float(i), float(i + 1)) for i in range(10)]
        table = pa.table({
            "geometry": [wkb.dumps(point) for point in points],
            "id": list(range(10)),
        })
        parquet_path = temp_dir / "points.parquet"
        pq.write_table(table, str(parquet_path), row_group_size=2)

        spatial_sample = read_spatial_sample(
            str(parquet_path),
            sample_cap=3,
            seed=42,
            geoparquet_workers=2,
        )

        assert spatial_sample.total_seen == 10
        assert spatial_sample.total_sampled == 3
        assert spatial_sample.batches_read == 5
        assert spatial_sample.sample_points.shape == (2, 3)

    def test_iter_geojson_xy_walks_geometry_collections(self):
        """Test stack traversal over geometry objects and nested coordinates."""
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "GeometryCollection",
                "geometries": [
                    {"type": "Point", "coordinates": [1.0, 2.0]},
                    {
                        "type": "LineString",
                        "coordinates": [[3.0, 4.0], [5.0, 6.0]],
                    },
                    {
                        "type": "GeometryCollection",
                        "geometries": [
                            {
                                "type": "Polygon",
                                "coordinates": [[
                                    [7.0, 8.0],
                                    [9.0, 10.0],
                                    [7.0, 8.0],
                                ]],
                            }
                        ],
                    },
                ],
            },
        }

        assert list(iter_geojson_xy(json.dumps(feature))) == [
            (1.0, 2.0),
            (3.0, 4.0),
            (5.0, 6.0),
            (7.0, 8.0),
            (9.0, 10.0),
            (7.0, 8.0),
        ]


class TestCSVSource:
    def test_xy_columns_are_converted_to_geometry(self, temp_dir):
        csv_path = temp_dir / "points.csv"
        csv_path.write_text("id,x,y\n1,0,1\n2,2,3\n")

        source = CSVSource(str(csv_path), x_col="x", y_col="y")
        tables = list(source.iter_tables())

        assert len(tables) == 1
        assert tables[0].column_names == ["id", "x", "y", "geometry"]
        assert tables[0]["id"].to_pylist() == [1, 2]
        assert wkb.loads(tables[0]["geometry"][0].as_py()).equals(Point(0, 1))

    def test_headerless_xy_indexes_are_converted_to_geometry(self, temp_dir):
        csv_path = temp_dir / "points.csv"
        csv_path.write_text("1,0,1\n2,2,3\n")

        source = CSVSource(str(csv_path), x_col=1, y_col=2)
        tables = list(source.iter_tables())

        assert len(tables) == 1
        assert tables[0].column_names == ["column_0", "column_1", "column_2", "geometry"]
        assert tables[0]["column_0"].to_pylist() == [1, 2]
        assert wkb.loads(tables[0]["geometry"][0].as_py()).equals(Point(0, 1))

    def test_headerless_txt_uses_whitespace_delimiter(self, temp_dir):
        csv_path = temp_dir / "points.txt"
        csv_path.write_text("1 0 1\n2 2 3\n")

        source = CSVSource(str(csv_path), x_col=1, y_col=2)
        tables = list(source.iter_tables())

        assert len(tables) == 1
        assert tables[0].column_names == ["column_0", "column_1", "column_2", "geometry"]
        assert tables[0]["column_0"].to_pylist() == [1, 2]
        assert wkb.loads(tables[0]["geometry"][0].as_py()).equals(Point(0, 1))

    def test_headerless_wkt_index_can_be_geometry_only(self, temp_dir):
        csv_path = temp_dir / "points.csv"
        csv_path.write_text("1,POINT (0 1)\n2,POINT (2 3)\n")

        source = CSVSource(str(csv_path), wkt_col=1, geometry_only=True)
        table = next(source.iter_tables())

        assert table.column_names == ["geometry"]
        assert wkb.loads(table["geometry"][0].as_py()).equals(Point(0, 1))

    def test_byte_splits_read_complete_rows_once(self, temp_dir):
        csv_path = temp_dir / "points.csv"
        csv_path.write_text(
            "id,x,y,name\n"
            "1,0,1,alpha\n"
            "2,2,3,beta\n"
            "3,4,5,gamma\n"
            "4,6,7,delta\n"
        )

        source = CSVSource(str(csv_path), x_col="x", y_col="y")
        splits = source.create_splits(num_splits=5)
        ids = [
            row_id
            for split in splits
            for table in source.iter_tables(split)
            for row_id in table["id"].to_pylist()
        ]

        assert ids == [1, 2, 3, 4]

    def test_bzip2_splits_read_complete_rows_once(self, temp_dir):
        csv_path = temp_dir / "points.csv.bz2"
        header = "id,x,y,name\n".encode("utf-8")
        rows = [
            f"{i},{i},{i + 1},{os.urandom(96).hex()}\n".encode("utf-8")
            for i in range(14_000)
        ]
        _write_bz2(csv_path, header + b"".join(rows), compresslevel=1)

        source = CSVSource(str(csv_path), x_col="x", y_col="y")
        splits = source.create_splits(num_splits=4)
        ids = [
            row_id
            for split in splits
            for table in source.iter_tables(split)
            for row_id in table["id"].to_pylist()
        ]

        assert len(splits) == 4
        assert ids == list(range(14_000))

    def test_headerless_byte_splits_read_complete_rows_once(self, temp_dir):
        csv_path = temp_dir / "points.csv"
        csv_path.write_text(
            "1,0,1,alpha\n"
            "2,2,3,beta\n"
            "3,4,5,gamma\n"
            "4,6,7,delta\n"
        )

        source = CSVSource(str(csv_path), x_col=1, y_col=2)
        splits = source.create_splits(num_splits=5)
        ids = [
            row_id
            for split in splits
            for table in source.iter_tables(split)
            for row_id in table["column_0"].to_pylist()
        ]

        assert ids == [1, 2, 3, 4]

    def test_create_splits_does_not_open_csv(self, temp_dir, monkeypatch):
        csv_path = temp_dir / "points.csv"
        csv_path.write_text("id,x,y\n1,0,1\n2,2,3\n")
        source = CSVSource(str(csv_path), x_col="x", y_col="y")

        def fail_open(*args, **kwargs):
            raise AssertionError("create_splits should not read CSV contents")

        monkeypatch.setattr("builtins.open", fail_open)

        splits = source.create_splits(num_splits=2)

        assert splits
        assert splits[0].offset == 0

    def test_wkt_column_can_be_geometry_only(self, temp_dir):
        csv_path = temp_dir / "points.csv"
        csv_path.write_text("id,wkt\n1,POINT (0 1)\n2,POINT (2 3)\n")

        source = CSVSource(str(csv_path), wkt_col="wkt", geometry_only=True)
        table = next(source.iter_tables())

        assert table.column_names == ["geometry"]
        assert wkb.loads(table["geometry"][0].as_py()).equals(Point(0, 1))

    def test_source_for_path_detects_csv(self, temp_dir):
        csv_path = temp_dir / "points.csv"
        csv_path.write_text("id,x,y\n1,0,1\n")

        source = source_for_path(str(csv_path), geom_col="geom", csv_x_col="x", csv_y_col="y")

        assert isinstance(source, CSVSource)
        assert next(source.iter_tables()).column_names[-1] == "geom"

    def test_source_for_path_detects_headerless_csv_with_indexes(self, temp_dir):
        csv_path = temp_dir / "points.csv"
        csv_path.write_text("1,0,1\n")

        source = source_for_path(str(csv_path), geom_col="geom", csv_x_index=1, csv_y_index=2)

        assert isinstance(source, CSVSource)
        table = next(source.iter_tables())
        assert table.column_names == ["column_0", "column_1", "column_2", "geom"]

    def test_source_for_path_detects_bzip2_csv(self, temp_dir):
        csv_path = temp_dir / "points.csv.bz2"
        _write_bz2(csv_path, b"id,x,y\n1,0,1\n")

        source = source_for_path(str(csv_path), geom_col="geom", csv_x_col="x", csv_y_col="y")

        assert isinstance(source, CSVSource)
        assert next(source.iter_tables()).column_names[-1] == "geom"

    def test_read_spatial_sample_uses_csv_geometry_columns(self, temp_dir):
        csv_path = temp_dir / "points.csv"
        csv_path.write_text("id,x,y\n1,0,1\n2,2,3\n")

        sample = read_spatial_sample(
            str(csv_path),
            csv_x_col="x",
            csv_y_col="y",
            source_workers=1,
        )

        assert sample.total_seen == 2
        assert sample.total_sampled == 2
        assert sample.sample_points.shape == (2, 2)

    def test_read_spatial_sample_uses_headerless_csv_indexes(self, temp_dir):
        csv_path = temp_dir / "points.csv"
        csv_path.write_text("1,0,1\n2,2,3\n")

        sample = read_spatial_sample(
            str(csv_path),
            csv_x_index=1,
            csv_y_index=2,
            source_workers=1,
        )

        assert sample.total_seen == 2
        assert sample.total_sampled == 2
        assert sample.sample_points.shape == (2, 2)
        assert sample.schema is not None
        assert sample.schema.names == ["column_0", "column_1", "column_2", "geometry"]

    def test_csv_sampling_unifies_schema_across_splits(self, temp_dir):
        csv_path = temp_dir / "mixed_types.csv"
        csv_path.write_text(
            "id,x,y,units,active,mixed,optional,code,zip_code\n"
            "1,0,1,10,true,1,,001,00123\n"
            "2,2,3,2.5,false,label,7,ABC,00456\n"
        )

        sample = read_spatial_sample(
            str(csv_path),
            csv_x_col="x",
            csv_y_col="y",
            csv_split_size=30,
            source_workers=1,
        )

        assert sample.schema is not None
        assert sample.schema.field("units").type == pa.float64()
        assert sample.schema.field("active").type == pa.bool_()
        assert sample.schema.field("mixed").type == pa.large_string()
        assert sample.schema.field("optional").type == pa.int64()
        assert sample.schema.field("code").type == pa.string()
        assert sample.schema.field("zip_code").type == pa.string()

        source = CSVSource(
            str(csv_path),
            x_col="x",
            y_col="y",
            split_size=30,
        )
        source.set_schema(sample.schema)
        tables = list(source.iter_tables())

        assert all(table.schema.equals(sample.schema) for table in tables)
        combined = pa.concat_tables(tables)
        assert combined["units"].to_pylist() == [10.0, 2.5]
        assert combined["active"].to_pylist() == [True, False]
        assert combined["mixed"].to_pylist() == ["1", "label"]
        assert combined["optional"].to_pylist() == [None, 7]
        assert combined["code"].to_pylist() == ["001", "ABC"]
        assert combined["zip_code"].to_pylist() == ["00123", "00456"]

    def test_csv_source_preserves_native_crs(self, temp_dir):
        lon, lat = -118.25, 34.05
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(lon, lat)
        csv_path = temp_dir / "mercator_points.csv"
        csv_path.write_text(f"id,x,y\n1,{x},{y}\n")

        source = CSVSource(
            str(csv_path),
            x_col="x",
            y_col="y",
            src_crs="EPSG:3857",
        )
        table = next(source.iter_tables())
        point = wkb.loads(table["geometry"][0].as_py())

        assert point.x == pytest.approx(x)
        assert point.y == pytest.approx(y)
        geo = json.loads(table.schema.metadata[b"geo"].decode("utf-8"))
        assert geo["columns"]["geometry"]["crs"] == "EPSG:3857"


class TestShapefileSource:
    def test_reads_shapefile_and_geometry_only_projection(self, temp_dir):
        shp_path = temp_dir / "points.shp"
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2], "name": ["a", "b"]},
            geometry=[Point(0, 1), Point(2, 3)],
            crs="EPSG:4326",
        )
        gdf.to_file(shp_path, engine="pyogrio")

        source = ShapefileSource(str(shp_path), geometry_only=True)
        table = next(source.iter_tables())

        assert table.column_names == ["geometry"]
        assert wkb.loads(table["geometry"][0].as_py()).equals(Point(0, 1))

    def test_source_for_path_detects_shapefile(self, temp_dir):
        shp_path = temp_dir / "points.shp"
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 1)], crs="EPSG:4326")
        gdf.to_file(shp_path, engine="pyogrio")

        source = source_for_path(str(shp_path))

        assert isinstance(source, ShapefileSource)

    def test_preserves_shapefile_source_crs(self, temp_dir):
        lon, lat = -118.25, 34.05
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(lon, lat)
        shp_path = temp_dir / "mercator_points.shp"
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Point(x, y)],
            crs="EPSG:3857",
        )
        gdf.to_file(shp_path, engine="pyogrio")

        source = ShapefileSource(str(shp_path), geometry_only=True)
        table = next(source.iter_tables())
        point = wkb.loads(table["geometry"][0].as_py())

        assert point.x == pytest.approx(x)
        assert point.y == pytest.approx(y)
        geo = json.loads(table.schema.metadata[b"geo"].decode("utf-8"))
        assert "3857" in str(geo["columns"]["geometry"]["crs"])

    def test_linearizes_nonlinear_geometry_on_read_error(self, temp_dir, monkeypatch, caplog):
        import pyogrio

        shp_path = temp_dir / "curves.shp"
        shp_path.touch()

        def read_arrow_with_actual_multicurve(*args, **kwargs):
            return (
                {"geometry_name": "SHAPE", "fid_column": "OBJECTID"},
                pa.table(
                    {
                        "OBJECTID": [123],
                        "SHAPE": [_multicurve_wkb([[(0.0, 0.0), (1.0, 1.0)]])],
                    }
                ),
            )

        monkeypatch.setattr(pyogrio, "list_layers", lambda path: [["curves", "CurvePolygon"]])
        monkeypatch.setattr(
            pyogrio,
            "read_info",
            lambda path, layer=None, force_feature_count=False: {
                "features": 1,
                "geometry_type": "CurvePolygon",
            },
        )
        monkeypatch.setattr(pyogrio, "read_arrow", read_arrow_with_actual_multicurve)

        source = ShapefileSource(str(shp_path))

        with caplog.at_level(logging.WARNING, logger="starlet._internal.tiling.vector_source"):
            tables = list(source.iter_tables())

        assert len(tables) == 1
        geom = wkb.loads(tables[0]["geometry"][0].as_py())
        assert geom.geom_type == "MultiLineString"
        assert list(geom.geoms[0].coords) == [(0.0, 0.0), (1.0, 1.0)]
        assert "curves.shp" in caplog.text
        assert "layer='curves'" in caplog.text
        assert "geometry_type=CurvePolygon" in caplog.text
        assert "linearized_records=1" in caplog.text
        assert "skip_features=0" in caplog.text

    def test_reads_all_shapefiles_from_zip_directly(self, temp_dir):
        source_dir = temp_dir / "shapes"
        source_dir.mkdir()
        first = gpd.GeoDataFrame(
            {"dataset": ["a", "a"], "id": [1, 2]},
            geometry=[Point(0, 1), Point(2, 3)],
            crs="EPSG:4326",
        )
        second = gpd.GeoDataFrame(
            {"dataset": ["b", "b", "b"], "id": [10, 11, 12]},
            geometry=[Point(4, 5), Point(6, 7), Point(8, 9)],
            crs="EPSG:4326",
        )
        first.to_file(source_dir / "alpha.shp", engine="pyogrio")
        second.to_file(source_dir / "beta.shp", engine="pyogrio")

        zip_path = temp_dir / "bundle.zip"
        _write_zip_from_dir(zip_path, source_dir)

        source = ShapefileSource(str(zip_path))
        tables = list(source.iter_tables())
        ids = sorted(
            row_id
            for table in tables
            for row_id in table["id"].to_pylist()
        )
        datasets = sorted(
            value
            for table in tables
            for value in table["dataset"].to_pylist()
        )

        assert len(source._layers) == 2
        assert ids == [1, 2, 10, 11, 12]
        assert datasets == ["a", "a", "b", "b", "b"]
        assert source.input_size_bytes() == zip_path.stat().st_size

    def test_directory_shapefile_schema_promotes_mismatched_integer_fields(
        self,
        temp_dir,
        monkeypatch,
    ):
        import pyogrio
        from starlet._internal import tiling as tiling_pkg

        shapes_dir = temp_dir / "shapes"
        shapes_dir.mkdir()
        alpha = shapes_dir / "alpha.shp"
        beta = shapes_dir / "beta.shp"
        alpha.touch()
        beta.touch()

        monkeypatch.setattr(pyogrio, "list_layers", lambda path: [["points", "Point"]])
        monkeypatch.setattr(
            pyogrio,
            "read_info",
            lambda path, layer=None, force_feature_count=False: {
                "features": 2,
                "geometry_type": "Point",
            },
        )

        def read_arrow(path, **kwargs):
            assert kwargs.get("max_features") != 1
            value_type = pa.int32() if str(path).endswith("alpha.shp") else pa.int64()
            max_features = kwargs.get("max_features")
            row_count = 1 if max_features == 1 else 2
            return (
                {
                    "geometry_name": "SHAPE",
                    "fid_column": "OBJECTID",
                    "crs": "EPSG:4326",
                },
                pa.table(
                    {
                        "OBJECTID": pa.array(range(row_count), type=pa.int64()),
                        "Technology": pa.array([1] * row_count, type=value_type),
                        "SHAPE": pa.array(
                            [wkb.dumps(Point(float(i), float(i))) for i in range(row_count)],
                            type=pa.binary(),
                        ),
                    }
                ),
            )

        monkeypatch.setattr(pyogrio, "read_arrow", read_arrow)
        monkeypatch.setattr(
            tiling_pkg.datasource,
            "create_process_executor",
            lambda max_workers=None, logger=None, context="parallel work": ThreadPoolExecutor(
                max_workers=max_workers
            ),
        )

        sample = read_spatial_sample(
            str(shapes_dir),
            sample_cap=None,
            source_workers=1,
        )
        source = ShapefileSource(str(shapes_dir))
        source.set_schema(sample.schema)
        tables = list(source.iter_tables())

        assert sample.schema is not None
        assert sample.schema.field("Technology").type == pa.int64()
        assert {table.schema.field("Technology").type for table in tables} == {pa.int64()}
        pa.concat_tables(tables, promote_options="default")

    def test_source_for_path_detects_zipped_shapefile(self, temp_dir):
        source_dir = temp_dir / "zip-shp"
        source_dir.mkdir()
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 1)], crs="EPSG:4326")
        gdf.to_file(source_dir / "points.shp", engine="pyogrio")
        zip_path = temp_dir / "points.zip"
        _write_zip_from_dir(zip_path, source_dir)

        source = source_for_path(str(zip_path))

        assert isinstance(source, ShapefileSource)


class TestGDBSource:
    def test_detects_gdb_directory_inside_zip(self, temp_dir):
        zip_path = temp_dir / "Export.gdb.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("CAMS-Export.gdb/gdb", "gdb\n")
            archive.writestr("CAMS-Export.gdb/a00000001.gdbtable", "")
            archive.writestr("CAMS-Export.gdb/a00000001.gdbtablx", "")

        assert _zip_gdb_member_dirs(zip_path) == ["CAMS-Export.gdb"]

    def test_source_for_path_detects_zipped_gdb(self, temp_dir, monkeypatch):
        import pyogrio

        zip_path = temp_dir / "Export.gdb.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("CAMS-Export.gdb/gdb", "gdb\n")
            archive.writestr("CAMS-Export.gdb/a00000001.gdbtable", "")
            archive.writestr("CAMS-Export.gdb/a00000001.gdbtablx", "")

        monkeypatch.setattr(pyogrio, "list_layers", lambda path: [["points", "Point"]])
        monkeypatch.setattr(
            pyogrio,
            "read_info",
            lambda path, layer=None, force_feature_count=False: {
                "features": 1,
                "geometry_type": "Point",
            },
        )

        source = source_for_path(str(zip_path))

        assert isinstance(source, GDBSource)
        assert source._layers[0].path.endswith("CAMS-Export.gdb")

    def test_source_for_path_detects_renamed_zipped_gdb(self, temp_dir, monkeypatch):
        import pyogrio

        zip_path = temp_dir / "Export.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("CAMS-Export.gdb/gdb", "gdb\n")
            archive.writestr("CAMS-Export.gdb/a00000001.gdbtable", "")
            archive.writestr("CAMS-Export.gdb/a00000001.gdbtablx", "")

        monkeypatch.setattr(pyogrio, "list_layers", lambda path: [["points", "Point"]])
        monkeypatch.setattr(
            pyogrio,
            "read_info",
            lambda path, layer=None, force_feature_count=False: {
                "features": 1,
                "geometry_type": "Point",
            },
        )

        source = source_for_path(str(zip_path))

        assert isinstance(source, GDBSource)
        assert source._layers[0].path.endswith("CAMS-Export.gdb")

    def test_source_for_path_detects_gdb_directory_by_marker_file(self, temp_dir, monkeypatch):
        import pyogrio

        gdb_path = temp_dir / "CAMS-Export.gdb"
        gdb_path.mkdir()
        (gdb_path / "gdb").write_text("gdb\n")
        (gdb_path / "a00000001.gdbtable").write_text("")
        (gdb_path / "a00000001.gdbtablx").write_text("")

        monkeypatch.setattr(pyogrio, "list_layers", lambda path: [["points", "Point"]])
        monkeypatch.setattr(
            pyogrio,
            "read_info",
            lambda path, layer=None, force_feature_count=False: {
                "features": 1,
                "geometry_type": "Point",
            },
        )

        source = source_for_path(str(gdb_path))

        assert isinstance(source, GDBSource)
        assert source._layers[0].path == str(gdb_path)


class TestDataSourceIntegration:
    """Integration tests across data sources."""

    def test_consistent_geometry_reading(self, sample_parquet_file, sample_polygons):
        """Test that geometries are read correctly and match original data."""
        source = GeoParquetSource(str(sample_parquet_file))

        # Read all geometries using iter_tables()
        all_geoms = []
        for table in source.iter_tables():
            geoms_wkb = table['geometry'].to_pylist()
            all_geoms.extend([wkb.loads(g) for g in geoms_wkb])

        # Decode and compare bounds
        for i, geom in enumerate(all_geoms):
            original = sample_polygons[i]
            # Compare bounds (should be identical)
            assert geom.bounds == original.bounds
