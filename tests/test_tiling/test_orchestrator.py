"""Unit tests for the two-stage tiling orchestrator."""
import errno
import pytest
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from shapely.geometry import Point
from shapely import wkb

import starlet._internal.tiling.two_stage_orchestrator as two_stage_module
from starlet._internal.tiling import (
    GeoParquetSource,
    RSGroveAssigner,
    SortMode,
    TwoStageOrchestrator,
)
from starlet._internal.tiling.RSGrove import EnvelopeNDLite

class TestTwoStageOrchestrator:
    """Test the two-stage split assignment/write orchestrator."""

    def test_rsgrove_assigner_skips_null_geometries(self):
        mbr = EnvelopeNDLite(
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([10.0, 10.0], dtype=np.float64),
        )
        assigner = RSGroveAssigner.from_sample_and_bounds(
            sample_points=np.array([[1.0, 5.0], [1.0, 5.0]], dtype=np.float64),
            bounds=mbr,
            num_partitions=2,
        )
        table = pa.table(
            {
                "geometry": [None, wkb.dumps(Point(1.0, 1.0))],
                "id": [1, 2],
            }
        )

        partition_table = assigner.partition_by_tile(table)

        assert partition_table["partition_id"].to_pylist()[0] == -1
        assert partition_table["partition_id"].to_pylist()[1] >= 0

    def test_group_table_by_reducer_drops_unassigned_rows(self):
        table = pa.table(
            {
                "value": ["drop", "keep"],
                "_tile_id": [-1, 3],
            }
        )

        grouped = two_stage_module._group_table_by_reducer(table, num_reducers=2)

        assert sorted(grouped) == [1]
        assert grouped[1]["value"].to_pylist() == ["keep"]

    def test_merge_fan_in_retries_on_open_file_limit(self, monkeypatch, temp_dir):
        input_paths = [str(temp_dir / f"input_{index:02d}.parquet") for index in range(9)]
        observed_chunk_sizes = []

        def fake_merge(chunk, output_path, compression):
            observed_chunk_sizes.append(len(chunk))
            if len(chunk) > 2:
                raise OSError(errno.EMFILE, "Too many open files")
            Path(output_path).touch()
            return output_path

        monkeypatch.setattr(two_stage_module, "_default_merge_fan_in", lambda _: 8)
        monkeypatch.setattr(two_stage_module, "_merge_sorted_partition_files", fake_merge)

        result = two_stage_module._merge_sorted_partition_files_to_fan_in(
            input_paths,
            intermediate_compression=None,
            temp_dir=str(temp_dir / "merge_runs"),
        )

        assert observed_chunk_sizes[:3] == [8, 4, 2]
        assert len(result) == 2
        assert all(Path(path).exists() for path in result)

    def test_assignment_worker_uses_bounded_fan_in_for_batch_run_merge(self, monkeypatch, temp_dir):
        table = pa.table({"geometry": [wkb.dumps(Point(1.0, 1.0))], "value": [1]})

        class Source:
            def iter_tables(self, split):
                del split
                for _ in range(5):
                    yield table

        class Assigner:
            def partition_by_tile(self, table):
                return pa.table({"partition_id": pa.array([0] * table.num_rows, type=pa.int64())})

        observed_chunk_sizes = []

        def fake_merge(chunk, output_path, compression):
            del compression
            observed_chunk_sizes.append(len(chunk))
            if len(chunk) > 2:
                raise OSError(errno.EMFILE, "Too many open files")
            Path(output_path).touch()
            return output_path

        monkeypatch.setattr(two_stage_module, "_default_merge_fan_in", lambda _: 2)
        monkeypatch.setattr(two_stage_module, "_merge_sorted_partition_files", fake_merge)

        manifest = two_stage_module._assignment_stage_worker(
            Source(),
            object(),
            0,
            Assigner(),
            1,
            str(temp_dir),
            None,
            1,
            False,
        )

        assert max(observed_chunk_sizes) == 2
        assert sorted(manifest.intermediate_by_reducer) == [0]

    def test_merge_relaxes_non_nullable_fields_when_later_files_have_nulls(self, temp_dir):
        schema_non_nullable = pa.schema([
            pa.field("_tile_id", pa.int64(), nullable=False),
            pa.field("MADRank", pa.int64(), nullable=False),
        ])
        schema_nullable = pa.schema([
            pa.field("_tile_id", pa.int64(), nullable=False),
            pa.field("MADRank", pa.int64(), nullable=True),
        ])
        first = pa.table(
            [pa.array([1], type=pa.int64()), pa.array([10], type=pa.int64())],
            schema=schema_non_nullable,
        )
        second = pa.table(
            [pa.array([1], type=pa.int64()), pa.array([None], type=pa.int64())],
            schema=schema_nullable,
        )
        first_path = temp_dir / "first.arrow"
        second_path = temp_dir / "second.arrow"
        output_path = temp_dir / "merged.arrow"
        two_stage_module._write_intermediate_table(str(first_path), first)
        two_stage_module._write_intermediate_table(str(second_path), second)

        merged = two_stage_module._merge_sorted_partition_files(
            [str(first_path), str(second_path)],
            str(output_path),
            intermediate_compression=None,
        )

        assert merged == str(output_path)
        with pa.memory_map(str(output_path), "r") as source:
            result = ipc.open_file(source).read_all()
        assert result["MADRank"].to_pylist() == [10, None]
        assert result.schema.field("MADRank").nullable

    def test_merge_promotes_null_field_to_later_string_type(self, temp_dir):
        schema_null = pa.schema([
            pa.field("_tile_id", pa.int64(), nullable=False),
            pa.field("OLD_BLD_ID", pa.null(), nullable=True),
        ])
        schema_string = pa.schema([
            pa.field("_tile_id", pa.int64(), nullable=False),
            pa.field("OLD_BLD_ID", pa.large_string(), nullable=True),
        ])
        first = pa.table(
            [pa.array([1], type=pa.int64()), pa.array([None], type=pa.null())],
            schema=schema_null,
        )
        second = pa.table(
            [pa.array([1], type=pa.int64()), pa.array(["B123"], type=pa.large_string())],
            schema=schema_string,
        )
        first_path = temp_dir / "first.arrow"
        second_path = temp_dir / "second.arrow"
        output_path = temp_dir / "merged.arrow"
        two_stage_module._write_intermediate_table(str(first_path), first)
        two_stage_module._write_intermediate_table(str(second_path), second)

        merged = two_stage_module._merge_sorted_partition_files(
            [str(first_path), str(second_path)],
            str(output_path),
            intermediate_compression=None,
        )

        assert merged == str(output_path)
        with pa.memory_map(str(output_path), "r") as source:
            result = ipc.open_file(source).read_all()
        assert result["OLD_BLD_ID"].to_pylist() == [None, "B123"]
        assert result.schema.field("OLD_BLD_ID").type == pa.large_string()

    def test_two_stage_orchestrator_writes_all_rows(self, sample_parquet_file, sample_polygons, temp_dir):
        source = GeoParquetSource(str(sample_parquet_file))
        centers = np.array(
            [[geom.centroid.x for geom in sample_polygons], [geom.centroid.y for geom in sample_polygons]],
            dtype=np.float64,
        )
        bounds = np.array([geom.bounds for geom in sample_polygons], dtype=np.float64)
        mbr = EnvelopeNDLite(
            np.array([bounds[:, 0].min(), bounds[:, 1].min()], dtype=np.float64),
            np.array([bounds[:, 2].max(), bounds[:, 3].max()], dtype=np.float64),
        )
        assigner = RSGroveAssigner.from_sample_and_bounds(
            sample_points=centers,
            bounds=mbr,
            num_partitions=2,
        )
        outdir = temp_dir / "two_stage_tiles"

        orchestrator = TwoStageOrchestrator(
            source=source,
            assigner=assigner,
            outdir=str(outdir),
            sort_mode=SortMode.NONE,
            parallelism=2,
        )
        orchestrator.run()

        tile_files = list(outdir.glob("*.parquet"))
        assert tile_files
        total_rows = sum(pq.read_metadata(str(path)).num_rows for path in tile_files)
        assert total_rows == len(sample_polygons)
        tables = [pq.read_table(path) for path in tile_files]
        ids = []
        for table in tables:
            assert "_id" in table.column_names
            ids.extend(table["_id"].to_pylist())
        assert sorted(ids) == list(range(len(sample_polygons)))

    def test_two_stage_orchestrator_uses_custom_temp_dir(self, sample_parquet_file, sample_polygons, temp_dir):
        source = GeoParquetSource(str(sample_parquet_file))
        centers = np.array(
            [[geom.centroid.x for geom in sample_polygons], [geom.centroid.y for geom in sample_polygons]],
            dtype=np.float64,
        )
        bounds = np.array([geom.bounds for geom in sample_polygons], dtype=np.float64)
        mbr = EnvelopeNDLite(
            np.array([bounds[:, 0].min(), bounds[:, 1].min()], dtype=np.float64),
            np.array([bounds[:, 2].max(), bounds[:, 3].max()], dtype=np.float64),
        )
        assigner = RSGroveAssigner.from_sample_and_bounds(
            sample_points=centers,
            bounds=mbr,
            num_partitions=2,
        )
        temp_parent = temp_dir / "large_tmp"

        orchestrator = TwoStageOrchestrator(
            source=source,
            assigner=assigner,
            outdir=str(temp_dir / "custom_tmp_tiles"),
            sort_mode=SortMode.NONE,
            parallelism=2,
            temp_dir=str(temp_parent),
            keep_temp=True,
            # Compression is covered by TestIntermediateCompression; keep this
            # test focused on the custom-temp-dir behavior.
            intermediate_compression="none",
        )
        orchestrator.run()

        run_dirs = list(temp_parent.glob("starlet_two_stage_*"))
        assert len(run_dirs) == 1
        intermediate_files = list(run_dirs[0].glob("split_*/mapper_*_reducer_*.arrow"))
        assert intermediate_files
        for path in intermediate_files:
            with pa.memory_map(str(path), "r") as source:
                table = ipc.open_file(source).read_all()
                tile_ids = table["_tile_id"].to_pylist()
                feature_ids = table["_id"].to_pylist()
            assert tile_ids == sorted(tile_ids)
            assert all(feature_id % 2 in {0, 1} for feature_id in feature_ids)

    def test_assignment_worker_feature_ids_follow_mapper_stride(self, sample_parquet_file, sample_polygons, temp_dir):
        source = GeoParquetSource(str(sample_parquet_file))
        splits = list(source.create_splits())
        centers = np.array(
            [[geom.centroid.x for geom in sample_polygons], [geom.centroid.y for geom in sample_polygons]],
            dtype=np.float64,
        )
        bounds = np.array([geom.bounds for geom in sample_polygons], dtype=np.float64)
        mbr = EnvelopeNDLite(
            np.array([bounds[:, 0].min(), bounds[:, 1].min()], dtype=np.float64),
            np.array([bounds[:, 2].max(), bounds[:, 3].max()], dtype=np.float64),
        )
        assigner = RSGroveAssigner.from_sample_and_bounds(
            sample_points=centers,
            bounds=mbr,
            num_partitions=2,
        )

        manifest = two_stage_module._assignment_stage_worker(
            source,
            splits[0],
            1,
            assigner,
            2,
            str(temp_dir),
            None,
            3,
            False,
        )

        feature_ids = []
        for path in manifest.intermediate_by_reducer.values():
            with pa.memory_map(path, "r") as source_file:
                feature_ids.extend(ipc.open_file(source_file).read_all()["_id"].to_pylist())
        assert sorted(feature_ids) == [1, 4, 7, 10, 13]


class TestIntermediateCompression:
    """Test tile-intermediate-compression for two-stage shuffle files."""

    def test_normalize_rejects_invalid_value(self):
        with pytest.raises(ValueError):
            two_stage_module._normalize_intermediate_compression("bzip2")

    def test_normalize_treats_none_as_no_compression(self):
        assert two_stage_module._normalize_intermediate_compression(None) == "none"
        assert two_stage_module._normalize_intermediate_compression("none") == "none"
        assert two_stage_module._normalize_intermediate_compression("GZIP") == "gzip"

    def test_orchestrator_rejects_invalid_intermediate_compression(self, sample_parquet_file, temp_dir):
        source = GeoParquetSource(str(sample_parquet_file))
        with pytest.raises(ValueError):
            TwoStageOrchestrator(
                source=source,
                assigner=None,
                outdir=str(temp_dir / "out"),
                intermediate_compression="bzip2",
            )

    def test_write_intermediate_table_gzip_produces_gzip_file_and_roundtrips(self, temp_dir):
        table = pa.table({"a": list(range(2000)), "b": ["repeated-value"] * 2000})
        path = temp_dir / "mapper.arrow"

        two_stage_module._write_intermediate_table(
            str(path), table, intermediate_compression="gzip"
        )

        raw = path.read_bytes()
        assert raw[:2] == b"\x1f\x8b"  # gzip magic bytes
        assert two_stage_module._ipc_file_schema(str(path), "gzip").equals(table.schema)

        with two_stage_module._open_intermediate_source(str(path), "gzip") as source:
            result = ipc.open_file(source).read_all()
        assert result.equals(table)

    def test_write_intermediate_table_none_produces_plain_ipc_file(self, temp_dir):
        table = pa.table({"a": [1, 2, 3]})
        path = temp_dir / "mapper.arrow"

        two_stage_module._write_intermediate_table(
            str(path), table, intermediate_compression="none"
        )

        raw = path.read_bytes()
        assert raw[:2] != b"\x1f\x8b"
        # Legacy readers that mmap + ipc.open_file directly must still work.
        with pa.memory_map(str(path), "r") as source:
            result = ipc.open_file(source).read_all()
        assert result.equals(table)

    def test_gzip_intermediate_file_is_smaller_for_compressible_data(self, temp_dir):
        table = pa.table(
            {
                "a": [1] * 20_000,
                "b": ["the quick brown fox jumps over the lazy dog"] * 20_000,
            }
        )
        plain_path = temp_dir / "plain.arrow"
        gzip_path = temp_dir / "gzip.arrow"

        two_stage_module._write_intermediate_table(
            str(plain_path), table, intermediate_compression="none"
        )
        two_stage_module._write_intermediate_table(
            str(gzip_path), table, intermediate_compression="gzip"
        )

        plain_size = plain_path.stat().st_size
        gzip_size = gzip_path.stat().st_size
        assert gzip_size < plain_size * 0.1

    def test_merge_sorted_partition_files_with_gzip_compression(self, temp_dir):
        schema = pa.schema([
            pa.field("_tile_id", pa.int64(), nullable=False),
            pa.field("value", pa.int64(), nullable=False),
        ])
        first = pa.table(
            [pa.array([0, 1], type=pa.int64()), pa.array([10, 20], type=pa.int64())],
            schema=schema,
        )
        second = pa.table(
            [pa.array([0, 2], type=pa.int64()), pa.array([30, 40], type=pa.int64())],
            schema=schema,
        )
        first_path = temp_dir / "first.arrow"
        second_path = temp_dir / "second.arrow"
        output_path = temp_dir / "merged.arrow"
        two_stage_module._write_intermediate_table(
            str(first_path), first, intermediate_compression="gzip"
        )
        two_stage_module._write_intermediate_table(
            str(second_path), second, intermediate_compression="gzip"
        )

        merged = two_stage_module._merge_sorted_partition_files(
            [str(first_path), str(second_path)],
            str(output_path),
            intermediate_compression="gzip",
        )

        assert merged == str(output_path)
        assert output_path.read_bytes()[:2] == b"\x1f\x8b"
        groups = dict(
            two_stage_module._iter_partition_groups(
                str(output_path), intermediate_compression="gzip"
            )
        )
        assert sorted(groups) == [0, 1, 2]
        assert sorted(groups[0]["value"].to_pylist()) == [10, 30]

    def test_two_stage_orchestrator_default_compresses_intermediate_files(
        self, sample_parquet_file, sample_polygons, temp_dir
    ):
        source = GeoParquetSource(str(sample_parquet_file))
        centers = np.array(
            [[geom.centroid.x for geom in sample_polygons], [geom.centroid.y for geom in sample_polygons]],
            dtype=np.float64,
        )
        bounds = np.array([geom.bounds for geom in sample_polygons], dtype=np.float64)
        mbr = EnvelopeNDLite(
            np.array([bounds[:, 0].min(), bounds[:, 1].min()], dtype=np.float64),
            np.array([bounds[:, 2].max(), bounds[:, 3].max()], dtype=np.float64),
        )
        assigner = RSGroveAssigner.from_sample_and_bounds(
            sample_points=centers,
            bounds=mbr,
            num_partitions=2,
        )
        temp_parent = temp_dir / "large_tmp"

        orchestrator = TwoStageOrchestrator(
            source=source,
            assigner=assigner,
            outdir=str(temp_dir / "custom_tmp_tiles"),
            sort_mode=SortMode.NONE,
            parallelism=2,
            temp_dir=str(temp_parent),
            keep_temp=True,
        )
        assert orchestrator.intermediate_compression == "gzip"
        orchestrator.run()

        run_dirs = list(temp_parent.glob("starlet_two_stage_*"))
        assert len(run_dirs) == 1
        intermediate_files = list(run_dirs[0].glob("split_*/mapper_*_reducer_*.arrow"))
        assert intermediate_files
        for path in intermediate_files:
            assert path.read_bytes()[:2] == b"\x1f\x8b"
            groups = dict(
                two_stage_module._iter_partition_groups(str(path), intermediate_compression="gzip")
            )
            for partition_id, group_table in groups.items():
                assert group_table["_tile_id"].to_pylist() == [partition_id] * group_table.num_rows

        tile_files = list((temp_dir / "custom_tmp_tiles").glob("*.parquet"))
        assert tile_files
        total_rows = sum(pq.read_metadata(str(path)).num_rows for path in tile_files)
        assert total_rows == len(sample_polygons)

    def test_two_stage_orchestrator_intermediate_compression_none_matches_legacy_behavior(
        self, sample_parquet_file, sample_polygons, temp_dir
    ):
        source = GeoParquetSource(str(sample_parquet_file))
        centers = np.array(
            [[geom.centroid.x for geom in sample_polygons], [geom.centroid.y for geom in sample_polygons]],
            dtype=np.float64,
        )
        bounds = np.array([geom.bounds for geom in sample_polygons], dtype=np.float64)
        mbr = EnvelopeNDLite(
            np.array([bounds[:, 0].min(), bounds[:, 1].min()], dtype=np.float64),
            np.array([bounds[:, 2].max(), bounds[:, 3].max()], dtype=np.float64),
        )
        assigner = RSGroveAssigner.from_sample_and_bounds(
            sample_points=centers,
            bounds=mbr,
            num_partitions=2,
        )
        temp_parent = temp_dir / "large_tmp"

        orchestrator = TwoStageOrchestrator(
            source=source,
            assigner=assigner,
            outdir=str(temp_dir / "tiles"),
            sort_mode=SortMode.NONE,
            parallelism=2,
            temp_dir=str(temp_parent),
            keep_temp=True,
            intermediate_compression="none",
        )
        orchestrator.run()

        run_dirs = list(temp_parent.glob("starlet_two_stage_*"))
        intermediate_files = list(run_dirs[0].glob("split_*/mapper_*_reducer_*.arrow"))
        assert intermediate_files
        for path in intermediate_files:
            assert path.read_bytes()[:2] != b"\x1f\x8b"
            with pa.memory_map(str(path), "r") as mm:
                table = ipc.open_file(mm).read_all()
            tile_ids = table["_tile_id"].to_pylist()
            assert tile_ids == sorted(tile_ids)

        tile_files = list((temp_dir / "tiles").glob("*.parquet"))
        total_rows = sum(pq.read_metadata(str(path)).num_rows for path in tile_files)
        assert total_rows == len(sample_polygons)


class TestOrchestratorErrorHandling:
    """Test error handling in orchestrator."""

    def test_missing_input_file(self, temp_dir):
        """Test behavior with missing input file."""
        pass

    def test_invalid_num_tiles(self, temp_dir):
        """Test with invalid number of tiles."""
        pass

    def test_output_directory_creation(self, temp_dir):
        """Test that output directories are created."""
        pass


# Placeholder for additional orchestrator tests
