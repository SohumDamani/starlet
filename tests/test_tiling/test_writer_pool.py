from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box
from shapely import wkb

from starlet._internal.server.tiler.parquet_index import parse_parquet_bbox
from starlet._internal.tiling import writer_pool
from starlet._internal.tiling.writer_pool import _WriterPoolConfig, _finalize_one_tile, SortMode


def test_finalize_one_tile_filename_bbox_is_outward_and_zero_padded(temp_dir):
    geom = box(-118.708998238698, 34.0281302182776, -118.012657194198, 34.2522126466092)
    table = pa.table({"geometry": [wkb.dumps(geom)], "id": [1]})
    config = _WriterPoolConfig(
        geom_col="geometry",
        sort_mode=SortMode.NONE,
        sort_keys=[],
        sfc_bits=16,
        global_extent=None,
        compression="zstd",
        pq_args={},
        outdir=str(temp_dir),
        covering_bbox=False,
    )

    out_path = Path(_finalize_one_tile(0, [table], config))

    assert out_path.name == "tile_000000__-118_708999_34_028130_-118_012657_34_252213.parquet"

    parsed = parse_parquet_bbox(out_path.name)
    assert parsed == (-118.708999, 34.02813, -118.012657, 34.252213)

    geo = json.loads((pq.ParquetFile(out_path).schema_arrow.metadata or {})[b"geo"].decode())
    assert tuple(geo["columns"]["geometry"]["bbox"]) == pytest.approx(
        (-118.708998238698, 34.0281302182776, -118.012657194198, 34.2522126466092)
    )


def test_finalize_one_tile_footer_guard_counts_nested_columns(temp_dir, monkeypatch):
    # Small budget and byte cap so the test is fast: without the guard this gives hundreds of tiny row groups.
    monkeypatch.setattr(writer_pool, "FOOTER_BUDGET_BYTES", 64 * 1024)
    monkeypatch.setattr(writer_pool, "DEFAULT_ROW_GROUP_BYTES", 16 * 1024)
    n, tags = 4000, 200
    table = pa.table({
        "geometry": [wkb.dumps(box(i, 0, i + 1, 1)) for i in range(n)],
        "tags": pa.StructArray.from_arrays([pa.array([1.0] * n)] * tags, [f"t{i}" for i in range(tags)]),
    })
    config = _WriterPoolConfig(geom_col="geometry", sort_mode=SortMode.NONE, sort_keys=[], sfc_bits=16,
                               global_extent=None, compression="zstd", pq_args={},
                               outdir=str(temp_dir), covering_bbox=True)

    md = pq.read_metadata(_finalize_one_tile(0, [table], config))

    assert md.num_columns > 200                      # one top-level struct, 200 leaf columns in the file
    assert md.serialized_size <= 64 * 1024           # footer stays within the budget
    assert md.num_rows == n
