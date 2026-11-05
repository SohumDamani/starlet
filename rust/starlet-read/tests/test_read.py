"""starlet_read must return exactly what pyarrow returns, in row-group order, at any thread count.

  python -m pytest rust/starlet-read/tests/test_read.py
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

starlet_read = pytest.importorskip("starlet_read")

LAYOUTS = Path(__file__).resolve().parents[3] / "benchmarks" / "on_the_fly" / "rowgroups"
BBOX = ["_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax"]


def expected(path, groups, columns):
    return pq.ParquetFile(path).read_row_groups(groups, columns=columns)


def same(ours, theirs):
    return ours.select(theirs.column_names).combine_chunks().equals(theirs.combine_chunks())


@pytest.fixture
def small_file(tmp_path):
    path = tmp_path / "small.parquet"
    table = pa.table({"id": list(range(1000)), "name": [f"n{i}" for i in range(1000)],
                      "nested": [{"a": i, "b": str(i)} for i in range(1000)]})
    pq.write_table(table, path, compression="zstd", row_group_size=100)
    return path


@pytest.mark.parametrize("threads", [1, 8])
def test_rows_match_pyarrow(small_file, threads):
    groups, columns = [0, 3, 7], ["id", "name", "nested"]
    assert same(starlet_read.ParquetFile(str(small_file)).read_row_groups(groups, columns, threads), expected(small_file, groups, columns))


def kept_by_python(path, groups, columns, q):
    """The rows read_kept_rows must return: read everything with pyarrow, then filter on the bbox columns."""
    t = pq.ParquetFile(path).read_row_groups(groups, columns=columns + BBOX)
    a, b, c, d = [t[col].to_numpy(zero_copy_only=False) for col in BBOX]
    return t.filter(pa.array((c >= q[0]) & (a <= q[2]) & (d >= q[1]) & (b <= q[3])))


@pytest.fixture
def bbox_file(tmp_path):
    """Rows along a line, so a small query matches a few neighbouring rows of one group."""
    path = tmp_path / "bbox.parquet"
    x = [float(i) for i in range(1000)]
    table = pa.table({"geometry": [bytes([i % 256]) * 50 for i in range(1000)], "name": [f"n{i}" for i in range(1000)],
                      "_bbox_xmin": x, "_bbox_ymin": x, "_bbox_xmax": [v + 1 for v in x], "_bbox_ymax": [v + 1 for v in x]})
    pq.write_table(table, path, compression="zstd", row_group_size=100, write_page_index=True, max_rows_per_page=10)
    return path


@pytest.mark.parametrize("use_pages", [False, True])
@pytest.mark.parametrize("q", [[105.5, 105.5, 107.5, 107.5], [0.0, 0.0, 999.0, 999.0], [5000.0, 5000.0, 5001.0, 5001.0]])
def test_kept_rows_match_python_filter(bbox_file, use_pages, q):
    columns = ["geometry", "name"]
    table, _, _, kept = starlet_read.ParquetFile(str(bbox_file)).read_kept_rows(list(range(10)), columns, q, 8, use_pages)
    theirs = kept_by_python(bbox_file, list(range(10)), columns, q)
    assert kept == theirs.num_rows
    assert table.select(theirs.column_names).combine_chunks().equals(theirs.combine_chunks())


def test_pages_read_fewer_rows_than_whole_groups(bbox_file):
    f = starlet_read.ParquetFile(str(bbox_file))
    q, columns = [105.5, 105.5, 107.5, 107.5], ["geometry", "name"]
    _, whole_bbox, whole_geom, _ = f.read_kept_rows(list(range(10)), columns, q, 1, False)
    _, page_bbox, page_geom, _ = f.read_kept_rows(list(range(10)), columns, q, 1, True)
    assert (whole_bbox, whole_geom) == (1000, 100)                  # every row's bbox; the one group with a match
    assert page_bbox == 10 and page_geom == 10                      # one page of 10 rows


@pytest.mark.parametrize("dataset", ["asia", "osm_parks"])
def test_real_page_files_keep_the_same_rows(dataset):
    files = sorted((LAYOUTS / dataset / "mb_2_pages_64" / "parquet_tiles").glob("*.parquet"))
    if not files:
        pytest.skip("no page-index layout here")
    path = files[len(files) // 2]
    pf = pq.ParquetFile(path)
    groups = list(range(min(4, pf.metadata.num_row_groups)))
    x0, y0 = pq.read_table(path, columns=BBOX[:2]).slice(0, 1).to_pylist()[0].values()
    q = [x0 - 0.01, y0 - 0.01, x0 + 0.01, y0 + 0.01]
    columns = [c for c in pf.schema_arrow.names if c not in BBOX]
    table, _, _, kept = starlet_read.ParquetFile(str(path)).read_kept_rows(groups, columns, q, 8, True)
    theirs = kept_by_python(path, groups, columns, q)
    assert kept == theirs.num_rows
    assert table.select(theirs.column_names).combine_chunks().equals(theirs.combine_chunks())


def test_row_groups_stay_in_the_order_asked(small_file):
    table = starlet_read.ParquetFile(str(small_file)).read_row_groups(list(range(10)), ["id"], 8)
    assert table["id"].to_pylist() == list(range(1000))


def test_missing_column_is_an_error(small_file):
    with pytest.raises(KeyError):
        starlet_read.ParquetFile(str(small_file)).read_row_groups([0], ["nope"], 1)


@pytest.mark.parametrize("dataset", ["asia", "osm_parks"])
@pytest.mark.parametrize("threads", [1, 8])
def test_real_files_match_pyarrow(dataset, threads):
    files = sorted((LAYOUTS / dataset / "mb_2" / "parquet_tiles").glob("*.parquet"))
    if not files:
        pytest.skip("no 2 MB layout here")
    path = files[len(files) // 2]
    pf = pq.ParquetFile(path)
    groups = list(range(min(4, pf.metadata.num_row_groups)))
    columns = pf.schema_arrow.names
    ours = starlet_read.ParquetFile(str(path)).read_row_groups(groups, columns, threads)
    assert ours.num_rows == sum(pf.metadata.row_group(g).num_rows for g in groups)
    assert same(ours, expected(path, groups, columns))
