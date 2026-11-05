"""Was Python slow because it could not read in parallel? The same Read, same row groups, same tiles, three ways:

  python_today   one row group at a time; pyarrow's default threads inside each group (read_steps from time_read_steps.py)
  python_8       8 row groups at a time from Python threads, one thread per group
  python_8_inner the same, but pyarrow also uses its default threads inside each group, as python_today does
  python_late    8 row groups at a time, bbox columns first; other columns only for groups with a match (late materialization)
  rust_8         8 row groups at a time from starlet_read, one thread per group
  rust_1         the same, 1 thread
  rust_late      8 threads, bbox columns first; other columns only for groups with a match (late materialization)
  rust_pages     rust_late, but only the pages with a match (page index; needs --layout mb_2_pages_64)

Each file's picked groups are read together; a tile's files are read one after another (same in python_8 and rust_8).
Footers cached, no row-group cache, each tile's files dropped from the OS file cache first, so row groups come from disk.
Run each mode in its own process: python compare_parallel_read.py --dataset asia --mode rust_8 --name parallel_read/asia/rust_8_1
Writes <name>/results.csv with the time_read_steps.py columns.
"""
import argparse
import csv
import gc
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import starlet_read

from starlet._internal.config import config_value
from starlet._internal.mvt.fast_tile import BBOX_COLS, RowGroupCache, TileFrame
from starlet._internal.server.tiler.parquet_index import ParquetIndex
from time_read_steps import (BUFFER, EXTENT, chunk_bytes, evict_from_os_cache, evict_tile_files, read_steps,
                             startup_schema_read)

HERE = Path(__file__).resolve().parent
THREADS = 8
POOL = ThreadPoolExecutor(THREADS)
RUST_FILES = {}                                              # path -> starlet_read.ParquetFile: footers cached


def python_groups(path, footer, rgs, columns):
    def one(rg):
        with pq.ParquetFile(path, metadata=footer) as f:     # own handle: a ParquetFile is not shared across threads
            return f.read_row_group(rg, columns=columns, use_threads=False)
    return list(POOL.map(one, rgs))                          # map keeps the groups in order


def python_groups_inner(path, footer, rgs, columns):
    def one(rg):
        with pq.ParquetFile(path, metadata=footer) as f:
            return f.read_row_group(rg, columns=columns)     # pyarrow's default: its own threads inside the group too
    return list(POOL.map(one, rgs))


def rust_groups(path, footer, rgs, columns):
    return [RUST_FILES[path].read_row_groups(rgs, columns, THREADS)] if rgs else []


def rust_1_groups(path, footer, rgs, columns):
    return [RUST_FILES[path].read_row_groups(rgs, columns, 1)] if rgs else []


def kept_rows_read(index, cache, frame, use_pages):
    """Rust does load, bbox and filter together, so they are all timed as load. Returns also rows whose bbox and geometry were read."""
    t = dict.fromkeys(("find", "load", "bbox", "filter"), 0.0)
    records_read = records_in_tile = bbox_rows = geometry_rows = 0

    s = time.perf_counter()
    plan = [(path, cache.file_info(path), cache.candidate_row_groups(path, frame.query_lonlat))
            for path in index.find_intersecting_files(frame.query_lonlat)]
    t["find"] += time.perf_counter() - s

    for path, (pf, geom_col, _has_bbox, _crs, attrs, _stats), rgs in plan:
        if not rgs:
            continue
        s = time.perf_counter()
        _table, bbox, geometry, kept = RUST_FILES[path].read_kept_rows(rgs, [geom_col] + attrs, list(frame.query_lonlat), THREADS, use_pages)
        t["load"] += time.perf_counter() - s
        records_read += sum(pf.metadata.row_group(rg).num_rows for rg in rgs)
        records_in_tile += kept
        bbox_rows += bbox
        geometry_rows += geometry
    return {k: v * 1000 for k, v in t.items()}, records_read, records_in_tile, bbox_rows, geometry_rows


def python_late_group(path, footer, rg, rest, q):
    """One row group: bbox columns first, filter, then the other columns only if a row matches. Returns (kept rows, bbox rows, geometry rows, kept)."""
    with pq.ParquetFile(path, metadata=footer) as f:
        bbox = f.read_row_group(rg, columns=list(BBOX_COLS))
        a, b, c, d = [bbox[col].to_numpy(zero_copy_only=False).astype(np.float64) for col in BBOX_COLS]
        hit = (c >= q[0]) & (a <= q[2]) & (d >= q[1]) & (b <= q[3])
        if not hit.any():
            return None, bbox.num_rows, 0, 0
        mask = pa.array(hit)
        table = f.read_row_group(rg, columns=rest).filter(mask)
    for col, values in zip(BBOX_COLS, bbox.filter(mask).columns):
        table = table.append_column(col, values)
    return table, bbox.num_rows, bbox.num_rows, int(hit.sum())


def python_late(index, cache, frame):
    """Late materialization in Python, 8 row groups at a time: the same work and timers as kept_rows_read without pages."""
    t = dict.fromkeys(("find", "load", "bbox", "filter"), 0.0)
    records_read = records_in_tile = bbox_rows = geometry_rows = 0

    s = time.perf_counter()
    plan = [(path, cache.file_info(path), cache.candidate_row_groups(path, frame.query_lonlat))
            for path in index.find_intersecting_files(frame.query_lonlat)]
    t["find"] += time.perf_counter() - s

    for path, (pf, geom_col, _has_bbox, _crs, attrs, _stats), rgs in plan:
        if not rgs:
            continue
        rest = [geom_col] + attrs
        s = time.perf_counter()
        groups = list(POOL.map(lambda rg: python_late_group(path, pf.metadata, rg, rest, frame.query_lonlat), rgs))
        kept_tables = [g[0] for g in groups if g[0] is not None]
        if kept_tables:
            pa.concat_tables(kept_tables)
        t["load"] += time.perf_counter() - s
        records_read += sum(pf.metadata.row_group(rg).num_rows for rg in rgs)
        records_in_tile += sum(g[3] for g in groups)
        bbox_rows += sum(g[1] for g in groups)
        geometry_rows += sum(g[2] for g in groups)
    return {k: v * 1000 for k, v in t.items()}, records_read, records_in_tile, bbox_rows, geometry_rows


def parallel_read(index, cache, frame, read_groups):
    """Same steps and timers as read_steps, but each file's row groups are loaded together by `read_groups`."""
    t = dict.fromkeys(("find", "load", "bbox", "filter"), 0.0)
    qx0, qy0, qx1, qy1 = frame.query_lonlat
    records_read = records_in_tile = bytes_requested = 0

    s = time.perf_counter()
    plan = [(path, cache.file_info(path), cache.candidate_row_groups(path, frame.query_lonlat))
            for path in index.find_intersecting_files(frame.query_lonlat)]
    t["find"] += time.perf_counter() - s

    for path, (pf, geom_col, _has_bbox, _crs, attrs, _stats), rgs in plan:
        columns = [geom_col] + list(BBOX_COLS) + attrs
        footer = pf.metadata
        s = time.perf_counter()
        tables = read_groups(path, footer, rgs, columns)
        t["load"] += time.perf_counter() - s
        for table in tables:
            s = time.perf_counter()
            a, b, c, d = [table[col].to_numpy(zero_copy_only=False).astype(np.float64) for col in BBOX_COLS]
            t["bbox"] += time.perf_counter() - s
            s = time.perf_counter()
            hit = (c >= qx0) & (a <= qx1) & (d >= qy0) & (b <= qy1)
            t["filter"] += time.perf_counter() - s
            records_read += table.num_rows
            records_in_tile += int(hit.sum())
        bytes_requested += sum(chunk_bytes(path, pf, rg, columns) for rg in rgs)
    return {k: v * 1000 for k, v in t.items()}, records_read, records_in_tile, bytes_requested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", required=True, choices=("python_today", "python_8", "python_8_inner", "python_late", "rust_8", "rust_1", "rust_late", "rust_pages"))
    ap.add_argument("--layout", default="mb_2", help="folder under rowgroups/<dataset>/")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tiles", default="tiles_z8plus.csv", help="tile list in <dataset>/")
    args = ap.parse_args()

    parquet_dir = HERE / "rowgroups" / args.dataset / args.layout / "parquet_tiles"
    with open(HERE / args.dataset / args.tiles, newline="") as fh:
        tiles = [(int(r["z"]), int(r["x"]), int(r["y"])) for r in csv.DictReader(fh)][:args.limit]
    random.Random(1).shuffle(tiles)
    for f in parquet_dir.glob("*.parquet"):                  # start clean
        evict_from_os_cache(f)
    index = ParquetIndex(parquet_dir)
    cache = RowGroupCache(index, int(config_value("mvt", "row_group_cache")))
    print(f"{args.dataset} {args.mode}: {len(tiles)} tiles. {startup_schema_read(index)}", flush=True)

    folder = HERE / args.name
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["z", "x", "y", "find_ms", "load_ms", "bbox_ms", "filter_ms", "read_ms", "records_read", "records_in_tile",
                    "bytes_requested", "disk_read_bytes", "background_bytes", "cpu_pct", "records_bbox_read", "records_geometry_read"])
        for z, x, y in tiles:
            frame = TileFrame(z, x, y, EXTENT, BUFFER)
            if args.mode.startswith("rust"):
                for path in index.find_intersecting_files(frame.query_lonlat):
                    if path not in RUST_FILES:
                        RUST_FILES[path] = starlet_read.ParquetFile(str(path))
            evict_tile_files(index, cache, frame)            # this tile is a first-time request
            psutil.cpu_percent()
            disk0 = psutil.disk_io_counters().read_bytes
            if args.mode == "python_today":
                t, records_read, records_in_tile, nbytes = read_steps(index, cache, frame, False)
            elif args.mode == "python_late":
                t, records_read, records_in_tile, bbox_rows, geometry_rows = python_late(index, cache, frame)
                nbytes = 0
            elif args.mode in ("rust_late", "rust_pages"):
                t, records_read, records_in_tile, bbox_rows, geometry_rows = kept_rows_read(index, cache, frame, args.mode == "rust_pages")
                nbytes = 0                                   # no fixed amount to compare with: skipping is the point
            else:
                reader = {"python_8": python_groups, "python_8_inner": python_groups_inner, "rust_8": rust_groups, "rust_1": rust_1_groups}[args.mode]
                t, records_read, records_in_tile, nbytes = parallel_read(index, cache, frame, reader)
            if args.mode not in ("python_late", "rust_late", "rust_pages"):
                bbox_rows = geometry_rows = records_read
            disk_bytes = psutil.disk_io_counters().read_bytes - disk0
            total = sum(t.values())
            b0 = psutil.disk_io_counters().read_bytes        # other programs' reads over an idle wait as long as the tile
            time.sleep(min(total / 1000, 2.0))
            w.writerow([z, x, y, *[round(t[s], 3) for s in ("find", "load", "bbox", "filter")], round(total, 3),
                        records_read, records_in_tile, nbytes, disk_bytes, psutil.disk_io_counters().read_bytes - b0,
                        psutil.cpu_percent(), bbox_rows, geometry_rows])
            fh.flush()
            gc.collect()
            pa.default_memory_pool().release_unused()
    print(f"saved {folder / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
