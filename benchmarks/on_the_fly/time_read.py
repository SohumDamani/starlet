"""Time the READ part of on-the-fly tile generation for every benchmark tile.

read_tile() is the read loop of fast_tile.generate_tile_ex (fast_tile.py:357-390), nothing more.
Cold: a new index and an empty row-group cache for every tile. OS file cache: warmed once at the start.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import psutil

from starlet._internal.config import config_value
from starlet._internal.mvt.fast_tile import RowGroupCache, TileFrame
from starlet._internal.server.tiler.parquet_index import ParquetIndex

HERE = Path(__file__).resolve().parent
EXTENT = int(config_value("mvt", "extent"))
BUFFER = int(config_value("mvt", "buffer"))


def read_tile(index, cache, z, x, y):
    """Returns (records read, records in the tile)."""
    frame = TileFrame(z, x, y, EXTENT, BUFFER)                              # z/x/y -> rectangle
    qx0, qy0, qx1, qy1 = frame.query_lonlat
    records_read = records_in_tile = 0

    for path in index.find_intersecting_files(frame.query_lonlat):          # 1. files that overlap
        for rg in cache.candidate_row_groups(path, frame.query_lonlat):     # 2. row groups that overlap
            table, a, b, c, d, crc = cache.row_group(path, rg)              # 3. read the row group
            hit = (c >= qx0) & (a <= qx1) & (d >= qy0) & (b <= qy1)         # 4. records that overlap
            records_read += table.num_rows
            records_in_tile += int(hit.sum())

    return records_read, records_in_tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="built dataset folder (contains parquet_tiles/)")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    parquet_dir = Path(args.dataset) / "parquet_tiles"
    tiles = [(int(r["z"]), int(r["x"]), int(r["y"])) for r in csv.DictReader(open(HERE / "tiles.csv"))]
    for f in parquet_dir.glob("*.parquet"):          # warm the OS file cache once
        f.read_bytes()
    print(f"{len(tiles)} tiles, {args.runs} runs; CPU load {psutil.cpu_percent(interval=1.0):.0f}%", flush=True)

    out = HERE / "results_read_python.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "z", "x", "y", "read_ms", "records_read", "records_in_tile"])
        for run in range(1, args.runs + 1):
            order = tiles[:]
            random.Random(run).shuffle(order)          # different, repeatable order per run
            for z, x, y in order:
                index = ParquetIndex(parquet_dir)                                   # cold
                cache = RowGroupCache(index, int(config_value("mvt", "row_group_cache")))   # cold
                start = time.perf_counter()
                records_read, records_in_tile = read_tile(index, cache, z, x, y)
                read_ms = (time.perf_counter() - start) * 1000
                w.writerow([run, z, x, y, round(read_ms, 2), records_read, records_in_tile])
            print(f"run {run} done", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
