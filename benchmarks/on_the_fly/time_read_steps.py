"""Time the steps inside READ for every benchmark tile (Read only, no later stages).

Same work as fast_tile.generate_tile_ex's read loop (fast_tile.py:374-393) with
RowGroupCache.row_group (fast_tile.py:286-295) copied in, so each step can be timed:
  find    files by name + row groups by footer min/max
  load    read_row_group: all rows, geometry + bbox + attribute columns (decompress -> Arrow)
  wkb     every geometry -> Python bytes
  bbox    the 4 bbox columns -> numpy
  crc     crc32 of every geometry
  filter  per-record bbox overlap
Cold: a new index and row-group cache per tile. OS file cache: warmed once at the start.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import psutil

from starlet._internal.config import config_value
from starlet._internal.mvt.fast_tile import BBOX_COLS, RowGroupCache, TileFrame, _crc32_array
from starlet._internal.server.tiler.parquet_index import ParquetIndex

HERE = Path(__file__).resolve().parent
EXTENT = int(config_value("mvt", "extent"))
BUFFER = int(config_value("mvt", "buffer"))
STEPS = ["find", "load", "wkb", "bbox", "crc", "filter"]


def read_steps(index, cache, frame):
    """Returns ({step: ms}, records read, records in tile)."""
    t = dict.fromkeys(STEPS, 0.0)
    qx0, qy0, qx1, qy1 = frame.query_lonlat
    records_read = records_in_tile = 0

    s = time.perf_counter()
    plan = [(path, cache.file_info(path), cache.candidate_row_groups(path, frame.query_lonlat))
            for path in index.find_intersecting_files(frame.query_lonlat)]          # a. files + row groups
    t["find"] += time.perf_counter() - s

    for path, (pf, geom_col, has_bbox, _crs, attrs, _stats), rgs in plan:
        columns = [geom_col] + list(BBOX_COLS) + attrs
        for rg in rgs:
            s = time.perf_counter()
            table = pf.read_row_group(rg, columns=columns)                          # b. load the row group
            t["load"] += time.perf_counter() - s

            s = time.perf_counter()
            wkb = table[geom_col].to_pylist()                                      # c. geometry -> Python bytes
            t["wkb"] += time.perf_counter() - s

            s = time.perf_counter()
            a, b, c, d = [table[col].to_numpy(zero_copy_only=False).astype(np.float64) for col in BBOX_COLS]  # d. bbox -> numpy
            t["bbox"] += time.perf_counter() - s

            s = time.perf_counter()
            _crc32_array(wkb)                                                      # e. crc32 of every geometry
            t["crc"] += time.perf_counter() - s

            s = time.perf_counter()
            hit = (c >= qx0) & (a <= qx1) & (d >= qy0) & (b <= qy1)                # f. records that overlap
            t["filter"] += time.perf_counter() - s

            records_read += table.num_rows
            records_in_tile += int(hit.sum())

    return {k: v * 1000 for k, v in t.items()}, records_read, records_in_tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="built dataset folder (contains parquet_tiles/)")
    ap.add_argument("--name", required=True, help="benchmark name: reads <name>/tiles.csv")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    folder = HERE / args.name
    parquet_dir = Path(args.dataset) / "parquet_tiles"
    with open(folder / "tiles.csv", newline="") as fh:
        tiles = [(int(r["z"]), int(r["x"]), int(r["y"])) for r in csv.DictReader(fh)]
    for f in parquet_dir.glob("*.parquet"):          # warm the OS file cache once
        f.read_bytes()
    print(f"{len(tiles)} tiles, {args.runs} run(s); CPU load {psutil.cpu_percent(interval=1.0):.0f}%", flush=True)

    out = folder / "results_read_steps.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "z", "x", "y", *[f"{s}_ms" for s in STEPS], "read_ms", "records_read", "records_in_tile"])
        for run in range(1, args.runs + 1):
            order = tiles[:]
            random.Random(run).shuffle(order)
            for n, (z, x, y) in enumerate(order, 1):
                index = ParquetIndex(parquet_dir)                                   # cold
                cache = RowGroupCache(index, int(config_value("mvt", "row_group_cache")))   # cold
                frame = TileFrame(z, x, y, EXTENT, BUFFER)
                t, records_read, records_in_tile = read_steps(index, cache, frame)
                w.writerow([run, z, x, y, *[round(t[s], 3) for s in STEPS], round(sum(t.values()), 3),
                            records_read, records_in_tile])
                if n % 50 == 0:
                    print(f"run {run}: {n}/{len(order)} tiles", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
