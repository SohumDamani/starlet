"""Time the steps inside READ for every benchmark tile (Read only, no later stages).

Same work as fast_tile.generate_tile_ex's read loop (fast_tile.py:374-393) with
RowGroupCache.row_group (fast_tile.py:286-297) copied in, so each step can be timed:
  find    files by name + row groups by footer min/max
  load    read_row_group: all rows, geometry + bbox + attribute columns (decompress -> Arrow)
  wkb     every geometry -> Python bytes            (only with crc32 priority; --no-crc removes it)
  bbox    the 4 bbox columns -> numpy
  crc     crc32 of every geometry                   (only with crc32 priority; --no-crc removes it)
  filter  per-record bbox overlap
Cold (default): a new index and row-group cache per tile. --warm: one index and footer cache for all tiles, like the
running server (row groups are still read for every tile). OS file cache: warmed once at the start.
"""
import argparse
import csv
import gc
import random
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa

from starlet._internal.config import config_value, get_loaded_config, set_loaded_config
from starlet._internal.mvt.fast_tile import BBOX_COLS, RowGroupCache, TileFrame, _crc32_array
from starlet._internal.server.tiler.parquet_index import ParquetIndex

HERE = Path(__file__).resolve().parent
EXTENT = int(config_value("mvt", "extent"))
BUFFER = int(config_value("mvt", "buffer"))


def step_names(use_crc):
    return ["find", "load", "wkb", "bbox", "crc", "filter"] if use_crc else ["find", "load", "bbox", "filter"]


def read_steps(index, cache, frame, use_crc):
    """Returns ({step: ms}, records read, records in tile)."""
    t = dict.fromkeys(step_names(use_crc), 0.0)
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

            if use_crc:
                s = time.perf_counter()
                wkb = table[geom_col].to_pylist()                                  # c. geometry -> Python bytes
                t["wkb"] += time.perf_counter() - s

            s = time.perf_counter()
            a, b, c, d = [table[col].to_numpy(zero_copy_only=False).astype(np.float64) for col in BBOX_COLS]  # d. bbox -> numpy
            t["bbox"] += time.perf_counter() - s

            if use_crc:
                s = time.perf_counter()
                _crc32_array(wkb)                                                  # e. crc32 of every geometry
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
    ap.add_argument("--name", required=True, help="benchmark name: reads <name>/tiles.csv, writes <name>/results_read_steps.csv")
    ap.add_argument("--tiles", type=Path, help="use this tile list instead of <name>/tiles.csv")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-crc", action="store_true",
                    help="mvt.crc_priority = false: no WKB-to-bytes copy and no crc32 (the wkb and crc steps are gone)")
    ap.add_argument("--warm", action="store_true",
                    help="keep one index and footer cache across tiles, like the running server")
    args = ap.parse_args()

    if args.no_crc:
        config = get_loaded_config()
        config.setdefault("mvt", {})["crc_priority"] = False
        set_loaded_config(config)
    use_crc = bool(config_value("mvt", "crc_priority", True))
    steps = step_names(use_crc)

    folder = HERE / args.name
    parquet_dir = Path(args.dataset) / "parquet_tiles"
    with open(args.tiles or folder / "tiles.csv", newline="") as fh:
        tiles = [(int(r["z"]), int(r["x"]), int(r["y"])) for r in csv.DictReader(fh)]
    for f in parquet_dir.glob("*.parquet"):          # warm the OS file cache once
        f.read_bytes()
    if args.warm:                                    # shared by every tile, like the server's cached index
        index = ParquetIndex(parquet_dir)
        cache = RowGroupCache(index, int(config_value("mvt", "row_group_cache")))
    print(f"{len(tiles)} tiles, {args.runs} run(s), crc priority {'on' if use_crc else 'off'}, "
          f"{'warm' if args.warm else 'cold'}; "
          f"CPU load {psutil.cpu_percent(interval=1.0):.0f}%", flush=True)

    out = folder / "results_read_steps.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "z", "x", "y", *[f"{s}_ms" for s in steps], "read_ms", "records_read", "records_in_tile"])
        for run in range(1, args.runs + 1):
            order = tiles[:]
            random.Random(run).shuffle(order)
            for n, (z, x, y) in enumerate(order, 1):
                if not args.warm:
                    index = ParquetIndex(parquet_dir)                               # cold
                    cache = RowGroupCache(index, int(config_value("mvt", "row_group_cache")))   # cold
                frame = TileFrame(z, x, y, EXTENT, BUFFER)
                t, records_read, records_in_tile = read_steps(index, cache, frame, use_crc)
                w.writerow([run, z, x, y, *[round(t[s], 3) for s in steps], round(sum(t.values()), 3),
                            records_read, records_in_tile])
                fh.flush()
                if not args.warm:
                    del index, cache                     # free this tile's files and footers (outside the timer)
                del frame
                gc.collect()
                pa.default_memory_pool().release_unused()
                if n % 50 == 0:
                    print(f"run {run}: {n}/{len(order)} tiles", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
