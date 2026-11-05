"""Time the steps inside READ for every benchmark tile (Read only, no later stages).

Same work as fast_tile.generate_tile_ex's read loop (fast_tile.py:374-393) with
RowGroupCache.row_group (fast_tile.py:286-297) copied in, so each step can be timed:
  find    files by name + row groups by footer min/max
  load    read_row_group: all rows, geometry + bbox + attribute columns (decompress -> Arrow)
  wkb     every geometry -> Python bytes            (only with crc32 priority; --no-crc removes it)
  bbox    the 4 bbox columns -> numpy
  crc     crc32 of every geometry                   (only with crc32 priority; --no-crc removes it)
  filter  per-record bbox overlap
--bbox-first: load_bbox (4 bbox columns) -> bbox -> filter -> load_rest (geometry + attributes, only for row groups
with at least one overlapping record). records_full = records in row groups loaded in full.
Cold (default): a new index and row-group cache per tile. --warm: one index and footer cache for all tiles, like the
running server (row groups are still read for every tile). OS file cache: warmed once at the start.
--cold-disk (with --warm): no warm-up, and before every tile (outside the timer) the files it reads are dropped from
Windows' file cache, so every row group comes from disk like a first-time request; footers stay cached in memory.
bytes_requested = compressed bytes of the column chunks read (from the footer); disk_read_bytes = bytes the disk
actually delivered during the tile (Windows disk counters, whole system); background_bytes (--cold-disk only) = bytes
the disk delivered in an idle wait of the same length right after the tile, i.e. reads by other programs;
cpu_pct = whole-system CPU load over the tile and that wait (other programs using the CPU would slow load).
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
import pyarrow.parquet as pq

from starlet._internal.config import config_value, get_loaded_config, set_loaded_config
from starlet._internal.mvt.fast_tile import BBOX_COLS, RowGroupCache, TileFrame, _crc32_array
from starlet._internal.server.tiler.parquet_index import ParquetIndex

HERE = Path(__file__).resolve().parent
EXTENT = int(config_value("mvt", "extent"))
BUFFER = int(config_value("mvt", "buffer"))


def evict_from_os_cache(path):
    """Drop one file's bytes from the Windows file cache: opening it unbuffered makes Windows purge its cached pages.
    Only works while no other handle has the file open (pyarrow's must be closed first). No Administrator needed."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    h = k32.CreateFileW(str(path), 0x80000000, 0x7, None, 3, 0x20000000, None)   # GENERIC_READ, share all, OPEN_EXISTING, NO_BUFFERING
    if h == wintypes.HANDLE(-1).value:
        raise OSError(ctypes.get_last_error(), f"could not open {path} unbuffered")
    k32.CloseHandle(h)


def evict_tile_files(index, cache, frame):
    """Make this tile's row groups come from disk while keeping footers cached: for each file the tile reads, close
    pyarrow's handle, evict the file from the OS cache, and reopen it with the footer already in memory (no disk read)."""
    for path in index.find_intersecting_files(frame.query_lonlat):
        info = cache._files.get(path)
        if info is not None:
            footer = info[0].metadata
            info[0].close()
        evict_from_os_cache(path)
        if info is not None:
            cache._files[path] = (pq.ParquetFile(path, metadata=footer), *info[1:])


def startup_schema_read(index):
    """ParquetIndex.find_intersecting_files reads the schema (for the CRS) of EVERY file on the first request, from disk.
    Done here once, timed, so it is not hidden in the first tile's eviction step."""
    disk0, s = psutil.disk_io_counters().read_bytes, time.perf_counter()
    for path, _bbox in index._entries:
        index._schema_info(path)
    return (f"startup: schema of all {len(index._entries)} files read from disk in {(time.perf_counter() - s) * 1000:.1f} ms "
            f"({(psutil.disk_io_counters().read_bytes - disk0) / 1e6:.2f} MB)")


_LEAVES = {}


def chunk_bytes(path, pf, rg, columns):
    """Compressed bytes of these columns in one row group, from the footer: what read_row_group fetches from the file."""
    leaves = _LEAVES.get(path)
    if leaves is None:
        md = pf.metadata
        leaves = _LEAVES[path] = {}
        for i in range(md.num_columns):                                         # nested columns have several leaves
            leaves.setdefault(md.schema.column(i).path.split(".")[0], []).append(i)
    r = pf.metadata.row_group(rg)
    return sum(r.column(i).total_compressed_size for c in columns for i in leaves[c])


def step_names(use_crc, bbox_first=False):
    if bbox_first:
        return ["find", "load_bbox", "bbox", "filter", "load_rest"]
    return ["find", "load", "wkb", "bbox", "crc", "filter"] if use_crc else ["find", "load", "bbox", "filter"]


def read_steps_bbox_first(index, cache, frame):
    """Bbox first: load only the 4 bbox columns, filter, then load geometry + attributes only for row groups
    with at least one overlapping record.
    Returns ({step: ms}, records read, records fully loaded, records in tile, bytes requested)."""
    t = dict.fromkeys(step_names(False, True), 0.0)
    qx0, qy0, qx1, qy1 = frame.query_lonlat
    records_read = records_full = records_in_tile = bytes_requested = 0

    s = time.perf_counter()
    plan = [(path, cache.file_info(path), cache.candidate_row_groups(path, frame.query_lonlat))
            for path in index.find_intersecting_files(frame.query_lonlat)]          # a. files + row groups
    t["find"] += time.perf_counter() - s

    for path, (pf, geom_col, has_bbox, _crs, attrs, _stats), rgs in plan:
        for rg in rgs:
            s = time.perf_counter()
            table = pf.read_row_group(rg, columns=list(BBOX_COLS))                  # b. load the 4 bbox columns
            t["load_bbox"] += time.perf_counter() - s

            s = time.perf_counter()
            a, b, c, d = [table[col].to_numpy(zero_copy_only=False).astype(np.float64) for col in BBOX_COLS]  # c. bbox -> numpy
            t["bbox"] += time.perf_counter() - s

            s = time.perf_counter()
            hit = (c >= qx0) & (a <= qx1) & (d >= qy0) & (b <= qy1)                # d. records that overlap
            t["filter"] += time.perf_counter() - s

            records_read += table.num_rows
            bytes_requested += chunk_bytes(path, pf, rg, list(BBOX_COLS))
            n_hit = int(hit.sum())
            records_in_tile += n_hit
            if n_hit:
                s = time.perf_counter()
                pf.read_row_group(rg, columns=[geom_col] + attrs)                   # e. the rest, only when needed
                t["load_rest"] += time.perf_counter() - s
                records_full += table.num_rows
                bytes_requested += chunk_bytes(path, pf, rg, [geom_col] + attrs)

    return {k: v * 1000 for k, v in t.items()}, records_read, records_full, records_in_tile, bytes_requested


def read_steps(index, cache, frame, use_crc):
    """Returns ({step: ms}, records read, records in tile, bytes requested)."""
    t = dict.fromkeys(step_names(use_crc), 0.0)
    qx0, qy0, qx1, qy1 = frame.query_lonlat
    records_read = records_in_tile = bytes_requested = 0

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
            bytes_requested += chunk_bytes(path, pf, rg, columns)

    return {k: v * 1000 for k, v in t.items()}, records_read, records_in_tile, bytes_requested


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
    ap.add_argument("--bbox-first", action="store_true",
                    help="load the 4 bbox columns, filter, then load the rest only for row groups with a hit (needs --no-crc)")
    ap.add_argument("--cold-disk", action="store_true",
                    help="no warm-up; drop each tile's files from the OS file cache first, so row groups come from disk (needs --warm)")
    ap.add_argument("--limit", type=int, help="time only the first N tiles of the list (quick checks)")
    args = ap.parse_args()
    if args.bbox_first and not args.no_crc:
        ap.error("--bbox-first needs --no-crc (the crc32 would need every geometry)")
    if args.cold_disk and not args.warm:
        ap.error("--cold-disk needs --warm (footers are cached; only row-group bytes are dropped from the OS cache)")

    if args.no_crc:
        config = get_loaded_config()
        config.setdefault("mvt", {})["crc_priority"] = False
        set_loaded_config(config)
    use_crc = bool(config_value("mvt", "crc_priority", True))
    steps = step_names(use_crc, args.bbox_first)

    folder = HERE / args.name
    parquet_dir = Path(args.dataset) / "parquet_tiles"
    with open(args.tiles or folder / "tiles.csv", newline="") as fh:
        tiles = [(int(r["z"]), int(r["x"]), int(r["y"])) for r in csv.DictReader(fh)]
    tiles = tiles[:args.limit] if args.limit else tiles
    if args.cold_disk:
        for f in parquet_dir.glob("*.parquet"):      # start clean: nothing left over from earlier runs
            evict_from_os_cache(f)
    else:
        for f in parquet_dir.glob("*.parquet"):      # warm the OS file cache once
            f.read_bytes()
    if args.warm:                                    # shared by every tile, like the server's cached index
        index = ParquetIndex(parquet_dir)
        cache = RowGroupCache(index, int(config_value("mvt", "row_group_cache")))
    if args.cold_disk:                               # the first request reads every file's schema (CRS); time it once
        print(startup_schema_read(index), flush=True)
    print(f"{len(tiles)} tiles, {args.runs} run(s), crc priority {'on' if use_crc else 'off'}, "
          f"{'warm' if args.warm else 'cold'}{', bbox first' if args.bbox_first else ''}"
          f"{', tile files dropped from the OS file cache before every tile' if args.cold_disk else ', OS file cache warmed'}; "
          f"CPU load {psutil.cpu_percent(interval=1.0):.0f}%", flush=True)

    out = folder / "results_read_steps.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "z", "x", "y", *[f"{s}_ms" for s in steps], "read_ms", "records_read", "records_full", "records_in_tile",
                    "bytes_requested", "disk_read_bytes", "background_bytes", "cpu_pct"])
        for run in range(1, args.runs + 1):
            order = tiles[:]
            random.Random(run).shuffle(order)
            for n, (z, x, y) in enumerate(order, 1):
                if not args.warm:
                    index = ParquetIndex(parquet_dir)                               # cold
                    cache = RowGroupCache(index, int(config_value("mvt", "row_group_cache")))   # cold
                frame = TileFrame(z, x, y, EXTENT, BUFFER)
                if args.cold_disk:
                    evict_tile_files(index, cache, frame)                           # this tile is a first-time request
                psutil.cpu_percent()                                                # start the CPU-load window
                disk0 = psutil.disk_io_counters().read_bytes
                if args.bbox_first:
                    t, records_read, records_full, records_in_tile, nbytes = read_steps_bbox_first(index, cache, frame)
                else:
                    t, records_read, records_in_tile, nbytes = read_steps(index, cache, frame, use_crc)
                    records_full = records_read                 # every record read is loaded in full
                disk_bytes = psutil.disk_io_counters().read_bytes - disk0
                background = ""
                if args.cold_disk:                       # other programs' disk reads over an idle wait as long as the tile
                    b0 = psutil.disk_io_counters().read_bytes
                    time.sleep(min(sum(t.values()) / 1000, 2.0))
                    background = psutil.disk_io_counters().read_bytes - b0
                w.writerow([run, z, x, y, *[round(t[s], 3) for s in steps], round(sum(t.values()), 3),
                            records_read, records_full, records_in_tile, nbytes, disk_bytes, background, psutil.cpu_percent()])
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
