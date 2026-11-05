"""Compare tile-generation scenarios (data layout + settings) on the same tiles, stage by stage.

A Scenario is a dataset folder plus the settings to apply. A Benchmark times the five on-the-fly stages
(Read, Pick, Decode, Transform, Encode) for every tile, cold per tile, using the real stage functions of
time_read.py (imported, not copied). The report puts the scenarios side by side and against last week's log.

--cold-disk: like a first-time request on a running server: footers stay cached across tiles, the row-group cache is
emptied and each tile's files are dropped from the OS file cache before the tile (outside the timers), so Read fetches
from disk. Adds bytes_requested / disk_read_bytes / background_bytes / cpu_pct columns (see time_read_steps.py) and
writes to tilebench_results_colddisk/ instead.

To compare something new, add one Scenario line to SCENARIOS. Example:
  python tilebench.py --zoom 8                  # zoom 8, both datasets, 3 runs
  python tilebench.py --zoom 8 10 --runs 1 --datasets asia
"""
import argparse
import csv
import gc
import random
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

import time_read as stages
from time_read_steps import chunk_bytes, evict_from_os_cache, evict_tile_files, startup_schema_read
from starlet._internal.config import config_value, get_loaded_config, set_loaded_config
from starlet._internal.mvt.fast_tile import (BBOX_COLS, RowGroupCache, TileFrame, generate_tile_ex, invalidate_caches,
                                             tile_geometries)
from starlet._internal.mvt.intermediate_tile import normalize_simplify_tolerance
from starlet._internal.server.tiler.parquet_index import ParquetIndex

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RG = HERE / "rowgroups"
STAGES = ("read", "pick", "decode", "transform", "encode")


@dataclass
class Scenario:
    name: str
    dataset: Path            # folder that contains parquet_tiles/
    crc: bool = True         # mvt.crc_priority (False: no WKB-to-bytes copy, no crc32)
    threads: bool = False    # load a tile's row groups 8 at a time before Read goes through them (crc off only)

    @property
    def slug(self):          # results file name
        return self.name.split("(")[0].strip().replace(" ", "_").replace("+", "plus").replace("/", "_")

    def apply(self):
        config = get_loaded_config()
        config.setdefault("mvt", {})["crc_priority"] = self.crc
        set_loaded_config(config)
        invalidate_caches()


ASIA = ROOT / "starlet_learning" / "profiling" / "out_mvt" / "dataset"
PARKS = ROOT / "starlet" / "datasets" / "OSM2021_parks"
OTHER = {"ne_states": "NE_states_provinces", "chicago": "Chicago_Crimes", "ebird": "ebird_california"}   # benchmark name: dataset
SCENARIOS = {
    "asia": [Scenario("baseline (original rows, crc on)", ASIA),
             Scenario("+ 2 MB row groups", RG / "asia" / "mb_2"),
             Scenario("+ no WKB/CRC", RG / "asia" / "mb_2", crc=False),
             Scenario("+ 8 threads", RG / "asia" / "mb_2", crc=False, threads=True)],
    "osm_parks": [Scenario("baseline (original rows, crc on)", PARKS),
                  Scenario("+ 2 MB row groups", RG / "osm_parks" / "mb_2"),
                  Scenario("+ no WKB/CRC", RG / "osm_parks" / "mb_2", crc=False),
                  Scenario("+ 8 threads", RG / "osm_parks" / "mb_2", crc=False, threads=True)],
    **{name: [Scenario("baseline (original rows, crc on)", ROOT / "starlet" / "datasets" / folder),
              Scenario("+ no WKB/CRC", RG / name / "mb_2", crc=False),
              Scenario("+ 8 threads", RG / name / "mb_2", crc=False, threads=True)] for name, folder in OTHER.items()},
}
POOL = ThreadPoolExecutor(8)


def load_group(cache, path, rg):
    """RowGroupCache.row_group for one group with crc off, on its own file handle: a ParquetFile is not shared across threads."""
    pf, geom_col, _has_bbox, _crs, attrs, _stats = cache.file_info(path)
    with pq.ParquetFile(path, metadata=pf.metadata) as f:
        table = f.read_row_group(rg, columns=[geom_col] + list(BBOX_COLS) + attrs)
    bb = [table[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in BBOX_COLS]
    entry = (table, *bb, np.zeros(table.num_rows, dtype=np.uint32))
    with cache._lock:
        cache._rgs[(path, rg)] = entry


def read_tile_parallel(index, cache, frame):
    """Load the tile's row groups 8 at a time into the cache, then run the real Read, which finds them all there."""
    cache.capacity = 10 ** 9                             # a tile can need more groups than the cache's limit
    jobs = [(path, rg) for path in index.find_intersecting_files(frame.query_lonlat)
            for rg in cache.candidate_row_groups(path, frame.query_lonlat) if (path, rg) not in cache._rgs]
    list(POOL.map(lambda job: load_group(cache, *job), jobs))
    return stages.read_tile(index, cache, frame)


class Benchmark:
    def __init__(self, dataset, zooms, runs, limit=None, cold_disk=False):
        with open(HERE / dataset / "tiles.csv", newline="") as fh:
            tiles = [(int(r["z"]), int(r["x"]), int(r["y"])) for r in csv.DictReader(fh)
                     if zooms[0] <= int(r["z"]) <= zooms[1]]
        self.dataset, self.zooms, self.runs, self.tiles = dataset, zooms, runs, tiles[:limit]
        self.cold_disk, self.shared = cold_disk, None

    def run(self, scenario):
        scenario.apply()
        self.read = read_tile_parallel if scenario.threads else stages.read_tile
        parquet_dir = scenario.dataset / "parquet_tiles"
        if self.cold_disk:
            self.shared = None
            for f in parquet_dir.glob("*.parquet"):      # start clean: nothing left over from earlier runs
                evict_from_os_cache(f)
            index = ParquetIndex(parquet_dir)            # footers cached across tiles, like the running server
            self.shared = (index, RowGroupCache(index, int(config_value("mvt", "row_group_cache"))))
            print(f"  {startup_schema_read(index)}", flush=True)
        else:
            for f in parquet_dir.glob("*.parquet"):      # warm the OS file cache once
                f.read_bytes()
        self.check_against_production(scenario, parquet_dir)
        rows = []
        for run in range(1, self.runs + 1):
            order = self.tiles[:]
            random.Random(run).shuffle(order)
            for tile in order:
                try:
                    rows.append(self.time_tile(parquet_dir, run, *tile))
                except MemoryError:                           # the biggest low-zoom tiles: skip, and say so
                    print(f"  run {run}: tile {tile} skipped, out of memory", flush=True)
                    gc.collect()
        return rows

    def stage_tile(self, parquet_dir, z, x, y, timed):
        """The five stages, each through `timed(name, function, *args)`; returns the staged tile."""
        if self.shared:
            index, cache = self.shared                                                # footers cached (cold disk)
        else:
            index = ParquetIndex(parquet_dir)                                         # cold
            cache = RowGroupCache(index, int(config_value("mvt", "row_group_cache")))  # cold
        frame = TileFrame(z, x, y, stages.EXTENT, stages.BUFFER)
        tolerance = normalize_simplify_tolerance(None, frame.cell)
        records_read, pieces, source = timed("read", self.read, index, cache, frame)
        records_in_tile, selection, source = timed("pick", stages.pick_tile, frame, pieces, source)
        decoded = timed("decode", stages.decode_tile, frame, source, selection)
        transformed = timed("transform", tile_geometries, frame, decoded[0], tolerance)
        tile, features = timed("encode", stages.encode_tile, frame, source, selection, decoded, transformed)
        return tile, features, records_read, records_in_tile

    def time_tile(self, parquet_dir, run, z, x, y):
        ms = {}

        def timed(name, function, *args):
            start = time.perf_counter()
            result = function(*args)
            ms[name] = (time.perf_counter() - start) * 1000
            return result

        if self.cold_disk:                                    # a first-time request: no row group in RAM
            index, cache = self.shared
            cache._rgs.clear()
            evict_tile_files(index, cache, TileFrame(z, x, y, stages.EXTENT, stages.BUFFER))
            psutil.cpu_percent()
            disk0 = psutil.disk_io_counters().read_bytes
        _, features, records_read, records_in_tile = self.stage_tile(parquet_dir, z, x, y, timed)
        disk = {}
        if self.cold_disk:                                    # every row group this tile loaded is in the emptied cache
            disk["disk_read_bytes"] = psutil.disk_io_counters().read_bytes - disk0
            disk["bytes_requested"] = sum(chunk_bytes(p, cache.file_info(p)[0], rg,
                                                      [cache.file_info(p)[1]] + list(BBOX_COLS) + cache.file_info(p)[4])
                                          for p, rg in cache._rgs)
            b0 = psutil.disk_io_counters().read_bytes        # other programs' reads over an idle wait as long as Read
            time.sleep(min(ms["read"] / 1000, 2.0))
            disk["background_bytes"] = psutil.disk_io_counters().read_bytes - b0
            disk["cpu_pct"] = psutil.cpu_percent()
        gc.collect()                                          # outside the timers
        pa.default_memory_pool().release_unused()
        return {"run": run, "z": z, "x": x, "y": y, **ms, "total": sum(ms.values()),
                "records_read": records_read, "records_in_tile": records_in_tile, "features": features, **disk}

    def check_against_production(self, scenario, parquet_dir):
        """The staged tile must equal what production generate_tile_ex makes (first tile only)."""
        z, x, y = self.tiles[0]
        tile, features, *_ = self.stage_tile(parquet_dir, z, x, y, lambda name, fn, *a: fn(*a))
        invalidate_caches()
        reference = generate_tile_ex(scenario.dataset, (z, x, y), feature_capacity=stages.FEATURE_CAPACITY,
                                     extent=stages.EXTENT, buffer=stages.BUFFER)
        invalidate_caches()
        if (tile, features) != reference:
            raise RuntimeError(f"{scenario.name}: staged tile differs from production for tile {z}/{x}/{y}")


def results_dir(dataset, zooms, cold_disk=False):
    return HERE / ("tilebench_results_colddisk" if cold_disk else "tilebench_results") / dataset / f"z{zooms[0]}-{zooms[1]}"


def median(rows, key):
    return st.median(float(r[key]) for r in rows)


def last_week(dataset, zooms):
    """Last week's logged five-stage run (results_python.csv), same zooms."""
    path = HERE / dataset / "results_python.csv"
    rows = [r for r in csv.DictReader(open(path)) if zooms[0] <= int(r["z"]) <= zooms[1]] if path.exists() else []
    if not rows:                                         # last week's log stops at zoom 10, and only has asia and parks
        return None
    for r in rows:
        r["total"] = sum(float(r[f"{s}_ms"]) for s in STAGES)
    return {**{s: median(rows, f"{s}_ms") for s in STAGES}, "total": median(rows, "total")}


def report(dataset, bench, results):
    base = median(results[0][1], "total")
    print(f"\n{dataset}, zoom {bench.zooms[0]}" + (f"-{bench.zooms[1]}" if bench.zooms[1] != bench.zooms[0] else "")
          + f" ({len(bench.tiles)} tiles, {bench.runs} runs): median ms per tile")
    print(f"{'':34}" + "".join(f"{s:>11}" for s in STAGES) + f"{'total':>9}{'vs baseline':>13}")
    old = last_week(dataset, bench.zooms)
    if old:
        print(f"{'last week (logged)':34}" + "".join(f"{old[s]:11.1f}" for s in STAGES) + f"{old['total']:9.1f}")
    for scenario, rows in results:
        total = median(rows, "total")
        print(f"{scenario.name:34}" + "".join(f"{median(rows, s):11.1f}" for s in STAGES)
              + f"{total:9.1f}{base / total:12.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--zoom", type=int, nargs="+", default=[8], help="one zoom, or LOW HIGH")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, help="only the first N tiles (smoke test)")
    ap.add_argument("--only", nargs="+", help="run just the scenarios with these slugs, e.g. plus_no_WKB_CRC")
    ap.add_argument("--cold-disk", action="store_true",
                    help="footers cached, no row-group cache, each tile's files dropped from the OS file cache first")
    args = ap.parse_args()
    zooms = (args.zoom[0], args.zoom[-1])
    for dataset in args.datasets:
        bench = Benchmark(dataset, zooms, args.runs, args.limit, args.cold_disk)
        results = []
        for scenario in [s for s in SCENARIOS[dataset] if not args.only or s.slug in args.only]:
            print(f"{dataset}: {scenario.name} ...", flush=True)
            rows = bench.run(scenario)
            results.append((scenario, rows))
            out = results_dir(dataset, zooms, args.cold_disk)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f"{scenario.slug}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        report(dataset, bench, results)


if __name__ == "__main__":
    main()
