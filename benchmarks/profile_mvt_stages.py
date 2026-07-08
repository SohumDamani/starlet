#!/usr/bin/env python3
"""Stage-level profiling of MVT geometry streaming.

Two modes:

  --mode stages (default)
      Re-implements the same row-group loop as GeometryStreamer._decode_table()
      but wraps each sub-step with wall-clock timers. Stages match the
      v0.3.1 vectorised pipeline: WKB decode / make_valid / reproject all
      happen at array level (one call per row group, not one per geometry).
      NOTE: if the internal architecture changes significantly, update the
      _profile_stages() function to match.

  --mode cprofile
      Calls starlet.generate_mvt() directly and profiles it with cProfile.
      This is always accurate regardless of code changes -- prefer this mode
      when you just want to know where time is going after an upstream update.

Usage:
    # Stage-level breakdown (requires a dataset directory with parquet_tiles/
    # and histograms/ already built by starlet tile):
    python benchmarks/profile_mvt_stages.py --dataset-dir path/to/dataset

    # Auto-accurate cProfile mode:
    python benchmarks/profile_mvt_stages.py --dataset-dir path/to/dataset --mode cprofile

    # Compare two saved stage profiles:
    python benchmarks/compare_profiles.py --baseline old.json --current new.json
"""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Default dataset directory — produced by running `starlet tile` on any input.
# Override with --dataset-dir on the command line.
DEFAULT_DATASET_DIR = PROJECT_ROOT / "benchmark_output_parquet"

# Stages measured by _profile_stages(). These match the v0.3.1 row-group
# vectorised pipeline. Update this list if the architecture changes.
STAGES = [
    "parquet_read",    # read_row_group() from disk
    "wkb_decode",      # shapely.from_wkb(array) — whole row group at once
    "make_valid",      # shapely.make_valid(array) — whole row group at once
    "reproject",       # shapely.transform(array, fn) — whole row group at once
    "attr_extract",    # extract non-geometry columns to Python lists
    "tile_assign",     # per-geometry: bounds + tile-range + bucket insert
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_histogram(dataset_dir: Path) -> Path:
    """Return the prefix-sum histogram if available, falling back to the raw one.

    The prefix-sum version enables TileAssigner to count features in any
    rectangular region with 4 array lookups instead of summing individual cells.
    Always prefer it when present.
    """
    hist_dir = dataset_dir / "histograms"
    prefix = hist_dir / "global_prefix.npy"
    raw = hist_dir / "global.npy"
    if prefix.exists():
        return prefix
    if raw.exists():
        return raw
    raise FileNotFoundError(f"No histogram found under {hist_dir}")


def _get_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=PROJECT_ROOT,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# Mode 1: Manual stage timing (v0.3.1 row-group vectorised architecture)
# ---------------------------------------------------------------------------

def _reproject_coords(transformer: Transformer, coords: np.ndarray) -> np.ndarray:
    # coords is an (N, 2) array of [lon, lat] values in EPSG:4326.
    # always_xy=True (set on the Transformer) ensures we pass lon first, lat second
    # regardless of what the CRS spec says — without it pyproj 3+ respects the
    # CRS axis order and silently swaps x/y for EPSG:4326.
    x, y = transformer.transform(coords[:, 0], coords[:, 1])
    return np.column_stack([x, y])


def _profile_stages(
    parquet_dir: Path,
    hist_path: Path,
    zoom: int,
    threshold: float,
) -> dict:
    # Imported here to keep the top-level namespace clean and to avoid import
    # errors when starlet is not yet installed (e.g. --help still works).
    from starlet._internal.histogram.loader import HistogramLoader
    from starlet._internal.mvt.assigner import TileAssigner
    from starlet._internal.mvt.helpers import mercator_bounds_to_tile_range

    stage_time = {s: 0.0 for s in STAGES}
    geom_count = 0
    skipped = 0
    row_group_count = 0

    # load() returns a 4096x4096 numpy array where each cell holds the prefix-sum
    # count of geometry vertices in that grid cell, enabling O(1) range queries.
    prefix = HistogramLoader(str(hist_path)).load()
    zooms = list(range(0, zoom + 1))
    assigner = TileAssigner(zooms, prefix, threshold)
    # compute_nonempty() uses the prefix sum to filter the 16M+ zoom-12 candidates
    # down to only the z/x/y tiles that exceed the density threshold.
    assigner.compute_nonempty()
    total_nonempty = sum(len(v) for v in assigner.nonempty.values())
    print(f"Non-empty tiles across {len(zooms)} zoom levels: {total_nonempty:,}")

    # always_xy=True forces lon/lat input order regardless of CRS axis convention.
    # Without it, pyproj 3+ respects the EPSG:4326 spec (lat/lon), which silently
    # produces wrong tile coordinates (x and y swapped) on the output.
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    # shapely.transform() calls this function with an (N, 2) coordinate array per geometry.
    reproject = lambda coords: _reproject_coords(to_3857, coords)

    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet tile files")

    t_loop_start = time.perf_counter()

    for pf_path in parquet_files:
        pf_obj = pq.ParquetFile(pf_path)
        for rg in range(pf_obj.num_row_groups):
            row_group_count += 1

            # --- parquet_read ---
            t0 = time.perf_counter()
            table = pf_obj.read_row_group(rg)
            stage_time["parquet_read"] += time.perf_counter() - t0

            # --- wkb_decode (array-level, v0.3.1) ---
            t0 = time.perf_counter()
            wkb_arr = table["geometry"].to_numpy(zero_copy_only=False)
            geoms = shapely.from_wkb(wkb_arr)
            stage_time["wkb_decode"] += time.perf_counter() - t0

            # --- make_valid (array-level, v0.3.1) ---
            t0 = time.perf_counter()
            geoms = shapely.make_valid(geoms)
            stage_time["make_valid"] += time.perf_counter() - t0

            # --- reproject (array-level, v0.3.1) ---
            t0 = time.perf_counter()
            geoms = shapely.transform(geoms, reproject)
            stage_time["reproject"] += time.perf_counter() - t0

            # --- attr_extract ---
            t0 = time.perf_counter()
            attrs = {
                col: table[col].to_pylist()
                for col in table.column_names
                if col != "geometry"
            }
            stage_time["attr_extract"] += time.perf_counter() - t0

            # --- tile_assign (per geometry) ---
            # Unlike the earlier stages, this cannot be vectorised: each geometry maps
            # to a different set of z/x/y tile buckets, so we must loop to do per-tile
            # inserts. This is typically the dominant stage on large datasets.
            t0 = time.perf_counter()
            for i, geom in enumerate(geoms):
                if geom is None or geom.is_empty:
                    skipped += 1
                    continue
                row_attrs = {k: attrs[k][i] for k in attrs}
                minx, miny, maxx, maxy = geom.bounds
                for z in zooms:
                    tx0, ty0, tx1, ty1 = mercator_bounds_to_tile_range(
                        z, minx, miny, maxx, maxy
                    )
                    for x in range(tx0, tx1 + 1):
                        for y in range(ty0, ty1 + 1):
                            if (x, y) in assigner.nonempty[z]:
                                assigner._priority_insert(
                                    z, x, y, i / max(len(geoms), 1), (geom, row_attrs)
                                )
                geom_count += 1
            stage_time["tile_assign"] += time.perf_counter() - t0

            if geom_count % 50_000 == 0 and geom_count > 0:
                print(f"  ...processed {geom_count:,} geometries")

    total_time = time.perf_counter() - t_loop_start

    return {
        "mode": "stages",
        "geom_count": geom_count,
        "row_group_count": row_group_count,
        "skipped": skipped,
        "total_loop_time_s": round(total_time, 3),
        "stage_time_s": {k: round(v, 3) for k, v in stage_time.items()},
    }


def _print_stage_report(result: dict):
    total = result["total_loop_time_s"]
    stages = result["stage_time_s"]
    geom_count = result["geom_count"]

    print()
    print(f"{'Stage':<18} | {'Time (s)':>10} | {'% of loop':>10} | {'geoms/s':>10}")
    print("-" * 56)
    for stage, t in stages.items():
        pct = (t / total * 100) if total else 0
        rate = geom_count / t if t > 0 else float("inf")
        print(f"{stage:<18} | {t:>10.3f} | {pct:>9.1f}% | {rate:>10,.0f}")
    print("-" * 56)
    measured = sum(stages.values())
    print(f"{'measured sum':<18} | {measured:>10.3f} | {(measured/total*100 if total else 0):>9.1f}%")
    print(f"{'total wall time':<18} | {total:>10.3f}")
    print()
    print(f"Geometries: {geom_count:,} processed, {result['skipped']:,} skipped")
    print(f"Row groups: {result['row_group_count']:,}")


# ---------------------------------------------------------------------------
# Mode 2: cProfile (always accurate, no maintenance needed)
# ---------------------------------------------------------------------------

# Functions to highlight in the cProfile summary. These are the key hot spots
# in the v0.3.1 pipeline — add/remove entries here if they change.
_HIGHLIGHT_FUNCTIONS = {
    "from_wkb",
    "make_valid",
    "transform",           # shapely.transform (reprojection)
    "read_row_group",
    "_priority_insert",
    "mercator_bounds_to_tile_range",
    "run",                 # BucketMVTGenerator.run
}


def _profile_cprofile(dataset_dir: Path, zoom: int, threshold: float) -> dict:
    import starlet

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    mvt_result = starlet.generate_mvt(
        tile_dir=str(dataset_dir),
        zoom=zoom,
        threshold=threshold,
    )
    pr.disable()
    total_time = time.perf_counter() - t0

    stream = StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    ps.print_stats(40)
    profile_text = stream.getvalue()

    # Extract timing for highlighted functions.
    # pstats.Stats.stats is a dict keyed by (file, lineno, funcname).
    # The value tuple is (primitive_calls, total_calls, tottime, cumtime, callers).
    # tottime = time spent inside the function itself (excludes callees).
    # cumtime = total time including all callees — use this to rank hot spots.
    highlights: dict[str, dict] = {}
    ps2 = pstats.Stats(pr)
    for func, (cc, nc, tt, ct, _) in ps2.stats.items():
        fname = func[2]
        if fname in _HIGHLIGHT_FUNCTIONS:
            highlights[fname] = {
                "calls": nc,
                "tottime_s": round(tt, 4),
                "cumtime_s": round(ct, 4),
            }

    return {
        "mode": "cprofile",
        "total_wall_time_s": round(total_time, 3),
        "mvt_tiles_generated": mvt_result.tile_count,
        "zoom_levels": mvt_result.zoom_levels,
        "highlights": highlights,
        "full_profile_text": profile_text,
    }


def _print_cprofile_report(result: dict):
    print()
    print(f"Total wall time: {result['total_wall_time_s']:.3f}s")
    print(f"MVT tiles generated: {result['mvt_tiles_generated']:,}")
    print(f"Zoom levels: {result['zoom_levels']}")
    print()
    print("Key function timings:")
    print(f"{'Function':<35} | {'Calls':>8} | {'tottime (s)':>12} | {'cumtime (s)':>12}")
    print("-" * 74)
    for fname, stats in sorted(
        result["highlights"].items(), key=lambda kv: -kv[1]["cumtime_s"]
    ):
        print(
            f"{fname:<35} | {stats['calls']:>8,} | "
            f"{stats['tottime_s']:>12.4f} | {stats['cumtime_s']:>12.4f}"
        )
    print()
    print("--- Full cProfile output (top 40 by cumulative time) ---")
    print(result["full_profile_text"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile MVT generation. Use --mode cprofile for automatic accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help=(
            "Dataset directory containing parquet_tiles/ and histograms/ "
            f"(default: {DEFAULT_DATASET_DIR})"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["stages", "cprofile"],
        default="stages",
        help=(
            "stages: manual per-stage wall-clock timing matching v0.3.1 architecture. "
            "cprofile: wrap starlet.generate_mvt() with Python cProfile - always accurate."
        ),
    )
    parser.add_argument("--zoom", type=int, default=5, help="Maximum zoom level (default: 5)")
    parser.add_argument("--threshold", type=float, default=50_000, help="Histogram density threshold (default: 50000)")
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "results" / "mvt_stage_profile.json"),
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    parquet_dir = dataset_dir / "parquet_tiles"

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")
    if not parquet_dir.exists():
        raise SystemExit(f"No parquet_tiles/ found under {dataset_dir}")

    print(f"Dataset : {dataset_dir}")
    print(f"Mode    : {args.mode}")
    print(f"Zoom    : {args.zoom}")
    print(f"Threshold: {args.threshold}")
    print()

    if args.mode == "stages":
        hist_path = _find_histogram(dataset_dir)
        print(f"Histogram: {hist_path}")
        result = _profile_stages(parquet_dir, hist_path, args.zoom, args.threshold)
        _print_stage_report(result)
    else:
        result = _profile_cprofile(dataset_dir, args.zoom, args.threshold)
        _print_cprofile_report(result)

    result["metadata"] = {
        "mode": args.mode,
        "git_commit": _get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": {
            "dataset_dir": str(dataset_dir),
            "zoom": args.zoom,
            "threshold": args.threshold,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "cprofile":
        # Don't save the full profile text in JSON — it's large and not useful for compare_profiles.py
        result_to_save = {k: v for k, v in result.items() if k != "full_profile_text"}
    else:
        result_to_save = result
    out_path.write_text(json.dumps(result_to_save, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
