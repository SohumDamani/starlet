#!/usr/bin/env python3
"""Single end-to-end starlet.build() run with explicit cache warmup.

Produces a clean before/after timing record. The warmup read is excluded
from the timed region so disk-cache state does not skew the result.
Output directory is deleted after the run unless --keep-output is set.

Usage:
    python benchmarks/run_e2e_once.py --tag pre --run-id 1
    python benchmarks/run_e2e_once.py --tag post --run-id 1 --keep-output
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = PROJECT_ROOT / "benchmark_data" / "asia_postal_codes.parquet"


def _warm_file_cache(path: Path) -> float:
    """Read the entire file to warm the OS file cache. Returns elapsed seconds.

    Without this, the first build() call hits cold storage, which can add several
    minutes on spinning disks and makes A/B comparisons unreliable. The warmup
    time is recorded separately and excluded from the build() measurement.
    """
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        # Read in 1 MB chunks — fast enough to saturate disk bandwidth without
        # allocating the whole file into memory at once.
        while f.read(1 << 20):
            pass
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(
        description="Single timed starlet.build() run for before/after comparison."
    )
    parser.add_argument("--tag", required=True, help="Label for this run, e.g. 'pre' or 'post'")
    parser.add_argument("--run-id", required=True, help="Repeat index, e.g. '1' or '2'")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to input GeoParquet or GeoJSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: benchmarks/results/_e2e_<tag>_<run-id>/)",
    )
    parser.add_argument(
        "--zoom", type=int, default=5,
        help="Maximum MVT zoom level (default: 5)",
    )
    parser.add_argument(
        "--threshold", type=float, default=50_000,
        help="Histogram density threshold for MVT tile filtering (default: 50000)",
    )
    parser.add_argument(
        "--max-parallel-files", type=int, default=8,
        help=(
            "Max concurrent tile writers. Kept low to avoid exhausting virtual "
            "memory on machines with a small pagefile (default: 8)"
        ),
    )
    parser.add_argument(
        "--keep-output", action="store_true",
        help="Do not delete the output directory after the run",
    )
    parser.add_argument(
        "--results-file",
        default=str(SCRIPT_DIR / "results" / "e2e_ab_runs.jsonl"),
        help="Path to append the JSON result record (default: benchmarks/results/e2e_ab_runs.jsonl)",
    )
    args = parser.parse_args()

    import starlet

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    outdir = Path(
        args.outdir or SCRIPT_DIR / "results" / f"_e2e_{args.tag}_{args.run_id}"
    ).resolve()
    if outdir.exists():
        # Remove stale output from a previous interrupted run so build() starts
        # clean and tile-write times are not inflated by overwrite overhead.
        shutil.rmtree(outdir)

    print(f"Input   : {input_path}")
    print(f"Outdir  : {outdir}")
    print(f"Tag     : {args.tag}  Run: {args.run_id}")
    print(f"Zoom    : {args.zoom}  Threshold: {args.threshold}")
    print()

    print("Warming file cache...")
    warmup_s = _warm_file_cache(input_path)
    print(f"Warmup done in {warmup_s:.2f}s (excluded from timing)")
    print()

    print("Running starlet.build()...")
    t0 = time.perf_counter()
    # starlet.build() returns (TileResult, MVTResult, pmtiles_path).
    # The third value is the path to the exported .pmtiles file; not needed here.
    tile_result, mvt_result, _ = starlet.build(
        input=str(input_path),
        outdir=str(outdir),
        zoom=args.zoom,
        threshold=args.threshold,
        max_parallel_files=args.max_parallel_files,
    )
    elapsed = time.perf_counter() - t0

    record = {
        "tag": args.tag,
        "run_id": args.run_id,
        "warmup_s": round(warmup_s, 3),
        "total_time_s": round(elapsed, 3),
        "parquet_tiles": tile_result.num_files,
        "total_rows": tile_result.total_rows,
        "mvt_tiles": mvt_result.tile_count,
        "zoom_levels": mvt_result.zoom_levels,
        "max_parallel_files": args.max_parallel_files,
        "version": starlet.__version__,
    }

    print(json.dumps(record, indent=2))

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # Append as JSONL (one JSON object per line) so multiple runs accumulate
    # in a single file without needing to parse and rewrite it each time.
    with open(results_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nResult appended to {results_path}")

    if not args.keep_output:
        shutil.rmtree(outdir, ignore_errors=True)
        print(f"Output directory removed (use --keep-output to retain it)")


if __name__ == "__main__":
    main()
