#!/usr/bin/env python3
"""Capacity scaling benchmark for IntermediateVectorTile.

Loads real features from benchmark_output_parquet/parquet_tiles/, feeds them
to IntermediateVectorTile instances at capacity 2_000, 5_000, and 10_000, and
compares encode time and tile byte size. No triage, no modifications — this is
the raw baseline before any heuristics are applied.

The same feature pool is fed to all three capacities; only the heap cap differs.
A higher capacity retains more features → larger tile → longer encode time.

Usage:
    python benchmarks/profile_capacity.py
    python benchmarks/profile_capacity.py --max-features 12000
    python benchmarks/profile_capacity.py --capacities 1000 2000 5000 10000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starlet._internal.mvt.helpers import mercator_bounds_to_tile_range
from starlet._internal.mvt.intermediate_tile import IntermediateVectorTile
from starlet._internal.mvt.mvt_generator import _iter_web_mercator_features

CAPACITIES = [2_000, 5_000, 10_000]
DEFAULT_TILE_DIR = PROJECT_ROOT / "benchmark_output_parquet" / "parquet_tiles"
DEFAULT_MAX_FEATURES = 12_000  # Load this many; must exceed max capacity


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_features(tile_dir: Path, max_features: int) -> list[tuple]:
    """Load up to max_features from parquet tile files, ready for IVT.

    Each element: (mercator_geometry, properties_dict, priority_int).

    Delegates entirely to _iter_web_mercator_features() from the library:
    - Reads CRS from file metadata (no hardcoded EPSG:4326 assumption)
    - Reprojects to Web Mercator via the library's reproject_geometries()
    - Extracts ALL non-geometry, non-internal columns automatically
    - Computes priority from source WKB bytes (pre-reproject) for
      seam-consistency with the main pipeline
    """
    parquet_files = sorted(tile_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {tile_dir}")

    print(f"Loading from {len(parquet_files)} parquet file(s) in {tile_dir.name}/")
    features: list[tuple] = []

    for pf_path in parquet_files:
        if len(features) >= max_features:
            break
        table = pq.read_table(pf_path)
        for geom, attrs, priority in _iter_web_mercator_features(table, geom_col="geometry"):
            features.append((geom, attrs, priority))
            if len(features) >= max_features:
                break
        print(f"  {pf_path.name[:55]}: {len(features):,} features total")

    print(f"\nLoaded {len(features):,} features (stopped at max={max_features:,})")
    return features


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------

def pick_tile_z5(features: list[tuple]) -> tuple[int, int, int]:
    """Return zoom-5 tile (z, x, y) whose centre is closest to the feature centroid.

    Uses up to 500 feature centroids so this stays fast regardless of pool size.
    World bounds come from helpers.WORLD_MINX/MAXX/MAXY (no local constants).
    """
    sample = features[:500]
    cx = float(np.mean([f[0].centroid.x for f in sample]))
    cy = float(np.mean([f[0].centroid.y for f in sample]))

    z = 5
    # Treat (cx, cy) as a degenerate bounding box (point = minx/maxx and miny/maxy are equal).
    # mercator_bounds_to_tile_range collapses it to a single tile; discard the redundant tx1/ty1.
    tx, ty, _, _ = mercator_bounds_to_tile_range(z, cx, cy, cx, cy)
    return z, tx, ty


# ---------------------------------------------------------------------------
# Single capacity run
# ---------------------------------------------------------------------------

def run_capacity(
    features: list[tuple],
    z: int,
    x: int,
    y: int,
    capacity: int,
    use_triage: bool = False,
    kld_threshold: float = 0.1,
) -> dict:
    """Feed all features to an IVT at the given capacity; encode and time it.

    The IVT heap naturally caps at `capacity` (top-k by priority), so every
    capacity level sees the same features but keeps a different number of them.
    If use_triage=True, calls triage() between add_feature and encode, applying
    both numeric quantization and string prefix triage.
    """
    tile = IntermediateVectorTile(z, x, y, feature_capacity=capacity)

    t0 = time.perf_counter()
    for geom, props, prio in features:
        tile.add_feature(geom, props, priority=prio)
    add_time = time.perf_counter() - t0

    if use_triage:
        tile.triage(kld_threshold=kld_threshold)

    t0 = time.perf_counter()
    mvt_bytes = tile.encode()
    encode_time = time.perf_counter() - t0

    return {
        "capacity": capacity,
        "use_triage": use_triage,
        "features_fed": len(features),
        "features_retained": tile.feature_count,
        "add_time_s": round(add_time, 3),
        "encode_time_s": round(encode_time, 3),
        "total_time_s": round(add_time + encode_time, 3),
        "tile_bytes": len(mvt_bytes),
        "tile_kb": round(len(mvt_bytes) / 1024, 2),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list[dict], tile: tuple[int, int, int]) -> None:
    z, x, y = tile
    header_cols = ["Capacity", "Retained", "add (s)", "encode (s)", "total (s)", "Size (KB)"]
    widths      = [10,          10,          10,         12,           10,           10]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    print()
    print(f"Tile: z={z} x={x} y={y}  (zoom-5 tile covering feature centroid)")
    print(sep)
    hrow = " | ".join(f"{h:^{w}}" for h, w in zip(header_cols, widths))
    print(f"| {hrow} |")
    print(sep)
    for r in results:
        vals = [
            f"{r['capacity']:,}",
            f"{r['features_retained']:,}",
            f"{r['add_time_s']:.3f}",
            f"{r['encode_time_s']:.3f}",
            f"{r['total_time_s']:.3f}",
            f"{r['tile_kb']:.2f}",
        ]
        row = " | ".join(f"{v:^{w}}" for v, w in zip(vals, widths))
        print(f"| {row} |")
    print(sep)
    print()

    # Quick ratios relative to capacity=2000 baseline
    base = results[0]
    print("Growth vs baseline (capacity=2,000):")
    for r in results[1:]:
        size_ratio = r["tile_kb"] / base["tile_kb"] if base["tile_kb"] else 0
        enc_ratio  = r["encode_time_s"] / base["encode_time_s"] if base["encode_time_s"] else 0
        print(
            f"  capacity={r['capacity']:,}: "
            f"size x{size_ratio:.2f}  "
            f"encode x{enc_ratio:.2f}"
        )
    print()


def print_triage_comparison(raw: list[dict], triaged: list[dict]) -> None:
    print("Triage comparison (raw vs triage, same capacity):")
    print(f"  {'Capacity':<12} {'Raw (KB)':>10} {'Triage (KB)':>12} {'Saved (KB)':>12} {'Reduction':>10}")
    print("  " + "-" * 58)
    for r, t in zip(raw, triaged):
        saved = r["tile_kb"] - t["tile_kb"]
        reduction = (saved / r["tile_kb"] * 100) if r["tile_kb"] else 0
        print(
            f"  {r['capacity']:<12,} {r['tile_kb']:>10.2f} {t['tile_kb']:>12.2f}"
            f" {saved:>12.2f} {reduction:>9.1f}%"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare IVT encode() at different feature capacities (no triage).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tile-dir",
        default=str(DEFAULT_TILE_DIR),
        help=f"Directory with parquet tile files (default: benchmark_output_parquet/parquet_tiles/)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help=f"Max features to load — must exceed the largest capacity (default: {DEFAULT_MAX_FEATURES:,})",
    )
    parser.add_argument(
        "--capacities",
        nargs="+",
        type=int,
        default=CAPACITIES,
        help=f"Capacities to test (default: {CAPACITIES})",
    )
    parser.add_argument(
        "--kld-threshold",
        type=float,
        default=0.1,
        help="Conditional-entropy threshold (bits) for string prefix triage (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "results" / "capacity_scaling.json"),
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    tile_dir = Path(args.tile_dir)
    if not tile_dir.exists():
        raise SystemExit(f"Tile directory not found: {tile_dir}")

    capacities = sorted(args.capacities)
    max_cap = max(capacities)
    if args.max_features <= max_cap:
        print(
            f"WARNING: --max-features ({args.max_features:,}) is not greater than the "
            f"largest capacity ({max_cap:,}). The top capacity bucket will not be fully "
            f"stressed. Re-run with --max-features {max_cap + 2000}."
        )

    # --- Load ---
    features = load_features(tile_dir, args.max_features)

    # --- Pick representative tile ---
    z, x, y = pick_tile_z5(features)
    print(f"Using tile z={z}, x={x}, y={y}\n")

    # --- Run each capacity: raw then triage ---
    raw_results = []
    triage_results = []
    for cap in capacities:
        print(f"Running capacity={cap:,} [raw]    ...", end=" ", flush=True)
        r = run_capacity(features, z, x, y, cap, use_triage=False)
        raw_results.append(r)
        print(f"encode={r['encode_time_s']:.3f}s  size={r['tile_kb']:.2f}KB")

        print(f"Running capacity={cap:,} [triage] ...", end=" ", flush=True)
        t = run_capacity(features, z, x, y, cap, use_triage=True, kld_threshold=args.kld_threshold)
        triage_results.append(t)
        print(f"encode={t['encode_time_s']:.3f}s  size={t['tile_kb']:.2f}KB")

    # --- Report ---
    print("\n--- RAW (no triage) ---")
    print_table(raw_results, (z, x, y))

    print("--- WITH TRIAGE ---")
    print_table(triage_results, (z, x, y))

    print_triage_comparison(raw_results, triage_results)

    # --- Save ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tile": {"z": z, "x": x, "y": y},
        "features_in_pool": len(features),
        "raw": raw_results,
        "triage": triage_results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
