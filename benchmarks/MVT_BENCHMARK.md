# MVT Benchmark Tools

Two scripts for measuring and comparing MVT generation performance. Both work on Windows and Linux without modification.

---

## Prerequisites

- Python virtualenv with starlet installed (`.venv/` in the project root)
- A dataset already tiled by `starlet tile` — the output directory must contain `parquet_tiles/` and `histograms/`
- The pre-built `benchmark_output_parquet/` directory in the project root works out of the box

Activate your environment before running anything:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

---

## Tool 1 — `profile_mvt_stages.py`

Measures where time goes inside MVT generation. Two modes:

**`stages` mode (default)** — times each step in the geometry-streaming loop at row-group level. Stages match the v0.3.1 vectorised pipeline:

| Stage | What it measures |
|---|---|
| `parquet_read` | Reading a row group from disk |
| `wkb_decode` | `shapely.from_wkb(array)` — whole row group at once |
| `make_valid` | `shapely.make_valid(array)` — whole row group at once |
| `reproject` | `shapely.transform(array, fn)` — whole row group at once |
| `attr_extract` | Extracting non-geometry columns to Python lists |
| `tile_assign` | Per-geometry: tile-range lookup + bucket insert |

**`cprofile` mode** — wraps the actual `starlet.generate_mvt()` call with Python's built-in profiler. No maintenance needed — it measures the real code regardless of internal changes.

### Usage

```bash
# Default: stage breakdown on the pre-built benchmark dataset
python benchmarks/profile_mvt_stages.py

# Different dataset
python benchmarks/profile_mvt_stages.py --dataset-dir path/to/your/dataset

# Always-accurate mode (use after any upstream code change)
python benchmarks/profile_mvt_stages.py --mode cprofile

# Higher zoom (more tiles, slower)
python benchmarks/profile_mvt_stages.py --zoom 7

# Save result to a specific path (for later comparison)
python benchmarks/profile_mvt_stages.py --output benchmarks/results/my_baseline.json
```

### Output

Prints a table to stdout and saves a JSON file to `benchmarks/results/mvt_stage_profile.json`:

```
Stage              |   Time (s) |  % of loop |    geoms/s
--------------------------------------------------------
parquet_read       |      1.234 |       1.4% |    149,789
wkb_decode         |      2.345 |       2.6% |     78,812
make_valid         |     15.678 |      17.4% |     11,803
reproject          |     13.234 |      14.7% |     13,975
attr_extract       |      3.456 |       3.8% |     53,712
tile_assign        |     54.123 |      60.1% |      3,421
```

The stage with the highest `%` is where optimisation effort pays off.

---

## Tool 2 — `run_e2e_once.py`

Runs the full `starlet.build()` pipeline once with a controlled cache warmup, then writes a timing record. Use this for clean before/after comparisons (e.g. when testing a code change).

The warmup reads the entire input file before the timer starts, so OS disk-cache state does not affect the result.

### Usage

```bash
# Label this run and run index, e.g. before applying a change
python benchmarks/run_e2e_once.py --tag pre --run-id 1 --input benchmark_data/asia_postal_codes.parquet

# Apply your code change, then run again
python benchmarks/run_e2e_once.py --tag post --run-id 1 --input benchmark_data/asia_postal_codes.parquet

# Run 2 reps of each, interleaved, to cancel drift
python benchmarks/run_e2e_once.py --tag pre  --run-id 2 --input benchmark_data/asia_postal_codes.parquet
python benchmarks/run_e2e_once.py --tag post --run-id 2 --input benchmark_data/asia_postal_codes.parquet
```

Each run appends one JSON record to `benchmarks/results/e2e_ab_runs.jsonl`. Output directory is deleted automatically after each run unless `--keep-output` is set.

### Output

```json
{
  "tag": "post",
  "run_id": "1",
  "warmup_s": 2.456,
  "total_time_s": 128.01,
  "parquet_tiles": 15,
  "total_rows": 184746,
  "mvt_tiles": 170,
  "zoom_levels": [0, 1, 2, 3, 4, 5],
  "max_parallel_files": 8,
  "version": "0.3.1"
}
```

---

## Tool 3 — `compare_profiles.py`

Diffs two saved `profile_mvt_stages.py` JSON files and flags regressions.

```bash
python benchmarks/compare_profiles.py \
    --baseline benchmarks/results/my_baseline.json \
    --current  benchmarks/results/my_current.json
```

Exits with code 1 if any stage got more than `--threshold-pct` (default 10%) slower.

---

## Typical workflow: measuring the impact of a code change

```bash
# 1. Save a baseline profile on the current code
python benchmarks/profile_mvt_stages.py --output benchmarks/results/baseline.json

# 2. Apply your change

# 3. Profile again
python benchmarks/profile_mvt_stages.py --output benchmarks/results/current.json

# 4. Compare
python benchmarks/compare_profiles.py \
    --baseline benchmarks/results/baseline.json \
    --current  benchmarks/results/current.json

# 5. (Optional) Confirm with a clean full-pipeline A/B
python benchmarks/run_e2e_once.py --tag pre  --run-id 1 --input benchmark_data/asia_postal_codes.parquet
python benchmarks/run_e2e_once.py --tag post --run-id 1 --input benchmark_data/asia_postal_codes.parquet
```

---

## Keeping the stage profiler current

`--mode stages` re-implements the geometry-streaming loop manually to isolate sub-step timings. If the internal pipeline architecture changes significantly, the `_profile_stages()` function in `profile_mvt_stages.py` will need updating to match.

`--mode cprofile` profiles the real `starlet.generate_mvt()` call directly and requires no maintenance. Use it to get an accurate picture after an upstream update before deciding whether `_profile_stages()` needs updating.

---

## Notes on parallelism (`--max-parallel-files`)

`run_e2e_once.py` defaults to `--max-parallel-files 8` (much lower than starlet's CLI default of 64). On machines with limited virtual memory or a small pagefile, spawning 64 worker processes simultaneously can exhaust committed memory. Lower this further if you see a `BrokenProcessPool` crash.
