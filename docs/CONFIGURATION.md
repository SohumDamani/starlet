# Starlet Configuration

Starlet can read a TOML configuration file so you do not have to keep passing
machine-level and project-level defaults on every command.

## How It Works

Starlet reads configuration in this order:

1. CLI arguments
2. Config file values
3. Built-in defaults

This means command-line arguments always win when both are provided.

## Config File Discovery

By default, Starlet looks for configuration files in the current working
directory in this order:

1. `starlet.toml`
2. `.starlet.toml`
3. `pyproject.toml` under `[tool.starlet]`

You can also point to a file explicitly:

```bash
starlet --config /path/to/starlet.toml build --input data.parquet --outdir datasets/mydata
```

## Recommended Layout

The simplest approach is to start from [starlet.toml.example](../starlet.toml.example)
and save your own copy as `starlet.toml`.

If you prefer `pyproject.toml`, use the same structure under `tool.starlet`, for
example:

```toml
[tool.starlet.global]
temp_dir = "/fast/tmp/starlet"

[tool.starlet.serve]
port = 9000
```

The config is split into sections:

- `[global]`: settings shared across commands such as `temp_dir`,
  `parallelism`, and `log_level`
- `[tile]`: defaults for `starlet tile`
- `[mvt]`: defaults for `starlet mvt`
- `[build]`: flags specific to `starlet build` itself
- `[serve]`: defaults for `starlet serve`

## Example

```toml
[global]
temp_dir = "/fast/tmp/starlet"
parallelism = 16
log_level = "INFO"

[tile]
partition_size = "256mb"
compression = "zstd"

[mvt]
zoom = 8
threshold = 50000
pmtiles = false

[serve]
host = "127.0.0.1"
port = 9000
cache_size = 512
```

With that file in place:

```bash
starlet build --input data.parquet --outdir datasets/mydata
starlet mvt --dir datasets/mydata
starlet serve --dir datasets
```

## Which Settings Still Belong On The Command Line

Some parameters are intentionally still command-line only because they describe
the specific job you are running:

- input paths such as `--input`
- dataset/output paths such as `--outdir` and `--dir`
- source schema details such as `--csv-x-col`, `--csv-y-col`, `--csv-wkt-col`,
  `--csv-x-index`, `--csv-y-index`, and `--csv-wkt-index`
- run-specific controls such as `--covering-bbox` and `--seed`

## Current Command Surface

The CLI now expects most persistent tuning values to come from config:

- `starlet tile` keeps run-specific inputs on the command line and loads
  partitioning, histogram, and compression defaults from config. Process-based
  parallelism always comes from `[global].parallelism`.
- `starlet mvt` keeps dataset path and optional `--zoom` on the command line and
  loads threshold, PMTiles export settings, buffer, extent, and feature
  reservoir settings from config.
- `starlet build` keeps source/output paths plus a few run-specific flags on the
  command line and loads the rest from config, forwarding MVT-related settings
  to the MVT stage.
- `starlet serve` keeps the dataset root on the command line and loads host,
  port, and cache settings from config. On-demand MVT generation reuses
  `[mvt].extent`.
