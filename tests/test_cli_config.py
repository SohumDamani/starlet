from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from starlet._cli import main


def test_tile_command_uses_config_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "starlet.toml"
    config_path.write_text(
        """
[global]
temp_dir = "/tmp/starlet"
parallelism = 6
log_level = "DEBUG"

[tile]
partition_size = "256mb"
sort = "hilbert"
compression = "gzip"
intermediate_compression = "none"
sample_cap = 123
csv_split_size = "64mb"
grid_size = 1024
dtype = "float32"
sfc_bits = 20
""".strip()
    )

    captured = {}

    def fake_tile(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            num_files=1,
            total_rows=2,
            outdir=kwargs["outdir"],
            histogram_path="hist.npy",
        )

    monkeypatch.setattr("starlet.tile", fake_tile)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config",
            str(config_path),
            "tile",
            "--input",
            "input.geojson",
            "--outdir",
            "dataset",
        ],
    )

    assert result.exit_code == 0
    assert captured["partition_size"] == 256 * 1024 * 1024
    assert captured["sort"] == "hilbert"
    assert captured["compression"] == "gzip"
    assert captured["intermediate_compression"] == "none"
    assert captured["sample_cap"] == 123
    assert captured["csv_split_size"] == 64 * 1024 * 1024
    assert captured["parallelism"] == 6
    assert captured["grid_size"] == 1024
    assert captured["histogram_dtype"] == "float32"
    assert captured["sfc_bits"] == 20
    assert captured["temp_dir"] == "/tmp/starlet"


def test_mvt_command_prefers_cli_zoom_and_uses_config_threshold(tmp_path, monkeypatch):
    config_path = tmp_path / "starlet.toml"
    config_path.write_text(
        """
[global]
parallelism = 4

[mvt]
zoom = 5
threshold = 777
pmtiles = true
feature_capacity = 333
mvt_memory_budget = "8gb"
extent = 2048
buffer = 64
""".strip()
    )

    captured = {}

    def fake_generate_mvt(*, tile_dir, zoom, threshold, pmtiles, pmtiles_compression, outdir, parallelism, temp_dir, feature_capacity, mvt_memory_budget, extent, buffer, tile_attributes=None):
        captured.update(
            tile_dir=tile_dir,
            zoom=zoom,
            threshold=threshold,
            pmtiles=pmtiles,
            pmtiles_compression=pmtiles_compression,
            outdir=outdir,
            parallelism=parallelism,
            temp_dir=temp_dir,
            feature_capacity=feature_capacity,
            mvt_memory_budget=mvt_memory_budget,
            extent=extent,
            buffer=buffer,
        )
        return SimpleNamespace(tile_count=1, outdir=outdir or "mvt", zoom_levels=[0, 1], pmtiles_path="dataset/tiles.pmtiles")

    monkeypatch.setattr("starlet.generate_mvt", fake_generate_mvt)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config",
            str(config_path),
            "mvt",
            "--dir",
            "dataset",
            "--zoom",
            "9",
        ],
    )

    assert result.exit_code == 0
    assert captured["zoom"] == 9
    assert captured["threshold"] == 777.0
    assert captured["pmtiles"] is True
    assert captured["pmtiles_compression"] == "gzip"
    assert captured["parallelism"] == 4
    assert captured["feature_capacity"] == 333
    assert captured["mvt_memory_budget"] == "8gb"
    assert captured["extent"] == 2048
    assert captured["buffer"] == 64


def test_build_command_prefers_mvt_pmtiles_config(tmp_path, monkeypatch):
    config_path = tmp_path / "starlet.toml"
    config_path.write_text(
        """
[mvt]
pmtiles = true
pmtiles_compression = "brotli"
""".strip()
    )

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return (
            SimpleNamespace(num_files=1, total_rows=2),
            SimpleNamespace(tile_count=3, zoom_levels=[0], pmtiles_path="dataset/tiles.pmtiles"),
            "dataset/tiles.pmtiles",
        )

    monkeypatch.setattr("starlet.build", fake_build)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config",
            str(config_path),
            "build",
            "--input",
            "input.geojson",
            "--outdir",
            "dataset",
        ],
    )

    assert result.exit_code == 0
    assert captured["pmtiles"] is True
    assert captured["pmtiles_compression"] == "brotli"
    assert captured["intermediate_compression"] == "gzip"


def test_serve_command_uses_config_host_port_and_cache(tmp_path, monkeypatch):
    config_path = tmp_path / "starlet.toml"
    config_path.write_text(
        """
[serve]
host = "127.0.0.1"
port = 9001
cache_size = 99
""".strip()
    )

    captured = {}

    class FakeApp:
        def run(self, host, port, debug, use_reloader, threaded):
            captured.update(
                host=host,
                port=port,
                debug=debug,
                use_reloader=use_reloader,
                threaded=threaded,
            )

    def fake_create_app(*, data_dir, cache_size, tile_attributes=None):
        captured.update(data_dir=data_dir, cache_size=cache_size)
        return FakeApp()

    monkeypatch.setattr("starlet.create_app", fake_create_app)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config",
            str(config_path),
            "serve",
            "--dir",
            "datasets",
        ],
    )

    assert result.exit_code == 0
    assert captured["data_dir"] == "datasets"
    assert captured["cache_size"] == 99
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9001
