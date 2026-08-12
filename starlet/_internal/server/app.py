"""Flask application factory for the starlet tile server."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional

import json
import logging
import os
import re

import gzip as _gzip

from flask import Flask, Response, render_template, send_from_directory, request
from flask_cors import CORS

from starlet._internal.config import config_value, ensure_config_loaded

from .tiler.tiler import VectorTiler
from .download_service import DatasetFeatureService

logger = logging.getLogger(__name__)


def create_app(
    data_dir: str,
    cache_size: int | None = None,
    extent: int | None = None,
    buffer: int | None = None,
    on_the_fly_implementation: str | None = None,
    log_level: Optional[str] = None,
) -> Flask:
    """Create and configure a Flask tile server application.

    Parameters
    ----------
    data_dir : str
        Root directory containing dataset subdirectories.
    cache_size : int, optional
        Number of tiles to keep in the in-memory LRU cache. When omitted,
        Starlet uses ``serve.cache_size`` from config, or the built-in default.
    extent : int, optional
        Vector tile extent used for on-demand MVT generation.
    buffer : int, optional
        Vector tile buffer in extent units used for on-demand generation.
    log_level : str, optional
        Logging level (e.g. "INFO", "DEBUG"). Defaults to ``LOG_LEVEL`` env
        var or ``"INFO"``.

    Returns
    -------
    Flask
        Configured Flask application ready to be served.
    """
    ensure_config_loaded()
    if on_the_fly_implementation not in {None, "new"}:
        raise ValueError("on_the_fly_implementation must be 'new'")

    resolved_cache_size = cache_size
    if resolved_cache_size is None:
        resolved_cache_size = int(config_value("serve", "cache_size"))
    resolved_extent = extent
    if resolved_extent is None:
        resolved_extent = int(config_value("mvt", "extent"))
    resolved_buffer = buffer
    if resolved_buffer is None:
        resolved_buffer = int(config_value("mvt", "buffer"))
    level = (
        log_level
        or os.environ.get("LOG_LEVEL")
        or str(config_value("global", "log_level"))
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    template_dir = str(Path(__file__).parent / "templates")
    app = Flask(__name__, template_folder=template_dir)
    CORS(app, resources={r"/*": {"origins": "*"}})

    data_root = Path(data_dir)
    tiler_cache: dict[str, VectorTiler] = {}
    feature_service = DatasetFeatureService(data_root)

    def get_tiler(dataset: str) -> VectorTiler:
        if dataset not in tiler_cache:
            tiler_cache[dataset] = VectorTiler(
                str(data_root / dataset),
                memory_cache_size=resolved_cache_size,
                extent=resolved_extent,
                buffer=resolved_buffer,
            )
        return tiler_cache[dataset]

    @app.get("/<dataset>/<int:z>/<int:x>/<int:y>.mvt")
    def serve_tile(dataset, z, x, y):
        t0 = perf_counter()
        tiler = get_tiler(dataset)
        tile_info: dict[str, object] = {}
        try:
            data = tiler.get_tile(z, x, y, output=tile_info)
        except FileNotFoundError as e:
            logger.warning("[TileRequest] %s/%d/%d/%d not found: %s", dataset, z, x, y, e)
            return {"error": str(e)}, 404
        except Exception as e:
            logger.exception("[TileRequest] %s/%d/%d/%d failed", dataset, z, x, y)
            return {"error": f"Internal error: {str(e)}"}, 500
        elapsed_s = perf_counter() - t0
        generation = tile_info.get("generation", "unknown")

        client_accepts_gzip = "gzip" in request.headers.get("Accept-Encoding", "")
        tile_is_gzip = tile_info.get("gzip_compressed", False)

        headers = {}
        if tile_is_gzip and client_accepts_gzip:
            headers["Content-Encoding"] = "gzip"
        elif tile_is_gzip and not client_accepts_gzip:
            data = _gzip.decompress(data)

        logger.info(
            "[TileRequest] %s/%d/%d/%d bytes=%d method=%s gzip=%s time=%.3fs",
            dataset,
            z,
            x,
            y,
            len(data),
            generation,
            tile_is_gzip,
            elapsed_s,
        )
        return Response(data, mimetype="application/vnd.mapbox-vector-tile", headers=headers)

    @app.get("/api/datasets")
    def list_datasets():
        datasets = []
        if data_root.exists():
            datasets = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
        return json.dumps({"datasets": datasets})

    @app.get("/")
    def index():
        logger.info("Serving index page")
        return render_template("index.html")

    @app.route("/<path:filename>")
    def serve_file(filename):
        server_dir = Path(__file__).parent
        file_path = server_dir / filename
        if file_path.exists() and file_path.is_file():
            return send_from_directory(str(server_dir), filename)
        return "File not found", 404

    @app.get("/datasets/<dataset>/features.<format>")
    def download_features(dataset, format):
        try:
            mbr_string = request.args.get('mbr', default=None)
            feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
            mime_type = feature_service.get_mime_type(format)
            if mbr_string:
                filename = f"{dataset}_{mbr_string.replace(',', '_')}.{format}"
            else:
                filename = f"{dataset}_full.{format}"
            return Response(
                feature_stream,
                mimetype=mime_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500

    @app.post("/datasets/<dataset>/features.<format>")
    def download_features_with_geometry(dataset, format):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            geojson_payload = request.get_json()
            mbr_string = request.args.get("mbr", default=None)
            if geojson_payload:
                geometry = geojson_payload.get("geometry")
                if not geometry:
                    return {"error": "Invalid GeoJSON payload: 'geometry' field is required"}, 400
                feature_stream = feature_service.get_features_stream(dataset, format, geometry=geometry)
            else:
                feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
            mime_type = feature_service.get_mime_type(format)
            filename = f"{dataset}_filtered.{format}" if geojson_payload else f"{dataset}_mbr.{format}"
            return Response(
                feature_stream,
                mimetype=mime_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500

    @app.get("/api/datasets/<dataset>/stats")
    def get_dataset_stats(dataset):
        stats_path = data_root / dataset / "stats" / "attributes.json"
        if not stats_path.exists():
            return {"error": "Stats not found for dataset"}, 404
        try:
            with open(stats_path, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load stats: {str(e)}"}, 500

    @app.get("/datasets.json")
    def search_datasets():
        query = request.args.get("q", default=None)
        datasets = []
        if data_root.exists():
            for d in data_root.iterdir():
                if d.is_dir():
                    dataset_metadata = {
                        "id": d.name,
                        "name": d.name.replace("_", " ").title(),
                        "size": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()),
                    }
                    if query is None or query.lower() in d.name.lower():
                        datasets.append(dataset_metadata)
        return json.dumps({"datasets": datasets}, indent=2)

    @app.get("/datasets/<dataset>.json")
    def get_dataset_metadata(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            metadata = {
                "id": dataset,
                "name": dataset.replace("_", " ").title(),
                "size": sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file()),
                "file_count": sum(1 for f in dataset_path.rglob("*") if f.is_file()),
            }
            return json.dumps(metadata, indent=2)
        except Exception as e:
            return {"error": f"Failed to retrieve metadata: {str(e)}"}, 500

    @app.get("/datasets/<dataset>.html")
    def visualize_dataset(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return "<h1>Dataset not found</h1>", 404
        from flask import redirect
        return redirect(f"/view_mvt.html?dataset={dataset}")

    @app.get("/datasets/<dataset>/features/sample.json")
    def get_sample_non_geometry_attributes(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            mbr_string = request.args.get("mbr", default=None)
            if not mbr_string:
                return {"error": "MBR query parameter is required"}, 400
            sample_record = feature_service.get_sample_record(dataset, mbr_string, include_geometry=False)
            if not sample_record:
                return {"error": "No matching record found"}, 404
            return json.dumps(sample_record, indent=2)
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500

    @app.get("/datasets/<dataset>/features/sample.geojson")
    def get_sample_with_geometry(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            mbr_string = request.args.get("mbr", default=None)
            if not mbr_string:
                return {"error": "MBR query parameter is required"}, 400
            sample_record = feature_service.get_sample_record(dataset, mbr_string, include_geometry=True)
            if not sample_record:
                return {"error": "No matching record found"}, 404
            return json.dumps(sample_record, indent=2)
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500



    return app
