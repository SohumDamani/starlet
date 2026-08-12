import logging
from pathlib import Path
from time import perf_counter
from typing import Any, MutableMapping
import gzip

from starlet._internal.config import config_value
from starlet._internal.mvt.tile_format import TILE_GZIP
from starlet._internal.pmtiles.paths import discover_pmtiles_path

from .tile_cache import TileCache
logger = logging.getLogger(__name__)


TileInfo = MutableMapping[str, Any]


def _update_output(output: TileInfo | None, **values: Any) -> None:
    if output is not None:
        output.update(values)


class VectorTiler:
    """On-demand MVT tile server with a three-tier lookup.

    For each tile request the lookup order is:
      1. **Memory** — LRU cache (``TileCache``, default 256 entries)
      2. **PMTiles** — pre-generated ``<dataset>.pmtiles`` archive
      3. **Disk** — pre-generated ``.mvt`` files under ``<dataset>/mvt/z/x/y.mvt``
      4. **Generate** — reads intersecting GeoParquet tiles, clips/transforms
         geometries, and encodes a fresh MVT on the fly

    Only on-the-fly generated tiles are promoted into the memory cache.
    """

    def __init__(
        self,
        dataset_root: str,
        memory_cache_size: int | None = None,
        extent: int | None = None,
        buffer: int | None = None,
        feature_capacity: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.mvt_dir = self.dataset_root / "mvt"
        self.pmtiles_path = discover_pmtiles_path(self.dataset_root)
        self.extent = int(extent if extent is not None else config_value("mvt", "extent"))
        self.buffer = int(buffer if buffer is not None else config_value("mvt", "buffer"))
        self.feature_capacity = int(
            feature_capacity if feature_capacity is not None else config_value("mvt", "feature_capacity")
        )
        self._pmtiles_file = None
        self._pmtiles_reader = None
        self._pmtiles_header = None

        cache_size = memory_cache_size if memory_cache_size is not None else config_value("serve", "cache_size")
        self.cache = TileCache(int(cache_size))

    def tile_path(self, z: int, x: int, y: int) -> Path:
        gz = self.mvt_dir / str(z) / str(x) / f"{y}.mvt.gz"
        if gz.exists():
            return gz
        return self.mvt_dir / str(z) / str(x) / f"{y}.mvt"

    def _pmtiles_tile(self, z: int, x: int, y: int) -> bytes | None:
        if self.pmtiles_path is None or not self.pmtiles_path.exists():
            return None
        if self._pmtiles_reader is None:
            from pmtiles.reader import MmapSource, Reader

            self._pmtiles_file = self.pmtiles_path.open("rb")
            self._pmtiles_reader = Reader(MmapSource(self._pmtiles_file))
            self._pmtiles_header = self._pmtiles_reader.header()

        tile_bytes = self._pmtiles_reader.get(z, x, y)
        if tile_bytes is None:
            return None
        return _decode_pmtiles_tile(tile_bytes, self._pmtiles_header)

    def get_tile(self, z: int, x: int, y: int, output: TileInfo | None = None) -> bytes:
        t0 = perf_counter()
        key = (z, x, y)

        cached = self.cache.get(key)
        if cached is not None:
            logger.debug("[Cache] HIT memory z=%d x=%d y=%d", z, x, y)
            _update_output(
                output,
                generation="mem-cache",
                elapsed_ms=(perf_counter() - t0) * 1000,
                gzip_compressed=TILE_GZIP,
            )
            return cached

        pmtiles_t0 = perf_counter()
        pmtiles_data = self._pmtiles_tile(z, x, y)
        if pmtiles_data is not None:
            elapsed_ms = (perf_counter() - pmtiles_t0) * 1000
            logger.debug("[Cache] HIT pmtiles z=%d x=%d y=%d elapsed=%.1fms", z, x, y, elapsed_ms)
            _update_output(
                output,
                source="pmtiles",
                generation="read_from_pmtiles",
                path=str(self.pmtiles_path),
                elapsed_ms=elapsed_ms,
            )
            return pmtiles_data

        path = self.tile_path(z, x, y)

        if path.exists():
            t0 = perf_counter()
            data = path.read_bytes()
            elapsed_ms = (perf_counter() - t0) * 1000
            logger.debug("[Cache] HIT disk z=%d x=%d y=%d elapsed=%.1fms", z, x, y, elapsed_ms)
            _update_output(
                output,
                source="disk",
                generation="read_from_disk",
                path=str(path),
                elapsed_ms=elapsed_ms,
                gzip_compressed=path.suffix == ".gz",
            )
            return data

        logger.info("[Cache] MISS z=%d x=%d y=%d — generating (memory cache only)", z, x, y)
        from starlet._internal.mvt.mvt_generator import generate_single_mvt_tile

        t0 = perf_counter()
        tile_bytes = generate_single_mvt_tile(
            str(self.dataset_root),
            (z, x, y),
            feature_capacity=self.feature_capacity,
            extent=self.extent,
            buffer=self.buffer,
        )
        is_gzip = TILE_GZIP
        if is_gzip:
            tile_bytes = gzip.compress(tile_bytes, compresslevel=6)
        _update_output(
            output,
            source="generated",
            generation="generated",
            feature_capacity=self.feature_capacity,
            extent=self.extent,
            buffer=self.buffer,
            elapsed_ms=(perf_counter() - t0) * 1000,
            gzip_compressed=is_gzip,
        )

        # On-demand tiles are cached in memory only; we deliberately do NOT
        # persist them to disk. Writing every generated tile would incrementally
        # materialise the full pyramid on disk over time — exactly what lazy
        # serving exists to avoid. Callers who want a durable on-disk pyramid
        # should pre-generate it explicitly (`starlet mvt` / `generate_mvt`,
        # e.g. with threshold=0 to materialise every non-empty tile).
        self.cache.put(key, tile_bytes)
        return tile_bytes

def _decode_pmtiles_tile(tile_bytes: bytes, header: dict[str, Any] | None) -> bytes:
    if not header:
        return tile_bytes

    from pmtiles.tile import Compression

    compression = header.get("tile_compression", Compression.UNKNOWN)
    if compression in {Compression.UNKNOWN, Compression.NONE}:
        return tile_bytes
    if compression == Compression.GZIP:
        return gzip.decompress(tile_bytes)

    raise ValueError(f"Unsupported PMTiles compression for server reads: {compression}")
