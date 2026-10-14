"""Process-wide Starlet configuration helpers."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


DEFAULT_CONFIG: dict[str, Any] = {
    "global": {
        "temp_dir": None,
        "parallelism": None,
        "log_level": "INFO",
    },
    "tile": {
        "partition_size": None,
        "sort": "zorder",
        "compression": "zstd",
        "intermediate_compression": "gzip",
        "sample_cap": 10_000,
        "csv_split_size": "32mb",
        "grid_size": 4096,
        "dtype": "float64",
        "sfc_bits": 16,
    },
    "mvt": {
        "zoom": 7,
        "threshold": 0,
        "pmtiles": False,
        "feature_capacity": 10_000,
        "tile_attributes": "all",
        "simplify_tolerance": "auto",
        "row_group_cache": 256,
        "python_pyramid": "pull",
        "batch_row_group_cache": 32,
        "push_max_tiles": 1024,
        "mvt_memory_budget": "auto",
        "extent": 4096,
        "buffer": 256,
        "pmtiles_compression": "gzip",
    },
    "build": {},
    "serve": {
        "host": "0.0.0.0",
        "port": 8765,
        "cache_size": 256,
    },
}

_loaded_config: dict[str, Any] = deepcopy(DEFAULT_CONFIG)
_loaded_config_path: Path | None = None
_loaded_config_initialized = False
_loaded_config_lock = threading.Lock()
_temp_dir: Path | None = None


def set_temp_dir(path: str | Path | None) -> Path | None:
    """Set the process-wide parent directory for temporary Starlet files."""
    global _temp_dir
    if path is None:
        _temp_dir = None
        return None
    _temp_dir = Path(path)
    _temp_dir.mkdir(parents=True, exist_ok=True)
    return _temp_dir


def get_temp_dir() -> Path | None:
    """Return the configured temp directory, if one was set."""
    return _temp_dir


def resolve_temp_dir(
    explicit: str | Path | None = None,
    default: str | Path | None = None,
) -> Path:
    """Return the temp parent directory for a step."""
    if explicit is not None:
        temp_dir = Path(explicit)
    elif _temp_dir is not None:
        temp_dir = _temp_dir
    elif default is not None:
        temp_dir = Path(default)
    else:
        temp_dir = Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def default_config() -> dict[str, Any]:
    """Return a deep copy of the built-in Starlet configuration defaults."""
    return deepcopy(DEFAULT_CONFIG)


def config_search_paths(cwd: str | Path | None = None) -> list[Path]:
    """Return config files checked by default, in priority order."""
    root = Path.cwd() if cwd is None else Path(cwd)
    return [
        root / "starlet.toml",
        root / ".starlet.toml",
        root / "pyproject.toml",
    ]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load Starlet config from a TOML file or return defaults."""
    config = default_config()
    resolved_path = _resolve_config_path(path)
    if resolved_path is None:
        return config

    raw = _read_config_file(resolved_path)
    if resolved_path.name == "pyproject.toml":
        raw = ((raw.get("tool") or {}).get("starlet") or {})
    _deep_update(config, raw)
    return config


def set_loaded_config(config: dict[str, Any], path: str | Path | None = None) -> None:
    """Install process-wide Starlet configuration for the current session."""
    global _loaded_config, _loaded_config_path, _loaded_config_initialized
    merged = default_config()
    _deep_update(merged, config)
    _loaded_config = merged
    _loaded_config_path = Path(path) if path is not None else None
    _loaded_config_initialized = True
    set_temp_dir(config_value("global", "temp_dir"))


def ensure_config_loaded(path: str | Path | None = None) -> None:
    """Load process-wide config once, if it has not already been installed.

    A malformed config file in the working directory must not make
    ``import starlet`` unusable, so parse errors fall back to defaults
    with a warning instead of propagating.
    """
    if _loaded_config_initialized:
        return
    with _loaded_config_lock:
        if _loaded_config_initialized:
            return
        resolved_path = _resolve_config_path(path)
        try:
            set_loaded_config(load_config(resolved_path), resolved_path)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to load Starlet config from %s (%s); using defaults",
                resolved_path,
                exc,
            )
            set_loaded_config(default_config(), None)


def get_loaded_config() -> dict[str, Any]:
    """Return the current process-wide Starlet configuration."""
    return deepcopy(_loaded_config)


def get_loaded_config_path() -> Path | None:
    """Return the path of the loaded config file, if any."""
    return _loaded_config_path


def _reset_loaded_config_for_tests() -> None:
    """Reset process-wide config state for isolated tests."""
    global _loaded_config, _loaded_config_path, _loaded_config_initialized
    _loaded_config = deepcopy(DEFAULT_CONFIG)
    _loaded_config_path = None
    _loaded_config_initialized = False
    set_temp_dir(None)


def config_value(section: str, key: str, fallback: Any = None) -> Any:
    """Return a value from the loaded config."""
    return (_loaded_config.get(section) or {}).get(key, fallback)


def resolve_command_value(
    command: str,
    key: str,
    explicit: Any,
    *,
    fallback_sections: tuple[str, ...] = (),
) -> Any:
    """Resolve one option using CLI, then loaded config sections."""
    if explicit is not None:
        return explicit

    for section in (command, *fallback_sections, "global"):
        section_values = _loaded_config.get(section) or {}
        if key in section_values and section_values[key] is not None:
            return section_values[key]
    return None


def command_parallelism(
    command: str,
    explicit: int | None = None,
    *,
    fallback_sections: tuple[str, ...] = (),
) -> int | None:
    """Resolve the shared configured parallelism value."""
    _ = command
    _ = fallback_sections
    value = resolve_command_value(
        "global",
        "parallelism",
        explicit,
    )
    if value is None:
        return None
    return int(value)


def parse_size_value(value: str | int | None) -> int | None:
    """Parse a size like ``32mb`` or pass through integer byte counts."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.isdigit():
            return int(s)
        suffixes = {
            "kb": 1024,
            "mb": 1024 ** 2,
            "gb": 1024 ** 3,
            "tb": 1024 ** 4,
        }
        for suffix, mul in suffixes.items():
            if s.endswith(suffix):
                num = s[: -len(suffix)].strip()
                return int(float(num) * mul)
    raise ValueError(f"Invalid size value: {value!r}")


def _resolve_config_path(path: str | Path | None) -> Path | None:
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Starlet config file not found: {resolved}")
        return resolved

    for candidate in config_search_paths():
        if candidate.exists():
            return candidate
    return None


def _read_config_file(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid Starlet config file: {path}")
    return _expand_env(loaded)


def _expand_env(value: Any) -> Any:
    """Expand ``~`` and environment variables in strings from a config file.

    Applied only to values parsed out of the TOML file, not to values
    passed on the command line (the shell already expands those).
    """
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
