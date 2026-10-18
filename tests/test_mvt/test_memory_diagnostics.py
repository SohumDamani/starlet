"""Memory diagnostics must not make MVT generation Unix-only."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from starlet._internal.mvt import mvt_generator


def test_generator_imports_without_resource():
    script = """
import builtins
original_import = builtins.__import__
def without_resource(name, *args, **kwargs):
    if name == 'resource':
        raise ImportError('resource is unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_resource
from starlet._internal.mvt import mvt_generator
assert mvt_generator.resource is None
import sys
if sys.platform != 'win32':
    assert mvt_generator._process_rss_mib()[1] is None
"""
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows process memory API')
def test_windows_current_and_peak_memory():
    current, peak = mvt_generator._process_rss_mib()
    assert current > 0
    assert peak >= current


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows process memory API')
def test_windows_memory_query_failure_is_nonfatal(monkeypatch):
    import ctypes

    library = SimpleNamespace(GetCurrentProcess=lambda: 1,
                              GetProcessMemoryInfo=lambda *args: 0)
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *args, **kwargs: library)
    assert mvt_generator._process_rss_mib() == (None, None)


@pytest.mark.parametrize('platform,raw_peak', [('linux', 2048), ('darwin', 2 * 1024**2)])
def test_unix_peak_memory_units_unchanged(monkeypatch, platform, raw_peak):
    monkeypatch.setattr(mvt_generator.sys, 'platform', platform)
    monkeypatch.setattr(mvt_generator, 'resource', SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=raw_peak)))
    assert mvt_generator._process_rss_mib()[1] == 2.0
