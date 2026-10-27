import pytest

from lygos.paths import (
    get_cache_path,
    get_data_path,
    get_repository_path,
    get_visuals_path,
)


def test_repository_runtime_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("LYGOS_PATH", str(tmp_path))

    assert get_repository_path() == tmp_path
    assert get_data_path() == tmp_path / "data"
    assert get_visuals_path() == tmp_path / "visuals"


def test_repository_path_is_required(monkeypatch):
    monkeypatch.delenv("LYGOS_PATH", raising=False)

    with pytest.raises(EnvironmentError, match="LYGOS_PATH"):
        get_repository_path()


def test_cache_path_uses_configured_repository_data(monkeypatch, tmp_path):
    monkeypatch.setenv("LYGOS_PATH", str(tmp_path))

    path = get_cache_path("roman")

    assert path == tmp_path / "data" / "cache" / "roman"
    assert path.is_dir()


def test_cache_path_uses_user_directory_without_repository(monkeypatch, tmp_path):
    monkeypatch.delenv("LYGOS_PATH", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    path = get_cache_path("roman")

    assert path == tmp_path / ".lygos" / "cache" / "roman"
    assert path.is_dir()


def test_cache_path_uses_explicit_directory(tmp_path):
    cache_dir = tmp_path / "custom-cache"

    assert get_cache_path("roman", cache_dir=cache_dir) == cache_dir
    assert cache_dir.is_dir()
