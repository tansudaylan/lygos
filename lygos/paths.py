"""Repository-local runtime paths for Lygos."""

import os
from pathlib import Path

from tdpy.paths import RepositoryPaths


PATH_ENV_VAR = "LYGOS_PATH"
_REPOSITORY_PATHS = RepositoryPaths(PATH_ENV_VAR)

get_repository_path = _REPOSITORY_PATHS.get_repository_path
get_data_path = _REPOSITORY_PATHS.get_data_path
get_visuals_path = _REPOSITORY_PATHS.get_visuals_path


def get_cache_path(
    mission: str, *, cache_dir: str | os.PathLike[str] | None = None
) -> Path:
    """Return a created custom cache directory or the default mission cache."""
    if cache_dir:
        path = Path(cache_dir)
    else:
        if os.environ.get(PATH_ENV_VAR):
            root = get_data_path()
        else:
            root = Path.home() / ".lygos"
        path = Path(root) / "cache" / mission
    path.mkdir(parents=True, exist_ok=True)
    return path
