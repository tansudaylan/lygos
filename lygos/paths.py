"""Repository-local runtime paths for Lygos."""

import os
from pathlib import Path

from tdpy.paths import RepositoryPaths


PATH_ENV_VAR = "LYGOS_PATH"
_REPOSITORY_PATHS = RepositoryPaths(PATH_ENV_VAR)

get_repository_path = _REPOSITORY_PATHS.get_repository_path
get_data_path = _REPOSITORY_PATHS.get_data_path
get_visuals_path = _REPOSITORY_PATHS.get_visuals_path


def get_cache_path(mission: str) -> Path:
    """Return the download cache for one mission, under ``$LYGOS_PATH/data`` or ``~/.lygos``."""
    root = get_data_path() if os.environ.get(PATH_ENV_VAR) else Path.home() / ".lygos"
    path = Path(root) / "cache" / mission
    path.mkdir(parents=True, exist_ok=True)
    return path