"""Repository-local runtime paths for Lygos."""

import os
from pathlib import Path


PATH_ENV_VAR = "LYGOS_PATH"


def get_repository_path() -> Path:
    path_value = os.environ.get(PATH_ENV_VAR)
    if not path_value or not path_value.strip():
        raise EnvironmentError(f"{PATH_ENV_VAR} is required and cannot be empty.")
    return Path(path_value).expanduser().resolve()


def get_data_path() -> Path:
    return get_repository_path() / "data"


def get_visuals_path() -> Path:
    return get_repository_path() / "visuals"