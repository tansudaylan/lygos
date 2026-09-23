"""Lygos package.

Lygos provides the supported image-based photometry and light-curve extraction
workflow for TESS data. The public API is intentionally kept compact so the
package boundary remains clear and the scientific workflow is easier to inspect.
"""

from .main import init
from .paths import get_data_path, get_repository_path, get_visuals_path

__all__ = ["get_data_path", "get_repository_path", "get_visuals_path", "init"]
