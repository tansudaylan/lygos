"""Lygos package.

Lygos provides the supported image-based photometry and light-curve extraction
workflow for TESS data. The public API is intentionally kept compact so the
package boundary remains clear and the scientific workflow is easier to inspect.
"""

from .main import init

__all__ = ["init"]
