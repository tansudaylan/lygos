"""Lygos package.

Lygos retrieves, visualizes, and processes TESS and Roman images. TESS cutouts come from
TESSCut and full-frame images from the public MAST bucket. Roman images come from the public
OpenUniverse 2024 simulation. Every retrieval returns an :class:`ImageStack`, which supports
background subtraction, difference imaging, aperture photometry, and animation. The
image-based TESS photometry pipeline remains available as :func:`init`.
"""

from .imagestack import ImageStack
from .main import init
from .paths import get_cache_path, get_data_path, get_repository_path, get_visuals_path
from .roman import find_roman_images, get_roman_cutout, list_roman_images, load_roman_pointings, read_roman_image
from .tess import (get_tess_cutout, get_tess_ffi, get_tess_ffi_cutout, list_tess_ffis, read_tess_cutout,
                   resolve_coordinate, tess_sectors)
from .visualization import animate_stack, image_normalization, plot_image, plot_light_curve, plot_mosaic
from .variability import compute_variability_products, plot_variability_summary

__all__ = [
    "ImageStack",
    "animate_stack",
    "compute_variability_products",
    "find_roman_images",
    "get_cache_path",
    "get_data_path",
    "get_repository_path",
    "get_roman_cutout",
    "get_tess_cutout",
    "get_tess_ffi",
    "get_tess_ffi_cutout",
    "get_visuals_path",
    "image_normalization",
    "init",
    "list_roman_images",
    "list_tess_ffis",
    "load_roman_pointings",
    "plot_image",
    "plot_light_curve",
    "plot_mosaic",
    "plot_variability_summary",
    "read_roman_image",
    "read_tess_cutout",
    "resolve_coordinate",
    "tess_sectors",
]
