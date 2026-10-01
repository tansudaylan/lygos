"""Visualize image stacks: single images, frame mosaics, light curves, and time-series animations.

All functions accept an :class:`~lygos.imagestack.ImageStack` from any mission. Images use an
arcsinh, logarithmic, square-root, or linear stretch between robust percentiles shared by all
frames, so brightness changes between frames are real and not an artifact of rescaling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import (AsinhStretch, ImageNormalize, LinearStretch, LogStretch, PercentileInterval,
                                   SqrtStretch)
from tdpy.plotting import figure_to_frame, save_figure, write_animation

from .imagestack import ImageStack

STRETCHES = {"asinh": AsinhStretch, "log": LogStretch, "sqrt": SqrtStretch, "linear": LinearStretch}


def image_normalization(images, stretch: str = "asinh", percentile: float = 99.5, symmetric: bool = False):
    """Return one Matplotlib normalization shared by all ``images``.

    With ``symmetric`` the limits are centered on zero, which suits difference images.
    """
    if stretch not in STRETCHES:
        raise ValueError(f"stretch must be one of {tuple(STRETCHES)}")
    values = np.asarray(images, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("images contain no finite pixels")
    if symmetric:
        limit = float(np.percentile(np.abs(values), percentile))
        return ImageNormalize(vmin=-limit, vmax=limit, stretch=LinearStretch())
    vmin, vmax = PercentileInterval(percentile).get_limits(values)
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=STRETCHES[stretch]())


def _time_label(stack: ImageStack) -> str:
    return f"Time [{stack.time_format}, day]" if stack.time_format else "Time [day]"


def _draw_image(axis, image, norm, stack, cmap, aperture=None, markers=None, colorbar=True, figure=None,
                colorbar_label=None):
    shown = axis.imshow(image, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    if aperture is not None:
        axis.contour(np.asarray(aperture, dtype=float), levels=[0.5], colors="#00A3E0", linewidths=1.2)
    for label, (column, row) in (markers or {}).items():
        axis.scatter(column, row, marker="+", s=120, color="#FFD100", linewidths=1.5, label=label)
    if markers:
        axis.legend(loc="upper right", fontsize=8, fancybox=True, framealpha=1.0)
    if colorbar and figure is not None:
        figure.colorbar(shown, ax=axis, shrink=0.85, label=colorbar_label or f"Flux [{stack.unit}]")
    axis.grid(False)
    return shown


def _label_sky_axes(axis):
    """Label world coordinates, which may run along either pixel axis for a rotated detector."""
    axis.coords[0].set_axislabel("Right ascension")
    axis.coords[1].set_axislabel("Declination")


def _new_axes(stack: ImageStack, sky: bool, figsize=(5.4, 4.6)):
    figure = plt.figure(figsize=figsize)
    if sky and stack.wcs is not None:
        axis = figure.add_subplot(111, projection=stack.wcs)
        _label_sky_axes(axis)
    else:
        axis = figure.add_subplot(111)
        axis.set_xlabel("Column [pixel]")
        axis.set_ylabel("Row [pixel]")
    return figure, axis


def plot_image(stack: ImageStack, path, frame: int | str = "median", stretch: str = "asinh", cmap: str = "magma",
               aperture=None, markers=None, sky: bool = True, title: str | None = None,
               typefileplot: str = "png") -> str:
    """Plot one frame (an index), the ``"median"`` image, or the ``"variability"`` map of a stack.

    ``aperture`` is a Boolean mask outlined in blue and ``markers`` maps legend labels to
    (column, row) positions. With ``sky`` the axes show right ascension and declination.
    """
    if frame == "median":
        image, default_title = stack.median_image(), f"{stack.label}, median of {len(stack)} frames"
    elif frame == "variability":
        image, default_title = stack.variability_map(), f"{stack.label}, variability"
    else:
        image = stack.flux[frame]
        default_title = f"{stack.label}, {stack.time_format} {stack.time[frame]:.3f}"
    figure, axis = _new_axes(stack, sky)
    label = ("Scatter / median error" if stack.error is not None else "Scatter / typical pixel scatter") \
        if frame == "variability" else None
    _draw_image(axis, image, image_normalization(image, stretch), stack, cmap, aperture, markers, figure=figure,
                colorbar_label=label)
    axis.set_title(title or default_title, fontsize=10)
    return save_figure(figure, path, typefileplot, close_figure=True)


def plot_mosaic(stack: ImageStack, path, number: int = 12, columns: int = 4, stretch: str = "asinh",
                cmap: str = "magma", difference: bool = False, typefileplot: str = "png") -> str:
    """Plot ``number`` evenly spaced frames in a grid on one shared intensity scale.

    With ``difference`` each frame has the median image removed, on a symmetric scale.
    """
    shown = stack.difference() if difference else stack
    indices = np.unique(np.linspace(0, len(shown) - 1, min(number, len(shown))).astype(int))
    rows = int(np.ceil(indices.size / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.3 * columns, 2.3 * rows + 0.4), squeeze=False,
                                constrained_layout=True)
    norm = image_normalization(shown.flux[indices], stretch, symmetric=difference)
    cmap = "RdBu_r" if difference else cmap
    for axis, index in zip(axes.flat, indices):
        image = axis.imshow(shown.flux[index], origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        axis.set_title(f"{shown.time[index]:.2f}", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[indices.size:]:
        axis.set_visible(False)
    figure.colorbar(image, ax=axes, shrink=0.8, label=f"{'Difference flux' if difference else 'Flux'} [{stack.unit}]")
    figure.suptitle(f"{stack.label}, frame titles give {_time_label(stack)}", fontsize=10)
    return save_figure(figure, path, typefileplot, close_figure=True)


def plot_light_curve(light_curve: dict, path, title: str = "", normalize: bool = True,
                     typefileplot: str = "png") -> str:
    """Plot an aperture light curve from :meth:`ImageStack.aperture_photometry`, marking flagged frames."""
    time, flux, error = light_curve["time"], np.asarray(light_curve["flux"], float), light_curve["flux_error"]
    good = (light_curve["quality"] == 0) & np.isfinite(flux)
    scale = np.nanmedian(flux[good]) if normalize else 1.0
    figure, axis = plt.subplots(figsize=(7.0, 3.0))
    axis.errorbar(time[good], flux[good] / scale, yerr=np.asarray(error)[good] / scale, fmt=".", ms=3, lw=0.5,
                  color="black", ecolor="0.6", label="Good frames")
    if (~good).any():
        axis.plot(time[~good], flux[~good] / scale, "x", color="#A51C30", ms=4, label="Flagged frames")
    axis.set_xlabel(f"Time [{light_curve.get('time_format') or 'day'}, day]")
    axis.set_ylabel("Relative flux" if normalize else f"Flux [{light_curve.get('unit', '')} pixel]")
    axis.set_title(title, fontsize=10)
    axis.legend(loc="best", fontsize=8, fancybox=True, framealpha=1.0)
    axis.grid(False)
    return save_figure(figure, path, typefileplot, close_figure=True)


def animate_stack(stack: ImageStack, path, stretch: str = "asinh", cmap: str = "magma", aperture=None,
                  light_curve: dict | None = None, difference: bool = False, max_frames: int = 120,
                  duration_ms: int = 150, sky: bool = False, title: str | None = None) -> Path:
    """Write a GIF animation of a stack, one frame per image, on one shared intensity scale.

    With ``light_curve`` (from :meth:`ImageStack.aperture_photometry`) a lower panel shows the
    light curve with the current frame marked. With ``difference`` each frame has the median
    image removed, so only changing sources remain. At most ``max_frames`` evenly spaced frames
    are rendered.
    """
    shown = stack.difference() if difference else stack
    indices = np.unique(np.linspace(0, len(shown) - 1, min(max_frames, len(shown))).astype(int))
    norm = image_normalization(shown.flux[indices], stretch, symmetric=difference)
    cmap = "RdBu_r" if difference else cmap
    if light_curve is not None:
        flux = np.asarray(light_curve["flux"], dtype=float)
        good = (np.asarray(light_curve["quality"]) == 0) & np.isfinite(flux)
        scale = np.nanmedian(flux[good])
    frames = []
    for index in indices:
        # 100 dpi frames keep animations of a hundred frames to a few megabytes
        figure = plt.figure(figsize=(5.4, 6.6 if light_curve is not None else 4.8), dpi=100)
        grid = figure.add_gridspec(2 if light_curve is not None else 1, 1,
                                   height_ratios=[3.2, 1.3] if light_curve is not None else [1])
        projection = stack.wcs if sky and stack.wcs is not None else None
        axis = figure.add_subplot(grid[0], projection=projection)
        _draw_image(axis, shown.flux[index], norm, stack, cmap, aperture, figure=figure,
                    colorbar_label=f"{'Difference flux' if difference else 'Flux'} [{stack.unit}]")
        axis.set_title(f"{title or stack.label}\n{stack.time_format} {shown.time[index]:.3f} "
                       f"({index + 1}/{len(shown)})", fontsize=9)
        if projection is None:
            axis.set_xlabel("Column [pixel]")
            axis.set_ylabel("Row [pixel]")
        else:
            _label_sky_axes(axis)
        if light_curve is not None:
            curve = figure.add_subplot(grid[1])
            curve.plot(light_curve["time"][good], flux[good] / scale, ".", ms=2, color="0.5")
            curve.plot(light_curve["time"][index], flux[index] / scale, "o", ms=6, color="#A51C30")
            curve.set_xlabel(_time_label(stack))
            curve.set_ylabel("Relative flux")
            curve.grid(False)
        figure.tight_layout()
        frames.append(figure_to_frame(figure))
    return write_animation(frames, path, duration_ms=duration_ms)
