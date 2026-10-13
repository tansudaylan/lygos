"""Robust variability measurements and summaries for time-series image stacks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tdpy.plotting import save_figure

from .imagestack import ImageStack


def _robust_scatter_map(values: np.ndarray) -> np.ndarray:
    """Return 1.4826 times the per-pixel median absolute deviation over time."""
    masked = np.ma.masked_invalid(values)
    center = np.ma.median(masked, axis=0)
    return (1.4826 * np.ma.median(np.ma.abs(masked - center), axis=0)).filled(np.nan)


def compute_variability_products(stack: ImageStack) -> dict[str, np.ndarray]:
    """Return temporal median and robust relative-scatter maps for good frames.

    Relative scatter is the median absolute deviation of each pixel's residuals from its
    temporal median, divided by that median flux and expressed as a percentage.
    """
    good = stack.good()
    if len(good) < 2:
        raise ValueError("stack must contain at least two good time samples")

    masked_flux = np.ma.masked_invalid(good.flux)
    temporal_median = np.ma.median(masked_flux, axis=0).filled(np.nan)
    valid_baseline = np.isfinite(temporal_median) & (temporal_median != 0.0)
    relative_residual = np.full(good.flux.shape, np.nan)
    np.divide(
        good.flux - temporal_median,
        temporal_median,
        out=relative_residual,
        where=np.isfinite(good.flux) & valid_baseline,
    )

    return {
        "temporal_median": temporal_median,
        "relative_variability_percent": 100.0 * _robust_scatter_map(relative_residual),
    }


def plot_variability_summary(
    stack: ImageStack,
    output_path: str | Path,
    title: str | None = None,
    typefileplot: str = "png",
) -> dict[str, np.ndarray]:
    """Plot the median image, relative variability map, and the most variable source pixel."""
    if typefileplot == "html":
        return plot_variability_summary_bokeh(
            stack,
            output_path,
            title=title,
        )
    if typefileplot not in {"png", "pdf"}:
        raise ValueError("typefileplot must be 'png', 'pdf', or 'html'")

    good = stack.good()
    products = compute_variability_products(good)
    temporal_median = products["temporal_median"]
    variability = products["relative_variability_percent"]
    if not np.isfinite(variability).any():
        raise ValueError("stack contains no pixel with a valid nonzero median flux")

    finite_median = temporal_median[np.isfinite(temporal_median)]
    background_level = np.median(finite_median)
    background_scatter = 1.4826 * np.median(np.abs(finite_median - background_level))
    source_mask = temporal_median > background_level + 3.0 * background_scatter
    if not source_mask.any():
        source_mask = np.isfinite(temporal_median)
    source_variability = np.where(source_mask, variability, np.nan)
    if not np.isfinite(source_variability).any():
        source_variability = variability
    pixel_row, pixel_column = np.unravel_index(
        np.nanargmax(source_variability), source_variability.shape
    )
    pixel_baseline = temporal_median[pixel_row, pixel_column]

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor="white", constrained_layout=True)
    median_image = axes[0].imshow(temporal_median, origin="lower", cmap="gray_r")
    median_label = f"Median flux [{stack.unit}]" if stack.unit else "Median pixel value"
    figure.colorbar(median_image, ax=axes[0], label=median_label)
    axes[0].set_title("Temporal median image")

    variability_image = axes[1].imshow(source_variability, origin="lower", cmap="magma")
    axes[1].scatter(pixel_column, pixel_row, marker="x", s=70, linewidth=2.0, color="white")
    figure.colorbar(variability_image, ax=axes[1], label="Robust relative variability [%]")
    axes[1].set_title("Source-pixel variability map")

    time_label = f"Time [{good.time_format}]" if good.time_format else "Time [day]"
    axes[2].plot(
        good.time,
        good.flux[:, pixel_row, pixel_column] / pixel_baseline,
        color="#1B6CA8",
        linewidth=0.8,
        label=f"Pixel ({pixel_column}, {pixel_row})",
    )
    axes[2].set_xlabel(time_label)
    axes[2].set_ylabel("Normalized flux")
    axes[2].set_title("Selected pixel time series")
    legend = axes[2].legend(frameon=True, fancybox=True, framealpha=1.0)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    for axis in axes[:2]:
        axis.set_xlabel("Detector x pixel")
        axis.set_ylabel("Detector y pixel")
    for axis in axes:
        axis.grid(False)
    figure.suptitle(title or stack.label or "Image-stack variability", fontweight="bold")
    save_figure(figure, output_path, typefileplot, close_figure=True)
    return products


def plot_variability_summary_bokeh(
    stack: ImageStack,
    output_path: str | Path,
    title: str | None = None,
) -> dict[str, np.ndarray]:
    """Write interactive median, relative-variability, and pixel-curve panels."""
    from bokeh.layouts import row
    from bokeh.models import ColorBar, HoverTool, LinearColorMapper
    from bokeh.palettes import Greys256, Magma256
    from bokeh.plotting import figure
    from tdpy.plotting import save_bokeh_html

    good = stack.good()
    products = compute_variability_products(good)
    temporal_median = products["temporal_median"]
    variability = products["relative_variability_percent"]
    if not np.isfinite(variability).any():
        raise ValueError("stack contains no pixel with a valid nonzero median flux")
    finite_median = temporal_median[np.isfinite(temporal_median)]
    background_level = np.median(finite_median)
    background_scatter = 1.4826 * np.median(np.abs(finite_median - background_level))
    source_mask = temporal_median > background_level + 3.0 * background_scatter
    if not source_mask.any():
        source_mask = np.isfinite(temporal_median)
    source_variability = np.where(source_mask, variability, np.nan)
    if not np.isfinite(source_variability).any():
        source_variability = variability
    pixel_row, pixel_column = np.unravel_index(
        np.nanargmax(source_variability), source_variability.shape
    )
    pixel_baseline = temporal_median[pixel_row, pixel_column]
    rows, columns = temporal_median.shape
    plots = []
    image_panels = (
        (temporal_median, "Temporal median image", Greys256, "Median flux"),
        (source_variability, "Source-pixel variability map", Magma256, "Relative variability [%]"),
    )
    for image, panel_title, palette, colorbar_title in image_panels:
        finite = image[np.isfinite(image)]
        low = float(np.min(finite))
        high = float(np.max(finite))
        if high <= low:
            high = low + 1.0
        mapper = LinearColorMapper(palette=palette, low=low, high=high, nan_color="#D9DDE0")
        plot = figure(
            title=panel_title,
            width=380,
            height=380,
            x_range=(0, columns),
            y_range=(0, rows),
            match_aspect=True,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        plot.image(
            image=[image], x=0, y=0, dw=columns, dh=rows,
            color_mapper=mapper,
        )
        plot.add_layout(ColorBar(color_mapper=mapper, title=colorbar_title), "right")
        plot.scatter([pixel_column + 0.5], [pixel_row + 0.5], marker="x", size=12,
                     line_color="#00A3E0", line_width=2.0)
        plot.xaxis.axis_label = "Column [pixel]"
        plot.yaxis.axis_label = "Row [pixel]"
        plot.grid.visible = False
        plot.add_tools(HoverTool(tooltips=[
            ("Column [pixel]", "$x{0.0}"), ("Row [pixel]", "$y{0.0}"),
            (colorbar_title, "$image"),
        ]))
        plots.append(plot)

    time_label = f"Time [{good.time_format}, day]" if good.time_format else "Time [day]"
    curve = figure(
        title=f"Pixel ({pixel_column}, {pixel_row}) time series",
        width=440,
        height=380,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    curve.scatter(
        good.time,
        good.flux[:, pixel_row, pixel_column] / pixel_baseline,
        size=5,
        color="#1B6CA8",
    )
    curve.xaxis.axis_label = time_label
    curve.yaxis.axis_label = "Normalized flux"
    curve.grid.visible = False
    curve.add_tools(HoverTool(tooltips=[(time_label, "$x"), ("Normalized flux", "$y")]))
    plots.append(curve)
    return products | {"path": save_bokeh_html(
        row(*plots, sizing_mode="stretch_width"),
        output_path,
        title=title or stack.label or "Image-stack variability",
    )}