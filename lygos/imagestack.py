"""A mission-independent stack of images with times, sky coordinates, and processing methods.

Every retrieval function in :mod:`lygos.tess` and :mod:`lygos.roman` returns an
:class:`ImageStack`, so the same photometry, background, difference-imaging, and
visualization tools apply to TESS cutouts, TESS full-frame images, and Roman images.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS


def tangent_plane_wcs(center: SkyCoord, size: int, pixel_scale: float) -> WCS:
    """Return a north-up, east-left gnomonic WCS of ``size`` by ``size`` pixels centered on ``center``.

    ``pixel_scale`` is in arcseconds per pixel.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [center.ra.deg, center.dec.deg]
    wcs.wcs.crpix = [(size + 1) / 2.0, (size + 1) / 2.0]
    wcs.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]  # [deg pixel^-1]
    wcs.pixel_shape = (size, size)
    return wcs


def resample_image(image: np.ndarray, wcs_in: WCS, wcs_out: WCS, shape: tuple[int, int], order: int = 3) -> np.ndarray:
    """Interpolate an image from one WCS onto the pixel grid of another.

    Pixels of the output that fall outside the input are NaN. The value at each output pixel
    center is a spline interpolation, so the result preserves surface brightness, and fluxes
    are preserved when the two grids share one pixel scale.
    """
    from scipy.ndimage import map_coordinates

    rows, columns = np.indices(shape)
    sky = wcs_out.pixel_to_world(columns.ravel(), rows.ravel())
    column_in, row_in = wcs_in.world_to_pixel(sky)
    # nearest-edge extension keeps the spline finite; pixels outside the input are masked below
    values = map_coordinates(np.nan_to_num(image), [row_in, column_in], order=order, mode="nearest")
    inside = (row_in >= -0.5) & (row_in <= image.shape[0] - 0.5) & (column_in >= -0.5) & (column_in <= image.shape[1] - 0.5)
    return np.where(inside, values, np.nan).reshape(shape)


@dataclass
class ImageStack:
    """A time-ordered cube of images on one pixel grid.

    Attributes
    ----------
    flux : numpy.ndarray
        Pixel values with shape (time, row, column), in ``unit``.
    time : numpy.ndarray
        Mid-exposure times [day], in the time system named by ``time_format``.
    error : numpy.ndarray or None
        One-sigma pixel uncertainties with the same shape and unit as ``flux``.
    quality : numpy.ndarray
        Integer quality flags per frame; zero marks a good frame.
    wcs : astropy.wcs.WCS or None
        Celestial coordinates of the pixel grid, shared by all frames.
    unit : str
        Unit of ``flux``, for example ``"e-/s"``.
    time_format : str
        Time system of ``time``, for example ``"BTJD"`` or ``"MJD"``.
    mission, band, label : str
        Instrument, filter, and a short human-readable description.
    metadata : dict
        Provenance such as sector, camera, CCD, pointing, detector, or file paths.
    """

    flux: np.ndarray
    time: np.ndarray
    error: np.ndarray | None = None
    quality: np.ndarray | None = None
    wcs: WCS | None = None
    unit: str = ""
    time_format: str = ""
    mission: str = ""
    band: str = ""
    label: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.flux = np.asarray(self.flux, dtype=float)
        if self.flux.ndim == 2:
            self.flux = self.flux[None]
        if self.flux.ndim != 3:
            raise ValueError("flux must have shape (time, row, column)")
        self.time = np.atleast_1d(np.asarray(self.time, dtype=float))
        if self.time.size != self.flux.shape[0]:
            raise ValueError("time must have one entry per frame")
        if self.error is not None:
            self.error = np.asarray(self.error, dtype=float).reshape(self.flux.shape)
        self.quality = (np.zeros(self.time.size, dtype=int) if self.quality is None
                        else np.asarray(self.quality, dtype=int).reshape(self.time.size))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.flux.shape

    def __len__(self) -> int:
        return self.flux.shape[0]

    def select(self, indices) -> "ImageStack":
        """Return the frames at ``indices`` (an index array, slice, or Boolean mask)."""
        indices = np.arange(len(self))[indices]
        return replace(self, flux=self.flux[indices], time=self.time[indices],
                       error=None if self.error is None else self.error[indices],
                       quality=self.quality[indices], metadata=dict(self.metadata))

    def good(self) -> "ImageStack":
        """Return the frames with zero quality flags and finite pixels."""
        finite = np.isfinite(self.flux).any(axis=(1, 2))
        return self.select((self.quality == 0) & finite)

    def sorted(self) -> "ImageStack":
        """Return the frames in increasing time order."""
        return self.select(np.argsort(self.time, kind="stable"))

    def concatenate(self, other: "ImageStack") -> "ImageStack":
        """Append the frames of another stack on the same pixel grid, in time order."""
        if other.flux.shape[1:] != self.flux.shape[1:]:
            raise ValueError("stacks must share one pixel grid")
        error = (None if self.error is None or other.error is None
                 else np.concatenate([self.error, other.error]))
        return replace(self, flux=np.concatenate([self.flux, other.flux]),
                       time=np.concatenate([self.time, other.time]), error=error,
                       quality=np.concatenate([self.quality, other.quality]),
                       metadata=dict(self.metadata)).sorted()

    def median_image(self) -> np.ndarray:
        """Return the per-pixel median over good frames, a low-noise reference image."""
        return np.nanmedian(self.good().flux, axis=0)

    def subtract_background(self, method: str = "median", mask: np.ndarray | None = None) -> "ImageStack":
        """Return the stack with a per-frame sky level removed.

        ``method="median"`` subtracts the sigma-clipped median of each frame, ignoring
        pixels in ``mask``. ``method="plane"`` fits and subtracts a tilted plane.
        """
        from astropy.stats import sigma_clipped_stats

        usable = np.ones(self.flux.shape[1:], dtype=bool) if mask is None else ~np.asarray(mask, dtype=bool)
        rows, columns = np.indices(self.flux.shape[1:])
        background = np.empty_like(self.flux)
        for index, frame in enumerate(self.flux):
            valid = usable & np.isfinite(frame)
            if method == "median":
                background[index] = sigma_clipped_stats(frame[valid], sigma=3.0)[1]
            elif method == "plane":
                design = np.column_stack([np.ones(valid.sum()), rows[valid], columns[valid]])
                coefficients = np.linalg.lstsq(design, frame[valid], rcond=None)[0]
                background[index] = coefficients[0] + coefficients[1] * rows + coefficients[2] * columns
            else:
                raise ValueError("method must be 'median' or 'plane'")
        return replace(self, flux=self.flux - background, metadata=dict(self.metadata, background=method))

    def difference(self, reference: np.ndarray | None = None) -> "ImageStack":
        """Return each frame minus a reference image, the median image by default.

        Static sources cancel, so variable stars, transients, and moving objects stand out.
        """
        reference = self.median_image() if reference is None else np.asarray(reference, dtype=float)
        return replace(self, flux=self.flux - reference[None], label=f"{self.label} difference".strip(),
                       metadata=dict(self.metadata, difference=True))

    def variability_map(self) -> np.ndarray:
        """Return the robust per-pixel scatter over time, in units of the expected noise.

        With pixel errors the scatter is divided by each pixel's median error, so constant
        sources sit near one and only excess variability stands out; subtract a varying sky
        level first. Without errors it is divided by the median scatter over the image. The
        median absolute deviation makes the map insensitive to a few outlier frames.
        """
        good = self.good()
        from .variability import _robust_scatter_map

        deviation = _robust_scatter_map(good.flux)
        if good.error is not None:
            return deviation / np.nanmedian(good.error, axis=0)
        return deviation / np.nanmedian(deviation)

    def relative_variability_map(self) -> np.ndarray:
        """Return robust per-pixel variability as a percentage of each pixel's median flux."""
        from .variability import compute_variability_products

        return compute_variability_products(self)["relative_variability_percent"]

    def pixel_of(self, coordinate: SkyCoord) -> tuple[float, float]:
        """Return the (column, row) pixel position of a sky coordinate."""
        if self.wcs is None:
            raise ValueError("this stack has no world coordinate system")
        column, row = self.wcs.world_to_pixel(coordinate)
        return float(column), float(row)

    def circular_aperture(self, radius: float, center: tuple[float, float] | None = None) -> np.ndarray:
        """Return a Boolean mask of pixels whose centers lie within ``radius`` pixels of ``center``.

        ``center`` is a (column, row) position and defaults to the stack center.
        """
        rows, columns = np.indices(self.flux.shape[1:])
        if center is None:
            center = ((self.flux.shape[2] - 1) / 2.0, (self.flux.shape[1] - 1) / 2.0)
        return (columns - center[0]) ** 2 + (rows - center[1]) ** 2 <= radius ** 2

    def threshold_aperture(self, threshold: float = 3.0, center: tuple[float, float] | None = None) -> np.ndarray:
        """Return the connected pixels around ``center`` brighter than ``threshold`` sky deviations.

        This mirrors the threshold apertures of TESS pipelines on the median image.
        """
        from astropy.stats import sigma_clipped_stats
        from scipy import ndimage

        image = self.median_image()
        _, median, deviation = sigma_clipped_stats(image[np.isfinite(image)], sigma=3.0)
        bright = np.isfinite(image) & (image > median + threshold * deviation)
        labels, _ = ndimage.label(bright)
        if center is None:
            center = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
        label = labels[int(round(center[1])), int(round(center[0]))]
        if label == 0:
            return self.circular_aperture(2.0, center)
        return labels == label

    def aperture_photometry(self, aperture: np.ndarray, background: np.ndarray | None = None) -> dict:
        """Sum each frame within a Boolean aperture and return a light curve.

        If ``background`` (a Boolean mask of sky pixels) is given, the per-pixel median of
        those pixels is subtracted from every aperture pixel first. The returned dictionary
        holds ``time``, ``flux`` and ``flux_error`` (in ``unit`` times pixels), and ``quality``.
        """
        aperture = np.asarray(aperture, dtype=bool)
        flux = self.flux
        if background is not None:
            flux = flux - np.nanmedian(self.flux[:, np.asarray(background, dtype=bool)], axis=1)[:, None, None]
        summed = np.nansum(flux[:, aperture], axis=1)
        error = (np.sqrt(np.nansum(self.error[:, aperture] ** 2, axis=1)) if self.error is not None
                 else np.full(len(self), np.nan))
        return {"time": self.time.copy(), "flux": summed, "flux_error": error, "quality": self.quality.copy(),
                "unit": self.unit, "time_format": self.time_format}

    def centroid(self, aperture: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the flux-weighted (column, row) centroid of every frame within an aperture."""
        mask = np.ones(self.flux.shape[1:], dtype=bool) if aperture is None else np.asarray(aperture, dtype=bool)
        rows, columns = np.indices(self.flux.shape[1:])
        weight = np.where(mask[None], np.clip(np.nan_to_num(self.flux), 0.0, None), 0.0)
        total = weight.sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (weight * columns).sum(axis=(1, 2)) / total, (weight * rows).sum(axis=(1, 2)) / total

    def bin_time(self, width: float) -> "ImageStack":
        """Return the stack averaged in time bins of ``width`` [day], keeping only good frames."""
        good = self.good()
        edges = np.arange(good.time.min(), good.time.max() + width, width)  # [day]
        index = np.digitize(good.time, edges) - 1
        keep = np.unique(index)
        flux = np.stack([np.nanmean(good.flux[index == k], axis=0) for k in keep])
        error = None
        if good.error is not None:
            error = np.stack([np.sqrt(np.nansum(good.error[index == k] ** 2, axis=0)) / np.sum(index == k)
                              for k in keep])
        time = np.array([good.time[index == k].mean() for k in keep])  # [day]
        return replace(good, flux=flux, time=time, error=error, quality=np.zeros(keep.size, dtype=int),
                       metadata=dict(good.metadata, binned_days=width))
