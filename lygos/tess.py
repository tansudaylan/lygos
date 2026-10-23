"""Retrieve TESS cutouts and full-frame images (FFIs) as :class:`~lygos.imagestack.ImageStack` objects.

Cutouts come from the Mikulski Archive for Space Telescopes (MAST) TESSCut service
(Brasseur et al. 2019). Full-frame images are read from the public MAST bucket on
Amazon Web Services (``s3://stpubdata/tess/public/ffi``) without credentials. Only the
pixels that are needed are transferred, because uncompressed FITS files allow
section reads, so long FFI time series of a small region stay fast.
"""

from __future__ import annotations

from tdpy.verbosity import print

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from tdpy.plotting import save_figure
from tdpy.tess import tess_target_visibility

from .imagestack import ImageStack

FFI_BUCKET = "stpubdata/tess/public/ffi"
# science pixels of an FFI calibrated image start after 44 columns of serial overscan
FFI_SCIENCE_COLUMNS = slice(44, 2092)
FFI_SCIENCE_ROWS = slice(0, 2048)


def resolve_coordinate(target) -> SkyCoord:
    """Return a SkyCoord for a SkyCoord, an (RA, Dec) pair [deg], or a name resolvable by Sesame."""
    if isinstance(target, SkyCoord):
        return target
    if isinstance(target, str):
        print(f"Resolving {target} with Sesame...")
        return SkyCoord.from_name(target)
    right_ascension, declination = target
    return SkyCoord(float(right_ascension), float(declination), unit="deg")


def tess_sectors(target) -> list[dict]:
    """Return the TESS sectors, cameras, and CCDs that observed a target, from TESSCut."""
    from astroquery.mast import Tesscut

    table = Tesscut.get_sectors(coordinates=resolve_coordinate(target))
    return [{"sector": int(row["sector"]), "camera": int(row["camera"]), "ccd": int(row["ccd"])} for row in table]


def plot_tess_sky_coverage(
    sectors,
    output_path,
    sample_count: int = 1024,
    random_seed: int = 0,
    typefileplot: str = "png",
    typeplotback: str = "white",
) -> dict[str, object]:
    """Plot cumulative unique sky area covered by ordered TESS sectors.

    Uniform random points on the sphere are tested against the mission WCS for
    every CCD. The reported uncertainty is the binomial 1-sigma sampling error.
    """
    if np.isscalar(sectors):
        sectors = (sectors,)
    sectors = tuple(int(sector) for sector in sectors)
    if not sectors or any(sector < 1 for sector in sectors):
        raise ValueError("sectors must contain positive sector numbers")
    if sectors != tuple(sorted(set(sectors))):
        raise ValueError("sectors must be unique and strictly increasing")
    if sample_count < 32:
        raise ValueError("sample_count must be at least 32")

    random = np.random.default_rng(random_seed)
    right_ascension_deg = random.uniform(0.0, 360.0, sample_count)  # [deg]
    sine_declination = random.uniform(-1.0, 1.0, sample_count)
    declination_deg = np.rad2deg(np.arcsin(sine_declination))  # [deg]
    targets = {
        f"sample-{index:06d}": (right_ascension_deg[index], declination_deg[index])
        for index in range(sample_count)
    }
    visibility = tess_target_visibility(targets, sectors)
    observed = np.zeros((sample_count, len(sectors)), dtype=bool)
    for index, target_name in enumerate(targets):
        visible_sectors = set(visibility[target_name])
        observed[index] = [sector in visible_sectors for sector in sectors]

    cumulative_observed = np.logical_or.accumulate(observed, axis=1)
    covered_fraction = np.mean(cumulative_observed, axis=0)
    total_sky_area_deg2 = 4.0 * np.pi * (180.0 / np.pi) ** 2  # [deg^2]
    cumulative_area_deg2 = total_sky_area_deg2 * covered_fraction  # [deg^2]
    area_uncertainty_deg2 = total_sky_area_deg2 * np.sqrt(
        covered_fraction * (1.0 - covered_fraction) / sample_count
    )  # [deg^2]

    longitude_rad = np.deg2rad(((180.0 - right_ascension_deg) % 360.0) - 180.0)
    latitude_rad = np.deg2rad(declination_deg)
    figure = plt.figure(figsize=(9.5, 7.0), constrained_layout=True)
    map_axis = figure.add_subplot(2, 1, 1, projection="mollweide")
    scatter = map_axis.scatter(
        longitude_rad,
        latitude_rad,
        c=observed.sum(axis=1),
        s=7,
        cmap="viridis",
        vmin=0,
        vmax=len(sectors),
        linewidths=0,
    )
    map_axis.set_xticks(np.deg2rad(np.arange(-150, 181, 30)))
    map_axis.set_xticklabels([str(int((360 - value) % 360)) for value in np.arange(-150, 181, 30)])
    map_axis.tick_params(labelsize=8)
    map_axis.set_xlabel("Right ascension [deg]", fontsize=9)
    map_axis.set_ylabel("Declination [deg]", fontsize=9)
    map_axis.set_title(f"TESS sky coverage through Sector {sectors[-1]}", fontsize=10)
    map_axis.grid(False)
    colorbar = figure.colorbar(
        scatter,
        ax=map_axis,
        orientation="horizontal",
        pad=0.12,
        shrink=0.7,
    )
    colorbar.set_label("Number of sectors per sampled position", fontsize=8)
    colorbar.set_ticks(np.linspace(0, len(sectors), min(len(sectors) + 1, 5), dtype=int))
    colorbar.ax.tick_params(labelsize=8)

    area_axis = figure.add_subplot(2, 1, 2)
    area_axis.errorbar(
        sectors,
        cumulative_area_deg2,
        yerr=area_uncertainty_deg2,
        marker="o",
        markersize=3,
        linewidth=1,
        capsize=2,
    )
    area_axis.set_xlabel("TESS sector", fontsize=9)
    area_axis.set_ylabel("Cumulative unique sky area [deg^2]", fontsize=9)
    area_axis.tick_params(labelsize=8)
    area_axis.set_ylim(bottom=0.0)
    area_axis.grid(False)

    path = save_figure(figure, output_path, typefileplot, typeplotback, close_figure=True)
    return {
        "sector_numbers": np.asarray(sectors, dtype=int),
        "cumulative_area_deg2": cumulative_area_deg2,
        "area_uncertainty_deg2": area_uncertainty_deg2,
        "sample_sector_counts": observed.sum(axis=1),
        "path": Path(path),
    }


def read_tess_cutout(path) -> ImageStack:
    """Read a TESSCut target pixel file into an ImageStack.

    Flux and errors are calibrated, background-subtracted rates [e-/s]; times are TESS
    barycentric Julian dates (BTJD = BJD - 2457000) [day].
    """
    path = Path(path)
    print(f"Reading from {path}...")
    with fits.open(path) as hdul:
        table = hdul[1].data
        wcs = _quiet_wcs(hdul[2].header)
        primary = hdul[0].header
        background = np.asarray(table["FLUX_BKG"], dtype=float)
        stack = ImageStack(
            flux=np.asarray(table["FLUX"], dtype=float), time=np.asarray(table["TIME"], dtype=float),
            error=np.asarray(table["FLUX_ERR"], dtype=float), quality=np.asarray(table["QUALITY"], dtype=int),
            wcs=wcs, unit="e-/s", time_format="BTJD", mission="TESS", band="TESS",
            label=f"TESS Sector {primary.get('SECTOR')} cutout",
            metadata={"sector": primary.get("SECTOR"), "camera": primary.get("CAMERA"), "ccd": primary.get("CCD"),
                      "path": str(path), "background": background, "source": "TESSCut"},
        )
    # frames without a time stamp cannot be placed in a time series
    return stack.select(np.isfinite(stack.time))


def get_tess_cutout(target, sector: int | None = None, size: int = 15, cache_dir=None) -> list[ImageStack]:
    """Download TESSCut cutouts of ``size`` by ``size`` pixels and return one ImageStack per sector.

    Files already in ``cache_dir`` are reused, so repeated calls do not contact MAST.
    """
    from astroquery.mast import Tesscut

    coordinate = resolve_coordinate(target)
    cache_dir = Path(cache_dir or Path.cwd() / "tesscut")
    cache_dir.mkdir(parents=True, exist_ok=True)
    pattern = (f"tess-s{sector:04d}-*_{coordinate.ra.deg:.6f}_{coordinate.dec.deg:.6f}_{size}x{size}_astrocut.fits"
               if sector else f"tess-s*_{coordinate.ra.deg:.6f}_{coordinate.dec.deg:.6f}_{size}x{size}_astrocut.fits")
    paths = sorted(cache_dir.glob(pattern))
    if not paths:
        print(f"Downloading TESSCut cutouts to {cache_dir}...")
        manifest = Tesscut.download_cutouts(coordinates=coordinate, size=size, sector=sector, path=str(cache_dir))
        paths = sorted(Path(path) for path in manifest["Local Path"])
    return [read_tess_cutout(path) for path in paths]


@lru_cache(maxsize=None)
def list_tess_ffis(sector: int, camera: int, ccd: int) -> tuple[str, ...]:
    """Return the time-ordered calibrated FFI paths of one sector, camera, and CCD on the public bucket."""
    import s3fs

    filesystem = s3fs.S3FileSystem(anon=True)
    paths = []
    for year in filesystem.ls(f"{FFI_BUCKET}/s{sector:04d}"):
        for day in filesystem.ls(year):
            folder = f"{day}/{camera}-{ccd}"
            if filesystem.exists(folder):
                paths.extend(path for path in filesystem.ls(folder) if path.endswith("ffic.fits"))
    # FFI names start with the time stamp yyyydddhhmmss, so names sort in time
    return tuple(sorted(paths, key=lambda path: Path(path).name))


def _quiet_wcs(header) -> WCS:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(header)


def _open_ffi(path: str):
    return fits.open("s3://" + path, use_fsspec=True, fsspec_kwargs={"anon": True})


def _ffi_time(header) -> float:
    """Return the mid-exposure time of an FFI [day], in BTJD."""
    return 0.5 * (header["TSTART"] + header["TSTOP"])


def get_tess_ffi(sector: int, camera: int, ccd: int, index: int = 0, science_only: bool = True) -> ImageStack:
    """Read one full-frame image of a camera and CCD, with its uncertainty, as a one-frame ImageStack.

    ``index`` counts calibrated FFIs in time order within the sector. With
    ``science_only`` the overscan and virtual rows are trimmed to the 2048 by 2048
    science area, and the WCS is shifted to match.
    """
    path = list_tess_ffis(sector, camera, ccd)[index]
    print(f"Reading from s3://{path}...")
    with _open_ffi(path) as hdul:
        header = hdul[1].header
        rows, columns = (FFI_SCIENCE_ROWS, FFI_SCIENCE_COLUMNS) if science_only else (slice(None), slice(None))
        flux = hdul[1].section[rows, columns]
        error = hdul[2].section[rows, columns]
        wcs = _quiet_wcs(header)[rows, columns]
    return ImageStack(flux=flux, time=[_ffi_time(header)], error=error, quality=[header.get("DQUALITY", 0)],
                      wcs=wcs, unit="e-/s", time_format="BTJD", mission="TESS", band="TESS",
                      label=f"TESS Sector {sector} camera {camera} CCD {ccd} FFI",
                      metadata={"sector": sector, "camera": camera, "ccd": ccd, "path": path, "source": "FFI"})


def get_tess_ffi_cutout(target, sector: int, size: int = 15, max_frames: int | None = None,
                        stride: int = 1) -> ImageStack:
    """Return a ``size`` by ``size`` time series cut directly from the public FFIs of one sector.

    The camera and CCD come from the predicted sector pointing, and the pixel position from
    the first FFI's own header WCS. ``stride`` keeps every n-th FFI and ``max_frames`` caps
    the number of frames read, to bound the transfer for quick looks.
    """
    from tdpy.tess import locate_tess_target

    coordinate = resolve_coordinate(target)
    location = locate_tess_target(sector, coordinate.ra.deg, coordinate.dec.deg)
    if location is None:
        raise ValueError(f"{coordinate.to_string('hmsdms')} is not on a TESS detector in Sector {sector}")
    camera, ccd = location[:2]
    paths = list_tess_ffis(sector, camera, ccd)[::stride][:max_frames]
    if not paths:
        raise FileNotFoundError(f"No public FFIs for Sector {sector}, camera {camera}, CCD {ccd}")
    with _open_ffi(paths[0]) as hdul:
        column, row = _quiet_wcs(hdul[1].header).world_to_pixel(coordinate)
        height, width = hdul[1].header["NAXIS2"], hdul[1].header["NAXIS1"]
    half = size // 2
    row0 = int(np.clip(round(float(row)) - half, 0, height - size))
    column0 = int(np.clip(round(float(column)) - half, 0, width - size))
    rows, columns = slice(row0, row0 + size), slice(column0, column0 + size)
    flux, error, time, quality = [], [], [], []
    print(f"Reading {len(paths)} FFI sections of {size} by {size} pixels from s3://{FFI_BUCKET}...")
    for path in paths:
        with _open_ffi(path) as hdul:
            header = hdul[1].header
            flux.append(hdul[1].section[rows, columns])
            error.append(hdul[2].section[rows, columns])
            time.append(_ffi_time(header))
            quality.append(header.get("DQUALITY", 0))
            if len(flux) == 1:
                wcs = _quiet_wcs(header)[rows, columns]
    return ImageStack(flux=np.array(flux), time=time, error=np.array(error), quality=quality, wcs=wcs,
                      unit="e-/s", time_format="BTJD", mission="TESS", band="TESS",
                      label=f"TESS Sector {sector} FFI cutout",
                      metadata={"sector": sector, "camera": camera, "ccd": ccd, "row0": row0, "column0": column0,
                                "paths": list(paths), "source": "FFI"})
