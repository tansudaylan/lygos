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

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

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
