"""Retrieve Nancy Grace Roman Space Telescope images as :class:`~lygos.imagestack.ImageStack` objects.

Roman has not yet launched, so lygos reads the public OpenUniverse 2024 simulation of
Roman Wide Field Instrument (WFI) imaging (OpenUniverse et al. 2025), hosted by the NASA/IPAC
Infrared Science Archive (IRSA) on Amazon Web Services without credentials. The time-domain
survey (``"TDS"``) revisits each field about every five days in seven filters, and the
wide-area survey (``"WAS"``) maps a larger area. Each file holds one exposure of one of the
18 detectors (sensor chip assemblies, SCAs) of 4088 by 4088 pixels at 0.11 arcsec per pixel.

Files are gzip-compressed, so pixel sections cannot be read directly. Cutouts are streamed
instead: the file is decompressed only until the last needed detector row, so a cutout near
the bottom of a detector transfers a small fraction of the file.
"""

from __future__ import annotations

from tdpy.verbosity import print

import gzip
import json
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .imagestack import ImageStack, resample_image, tangent_plane_wcs
from .paths import get_cache_path
from .tess import _quiet_wcs, resolve_coordinate

BUCKET = "nasa-irsa-simulations/openuniverse2024/roman"
SEQUENCES = {"TDS": "Roman_TDS_obseq_11_6_23", "WAS": "Roman_WAS_obseq_11_1_23"}
BANDS = ("R062", "Z087", "Y106", "J129", "H158", "F184", "K213")
DETECTOR_SIDE = 4088  # [pixel]
PIXEL_SCALE = 0.11  # [arcsec pixel^-1]
# a target closer than this to a detector center can fall on that detector, half its diagonal
DETECTOR_RADIUS = DETECTOR_SIDE * PIXEL_SCALE / 3600.0 / np.sqrt(2.0)  # [deg]
FITS_BLOCK = 2880  # [byte]


def _filesystem():
    import s3fs

    return s3fs.S3FileSystem(anon=True, config_kwargs={"connect_timeout": 30, "read_timeout": 120})


def _survey_root(survey: str, release: str) -> str:
    if survey not in SEQUENCES:
        raise ValueError(f"survey must be one of {tuple(SEQUENCES)}")
    if release not in ("preview", "full"):
        raise ValueError("release must be 'preview' or 'full'")
    return f"{BUCKET}/{release}/Roman{survey}"


def load_roman_pointings(survey: str = "TDS", release: str = "preview", cache_dir=None) -> dict:
    """Return the simulated observing sequence: one row per pointing.

    The dictionary holds ``ra`` and ``dec`` [deg] of the 18 detector centers with shape
    (pointing, detector), and per pointing ``band``, ``mjd`` [day], and ``exptime`` [s].
    """
    cache_dir = Path(cache_dir or get_cache_path("roman"))
    root = _survey_root(survey, release)
    tables = {}
    for suffix in ("", "_radec"):
        name = f"{SEQUENCES[survey]}{suffix}.fits"
        local = cache_dir / name
        if not local.exists():
            print(f"Writing to {local}...")
            _filesystem().get(f"{root}/{name}", str(local))
        print(f"Reading from {local}...")
        tables[suffix] = Table.read(local)
    sequence, centers = tables[""], tables["_radec"]
    return {"ra": np.asarray(centers["ra"], dtype=float), "dec": np.asarray(centers["dec"], dtype=float),
            "band": np.char.strip(np.asarray(sequence["filter"]).astype(str)),
            "mjd": np.asarray(sequence["date"], dtype=float), "exptime": np.asarray(sequence["exptime"], dtype=float)}


def list_roman_images(band: str, survey: str = "TDS", release: str = "preview", kind: str = "simple_model",
                      cache_dir=None) -> dict[tuple[int, int], str]:
    """Return the archived images of one band as {(pointing, detector): path}, cached as JSON.

    ``kind="simple_model"`` selects calibrated images with noise, and ``kind="truth"`` the
    noiseless scenes. Detectors are numbered 1 to 18.
    """
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}")
    cache_dir = Path(cache_dir or get_cache_path("roman"))
    index_path = cache_dir / f"index_{survey}_{release}_{kind}_{band}.json"
    if index_path.exists():
        print(f"Reading from {index_path}...")
        return {tuple(map(int, key.split("_"))): path for key, path in json.loads(index_path.read_text()).items()}
    prefix = f"{_survey_root(survey, release)}/images/{kind}/{band}"
    index = {}
    for path in _filesystem().find(prefix):
        if path.endswith(".fits.gz"):
            pointing, detector = Path(path).name.removesuffix(".fits.gz").split("_")[-2:]
            index[(int(pointing), int(detector))] = path
    print(f"Writing to {index_path}...")
    index_path.write_text(json.dumps({f"{p}_{d}": path for (p, d), path in index.items()}))
    return index


def find_roman_images(target, bands=BANDS, survey: str = "TDS", release: str = "preview",
                      kind: str = "simple_model", mjd_range: tuple[float, float] | None = None,
                      cache_dir=None) -> list[dict]:
    """Return every archived exposure whose detector may contain a target, in time order.

    ``mjd_range`` keeps exposures between two modified Julian dates [day]. Each record holds
    ``band``, ``pointing``, ``detector``, ``mjd`` [day], ``exptime`` [s], ``separation`` [deg]
    from the detector center, and the archive ``path``.
    """
    coordinate = resolve_coordinate(target)
    pointings = load_roman_pointings(survey, release, cache_dir)
    centers = SkyCoord(pointings["ra"].ravel(), pointings["dec"].ravel(), unit="deg")
    separation = centers.separation(coordinate).deg.reshape(pointings["ra"].shape)  # [deg]
    records = []
    for band in np.atleast_1d(bands):
        index = list_roman_images(band, survey, release, kind, cache_dir)
        for pointing, detector in zip(*np.nonzero((separation < DETECTOR_RADIUS)
                                                  & (pointings["band"][:, None] == band))):
            path = index.get((int(pointing), int(detector) + 1))
            mjd = float(pointings["mjd"][pointing])  # [day]
            if path is not None and (mjd_range is None or mjd_range[0] <= mjd <= mjd_range[1]):
                records.append({"band": str(band), "pointing": int(pointing), "detector": int(detector) + 1,
                                "mjd": mjd, "exptime": float(pointings["exptime"][pointing]),
                                "separation": float(separation[pointing, detector]), "path": path})
    return sorted(records, key=lambda record: record["mjd"])


def _read_header(stream) -> fits.Header:
    """Read one FITS header, block by block, from a decompressing stream."""
    raw = b""
    while True:
        block = stream.read(FITS_BLOCK)
        if len(block) < FITS_BLOCK:
            raise EOFError("FITS stream ended inside a header")
        raw += block
        if any(block[start:start + 80] == b"END" + b" " * 77 for start in range(0, FITS_BLOCK, 80)):
            return fits.Header.fromstring(raw.decode("ascii"))


def _stream_rows(path: str, header_only: bool, row_stop: int = 0) -> tuple[fits.Header, fits.Header, np.ndarray]:
    """Return the primary and science headers and science rows [0, row_stop) of a remote image."""
    with _filesystem().open(path, "rb", block_size=2**20, cache_type="readahead") as remote:
        with gzip.GzipFile(fileobj=remote) as stream:
            primary = _read_header(stream)
            science = _read_header(stream)
            if header_only:
                return primary, science, np.empty((0, science["NAXIS1"]))
            dtype = np.dtype(f">f{abs(science['BITPIX']) // 8}")
            width = science["NAXIS1"]
            buffer = stream.read(row_stop * width * dtype.itemsize)
            return primary, science, np.frombuffer(buffer, dtype=dtype).reshape(row_stop, width).astype(float)


def _roman_metadata(record: dict, primary: fits.Header) -> dict:
    return {key: record[key] for key in ("band", "pointing", "detector", "path") if key in record} | {
        "exptime": primary.get("EXPTIME"), "zero_point_ab": primary.get("ZPTMAG")}


def read_roman_image(record, cache_dir=None, keep_file: bool = True) -> ImageStack:
    """Read one full Roman detector image (4088 by 4088 pixels) with its errors and pixel flags.

    ``record`` is an entry of :func:`find_roman_images` or an archive path. Pixel values are
    count rates [counts/s], the science array divided by the exposure time. With
    ``keep_file`` the compressed file is kept in ``cache_dir`` for later reads.
    """
    record = {"path": record} if isinstance(record, str) else dict(record)
    cache_dir = Path(cache_dir or get_cache_path("roman"))
    local = cache_dir / Path(record["path"]).name
    if not local.exists():
        print(f"Writing to {local}...")
        _filesystem().get(record["path"], str(local))
    print(f"Reading from {local}...")
    with fits.open(local) as hdul:
        primary = hdul[0].header
        exptime = primary["EXPTIME"]  # [s]
        stack = ImageStack(
            flux=hdul["SCI"].data / exptime, time=[primary["MJD-OBS"]], error=hdul["ERR"].data / exptime,
            wcs=_quiet_wcs(hdul["SCI"].header), unit="counts/s", time_format="MJD", mission="Roman",
            band=primary["FILTER"].strip(), label=f"Roman WFI {primary['FILTER'].strip()} SCA{primary['SCA_NUM']:02d}",
            metadata=_roman_metadata(record, primary) | {"pixel_quality": np.asarray(hdul["DQ"].data)},
        )
    if not keep_file:
        local.unlink()
    return stack


def get_roman_cutout(target, band: str = "J129", size: int = 41, max_frames: int | None = None,
                     mjd_range: tuple[float, float] | None = None, align: bool = True, survey: str = "TDS",
                     release: str = "preview", kind: str = "simple_model", cache_dir=None) -> ImageStack:
    """Return a time series of ``size`` by ``size`` pixel Roman cutouts centered on a target.

    Every archived exposure of the band that contains the target, optionally between the
    modified Julian dates in ``mjd_range`` [day], contributes one frame, streamed from the
    archive up to the last needed detector row. Consecutive exposures come from different
    pointings, detectors, and roll angles. With ``align`` every frame is resampled onto one
    north-up grid at the native 0.11 arcsec pixel scale, so frames can be differenced and
    animated; otherwise frames stay on their detector grids and their WCSs are kept in
    ``metadata["frame_wcs"]``. Values are count rates [counts/s].
    """
    coordinate = resolve_coordinate(target)
    records = find_roman_images(coordinate, (band,), survey, release, kind, mjd_range, cache_dir)
    # a margin of a factor sqrt(2) leaves enough pixels for any roll angle of the aligned grid
    half = int(np.ceil(size * np.sqrt(2.0) / 2.0)) + 2 if align else size // 2
    grid = tangent_plane_wcs(coordinate, size, PIXEL_SCALE) if align else None
    flux, time, frame_wcs, kept = [], [], [], []
    print(f"Streaming up to {len(records)} Roman {band} exposures for a {size} by {size} pixel cutout...")
    for record in records:
        if max_frames is not None and len(flux) >= max_frames:
            break
        primary, science, _ = _stream_rows(record["path"], header_only=True)
        wcs = _quiet_wcs(science)
        column, row = (int(round(float(value))) for value in wcs.world_to_pixel(coordinate))
        # the detector-center search is approximate, so skip exposures where the cutout leaves the detector
        if not (half <= column < science["NAXIS1"] - half and half <= row < science["NAXIS2"] - half):
            continue
        _, _, rows = _stream_rows(record["path"], header_only=False, row_stop=row + half + 1)
        cutout = rows[row - half:row + half + 1, column - half:column + half + 1] / primary["EXPTIME"]
        cutout_wcs = wcs[row - half:row + half + 1, column - half:column + half + 1]
        flux.append(resample_image(cutout, cutout_wcs, grid, (size, size)) if align else cutout)
        time.append(primary["MJD-OBS"])
        frame_wcs.append(grid if align else cutout_wcs)
        kept.append(record)
    if not flux:
        raise FileNotFoundError(f"No archived Roman {band} exposure contains {coordinate.to_string('hmsdms')}")
    return ImageStack(flux=np.array(flux), time=time, wcs=frame_wcs[0], unit="counts/s", time_format="MJD",
                      mission="Roman", band=band, label=f"Roman WFI {band} cutout",
                      metadata={"frame_wcs": frame_wcs, "records": kept, "survey": survey, "kind": kind,
                                "aligned": align})
