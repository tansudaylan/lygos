"""Offline tests of the lygos imaging layer on synthetic images; network tests need LYGOS_TEST_NETWORK=1."""

import gzip
import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from PIL import Image

import lygos
from lygos import roman
from lygos.imagestack import ImageStack, resample_image, tangent_plane_wcs

CENTER = SkyCoord(150.0, 2.0, unit="deg")


def gaussian_scene(size=21, center=(10.0, 10.0), amplitude=100.0, sigma=1.5, sky=10.0):
    rows, columns = np.indices((size, size))
    return sky + amplitude * np.exp(-0.5 * ((columns - center[0]) ** 2 + (rows - center[1]) ** 2) / sigma**2)


def synthetic_stack(number=30, size=21, seed=0):
    """A constant star at the center, a sinusoidally variable star off center, and Poisson noise."""
    random = np.random.default_rng(seed)
    time = np.linspace(0.0, 3.0, number)  # [day]
    flux = np.stack([gaussian_scene(size) + gaussian_scene(size, (4.0, 15.0), 60.0 * (1.0 + 0.5 * np.sin(2 * np.pi * t)), sky=0.0)
                     for t in time])
    flux = random.poisson(flux).astype(float)
    quality = np.zeros(number, dtype=int)
    quality[3] = 1
    return ImageStack(flux=flux, time=time, error=np.sqrt(flux), quality=quality,
                      wcs=tangent_plane_wcs(CENTER, size, 21.0), unit="e-/s", time_format="BTJD",
                      mission="TESS", label="synthetic")


def test_stack_selection_quality_and_concatenation():
    stack = synthetic_stack()
    assert stack.shape == (30, 21, 21)
    assert len(stack.good()) == 29
    halves = stack.select(slice(15, None)).concatenate(stack.select(slice(0, 15)))
    np.testing.assert_allclose(halves.time, stack.time)
    with pytest.raises(ValueError):
        ImageStack(flux=np.zeros((3, 4, 4)), time=[0.0, 1.0])


def test_background_difference_and_variability_isolate_the_variable_star():
    stack = synthetic_stack()
    background_free = stack.subtract_background()
    assert abs(np.median(background_free.flux)) < 2.0
    variability = stack.variability_map()
    # the variable star at column 4, row 15 is far more variable than the constant star at the center
    assert variability[15, 4] > 5.0 * variability[10, 10]
    difference = stack.difference()
    assert abs(np.median(difference.flux[:, 10, 10])) < 5.0


def test_relative_variability_products_use_good_frames_and_ignore_invalid_pixels():
    flux = np.array([
        [[9.0, 20.0], [30.0, np.nan]],
        [[10.0, 20.0], [33.0, np.nan]],
        [[11.0, 20.0], [27.0, np.nan]],
        [[100.0, 40.0], [20.0, np.nan]],
    ])
    stack = ImageStack(flux=flux, time=np.arange(4), quality=[0, 0, 0, 1])

    products = lygos.compute_variability_products(stack)

    np.testing.assert_allclose(products["temporal_median"][:2, :1], [[10.0], [30.0]])
    np.testing.assert_allclose(products["relative_variability_percent"][:2, :1], [[14.826], [14.826]])
    assert products["relative_variability_percent"][0, 1] == 0.0
    assert np.isnan(products["temporal_median"][1, 1])
    np.testing.assert_allclose(stack.relative_variability_map(), products["relative_variability_percent"])


def test_relative_variability_plot_writes_nonblank_image(tmp_path):
    time = np.arange(5, dtype=float)
    flux = np.full((5, 3, 3), 100.0)
    flux[:, 1, 2] = np.array([80.0, 100.0, 120.0, 100.0, 80.0])
    stack = ImageStack(flux=flux, time=time, time_format="BTJD", label="Test stack")
    output_path = tmp_path / "variability.png"

    products = lygos.plot_variability_summary(stack, output_path)

    image = Image.open(output_path)
    assert image.width > 100 and image.height > 100
    assert np.nanargmax(products["relative_variability_percent"]) == 5


def test_relative_variability_rejects_fewer_than_two_good_frames():
    stack = ImageStack(flux=np.ones((2, 3, 3)), time=[0.0, 1.0], quality=[0, 1])

    with pytest.raises(ValueError, match="two good time samples"):
        lygos.compute_variability_products(stack)


def test_aperture_photometry_recovers_injected_flux_and_variability():
    stack = synthetic_stack().subtract_background()
    aperture = stack.circular_aperture(5.0)
    light_curve = stack.aperture_photometry(aperture)
    # a radius of 3.3 sigma encloses 99.6% of a Gaussian star of total flux 2 pi sigma^2 amplitude
    expected = 2.0 * np.pi * 1.5**2 * 100.0  # [e-/s pixel]
    assert np.median(light_curve["flux"]) == pytest.approx(expected, rel=0.05)
    variable = stack.aperture_photometry(stack.circular_aperture(4.0, center=(4.0, 15.0)))
    assert np.ptp(variable["flux"]) / np.median(variable["flux"]) > 0.6
    assert np.all(light_curve["flux_error"] > 0.0)
    threshold = stack.threshold_aperture(5.0)
    assert threshold[10, 10] and not threshold[0, 0]


def test_centroid_and_time_binning():
    stack = synthetic_stack().subtract_background()
    column, row = stack.centroid(stack.circular_aperture(4.0))
    np.testing.assert_allclose(np.median(column), 10.0, atol=0.1)
    np.testing.assert_allclose(np.median(row), 10.0, atol=0.1)
    binned = stack.bin_time(0.5)
    assert len(binned) < len(stack) and np.all(np.diff(binned.time) > 0.0)


def test_resampling_preserves_identity_and_tracks_sky_offsets():
    image = gaussian_scene(31, (15.0, 15.0), sky=0.0)
    grid = tangent_plane_wcs(CENTER, 31, 0.11)
    np.testing.assert_allclose(resample_image(image, grid, grid, (31, 31)), image, atol=1e-6)
    # on a north-up, east-left grid centered 2 pixels east of the star, the star moves 2 pixels west (right)
    offset = 2 * 0.11 / 3600.0 / np.cos(CENTER.dec.radian)  # [deg] of right ascension
    east = SkyCoord(CENTER.ra.deg + offset, CENTER.dec.deg, unit="deg")
    shifted = resample_image(image, grid, tangent_plane_wcs(east, 31, 0.11), (31, 31))
    rows, columns = np.indices(shifted.shape)
    weight = np.nan_to_num(shifted)
    assert (weight * columns).sum() / weight.sum() == pytest.approx(17.0, abs=0.05)


def test_read_tess_cutout_parses_tesscut_files(tmp_path):
    stack = synthetic_stack(number=5)
    columns = fits.ColDefs([
        fits.Column(name="TIME", format="D", array=np.r_[stack.time[:4], np.nan]),
        fits.Column(name="FLUX", format="441E", dim="(21, 21)", array=stack.flux),
        fits.Column(name="FLUX_ERR", format="441E", dim="(21, 21)", array=stack.error),
        fits.Column(name="FLUX_BKG", format="441E", dim="(21, 21)", array=np.zeros_like(stack.flux)),
        fits.Column(name="QUALITY", format="J", array=stack.quality),
    ])
    primary = fits.PrimaryHDU(header=fits.Header({"SECTOR": 14, "CAMERA": 2, "CCD": 3}))
    aperture = fits.ImageHDU(np.zeros((21, 21), dtype=np.int32), header=stack.wcs.to_header(), name="APERTURE")
    path = tmp_path / "tess-s0014-2-3_cutout.fits"
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns, name="PIXELS"), aperture]).writeto(path)
    cutout = lygos.read_tess_cutout(path)
    # the frame without a time stamp is dropped
    assert cutout.shape == (4, 21, 21)
    assert cutout.metadata["sector"] == 14 and cutout.time_format == "BTJD"
    column, row = cutout.pixel_of(CENTER)
    assert column == pytest.approx(10.0, abs=1e-6) and row == pytest.approx(10.0, abs=1e-6)


def write_roman_file(path, science, exptime=300.0):
    """Write a gzipped file shaped like an OpenUniverse 2024 Roman image."""
    header = tangent_plane_wcs(CENTER, science.shape[0], 0.11).to_header()
    primary = fits.PrimaryHDU(header=fits.Header({"EXPTIME": exptime, "MJD-OBS": 62000.5, "FILTER": "J129",
                                                  "SCA_NUM": 7, "ZPTMAG": 26.0}))
    hdul = fits.HDUList([primary, fits.ImageHDU(science * exptime, header=header, name="SCI"),
                         fits.ImageHDU(np.ones_like(science, dtype=np.float32), header=header, name="ERR"),
                         fits.ImageHDU(np.zeros(science.shape, dtype=np.int32), name="DQ")])
    with gzip.open(path, "wb") as stream:
        hdul.writeto(stream)


class LocalFilesystem:
    def open(self, path, mode="rb", **kwargs):
        return open(path, mode)

    def get(self, source, destination):
        import shutil

        shutil.copyfile(source, destination)


def test_roman_streaming_cutout_matches_the_full_image(tmp_path, monkeypatch):
    science = gaussian_scene(201, (100.0, 60.0), amplitude=5.0, sky=0.4)  # [counts/s]
    path = tmp_path / "Roman_TDS_simple_model_J129_10_7.fits.gz"
    write_roman_file(path, science)
    monkeypatch.setattr(roman, "_filesystem", LocalFilesystem)
    _, header, rows = roman._stream_rows(str(path), header_only=False, row_stop=80)
    assert header["NAXIS1"] == 201 and rows.shape == (80, 201)
    np.testing.assert_allclose(rows, science[:80] * 300.0)
    record = {"band": "J129", "pointing": 10, "detector": 7, "mjd": 62000.5, "exptime": 300.0,
              "separation": 0.0, "path": str(path)}
    monkeypatch.setattr(roman, "find_roman_images", lambda *args, **kwargs: [record])
    target = tangent_plane_wcs(CENTER, 201, 0.11).pixel_to_world(100, 60)
    stack = roman.get_roman_cutout(target, size=21, align=False)
    np.testing.assert_allclose(stack.flux[0], science[50:71, 90:111])
    aligned = roman.get_roman_cutout(target, size=21, align=True)
    assert np.nanargmax(aligned.flux[0]) == np.ravel_multi_index((10, 10), (21, 21))
    full = roman.read_roman_image(record, cache_dir=tmp_path / "cache")
    np.testing.assert_allclose(full.flux[0], science)
    assert full.unit == "counts/s" and full.band == "J129"


def test_find_roman_images_selects_covering_detectors(monkeypatch):
    pointings = {"ra": np.array([[150.0, 151.0], [150.01, 152.0]]), "dec": np.array([[2.0, 2.0], [2.0, 2.0]]),
                 "band": np.array(["J129", "H158"]), "mjd": np.array([62010.0, 62000.0]),
                 "exptime": np.array([300.0, 300.0])}
    monkeypatch.setattr(roman, "load_roman_pointings", lambda *args, **kwargs: pointings)
    monkeypatch.setattr(roman, "list_roman_images",
                        lambda band, *args, **kwargs: {(0, 1): "a.fits.gz", (1, 1): "b.fits.gz"})
    records = roman.find_roman_images(CENTER, bands=("J129", "H158"))
    assert [(record["band"], record["pointing"], record["detector"]) for record in records] == [
        ("H158", 1, 1), ("J129", 0, 1)]
    assert roman.find_roman_images(CENTER, bands=("J129", "H158"), mjd_range=(62005.0, 62020.0))[0]["band"] == "J129"


def test_visualizations_write_images_and_animations(tmp_path):
    stack = synthetic_stack(number=8)
    aperture = stack.circular_aperture(3.0)
    light_curve = stack.aperture_photometry(aperture)
    for path in (lygos.plot_image(stack, tmp_path / "median", aperture=aperture, markers={"Star": (10.0, 10.0)}),
                 lygos.plot_image(stack, tmp_path / "variability", frame="variability", sky=False),
                 lygos.plot_mosaic(stack, tmp_path / "mosaic", difference=True),
                 lygos.plot_light_curve(light_curve, tmp_path / "light_curve")):
        assert os.path.isfile(path)
    animation = lygos.animate_stack(stack, tmp_path / "movie", aperture=aperture, light_curve=light_curve,
                                    difference=True)
    with Image.open(animation) as gif:
        assert gif.n_frames == len(stack)
    with pytest.raises(ValueError):
        lygos.image_normalization(np.full((3, 3), np.nan))


@pytest.mark.skipif(os.environ.get("LYGOS_TEST_NETWORK") != "1", reason="set LYGOS_TEST_NETWORK=1 to query archives")
def test_public_archives_return_tess_and_roman_images(tmp_path):
    cutout = lygos.get_tess_cutout((291.366304, 42.78435924), sector=14, size=5, cache_dir=tmp_path)[0]
    assert cutout.shape[1:] == (5, 5) and len(cutout) > 1000
    roman_stack = lygos.get_roman_cutout((9.6632, -43.88128), "J129", size=11, max_frames=2,
                                         cache_dir=tmp_path)
    assert roman_stack.shape == (2, 11, 11)
