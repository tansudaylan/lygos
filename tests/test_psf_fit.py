import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
import os

from lygos import main as lygos_main
from pcat.fixed import sample_posterior as pcat_sample_posterior


@pytest.mark.parametrize('indexing', ['xy', 'ij'])
@pytest.mark.parametrize('multi_frame', [False, True])
def test_polynomial_psf_uses_supplied_coefficients_and_preserves_background(indexing, multi_frame):
    side = 9
    coordinates = np.arange(side, dtype=float)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates, indexing=indexing)
    coefficients = np.array([0.001, 0.002, 0.003, 0.004, 0.005,
                             0.006, 0.007, 0.008, 0.009, 3.0, 2.0, 1.5])
    xpos, ypos = np.array([4.2]), np.array([4.4])
    flux = np.array([[2.0], [5.0]]) if multi_frame else np.array([2.0])
    background = np.array([0.5, 1.5]) if multi_frame else np.array(0.5)
    originals = [values.copy() for values in (coefficients, xpos, ypos, flux, background)]
    state = SimpleNamespace(
        fitt=SimpleNamespace(typepsfnshap='pfre'), numbside=np.array([side]),
        numbsideevalhalf=3, xposimag=[x_grid], yposimag=[y_grid], typepsfnsubp='eval',
    )
    expected = np.zeros((side, side, 2) if multi_frame else (side, side)) + background
    powers = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2),
              (2, 1), (1, 2), (3, 0), (0, 3)]
    for row in range(1, 6):
        for column in range(1, 6):
            dx = x_grid[row, column] - xpos[0]
            dy = y_grid[row, column] - ypos[0]
            profile = sum(value * dx**power_x * dy**power_y
                          for value, (power_x, power_y) in zip(coefficients[:9], powers))
            profile += coefficients[9] * np.exp(-dx**2 / coefficients[10] - dy**2 / coefficients[11])
            expected[row, column] += profile * (flux[:, 0] if multi_frame else flux[0])

    actual = lygos_main.retr_cntpmodl(
        state, 0, 'fitt', xpos, ypos, flux, background, coefficients, 'pnts',
    )

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    for values, original in zip((coefficients, xpos, ypos, flux, background), originals):
        np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize('multi_frame', [False, True])
@pytest.mark.parametrize('source_count', [0, 1, 2])
def test_cubic_psf_model_preserves_source_frame_axes_and_background(multi_frame, source_count):
    side = 3
    coefficients = np.arange(10 * side * side, dtype=float).reshape(10, -1) / 1000.0
    coefficients[0] += 1.0
    xpos = np.array([1.2, 1.7])[:source_count]
    ypos = np.array([1.6, 1.1])[:source_count]
    flux = (np.array([[2.0, 5.0], [3.0, 7.0], [11.0, 13.0]])[:, :source_count]
            if multi_frame else np.array([2.0, 5.0])[:source_count])
    background = np.array([0.5, 1.5, 2.5]) if multi_frame else np.array(0.5)
    originals = [values.copy() for values in (coefficients, xpos, ypos, flux, background)]
    state = SimpleNamespace(
        fitt=SimpleNamespace(typepsfnshap='gauscirc'),
        numbside=np.array([side]), typepsfnsubp='regrcubipixl',
        coefspix=coefficients, booldiag=True,
    )
    shape = (side, side, 3) if multi_frame else (side, side)
    expected = np.zeros(shape) + background
    for source in range(source_count):
        dx = xpos[source] - int(xpos[source]) - 0.5
        dy = ypos[source] - int(ypos[source]) - 0.5
        basis = [1.0, dx, dy, dx**2, dx * dy, dy**2,
                 dx**3, dx**2 * dy, dx * dy**2, dy**3]
        profile = sum(value * coefficients[term].reshape(side, side)
                      for term, value in enumerate(basis))
        if multi_frame:
            for frame in range(3):
                expected[:, :, frame] += flux[frame, source] * profile
        else:
            expected += flux[source] * profile

    actual = lygos_main.retr_cntpmodl(
        state, 0, 'fitt', xpos, ypos, flux, background, np.array([1.0]), 'pnts',
    )

    assert actual.shape == shape
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    for values, original in zip((coefficients, xpos, ypos, flux, background), originals):
        np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize('indexing', ['xy', 'ij'])
@pytest.mark.parametrize('source_x', [4.0, 5.0])
@pytest.mark.parametrize('multi_frame', [False, True])
@pytest.mark.parametrize('psf_shape', ['empi', 'gauscirc'])
def test_psf_interpolation_tracks_source_position_and_frame_flux(indexing, source_x, multi_frame, psf_shape):
    side = 9
    coordinate = np.arange(side, dtype=float)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate, indexing=indexing)
    psf_x = x_grid[2:7, 2:7]
    psf_y = y_grid[2:7, 2:7]
    psf = np.exp(-0.5 * ((psf_x - 4.0)**2 + (psf_y - 4.0)**2))
    gdat = SimpleNamespace(
        fitt=SimpleNamespace(typepsfnshap=psf_shape),
        numbside=np.array([side]), numbsideevalhalf=3, numbsidepsfnfitt=5,
        xposimag=[x_grid], yposimag=[y_grid],
        xposimagpsfn=[psf_x], yposimagpsfn=[psf_y],
        typepsfnsubp='eval',
    )
    flux = np.array([[20.0], [40.0]]) if multi_frame else np.array([20.0])
    background = np.array([2.0, 3.0]) if multi_frame else 2.0
    model = lygos_main.retr_cntpmodl(
        gdat, 0, 'fitt', np.array([source_x]), np.array([4.0]),
        flux, background, psf.ravel() if psf_shape == 'empi' else np.array([1.0]), 'pnts',
    )
    pixel = (4, int(source_x)) if indexing == 'xy' else (int(source_x), 4)
    np.testing.assert_allclose(model[pixel], [22.0, 43.0] if multi_frame else 22.0, atol=1e-12)
    assert model.shape == ((side, side, 2) if multi_frame else (side, side))
    assert np.isfinite(model).all()
    np.testing.assert_allclose(model[0, 0], background, atol=1e-12)


def test_epic_rectangular_grid_preserves_image_orientation_and_zero_fill():
    from tdpy.numerics import RectilinearInterpolator2D

    image = np.arange(30.0).reshape(5, 6)
    x_grid, y_grid = lygos_main._retr_epic_image_grid(image)
    assert x_grid.shape == y_grid.shape == image.shape
    np.testing.assert_array_equal(x_grid[0], [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
    np.testing.assert_array_equal(y_grid[:, 0], [-4.0, -2.0, 0.0, 2.0, 4.0])
    interpolate = RectilinearInterpolator2D(x_grid, y_grid, image)
    np.testing.assert_allclose(interpolate(x_grid[0], y_grid[:, 0]), image, atol=1e-12)
    np.testing.assert_array_equal(interpolate([-10.0], y_grid[:, 0]), np.zeros((5, 1)))


def test_psf_image_inference_uses_pcat_sampler():
    assert lygos_main.sample_posterior is pcat_sample_posterior


def test_psf_callbacks_use_explicit_instrument_and_write_scalar_medians(tmp_path, monkeypatch):
    evaluated_instruments = []
    gdat = SimpleNamespace(
        typefittpsfnposi='fixd',
        fitt=SimpleNamespace(catl={'xpos': np.array([2.0]), 'ypos': np.array([2.0])}),
        indxparapsfnflux=np.array([0]),
        indxparapsfnback=np.array([1]),
        indxparapsfnpsfn=np.array([2]),
        numbside=np.array([3, 5]),
        booldiag=False,
        typesour='pnts',
        cntpdatasexp=np.ones((5, 5)),
    )

    def model(gdat, instrument_index, *args):
        evaluated_instruments.append(instrument_index)
        return np.ones((gdat.numbside[instrument_index], gdat.numbside[instrument_index]))

    monkeypatch.setattr(lygos_main, 'retr_cntpmodl', model)
    monkeypatch.setattr(lygos_main, 'retr_raticonttotl', lambda *args: 0.25)
    likelihood, derived_parameters = lygos_main._bind_psf_fit_callbacks(1)
    parameters = np.array([1.0, 0.0, 1.0])

    assert likelihood(parameters, gdat) == 0.0
    derived = derived_parameters(parameters, gdat)

    assert evaluated_instruments == [1, 1, 1]
    assert derived['cntpmodlsexp'].shape == (5, 5)
    assert derived['fraccent'] == 1.0 / 25.0

    path = tmp_path / 'posterior_medians.csv'
    medians = {'sigmpsfnxpos': 1.0, 'sigmpsfnypos': 1.2,
               'fracskewpsfnxpos': 0.1, 'fracskewpsfnypos': -0.1}
    lygos_main._write_psf_posterior_medians(medians, path)
    table = pd.read_csv(path)
    assert len(table) == 1
    np.testing.assert_allclose(
        lygos_main._retr_psf_parameters_from_medians('gauselli', table.iloc[0].to_dict()),
        [1.0, 1.2, 0.1, -0.1],
    )


def test_psf_fit_runs_against_real_image_model_for_nonzero_instrument(tmp_path):
    side = 7
    coordinate = np.arange(side, dtype=float)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate, indexing='ij')
    gdat = SimpleNamespace(
        typefittpsfnposi='fixd',
        fitt=SimpleNamespace(
            catl={'xpos': np.array([3.0]), 'ypos': np.array([3.0])},
            typepsfnshap='gauscirc',
        ),
        indxparapsfnflux=np.array([0]),
        indxparapsfnback=np.array([1]),
        indxparapsfnpsfn=np.array([2]),
        numbside=np.array([5, side]),
        numbsideevalhalf=3,
        xposimag=[x_grid[:5, :5], x_grid],
        yposimag=[y_grid[:5, :5], y_grid],
        booldiag=False,
        typepsfnsubp='eval',
        typesour='pnts',
        typeverb=-1,
    )
    truth = np.array([100.0, 2.0, 0.8])
    gdat.cntpdatasexp = lygos_main.retr_cntpmodl(
        gdat, 1, 'fitt', gdat.fitt.catl['xpos'], gdat.fitt.catl['ypos'],
        truth[[0]], truth[[1]], truth[[2]], 'pnts',
    )
    likelihood, derived = lygos_main._bind_psf_fit_callbacks(1)

    samples = lygos_main.sample_posterior(
        gdat,
        8,
        likelihood,
        ('source_flux', 'background', 'sigma'),
        (['Source flux', 'count'], ['Background', 'count'], ['PSF width', 'pixel']),
        ('self', 'self', 'self'),
        np.array([50.0, 0.0, 0.5]),
        np.array([150.0, 5.0, 1.2]),
        pathbase=str(tmp_path) + os.sep,
        boolforcrepr=True,
        boolplot=False,
        numbsamppostwalk=8,
        numbsampburnwalk=2,
        retr_dictderi=derived,
        booltqdm=False,
        booldiag=False,
        strgextn='synthetic_psf',
        typeverb=-1,
    )

    assert np.isfinite(samples['source_flux']).all()
    assert np.isfinite(samples['cntpmodlsexp']).all()
    assert samples['cntpmodlsexp'].shape[1:] == (side, side)