import numpy as np
import pandas as pd
from types import SimpleNamespace
import os

from lygos import main as lygos_main


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