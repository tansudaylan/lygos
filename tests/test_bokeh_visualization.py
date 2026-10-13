import numpy as np
import pytest

from lygos.imagestack import ImageStack
from lygos.variability import plot_variability_summary


def test_variability_summary_writes_bokeh_html(tmp_path):
    pytest.importorskip("bokeh")
    flux = np.full((4, 5, 5), 5.0)
    flux[:, 2, 2] = [5.0, 6.0, 7.0, 8.0]
    stack = ImageStack(
        flux=flux,
        time=np.arange(4.0),
        label="Synthetic stack",
        unit="e-/s",
    )

    output_path = tmp_path / "variability.html"
    products = plot_variability_summary(
        stack,
        output_path,
        title="Synthetic variability",
        typefileplot="html",
    )

    assert output_path.is_file()
    assert products["relative_variability_percent"].shape == (5, 5)
    html = output_path.read_text(encoding="utf-8")
    assert "Temporal median image" in html
    assert "Source-pixel variability map" in html
    assert "Pixel (2, 2) time series" in html