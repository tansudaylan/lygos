import numpy as np
import pytest

from lygos.tess import plot_tess_sky_coverage


def test_tess_sky_coverage_estimates_cumulative_unique_area(monkeypatch, tmp_path):
    import lygos.tess as tess

    def mocked_visibility(targets, sectors):
        return {
            target_name: (sector,)
            for index, target_name in enumerate(targets)
            if (sector := (index // 8 + 1)) <= 3 and index < 24
        } | {
            target_name: ()
            for index, target_name in enumerate(targets)
            if index >= 24
        }

    monkeypatch.setattr(tess, "tess_target_visibility", mocked_visibility)
    output_path = tmp_path / "coverage.png"
    result = plot_tess_sky_coverage((1, 2, 3), output_path, sample_count=32)

    total_sky_area_deg2 = 4.0 * np.pi * (180.0 / np.pi) ** 2
    np.testing.assert_array_equal(result["sector_numbers"], (1, 2, 3))
    np.testing.assert_allclose(
        result["cumulative_area_deg2"], total_sky_area_deg2 * np.array((0.25, 0.5, 0.75))
    )
    np.testing.assert_array_equal(result["sample_sector_counts"], [1] * 24 + [0] * 8)
    assert np.all(result["area_uncertainty_deg2"] > 0.0)
    assert output_path.is_file()


@pytest.mark.parametrize("sectors", [(), (2, 1), (1, 1), (0, 1)])
def test_tess_sky_coverage_rejects_invalid_sector_order(sectors, tmp_path):
    with pytest.raises(ValueError):
        plot_tess_sky_coverage(sectors, tmp_path / "coverage.png", sample_count=32)


def test_tess_sky_coverage_requires_enough_samples(tmp_path):
    with pytest.raises(ValueError, match="at least 32"):
        plot_tess_sky_coverage((1,), tmp_path / "coverage.png", sample_count=31)