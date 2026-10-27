#!/usr/bin/env python3
"""Exercise the ULTRASAT-mode image workflow with illustrative expected counts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from miletos.simulation import generate_ultrasat_time_series
from tdpy.cli import add_plot_arguments
from tdpy.verbosity import print

import lygos
from lygos import ImageStack


EXAMPLE_PATH = Path(__file__).resolve().parent
VISUAL_PATH = EXAMPLE_PATH / "visuals"
DEFAULT_START_JD = 2462500.0  # [day]
ULTRASAT_CADENCE_SECONDS = 300.0  # [s]
ULTRASAT_EXPOSURE_SECONDS = 300.0  # [s]


def _target_slug(target: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", target).strip("_").lower()
    return slug or "target"


def simulate_target(
    target: str,
    duration_hours: float = 6.0,
    number_side: int = 21,
    maximum_catalog_magnitude: float = 16.0,
    start_jd: float = DEFAULT_START_JD,
    seed: int = 0,
    typefileplot: str = "png",
) -> dict[str, object]:
    """Simulate illustrative count-rate images for a MAST-resolvable target."""
    if not target.strip():
        raise ValueError("target must be a nonempty MAST-resolvable name")
    if not np.isfinite([duration_hours, maximum_catalog_magnitude, start_jd]).all():
        raise ValueError("duration, magnitude limit, and start time must be finite")
    if duration_hours <= 0.0 or number_side < 5 or number_side % 2 != 1:
        raise ValueError("duration must be positive and cutout side must be odd and at least 5 pixels")
    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")

    lygos.get_data_path()
    time_jd, cadence_seconds, exposure_seconds = generate_ultrasat_time_series(
        start_jd=start_jd,
        duration_days=duration_hours / 24.0,
        cadence_seconds=ULTRASAT_CADENCE_SECONDS,
        exposure_seconds=ULTRASAT_EXPOSURE_SECONDS,
    )
    slug = _target_slug(target)
    result = lygos.init(
        strgmast=target,
        labltarg=f"{target} (synthetic ULTRASAT field)",
        strgtarg=f"ultrasat_{slug}",
        liststrginst=["ULTRASAT"],
        liststrgtypedata=["simutargpartsynt"],
        listtime=[[time_jd]],
        timeexpo=[[exposure_seconds / 86400.0]],
        numbside=[number_side],
        maxmtmagcatl=maximum_catalog_magnitude,
        boolfitt=False,
        boolplot=False,
        boolplotcntp=False,
        boolplotrflx=False,
        boolanim=False,
        seedrand=seed,
    )

    count_rate_cube = np.asarray(result["cntpdata"], dtype=float)  # [e- s^-1 pixel^-1]
    simulated_times = np.asarray(result["listtime"][0][0], dtype=float)  # [day]
    exposure_seconds = float(result["timeexpo"][0][0]) * 86400.0  # [s]
    if not np.isfinite(exposure_seconds) or not np.isclose(exposure_seconds, ULTRASAT_EXPOSURE_SECONDS):
        raise RuntimeError("Lygos did not retain the requested ULTRASAT exposure")
    if count_rate_cube.shape != (number_side, number_side, simulated_times.size):
        raise RuntimeError("Lygos returned an ULTRASAT cube with an unexpected shape")
    if not np.isfinite(count_rate_cube).all() or not np.any(count_rate_cube > 0.0):
        raise RuntimeError("Lygos did not produce finite, positive ULTRASAT count rates")

    stack = ImageStack(
        flux=np.moveaxis(count_rate_cube, -1, 0),
        time=simulated_times,
        unit="e-/s",
        time_format="JD",
        mission="ULTRASAT",
        band="NUV 230-290 nm",
        label=f"{target} synthetic expected count rates",
        metadata={
            "target": target,
            "photometry": "synthetic expectation; no shot or read noise added",
            "cadence_seconds": cadence_seconds,
            "exposure_seconds": exposure_seconds,
            "band": "NUV 230-290 nm",
            "magnitude_conversion": "TIC Tmag with Lygos's ULTRASAT zeropoint; not an NUV spectral conversion",
            "instrument_model": "Illustrative pixel, PSF, and count-rate assumptions; not a mission response",
            "timing_model": "Nominal continuous 300-second exposures; no readout gaps or timing jitter modeled",
            "mission_observing_modes_url": "https://www.weizmann.ac.il/ultrasat/science-mission/modes-of-operation/modes-of-operation",
        },
    )
    catalog = result["refr"].catl[0][0][0]
    target_center = (float(catalog["xpos"][0]), float(catalog["ypos"][0]))
    aperture = stack.circular_aperture(radius=2.5, center=target_center)
    background_subtracted = stack.subtract_background()
    light_curve = background_subtracted.aperture_photometry(aperture)
    VISUAL_PATH.mkdir(parents=True, exist_ok=True)

    image_path = VISUAL_PATH / f"{slug}_ultrasat_median.{typefileplot}"
    curve_path = VISUAL_PATH / f"{slug}_ultrasat_aperture_light_curve.{typefileplot}"
    lygos.plot_image(
        stack,
        image_path,
        aperture=aperture,
        markers={target: target_center},
        typefileplot=typefileplot,
    )
    lygos.plot_light_curve(
        light_curve,
        curve_path,
        title=f"{target}, synthetic ULTRASAT, {int(aperture.sum())}-pixel aperture",
        typefileplot=typefileplot,
    )
    print(
        "Simulated {} expected count-rate images at {} s cadence with {} s exposure.".format(
            len(stack), ULTRASAT_CADENCE_SECONDS, ULTRASAT_EXPOSURE_SECONDS
        )
    )
    return {
        "stack": stack,
        "aperture": aperture,
        "light_curve": light_curve,
        "paths": (image_path, curve_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="Vega", help="MAST-resolvable target name.")
    parser.add_argument("--duration-hours", type=float, default=6.0)
    parser.add_argument("--number-side", type=int, default=21)
    parser.add_argument("--maximum-catalog-magnitude", type=float, default=16.0)
    parser.add_argument("--start-jd", type=float, default=DEFAULT_START_JD)
    parser.add_argument("--seed", type=int, default=0)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    simulate_target(
        arguments.target,
        duration_hours=arguments.duration_hours,
        number_side=arguments.number_side,
        maximum_catalog_magnitude=arguments.maximum_catalog_magnitude,
        start_jd=arguments.start_jd,
        seed=arguments.seed,
        typefileplot=arguments.typefileplot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())