#!/usr/bin/env python3
"""Find, align, and animate Roman images of a simulated Type Ia supernova.

Data: the public OpenUniverse 2024 simulation of the Roman time-domain survey (TDS)
(OpenUniverse et al. 2025), read anonymously from the NASA/IPAC Infrared Science Archive
bucket on Amazon Web Services. Roman has not yet launched, so all images are simulated.
SN 20034224, a Type Ia supernova at redshift 0.26 with a simulated peak at MJD 62312, lies
on a compact host galaxy. lygos finds every J129 exposure covering it, streams 41 by 41
pixel (4.5 arcsec) cutouts, resamples them onto one north-up grid, and subtracts the median
image so that the supernova stands out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tdpy.cli import add_plot_arguments

import lygos

EXAMPLE_PATH = Path(__file__).resolve().parent
VISUAL_PATH = EXAMPLE_PATH / "visuals"
SUPERNOVA = (9.6632, -43.88128)  # [deg], SN 20034224 in the OpenUniverse 2024 SNANA catalog
PEAK_MJD = 62312.2  # [day]
BAND = "J129"
SIZE = 41  # [pixel]
WINDOW = (62150.0, 62550.0)  # [day], MJD range around the peak


def run_example(typefileplot: str = "png") -> dict:
    """Read one full detector, stream aligned cutouts, and write figures and a difference animation."""
    records = lygos.find_roman_images(SUPERNOVA, bands=(BAND,), mjd_range=WINDOW)
    # the full detector image of the exposure closest to the simulated peak
    peak = min(records, key=lambda record: abs(record["mjd"] - PEAK_MJD))
    detector = lygos.read_roman_image(peak)
    stack = lygos.get_roman_cutout(SUPERNOVA, BAND, size=SIZE, mjd_range=WINDOW).subtract_background()
    aperture = stack.circular_aperture(3.0)
    light_curve = stack.aperture_photometry(aperture)
    supernova = lygos.resolve_coordinate(SUPERNOVA)
    paths = [
        lygos.plot_image(detector, VISUAL_PATH / "roman_detector_image", frame=0,
                         markers={"SN 20034224": detector.pixel_of(supernova)}, typefileplot=typefileplot),
        lygos.plot_mosaic(stack, VISUAL_PATH / "roman_supernova_cutouts", typefileplot=typefileplot),
        lygos.plot_mosaic(stack, VISUAL_PATH / "roman_supernova_difference_mosaic", difference=True,
                          typefileplot=typefileplot),
        lygos.plot_light_curve(light_curve, VISUAL_PATH / "roman_supernova_light_curve",
                               title=f"SN 20034224, Roman {BAND}, 3 pixel radius aperture", normalize=False,
                               typefileplot=typefileplot),
        lygos.animate_stack(stack, VISUAL_PATH / "roman_supernova_difference_animation", aperture=aperture,
                            light_curve=light_curve, difference=True, duration_ms=500,
                            title=f"SN 20034224, Roman {BAND} minus median"),
    ]
    return {"detector": detector, "stack": stack, "light_curve": light_curve, "paths": paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    run_example(arguments.typefileplot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
