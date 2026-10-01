#!/usr/bin/env python3
"""Retrieve, process, and animate a TESSCut time series of the RR Lyrae star RR Lyr.

Data: real TESS Sector 14 full-frame-image cutouts from the MAST TESSCut service
(Brasseur et al. 2019). RR Lyr pulsates with a 0.567 day period, which is clear in
every TESS orbit. The 15 by 15 pixel cutout covers 5.25 arcmin on a side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tdpy.cli import add_plot_arguments

import lygos

EXAMPLE_PATH = Path(__file__).resolve().parent
VISUAL_PATH = EXAMPLE_PATH / "visuals"
TARGET = (291.366304, 42.78435924)  # [deg], RR Lyr (ICRS)
SECTOR = 14
SIZE = 15  # [pixel]


def run_example(typefileplot: str = "png", max_frames: int = 120) -> dict:
    """Download the cutout, build an aperture light curve, and write figures and an animation."""
    stack = lygos.get_tess_cutout(TARGET, sector=SECTOR, size=SIZE, cache_dir=lygos.get_cache_path("tess"))[0]
    good = stack.good()
    aperture = good.threshold_aperture(threshold=5.0)
    light_curve = good.aperture_photometry(aperture)
    target_pixel = good.pixel_of(lygos.resolve_coordinate(TARGET))
    paths = [
        lygos.plot_image(good, VISUAL_PATH / "rrlyr_tesscut_median", aperture=aperture,
                         markers={"RR Lyr": target_pixel}, typefileplot=typefileplot),
        lygos.plot_light_curve(light_curve, VISUAL_PATH / "rrlyr_tesscut_light_curve",
                               title=f"RR Lyr, TESS Sector {SECTOR}, {int(aperture.sum())} pixel aperture",
                               typefileplot=typefileplot),
        lygos.plot_mosaic(good.select(slice(0, 120)), VISUAL_PATH / "rrlyr_tesscut_difference_mosaic",
                          difference=True, typefileplot=typefileplot),
        # the first 2.5 days at 10 minute cadence span four pulsation cycles
        lygos.animate_stack(good.select(slice(0, max_frames)), VISUAL_PATH / "rrlyr_tesscut_animation",
                            aperture=aperture, light_curve=good.select(slice(0, max_frames)).aperture_photometry(aperture),
                            title=f"RR Lyr, TESS Sector {SECTOR}"),
    ]
    return {"stack": good, "aperture": aperture, "light_curve": light_curve, "paths": paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    run_example(arguments.typefileplot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
