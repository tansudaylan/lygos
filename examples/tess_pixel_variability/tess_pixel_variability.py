#!/usr/bin/env python3
"""Map pixel-level variability in public TESS observations of pi Mensae."""

from __future__ import annotations

import argparse
from pathlib import Path

from tdpy.cli import add_plot_arguments

import lygos

EXAMPLE_PATH = Path(__file__).resolve().parent
VISUAL_PATH = EXAMPLE_PATH / "visuals"
TARGET = (84.291188, -80.469119)  # [deg], pi Mensae (ICRS)
SECTOR = 1


def run_example(typefileplot: str = "png") -> dict:
    """Retrieve a public TESS cutout and plot its variability products."""
    stack = lygos.get_tess_cutout(
        TARGET,
        sector=SECTOR,
        size=15,
        cache_dir=lygos.get_cache_path("tesscut"),
    )[0].good()
    output_path = VISUAL_PATH / f"tess_pixel_variability.{typefileplot}"
    products = lygos.plot_variability_summary(
        stack,
        output_path,
        title=f"TESS Sector {SECTOR} pixel variability around pi Mensae",
        typefileplot=typefileplot,
    )
    return {"stack": stack, "products": products, "path": output_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    run_example(arguments.typefileplot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())