#!/usr/bin/env python3
"""Read TESS full-frame images directly from the public MAST bucket and animate a region of them.

Data: real TESS Sector 14 calibrated full-frame images (FFIs) of camera 2, CCD 3, read
anonymously from ``s3://stpubdata/tess/public/ffi``. Only the pixels that are needed are
transferred. The example shows one full 2048 by 2048 pixel science frame, then cuts a
41 by 41 pixel region around RR Lyr from every fifth FFI of the first orbit and maps which
pixels vary in time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tdpy.cli import add_plot_arguments

import lygos

EXAMPLE_PATH = Path(__file__).resolve().parent
VISUAL_PATH = EXAMPLE_PATH / "visuals"
TARGET = (291.366304, 42.78435924)  # [deg], RR Lyr (ICRS)
SECTOR, CAMERA, CCD = 14, 2, 3
SIZE = 41  # [pixel]


def run_example(typefileplot: str = "png", max_frames: int = 120, stride: int = 5) -> dict:
    """Read one full frame and an FFI cutout time series, and write figures and an animation."""
    full_frame = lygos.get_tess_ffi(SECTOR, CAMERA, CCD, index=0)
    target = lygos.resolve_coordinate(TARGET)
    cutout = lygos.get_tess_ffi_cutout(TARGET, SECTOR, size=SIZE, max_frames=max_frames, stride=stride).good()
    aperture = cutout.threshold_aperture(threshold=5.0, center=cutout.pixel_of(target))
    light_curve = cutout.aperture_photometry(aperture)
    paths = [
        lygos.plot_image(full_frame, VISUAL_PATH / "tess_ffi_full_frame", frame=0,
                         markers={"RR Lyr": full_frame.pixel_of(target)}, typefileplot=typefileplot),
        lygos.plot_image(cutout, VISUAL_PATH / "tess_ffi_variability_map", frame="variability",
                         markers={"RR Lyr": cutout.pixel_of(target)}, cmap="viridis", stretch="linear",
                         typefileplot=typefileplot),
        lygos.animate_stack(cutout, VISUAL_PATH / "tess_ffi_cutout_animation", aperture=aperture,
                            light_curve=light_curve, sky=True,
                            title=f"TESS Sector {SECTOR} FFIs, camera {CAMERA}, CCD {CCD}"),
    ]
    return {"full_frame": full_frame, "cutout": cutout, "light_curve": light_curve, "paths": paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    run_example(arguments.typefileplot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
