#!/usr/bin/env python3
"""Estimate cumulative TESS sky coverage and animate dated sector footprints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tdpy.tess import PLANNED_SECTORS, animate_tess_sectors
from tdpy.verbosity import print

import lygos

EXAMPLE_PATH = Path(__file__).resolve().parent
VISUAL_PATH = EXAMPLE_PATH / "visuals"
TARGETS = {"TOI-1233": (186.574548, -51.362837)}


def run_example(
    sectors=range(1, 5),
    animation_sectors=None,
    sample_count: int = 1024,
    duration_ms: int = 300,
    typefileplot: str = "png",
) -> dict[str, object]:
    """Write a coverage map and animation; coverage samples the requested sectors."""
    sectors = tuple(int(sector) for sector in sectors)
    animation_sectors = (
        tuple(range(1, max(PLANNED_SECTORS) + 1))
        if animation_sectors is None
        else tuple(int(sector) for sector in animation_sectors)
    )
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_tag = run_stamp
    suffix = 2
    while (
        VISUAL_PATH / f"tess_cumulative_sky_coverage_{run_tag}.{typefileplot}"
    ).exists() or (VISUAL_PATH / f"tess_sector_timeline_{run_tag}.gif").exists():
        run_tag = f"{run_stamp}_{suffix:02d}"
        suffix += 1
    coverage = lygos.plot_tess_sky_coverage(
        sectors,
        VISUAL_PATH / f"tess_cumulative_sky_coverage_{run_tag}",
        sample_count=sample_count,
        typefileplot=typefileplot,
    )
    animation_path = animate_tess_sectors(
        animation_sectors,
        VISUAL_PATH / f"tess_sector_timeline_{run_tag}.gif",
        targets=TARGETS,
        highlighted_targets=("TOI-1233",),
        duration_ms=duration_ms,
    )
    final_area = coverage["cumulative_area_deg2"][-1]
    final_uncertainty = coverage["area_uncertainty_deg2"][-1]
    print(f"Cumulative sky area after Sector {sectors[-1]}: {final_area:.0f} +/- {final_uncertainty:.0f} deg^2")
    return {"coverage": coverage, "animation_path": animation_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sectors", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--animation-sectors", type=int, nargs="+")
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--duration-ms", type=int, default=300)
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()
    run_example(
        sectors=arguments.sectors,
        animation_sectors=arguments.animation_sectors,
        sample_count=arguments.sample_count,
        duration_ms=arguments.duration_ms,
        typefileplot=arguments.typefileplot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())