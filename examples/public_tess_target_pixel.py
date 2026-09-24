#!/usr/bin/env python3
"""Generate a public TESS target-pixel diagnostic for WASP-121."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import lygos
from lygos.paths import get_data_path


TARGET_TIC = 22529346
TESS_SECTOR = 7


def run_example(runtime_path: Path, output_path: Path) -> Path:
    """Run Lygos on the public Sector 7 SPOC target-pixel product."""

    lygos.init(
        liststrgtypedata=["obsd"],
        listtsecsele=[TESS_SECTOR],
        boolutiltpxf=True,
        strgmast=f"TIC {TARGET_TIC}",
        liststrginst=["TESS"],
        pathtarg=f"{runtime_path.resolve()}/",
        strgruns="public_tess_sector7",
        listnameanls=["aper"],
        boolfitt=False,
        boolplot=True,
        boolanim=False,
        typefileplot=output_path.suffix.lstrip("."),
        typeverb=1,
    )

    candidates = list(
        (runtime_path / "visuals" / "sexp").glob(
            f"cntpdatasexp_*TIC{TARGET_TIC}_TESS_0732_*{output_path.suffix}"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one generated target-pixel diagnostic, found {len(candidates)}."
        )

    print(f"Reading from {candidates[0]}...")
    print(f"Writing to {output_path}...")
    shutil.copyfile(candidates[0], output_path)
    return output_path


def main() -> None:
    """Parse output options and run the public-data example."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    parser.add_argument(
        "--runtime-path",
        type=Path,
        default=get_data_path() / "examples" / "wasp121_sector7",
        help="ignored directory for downloaded public products and pipeline outputs",
    )
    arguments = parser.parse_args()
    output_path = Path(__file__).with_name(
        f"public_tess_target_pixel.{arguments.typefileplot}"
    )
    run_example(arguments.runtime_path, output_path)


if __name__ == "__main__":
    main()