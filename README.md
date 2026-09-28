# Lygos

## Purpose
Lygos is a TESS image-based photometry and light-curve extraction workflow. It is designed to model or extract stellar fluxes from image stacks and produce light curves with diagnostics for subsequent time-domain analyses.

## Image-based photometry
Lygos inspects target-pixel images, models or extracts stellar fluxes from image stacks, assesses contamination from nearby sources, and produces light curves with image-level diagnostics.

## Installation

```bash
cd /path/to/lygos
pip install -e .
export LYGOS_PATH=/path/to/lygos
```

`LYGOS_PATH` identifies the repository root, whose `data/` and `visuals/` directories are ignored by Git. The workflow separately expects a configured working-data directory, typically via `LYGOS_DATA_PATH` or an explicit project path; that existing variable retains its dataset and output semantics.

## Minimal usage

```python
import lygos

# Example workflow entry point
# lygos.main.init(strgmast='WASP-121')
# or
# lygos.main.init(toiitarg=1233)
```

The package is designed to run through the importable workflow entry points rather than through ad hoc local scripts.

## Public TESS target-pixel example

With `LYGOS_PATH` set to the repository root, run:

```bash
python examples/public_tess_target_pixel.py --typefileplot png
```

![WASP-121 TESS Sector 7 target-pixel diagnostic](examples/public_tess_target_pixel.png)

The example queries the public Mikulski Archive for Space Telescopes (MAST) for the WASP-121 SPOC target-pixel products, selects TESS Sector 7, masks nonzero quality flags, and calculates the median 11 by 11 pixel count map with nearby TESS Input Catalog sources overlaid. The figure contains observed TESS data rather than a simulation and quantifies the image-level contamination context. This example does not extract a light curve.

## Input, intermediate products, and outputs
A good Lygos run should produce a visible chain of products:

- input image or target metadata;
- extracted or modeled source apertures;
- intermediate photometric diagnostics;
- final target light curves and summary plots.

Lygos writes these products under the configured data and visualization directories.
