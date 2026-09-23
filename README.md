# Lygos

## Scientific purpose
Lygos is a TESS image-based photometry and light-curve extraction workflow. It is designed to model or extract stellar fluxes from image stacks and produce light curves with diagnostics for subsequent time-domain analyses.

## Repository role in the ecosystem
Lygos sits at the interface between image-domain data products and higher-level time-series analysis workflows. It is intended to work with the shared numerical and path conventions in the broader astrophysics ecosystem, especially the project-level data directories and plotting utilities.

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

## Input, intermediate products, and outputs
A good Lygos run should produce a visible chain of products:

- input image or target metadata;
- extracted or modeled source apertures;
- intermediate photometric diagnostics;
- final target light curves and summary plots.

These outputs should be written to a reproducible project directory rather than relying on personal workstation paths.

## Current maintenance status
The repository remains research-grade and workflow-oriented rather than a general-purpose black-box package. The supported interface is the package import plus the documented workflow entry points; older or prototype analysis code should be treated as legacy unless clearly migrated.
