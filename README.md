# Lygos

## Overview
I start with the pixels when tracing a light curve back to its image. Lygos retrieves image time series from the Transiting Exoplanet Survey Satellite (TESS) and the Nancy Grace Roman Space Telescope, then extracts TESS light curves with image-based photometry. A neighboring star can contribute flux to a target aperture, so the pixels show who shares it. Documentation is at https://lygos.readthedocs.io, with its source in `docs/`.

Pliny records Lygos as an earlier name for Byzantium in [*Natural History* 4.11](https://en.wikisource.org/wiki/Natural_History_(Rackham,_Jones,_%26_Eichholz)/Book_4). The name's deeper etymology is uncertain. It suits a pipeline that traces light curves back to the image pixels that produced them.

Run `python -m pytest -q` from the checkout to execute the tests. The project
uses pytest's importlib mode by default, allowing test directories from multiple
checkouts to be collected together even when test filenames match.

## Images from TESS and Roman
Every retrieval returns one `ImageStack`, a time-ordered cube of images with times, uncertainties, quality flags, and sky coordinates. Background subtraction, difference imaging, variability maps, apertures, aperture photometry, centroids, plots, and animations therefore work the same way for every product.

| Product | Source | Function |
| --- | --- | --- |
| TESS cutouts | MAST TESSCut | `get_tess_cutout` |
| TESS full-frame images | public MAST bucket on Amazon Web Services | `get_tess_ffi`, `get_tess_ffi_cutout` |
| Roman detector images | OpenUniverse 2024 simulation at IRSA | `read_roman_image` |
| Roman cutout time series | OpenUniverse 2024 simulation at IRSA | `get_roman_cutout` |

Live archive checks on October 25, 2026 retrieved measured TESS sector 14
cutouts, a full 2048-by-2048 TESS detector image with uncertainties, and small
sections read directly from two FFIs. The Roman checks used OpenUniverse 2024
J129 simulations, including two aligned 11-by-11 cutouts and a full
4088-by-4088 detector image with uncertainties and pixel flags. The Roman
portion exercises the simulation-product reader with simulated data. Background
subtraction preserved the original cubes, and native GIF generation retained
fixed frame dimensions.

No archive account is needed. Cutout methods limit the requested image area.
TESS FFI methods use remote section reads. Compressed Roman products are
streamed and decompressed through the needed detector rows. Those transfers
include headers and intervening rows. Full-detector readers transfer the full
image products.

```python
import lygos

stack = lygos.get_tess_cutout("RR Lyr", sector=14, size=15)[0].good()
aperture = stack.threshold_aperture(threshold=5.0)
lygos.animate_stack(stack.select(slice(0, 120)), "rrlyr", aperture=aperture,
                    light_curve=stack.select(slice(0, 120)).aperture_photometry(aperture))

column, row = stack.pixel_of(lygos.resolve_coordinate("RR Lyr"))
vertices = [(column - 4, row - 3), (column + 3, row - 3), (column + 4, row + 1),
            (column + 2, row + 4), (column - 4, row + 3)]
custom_aperture = stack.polygon_aperture(vertices)
custom_light_curve = stack.aperture_photometry(custom_aperture)

roman = lygos.get_roman_cutout((9.6632, -43.88128), band="J129", size=41, mjd_range=(62150, 62550))
lygos.animate_stack(roman.subtract_background(), "supernova", difference=True)
```

![Roman J129 difference images of a simulated Type Ia supernova with its light curve](examples/roman_time_domain_supernova/visuals/roman_supernova_difference_animation.gif)

Examples in `examples/tess_cutout_time_series/`, `examples/tess_full_frame_images/`, and `examples/roman_time_domain_supernova/` use only public data.

## TESS sector coverage

Lygos can estimate the cumulative unique sky area covered by ordered TESS sectors. It samples the celestial sphere uniformly, tests each sample against the mission's CCD WCS geometry, and plots the accumulated area with its Monte Carlo uncertainty. The example also writes a dated sector-footprint animation, with TOI-1233 highlighted.

The estimate covers nominal detector footprints. It does not measure usable exposure time or data quality. Runtime scales with the number of sky samples and sectors. The default 1,024 samples keep the estimate quick, and larger samples reduce Monte Carlo uncertainty.

```bash
python examples/tess_sector_coverage/tess_sector_coverage.py --sectors 1 2 3 4 --animation-sectors 1 2 3 4
```

![Cumulative TESS sky coverage and dated sector animation](examples/tess_sector_coverage/visuals/tess_cumulative_sky_coverage_20261023_135600.png)

![Dated TESS sector footprints with TOI-1233 highlighted](examples/tess_sector_coverage/visuals/tess_sector_timeline_20261023_135600.gif)

## ULTRASAT synthetic images

The `examples/ultrasat_synthetic/ultrasat_synthetic.py` example runs Lygos's
image simulator for any target name resolvable by MAST and present in the TIC.
It simulates the ULTRASAT near-ultraviolet band (230-290 nm) with 2-arcsec
pixels and continuous 300-second exposures, matching the timing described for
the mission's [high-cadence survey](https://www.weizmann.ac.il/ultrasat/science-mission/modes-of-operation/modes-of-operation).
Frames are sampled at the exposure interval, with no readout gaps or timing
jitter. The example writes a median image and an aperture light curve under its
`visuals/` directory.

Set `LYGOS_PATH` before running the example so the simulated image cube and
catalog cache go under the Lygos data directory:

```bash
export LYGOS_PATH=/path/to/lygos
python examples/ultrasat_synthetic/ultrasat_synthetic.py --target Vega
```

The simulated count rates are deterministic expectations based on TIC Tmag and
Lygos's ULTRASAT zeropoint. Tmag is not converted from a measured ultraviolet
spectrum, and the simulation adds neither shot nor read noise. It has no
measured instrument response, so the pixel scale, PSF, and count rates are
illustrative.

The `timeexpo` argument to `lygos.init` takes values in days, nested by
instrument and pointing. When `booldiag` is true, each supplied value must be
finite and positive. Lygos retains explicit values as given and applies the
existing 0.8 factor only to inferred exposures.

## Mauve data status

Mauve's published science product is stellar UV-visible spectrophotometry over
200-700 nm. The mission uses a telescope coupled to spectrometers. Selected
commissioning spectra and calibration information are offered as FITS files
through a Stardrive account, as described at `https://bssl.space/mauve/` and
`https://bssl.space/mauve/preview-data/`.

Lygos has no Mauve image reader or imaging example. A spectral detector array
cannot be treated as a sky image without a documented mapping and response. The
synthetic ImageStack examples do not model Mauve. A mission-specific Miletos
analysis needs a defined product schema, wavelength and time conventions, and
uncertainty treatment.

## Pixel-level variability

`ImageStack.variability_map()` measures scatter relative to the pixel uncertainties when available. `ImageStack.relative_variability_map()` instead reports robust scatter as a percentage of each pixel's median flux. The public Sector 1 example plots the median image, relative variability, and the most variable source pixel's light curve.

```bash
python examples/tess_pixel_variability/tess_pixel_variability.py
python examples/tess_pixel_variability/tess_pixel_variability.py --bokeh
```

Install `python -m pip install -e '.[bokeh]'` to write the interactive variability maps and pixel light curve as standalone HTML.

![TESS Sector 1 pixel variability around pi Mensae](examples/tess_pixel_variability/visuals/tess_pixel_variability.png)

## Synthetic ImageStack notebook

The [synthetic ImageStack notebook](examples/synthetic_image_stack.ipynb)
builds a two-source image time series, compares threshold and fixed apertures,
and plots pixel-level variability. It draws Poisson counts from a specified
exposure time and converts them to electron rates and uncertainties. The data
are synthetic and use no archive observations. Static figures and optional
Bokeh output go under `examples/visuals/`.

![Synthetic aperture light curves from the ImageStack workflow](examples/visuals/lygos_synthetic_aperture_light_curves.png)

![Synthetic median image, variability map, and pixel light curve](examples/visuals/lygos_synthetic_variability_summary.png)

[Interactive variability summary](examples/visuals/lygos_synthetic_variability_summary.html)

## Image-based photometry
Target-pixel images can be used to model or extract stellar flux, assess nearby-source contamination, and trace a light curve to its aperture or image model.

For PSF fitting, Lygos evaluates the image likelihood and delegates posterior inference to PCAT. PCAT is a required dependency, and the dependency map marks this direct image-inference path.

## Installation

```bash
cd /path/to/lygos
pip install -e .
export LYGOS_PATH=/path/to/lygos
```

`LYGOS_PATH` identifies the repository root. Runtime data and downloaded mission caches belong under its ignored `data/` directory, while generated figures belong under its ignored `visuals/` directory. Functions accept explicit paths when a workflow needs an isolated run directory.

## Public TESS target-pixel example

With `LYGOS_PATH` set to the repository root, run:

```bash
python examples/public_tess_target_pixel.py --typefileplot png
```

![WASP-121 TESS Sector 7 target-pixel image](examples/public_tess_target_pixel.png)

The example queries the public Mikulski Archive for Space Telescopes (MAST) for the WASP-121 SPOC target-pixel products, selects TESS Sector 7, masks nonzero quality flags, and calculates the median 11 by 11 pixel count map with nearby TESS Input Catalog sources overlaid. The figure contains observed TESS data rather than a simulation and quantifies the image-level contamination context. This example does not extract a light curve.
