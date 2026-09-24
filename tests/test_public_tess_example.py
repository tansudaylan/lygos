from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "public_tess_target_pixel.py"
SPEC = spec_from_file_location("public_tess_target_pixel", EXAMPLE_PATH)
EXAMPLE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXAMPLE)


def test_example_uses_documented_public_target():
    assert EXAMPLE.TARGET_TIC == 22529346
    assert EXAMPLE.TESS_SECTOR == 7


def test_committed_public_tess_figure_is_nonblank():
    image_path = EXAMPLE_PATH.with_name("public_tess_target_pixel.png")

    image = mpimg.imread(image_path)
    assert image.shape[0] > 500
    assert image.shape[1] > 500
    assert np.std(image[..., :3]) > 0.05