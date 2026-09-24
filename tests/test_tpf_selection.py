import numpy as np

from lygos.main import identify_tpf_sectors


def test_identify_tpf_sectors_marks_available_spoc_products():
    selected_sectors = np.array([7, 8, 33])
    available_tpf_sectors = np.array([7, 33, 34, 61])

    np.testing.assert_array_equal(
        identify_tpf_sectors(selected_sectors, available_tpf_sectors),
        [True, False, True],
    )