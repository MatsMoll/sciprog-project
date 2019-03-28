"""
This is the unit test file for functions within the globalHDR.py file.

"""

import unittest as test
import numpy as np
from kode import globalHDR


class GlobalHDRTest(test.TestCase):
    """
    Tests the major functions from globalHDR.py.

    Note! A lower and upper boundary is set with a expected image.
    """
    def test_edit_globally(self):
        input_image = np.array([
            0.01, 0.1, 0.2, 0.5, 0.75, 0.99
        ])
        expected_image_lower = np.array([
            0.099, 0.316, 0.447, 0.707, 0.866, 0.994
        ])
        expected_image_upper = np.array([
            0.101, 0.317, 0.448, 0.708, 0.867, 0.995
        ])
        output = globalHDR.edit_globally(input_image, "sqrt")
        np.allclose(output, expected_image_lower)
        np.allclose(output, expected_image_upper)

    def test_edit_luminance(self):
        input_image = np.array([
            0.01, 0.1, 0.2, 0.5, 0.75, 0.99
        ])
        expected_image_lower = np.array([
            0.078, 0.252, 0.357, 0.565, 0.692, 0.795
        ])
        expected_image_upper = np.array([
            0.082, 0.253, 0.358, 0.566, 0.693, 0.796
        ])
        lum_scale = 5
        chrom_scale = .8
        output = globalHDR.edit_luminance(input_image, lum_scale, chrom_scale)
        np.allclose(output, expected_image_lower)
        np.allclose(output, expected_image_upper)


if __name__ == '__main__':
    test.main()
