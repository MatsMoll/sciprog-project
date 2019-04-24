"""
This is the unit test file for functions within the globalHDR.py file.

"""

import unittest
import numpy as np
import globalHDR
from filter_config import EffectConfig


class GlobalHDRTest(unittest.TestCase):
    """
    Tests the major functions from globalHDR.py.

    Note! A lower and upper boundary is set with an expected image.
    """
    def test_edit_globally(self):
        """
        Test the global rendering with the sqrt-function.
        """
        input_image = np.array([
            0.01, 0.1, 0.2, 0.5, 0.75, 0.99
        ])
        expected_image_lower = np.array([
            0.099, 0.316, 0.447, 0.707, 0.866, 0.994
        ])
        expected_image_upper = np.array([
            0.101, 0.317, 0.448, 0.708, 0.867, 0.995
        ])
        output_config = EffectConfig()
        output_config.func = "sqrt"
        output = globalHDR.edit_globally(input_image, output_config)
        self.assertTrue(np.allclose(output, expected_image_lower, atol=6e-03))
        self.assertTrue(np.allclose(output, expected_image_upper, atol=6e-03))

    def test_edit_luminance(self):
        """
        Test the luminance channel with a luminance-chromasity ratio and the default sqrt-function.
        """
        input_image = np.array([
            0.01, 0.1, 0.2, 0.5, 0.75, 0.99
        ])
        expected_image_lower = np.array([
            0.399, 1.264, 1.788, 2.828, 3.464, 3.979
        ])
        expected_image_upper = np.array([
            0.400, 1.265, 1.789, 2.829, 3.465, 3.980
        ])
        output_config = EffectConfig()
        output_config.lum_scale = 5
        output_config.chrom_scale = .8

        input_lum = globalHDR.luminance(input_image)
        input_chrom = globalHDR.chromasity(input_image, input_lum)

        output = globalHDR.edit_luminance(input_lum, input_chrom, output_config)
        self.assertTrue(np.allclose(output, expected_image_lower, atol=6e-03))
        self.assertTrue(np.allclose(output, expected_image_upper, atol=6e-03))


if __name__ == '__main__':
    unittest.main()
