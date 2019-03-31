"""
This is the unit test file for functions within the localHDR.py file.

"""

import unittest
import numpy as np
import localHDR


class LocalHDRTest(unittest.TestCase):
    """
    Tests the major functions from localHDR.py.

    Note! A lower and upper boundary is set with a expected image.
    """
    def test_blur_image(self):
        """
        Test the blur image function with sigma value 3.
        """
        input_image = np.array([
            0.01, 0.1, 0.2, 0.5, 0.75, 0.99
        ])
        expected_image_lower = np.array([
            0.286, 0.323, 0.387, 0.461, 0.526, 0.564
        ])
        expected_image_upper = np.array([
            0.287, 0.324, 0.388, 0.462, 0.527, 0.565
        ])
        output = localHDR.blur_image(input_image, 3)
        np.allclose(output, expected_image_lower)
        np.allclose(output, expected_image_upper)

    def test_find_details(self):
        """
        Tests the find details function with a detail level set to the 70th-percentile.
        """
        input_image = np.array([
            0.01, 0.1, 0.2, 0.5, 0.75, 0.99
        ])
        blur_input_image = np.array([
            0.28641213, 0.32315277, 0.3871898, 0.46174035, 0.52684723, 0.56466555
        ])
        expected_image = np.array([
            0, 0, 0, 0, 1, 1
        ])
        output = localHDR.find_details(input_image, blur_input_image, 70)
        np.allclose(output, expected_image)


    def test_edit_blurred_image(self):
        """
        Tests the edit blurred image function with global editing
            and sets weighting on the luminance and chromasity channels.
        """
        blur_input_image = np.array([
            0.28641213, 0.32315277, 0.3871898, 0.46174035, 0.52684723, 0.56466555
        ])
        expected_image_lower = np.array([
            0.535, 0.568, 0.622, 0.679, 0.725, 0.751
        ])
        expected_image_upper = np.array([
            0.536, 0.569, 0.623, 0.680, 0.726, 0.752
        ])
        output = localHDR.edit_blurred_image(blur_input_image, "global", 10, .2)
        np.allclose(output, expected_image_lower)
        np.allclose(output, expected_image_upper)


    def test_reconstruct_image(self):
        """
        Tests the reconstruction of the image. Basically a math function.
        Parameters a, b, c:
            a * c + b = reconstructed image.
        """
        self.assertEqual(localHDR.reconstruct_image(1, 2, 3), 5)
        self.assertEqual(localHDR.reconstruct_image(2, 3, 3), 9)
        self.assertEqual(localHDR.reconstruct_image(0, 0, 99), 0)
        self.assertEqual(localHDR.reconstruct_image(13, 3, 2), 29)
        self.assertEqual(localHDR.reconstruct_image(1, -3, 2), -1)


    def test_filter_linear(self):
        """
        Tests the main function calling all the other minor helping functions.
        """
        input_image = np.array([
            0.01, 0.1, 0.2, 0.5, 0.75, 0.99
        ])
        expected_image_lower = np.array([
            0.639, 0.643, 0.648, 0.655, 0.660, 5.663
        ])
        expected_image_upper = np.array([
            0.640, 0.644, 0.649, 0.656, 0.661, 5.664
        ])
        output = localHDR.filter_linear(input_image, 5, 99, "global", 5, .1, 5)
        np.allclose(output, expected_image_lower)
        np.allclose(output, expected_image_upper)


if __name__ == '__main__':
    unittest.main()
