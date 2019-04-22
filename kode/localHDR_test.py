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
    def test_check_dim(self):
        """
        Tests the last dimension of input arrays to see if they have
            a fourth channel (dimension) present.
        """
        image_123 = np.array([[  # One image with shape (1, 2, 3) ## Rename to image_3d
            [1, 2, 3],
            [4, 5, 6]
        ]])
        image_134 = np.array([[  # One image with shape (1, 3, 4) ## Rename to image_4d
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]])
        image_115 = np.array([[  # One image with shape (1, 1, 5) ## Rename to image_5d
            [1, 2, 3, 4, 5]
        ]])
        self.assertEqual(localHDR.check_dim(image_123), False)  # Riktig
        self.assertNotEqual(localHDR.check_dim(image_123), True)  # Sjekker at "NotEqual" == True

        self.assertEqual(localHDR.check_dim(image_134), True)  # Riktig
        self.assertNotEqual(localHDR.check_dim(image_134), False)  # Sjekker at "NotEqual" == True

        self.assertEqual(localHDR.check_dim(image_115), False)  # Riktig
        self.assertNotEqual(localHDR.check_dim(image_115), True)  # Sjekker at "NotEqual" == True

    def test_extract_alpha(self):
        """
        Tests extraction of the alpha channel (dimension).
        Will only be called if four channels (dimensions) is present, therefore there's no check.
        """
        image_134 = np.array([[  # One image with shape (1, 3, 4) ## Rename to image_4d
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]])
        im, alpha = localHDR.extract_alpha(image_134)
        np.allclose(alpha, image_134[:, :, 3])

    def test_blur_image(self): # utvid test til å gjelde non-linear også
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

    def test_find_details(self): # Rewrite!!
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
        output = localHDR.find_details(input_image, blur_input_image)
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

    def test_reconstruct_image(self):  # Skriv om test
        """
        Tests the reconstruction of the image. Basically a math function.
        Parameters a, b, c:
            a * c + b = reconstructed image.
        """
        detail_image = np.array([
            1
        ])
        blurry_image = np.array([
            2
        ])
        self.assertEqual(localHDR.reconstruct_image(detail_image, blurry_image, 3), 5)



        """
        Skal jeg skrive testene slik som ovenfor??
        self.assertEqual(localHDR.reconstruct_image(1, 2, 3), 5)
        self.assertEqual(localHDR.reconstruct_image(2, 3, 3), 9)
        self.assertEqual(localHDR.reconstruct_image(0, 0, 99), 0)
        self.assertEqual(localHDR.reconstruct_image(13, 3, 2), 29)
        self.assertEqual(localHDR.reconstruct_image(1, -3, 2), -1)
        """

    def test_append_alpha(self):
        """
        ...
        """
        image_134 = np.array([[  # One image with shape (1, 3, 4) ## Rename to image_4d
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]])
        im, alpha = localHDR.extract_alpha(image_134)
        appended = localHDR.append_alpha(im, alpha)
        np.allclose(image_134, appended)


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
        output = localHDR.filter_image(input_image, True, 5, 99, "global", 5, .1, 5)
        np.allclose(output, expected_image_lower)
        np.allclose(output, expected_image_upper)


if __name__ == '__main__':
    unittest.main()
