"""
This is the unit test file for functions within the localHDR.py file.

"""

import unittest
import numpy as np
import localHDR
from filter_config import FilterImageConfig, BlurImageConfig


class LocalHDRTest(unittest.TestCase):
    """
    Tests the major functions from localHDR.py.

    """
    def test_has_alpha(self):
        """
        Tests the last dimension of input arrays to see if they have
            a fourth channel (dimension) present.
        """
        image_3d = np.array([[  # One image with shape (1, 2, 3) ## Rename to image_3d
            [1, 2, 3],
            [4, 5, 6]
        ]])
        image_4d = np.array([[  # One image with shape (1, 3, 4) ## Rename to image_4d
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]])
        image_5d = np.array([[  # One image with shape (1, 1, 5) ## Rename to image_5d
            [1, 2, 3, 4, 5]
        ]])
        self.assertEqual(localHDR.has_alpha(image_3d), False)
        self.assertEqual(localHDR.has_alpha(image_4d), True)
        self.assertEqual(localHDR.has_alpha(image_5d), False)

    def test_extract_alpha(self):
        """
        Tests extraction of the alpha channel (dimension).
        Will only be called if four channels (dimensions) is present.
        """
        image_4d = np.array([[
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]])
        im, alpha = localHDR.extract_alpha(image_4d)
        self.assertTrue(np.allclose(alpha, image_4d[:, :, 3]))

    def test_blur_image_linear(self):
        """
        Test the linear and non-linear blur image function.
        """
        input_image = np.array([
            0.01, 0.1, 0.2, 0.5, 0.75, 0.99
        ])
        expected_linear_image_lower = np.array([
            0.286, 0.323, 0.387, 0.461, 0.526, 0.564
        ])
        expected_linear_image_upper = np.array([
            0.287, 0.324, 0.388, 0.462, 0.527, 0.565
        ])
        output_linear_config = BlurImageConfig()
        output_linear_config.sigma = 3
        output_linear = localHDR.blur_image(input_image, output_linear_config)
        self.assertTrue(np.allclose(output_linear, expected_linear_image_lower, atol=6e-03))
        self.assertTrue(np.allclose(output_linear, expected_linear_image_upper, atol=6e-03))

    def test_blur_image_nonlinear(self):
        input_image = np.array([
            0.01, 0.1, 0.2, 0.5, 0.75, 0.99
        ])
        expected_nonlinear_image = np.array([
            0.22020, 0.26142, 0.36163, 0.47307, 0.55185, 0.58400
        ])
        output_nonlinear_config = BlurImageConfig()
        output_nonlinear_config.sigma = 3
        output_nonlinear_config.linear = False
        output_nonlinear = localHDR.blur_image(input_image, output_nonlinear_config)
        self.assertTrue(np.allclose(output_nonlinear, expected_nonlinear_image, atol=6e-01))

    def test_find_details(self):
        """
        Tests the find details function.
        """
        input_image = np.array([
            0.01, 0.1, 0.2, 0.5, 0.75, 0.99
        ])
        blur_input_image = np.array([
            0.28641213, 0.32315277, 0.3871898, 0.46174035, 0.52684723, 0.56466555
        ])
        expected_image_lower = np.array([
            -0.277, -0.224, -0.188, 0.038, 0.223, 0.425
        ])
        expected_image_upper = np.array([
            -0.276, -0.223, -0.187, 0.039, 0.224, 0.426
        ])
        output = localHDR.find_details(input_image, blur_input_image)
        self.assertTrue(np.allclose(output, expected_image_lower, atol=6e-03))
        self.assertTrue(np.allclose(output, expected_image_upper, atol=6e-03))

    def test_edit_blurred_image(self):
        """
        Tests the edit blurred image function with global editing
            and sets weighting on the luminance and chromasity channels.
        """
        blur_input_image = np.array([
            0.28641213, 0.32315277, 0.3871898, 0.46174035, 0.52684723, 0.56466555
        ])
        expected_image_lower = np.array([
            0.384, 0.395, 0.414, 0.435, 0.454, 0.465
        ])
        expected_image_upper = np.array([
            0.385, 0.396, 0.415, 0.436, 0.455, 0.466
        ])
        output_config = FilterImageConfig()
        output_config.effect.sigma = 3
        output_config.blur.linear = True
        output = localHDR.blur_image(blur_input_image, output_config.blur)
        output_config.mode = "global"
        output_config.lum_scale = 10
        output_config.chrom_scale = .2
        self.assertTrue(np.allclose(output, expected_image_lower, atol=6e-03))
        self.assertTrue(np.allclose(output, expected_image_upper, atol=6e-03))

    def test_reconstruct_image(self):
        """
        Tests the reconstruction of the image. Basically a math function.
        Parameters a, b, c:
            a * c + b = reconstructed image.
        """
        first_product = np.array([1])
        link = np.array([2])
        second_product = FilterImageConfig()
        second_product.gamma = 3  # 1*3+2
        self.assertEqual(localHDR.reconstruct_image(first_product, link, second_product), 5)

        first_product = np.array([2])
        link = np.array([3])
        second_product = FilterImageConfig()
        second_product.gamma = 3  # 2*3+3
        self.assertEqual(localHDR.reconstruct_image(first_product, link, second_product), 9)

        first_product = np.array([0])
        link = np.array([0])
        second_product = FilterImageConfig()
        second_product.gamma = 99  # 0*99+0
        self.assertEqual(localHDR.reconstruct_image(first_product, link, second_product), 0)

        first_product = np.array([13])
        link = np.array([3])
        second_product = FilterImageConfig()
        second_product.gamma = 2  # 13*2+3
        self.assertEqual(localHDR.reconstruct_image(first_product, link, second_product), 29)

        first_product = np.array([1])
        link = np.array([-3])
        second_product = FilterImageConfig()
        second_product.gamma = 2  # 1*2+(-2)
        self.assertEqual(localHDR.reconstruct_image(first_product, link, second_product), -1)

    def test_append_channel(self):
        """
        Tests appending a forth (commonly alpha) channel after it has been extracted.
        """
        image_4d = np.array([[
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]])
        im, alpha = localHDR.extract_alpha(image_4d)
        appended = localHDR.append_channel(im, alpha)
        self.assertTrue(np.allclose(image_4d, appended))

    def test_filter_linear(self):
        """
        Tests the linear filtering of an input image.
        This is the outer-most function with the most customizability.
        """
        input_image = np.array([
            0.01, 0.1, 0.2, 0.5, 0.75, 0.99
        ])
        expected_image_lower = np.array([
            0.240, 0.329, 0.427, 0.725, 0.974, 1.213
        ])
        expected_image_upper = np.array([
            0.241, 0.330, 0.428, 0.726, 0.975, 1.214
        ])
        output_config = FilterImageConfig()
        output_config.blur.linear = True
        output_config.blur.sigma = 5
        output_config.effect.mode = "global"
        output_config.effect.lum_scale = 5
        output_config.effect.chrom_scale = .1
        output_config.effect.level = 5
        output = localHDR.filter_image(input_image, output_config)
        self.assertTrue(np.allclose(output, expected_image_lower, atol=6e-03))
        self.assertTrue(np.allclose(output, expected_image_upper, atol=6e-03))


if __name__ == '__main__':
    unittest.main()
