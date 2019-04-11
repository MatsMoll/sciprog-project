"""
Tests for the file hdr.py
"""
import unittest
import numpy as np
import hdr
# import matplotlib.pyplot as plt  # Used for debugging


class HDRTest(unittest.TestCase):

    def test_weighting_function(self):
        """
        Tests the hdr.standard_weighting(...) function
        """
        self.assertEqual(hdr.standard_weighting(0), 0)
        self.assertEqual(hdr.standard_weighting(256), 0)
        self.assertEqual(hdr.standard_weighting(1), 1)
        self.assertEqual(hdr.standard_weighting(255), 1)
        self.assertEqual(hdr.standard_weighting(129), 127)

    def test_weighting_function_vector(self):
        """
        Tests the hdr.standard_weighting_vector(...) function
        """
        input_array = np.array([
            0, 256, 1, 255, 129
        ])
        expected = np.array([
            0, 0, 1, 1, 127
        ])
        output = hdr.standard_weighting_vector(input_array)
        self.assertTrue(np.array_equal(output, expected))

    def test_find_reference_points_for(self):
        """
        Tests the find_reference_points_for(...)
        """
        gray_input = hdr.ImageSet(np.zeros((10, 10, 3)))  # 10 images with shape (10, 3)
        gray_expected = np.arange(0, 30)
        gray_output = hdr.find_reference_points_for(gray_input)
        self.assertTrue(np.array_equal(gray_output, gray_expected))

        color_input = hdr.ImageSet(np.zeros((10, 1000, 300, 3)))  # 10 images with shape (1000, 300, 3)
        color_expected = np.arange(0, 300000, 300)
        color_output = hdr.find_reference_points_for(color_input)
        self.assertTrue(np.array_equal(color_output, color_expected))

    def test_gray_images(self):
        """
        Tests ImageSet.gray_images()
        """
        image = np.array([[[ # One image with shape (1, 3, 3)
            [1, 2, 3],
            [2, 3, 4],
            [100, 100, 100]
        ]]])
        image_set = hdr.ImageSet(image)
        image_set.original_shape = (1, 3, 3)
        image_set.shutter_speed = [1]
        expected_image = np.array([[ # Shape (1, 3)
            2, 3, 100
        ]]).astype(float)
        output = image_set.gray_images()
        self.assertTrue(np.array_equal(output.images[0], expected_image))

    def test_hdr_reconstruction(self):
        """
        Tests if the reconstruction works
        """
        x = 20
        y = 20
        n = 10
        max_value = 255
        rand_im = np.random.rand(x, y) * max_value
        rand_im.astype(int)
        exposure_im = np.zeros((n, x, y))
        exposures = np.zeros(n)
        for i in range(1, n + 1):
            exposure_im[i - 1] = rand_im * 2 * i / n # n / 2 == mid exposure
            exposure_im[i - 1][exposure_im[i - 1] > max_value] = max_value
            # plt.imshow(exposure_im[i - 1], plt.cm.gray) # Used for debugging
            # plt.show()
            exposures[i - 1] = np.log(i)

        im_set = hdr.ImageSet(exposure_im)
        im_set.shutter_speed = exposures
        im_set.original_shape = (x, y)

        hdr_im = im_set.hdr_image(10)

        hdr_im = hdr_im * rand_im.max() / hdr_im.max()

        # plt.imshow(rand_im, plt.cm.gray)
        # plt.show()
        # plt.imshow(hdr_im, plt.cm.gray)
        # plt.show()

        diff = np.abs(rand_im - hdr_im).sum()
        self.assertTrue(diff < 0.5 * x * y)  # 0.5 value offset per pixel

    def test_hdr_reconstruction_color(self):
        """
        Tests if the reconstruction works for color images
        """
        x = 20
        y = 20
        n = 10
        c_dim = 3
        max_value = 255
        rand_im = np.random.rand(x, y, c_dim) * max_value
        rand_im = rand_im.astype(int)
        exposure_im = np.zeros((n, x, y, c_dim)).astype(int)
        exposures = np.zeros(n)
        for i in range(1, n + 1):
            exposure_im[i - 1] = rand_im * 2 * i / n # n / 2 == mid exposure
            exposure_im[i - 1][exposure_im[i - 1] > max_value] = max_value
            # plt.imshow(exposure_im[i - 1]) # Used for debugging
            # plt.show()
            exposures[i - 1] = np.log(i)

        im_set = hdr.ImageSet(exposure_im)
        im_set.shutter_speed = exposures
        im_set.original_shape = (x, y, c_dim)

        hdr_im = im_set.hdr_image(10)

        hdr_im = hdr_im * rand_im.max() / hdr_im.max()
        hdr_im = hdr_im.astype(int)

        # plt.imshow(rand_im)
        # plt.show()
        # plt.imshow(hdr_im)
        # plt.show()

        diff = np.abs(rand_im - hdr_im).sum()
        self.assertTrue(diff < 0.5 * x * y * c_dim)  # 0.5 value offset per pixel and channel


if __name__ == "__main__":
    unittest.main()
