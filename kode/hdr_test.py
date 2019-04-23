"""
Tests for the file hdr.py
"""
import unittest
import numpy as np
import hdr


class HDRTest(unittest.TestCase):
    """
    The tests for the hdr.py module
    """

    @staticmethod
    def random_image_set(x, y, colors, n):
        """
        Creates a image set
        :param x: The width of the image
        :param y: The height of the image
        :param colors: IF the image needs colors
        :param n: The number of images
        :return: A ImageSet
        """

        max_value = 255
        rand_im = np.random.rand(x, y)
        exposure_im = np.zeros((n, x, y), dtype=int)
        if colors:
            rand_im = np.random.rand(x, y, 3)
            exposure_im = np.zeros((n, x, y, 3), dtype=int)

        rand_im = rand_im
        exposures = np.zeros(n)
        for i in range(1, n + 1):
            exposure_im[i - 1] = (rand_im * max_value * 2 ** (i - 1)).astype(int)
            exposure_im[i - 1][exposure_im[i - 1] > max_value] = max_value
            exposures[i - 1] = 2 ** (i - 1)

        im_set = hdr.ImageSet(exposure_im)
        im_set.shutter_speed = np.log(exposures)
        im_set.original_shape = rand_im.shape

        return rand_im, im_set

    def test_weighting_function(self):
        """
        Tests the hdr.standard_weighting(...) function
        """
        self.assertEqual(hdr.standard_weighting(0), 0)
        self.assertEqual(hdr.standard_weighting(255), 0)
        self.assertEqual(hdr.standard_weighting(1), 1)
        self.assertEqual(hdr.standard_weighting(254), 1)
        self.assertEqual(hdr.standard_weighting(129), 126)

    def test_weighting_function_vector(self):
        """
        Tests the hdr.standard_weighting_vector(...) function

        This uses different values in the reconstruction function to avoid a graphical glitch
        """
        input_array = np.array([
            1, 256, 1, 254, 129
        ])
        expected = np.array([
            1, 0, 1, 2, 127
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
        expected_image = np.array([[  # Shape (1, 3)
            2, 3, 100
        ]]).astype(float)
        output = image_set.gray_images()
        self.assertTrue(np.array_equal(output.images[0], expected_image))

    def test_hdr_reconstruction(self):
        """
        Tests if the reconstruction works
        """
        x = 30
        y = 30
        n = 6
        expected_graph = np.log(np.arange(1, 257) / 127)
        rand_im, im_set = self.random_image_set(x, y, False, n)
        output_graph = im_set.hdr_curve(10)[0]

        diff = np.abs(np.exp(expected_graph) - np.exp(output_graph)).sum()

        self.assertTrue(diff < 5)

        hdr_im = im_set.hdr_image(10)
        rand_im = rand_im * 255
        hdr_im = hdr_im * rand_im.max() / hdr_im.max()

        diff = np.abs(rand_im - hdr_im).sum()
        self.assertTrue(diff < 0.75 * x * y)

    def test_hdr_reconstruction_color(self):
        """
        Tests if the reconstruction works for color images
        """
        x = 20
        y = 20
        n = 6
        rand_im, im_set = self.random_image_set(x, y, True, n)
        rand_im = rand_im * 255

        hdr_im = im_set.hdr_image(10)

        hdr_im = hdr_im * rand_im.max() / hdr_im.max()

        diff = np.abs(rand_im - hdr_im).sum()
        self.assertTrue(diff < 0.5 * x * y * 3)  # 0.5 value offset per pixel and channel

    def test_reconstruct_image_from_graph(self):
        """
        Tests if the graph is correct in a gray image
        """
        x = 30
        y = 30
        n = 6
        rand_im, im_set = self.random_image_set(x, y, False, n)
        graph = np.log(np.arange(1, 257) / 256)

        output = hdr.reconstruct_image(
            im_set.channels(), hdr.standard_weighting_vector, graph, im_set.shutter_speed)

        output = output - output.min()
        scaled = output * rand_im.max() / output.max()

        diff = np.abs(rand_im - scaled).sum()
        self.assertEqual(output.shape, (x, y))
        self.assertTrue(diff < 0.5 * x * y)

    def test_reconstruct_image_from_graph_color(self):
        """
        Tests if the graphs is correct in a color image
        """
        x = 30
        y = 30
        n = 6
        rand_im, im_set = self.random_image_set(x, y, True, n)
        graph = np.log(np.arange(1, 257) / 128)

        output = np.zeros(im_set.original_shape)
        for i in range(0, im_set.original_shape[-1] - 1):
            output[:, :, i] = hdr.reconstruct_image(
            im_set.channels()[i], hdr.standard_weighting_vector, graph, im_set.shutter_speed)

        output = output - output.min()
        scaled = output * rand_im.max() / output.max()

        diff = np.abs(rand_im - scaled).sum()
        self.assertEqual(output.shape, (x, y, 3))
        self.assertTrue(diff < 0.5 * x * y * 3)


if __name__ == "__main__":
    unittest.main()
