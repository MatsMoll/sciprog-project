"""
A module testing `gradient_compression.py`
"""
import unittest
import gradient_compression
from filter_config import GradientFilterConfig
import numpy as np


class GradientCompressionTests(unittest.TestCase):
    """
    The tests for `gradient_compression`
    """

    center = 9.5

    @staticmethod
    def gradient_length_at(x, y):
        """
        Calculates the length of the gradient on a point

        :param x: The x position
        :type x: Float

        :param y: The y position
        :type y: Float

        :return: The length in a float
        """
        return np.sqrt(GradientCompressionTests.gradient_at(x) ** 2 + GradientCompressionTests.gradient_at(y) ** 2)

    @staticmethod
    def gradient_at(x):
        """
        Calculates the gradient at a given position

        :param x: The x position
        :type x: Float

        :return: The gradient as a Float
        """
        return 2 * x

    @staticmethod
    def generate_test_im():
        """
        Generates a test image

        :return: The test image as a numpy array
        """
        values = np.zeros((20, 20))

        for i in range(0, 19):
            values[i + 1, 0] = values[i, 0] + GradientCompressionTests.gradient_at(i - GradientCompressionTests.center)
        for j in range(0, 19):
            values[:, j + 1] = values[:, j] + GradientCompressionTests.gradient_at(j - GradientCompressionTests.center)
        return values

    def test_gradient_vectors(self):
        """
        Tests that the gradient vectors and length is correct
        """
        test_im = self.generate_test_im()
        im_len, im_vector = gradient_compression.gradient_vectors(test_im)

        for x in range(1, im_len.shape[0] - 1):  # omitting the rand
            for y in range(1, im_len.shape[1] - 1):
                self.assertEqual(im_vector[x, y, 0], self.gradient_at(x - self.center))
                self.assertEqual(im_vector[x, y, 1], self.gradient_at(y - self.center))
                self.assertEqual(im_len[x, y], self.gradient_length_at(x - self.center, y - self.center))

    def test_div_matrix(self):
        """
        Test that the div matrix is correct
        """
        test_im = self.generate_test_im()
        _, im_vector = gradient_compression.gradient_vectors(test_im)
        div = gradient_compression.divergence_matrix(im_vector)

        for x in range(0, div.shape[0]):
            for y in range(0, div.shape[1]):

                expected_x = 2
                expected_y = 2
                if x <= 1 or x == div.shape[0]:
                    expected_x = 0
                if y <= 1 or y == div.shape[1]:
                    expected_y = 0

                self.assertEqual(div[x, y, 0], expected_x)
                self.assertEqual(div[x, y, 1], expected_y)

    def test_compress_gradient_color(self):
        """
        Tests that `gradient_compress_color(image)` returns the correct type
        """
        test_im = self.generate_test_im()

        config = GradientFilterConfig()
        config.func = lambda x: x
        result = gradient_compression.gradient_compress_color(test_im, config)
        self.assertEqual(test_im.shape, result.shape)

        config.use_pyramid = True
        result = gradient_compression.gradient_compress_color(test_im, config)
        self.assertEqual(test_im.shape, result.shape)
