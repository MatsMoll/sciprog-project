"""
Some unit-tests for hdr.py
"""
import unittest
import numpy as np
import images


class HDRUnitTest(unittest.TestCase):

    def test_channel_separation(self):
        """
        Tests if Image.separate_channels() is working correctly
        """
        im_input = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            [[12, 11, 10], [9, 8, 7], [6, 5, 4], [3, 2, 1]]
        ])
        expected_output = np.array([
            [1, 4, 7, 10, 12, 9, 6, 3],
            [2, 5, 8, 11, 11, 8, 5, 2],
            [3, 6, 9, 12, 10, 7, 4, 1]
        ])
        im = images.Image(im_input, 0, np.shape(im_input))
        self.assertTrue(np.array_equal(im.separate_channels(), expected_output))

    def test_gray_image(self):
        """
        Tests if Image.grey_image() is working correctly
        """
        im_input = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            [[12, 11, 10], [9, 8, 7], [6, 5, 4], [3, 2, 1]]
        ])
        expected_output = np.array([
            [2, 5, 8, 11],
            [11, 8, 5, 2]
        ])
        im = images.Image(im_input, 0, np.shape(im_input)).grey_image()
        self.assertTrue(np.array_equal(im.image, expected_output))
