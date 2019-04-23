"""
Testsing the align_image.py file
"""
import unittest
import numpy as np
import align_image
from hdr import ImageSet


class AlignImageTests(unittest.TestCase):
    """
    The different tests for align_image.py
    """

    def test_image_cropping(self):
        """
        Tests the cropping function
        """
        input_im = np.random.rand(4, 20, 20, 4)
        input_im[:, :, :, -1] = 1
        input_im[0, -3:, :, -1] = 0
        input_im[1, -3:, :, -1] = 0
        input_im[3, -1:, -3:, -1] = 0
        expected_im = input_im[:, :-3, :-3, :]
        output_im = align_image.crop_image_set(input_im)
        self.assertTrue(np.array_equal(expected_im, output_im))

    def test_alignment_function(self):
        """
        Tests the alignment function
        """
        first_input_im = np.random.rand(225, 225, 3) * 255
        second_input_im = np.rot90(first_input_im)
        expected_im = first_input_im.astype(np.uint8)
        output_im, _ = align_image.align_images(second_input_im, first_input_im)
        self.assertTrue(np.array_equal(expected_im, output_im))

    def test_alignment(self):
        """
        Tests if the ImageSet info is correct except the image, as this is a little unpredictable
        """
        rand_im = (np.random.rand(225, 225, 4) * 255).astype(np.uint8)
        rand_im[:, :, -1] = 1
        input_im = np.zeros((4,) + rand_im.shape, dtype=np.uint8)
        for i in range(0, input_im.shape[0]):
            input_im[i] = rand_im

        input_im[1, :-2, :] = input_im[1, 2:, :]
        input_im[1, -3:, :] = 0
        input_im[3] = np.rot90(input_im[3])

        input_im_set = ImageSet(input_im)
        input_im_set.original_shape = input_im.shape[1:]
        input_im_set.shutter_speed = np.array([1, 2, 3, 4])

        expected_im_set = ImageSet(input_im)  # ignoring the returned image here
        expected_im_set.original_shape = (222, 222, 4)
        expected_im_set.shutter_speed = input_im_set.shutter_speed.copy()

        output_image_set = align_image.align_image_set(input_im_set)

        self.assertEqual(expected_im_set.original_shape, output_image_set.original_shape)
        self.assertTrue(np.array_equal(expected_im_set.shutter_speed, output_image_set.shutter_speed))
