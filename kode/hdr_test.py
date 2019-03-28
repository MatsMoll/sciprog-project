import unittest
import numpy as np
import hdr


class HDRTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_weighting_function(self):
        self.assertEqual(hdr.standard_weighting(0), 0)
        self.assertEqual(hdr.standard_weighting(256), 0)
        self.assertEqual(hdr.standard_weighting(1), 1)
        self.assertEqual(hdr.standard_weighting(255), 1)
        self.assertEqual(hdr.standard_weighting(129), 127)

    def test_weighting_function_vector(self):
        input_array = np.array([
            0, 256, 1, 255, 129
        ])
        expected = np.array([
            0, 0, 1, 1, 127
        ])
        output = hdr.standard_weighting_vector(input_array)
        self.assertTrue(np.array_equal(output, expected))

    def test_find_reference_points_for(self):
        gray_input = hdr.ImageSet(np.zeros((10, 10, 3)))  # 10 images with shape (10, 3)
        gray_expected = np.arange(0, 30)
        gray_output = hdr.find_reference_points_for(gray_input)
        self.assertTrue(np.array_equal(gray_output, gray_expected))

        color_input = hdr.ImageSet(np.zeros((10, 1000, 300, 3)))  # 10 images with shape (1000, 300, 3)
        color_expected = np.arange(0, 300000, 300)
        color_output = hdr.find_reference_points_for(color_input)
        self.assertTrue(np.array_equal(color_output, color_expected))

    def test_gray_images(self):
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


if __name__ == "__main__":
    unittest.main()
