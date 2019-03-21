"""
A Module for loading images for a file
"""
import imageio
import numpy as np
from hdr import hdr_color_channels, hdr_channel, standard_weighting
from image_align import find_reference_points_for


class ImageSet:
    """
    A class containing a set of images
    This makes it easier to handel the images
    """
    def __init__(self, images):
        if isinstance(images, list):
            self.images = np.array([])
            self.shutter_speed = []
            for path, shutter in images:
                self.shutter_speed.append(np.log(float(shutter) * 10 ** (-5)))
                if len(self.images) == 0:
                    self.images = np.array([load_image(path, shutter)])
                else:
                    self.images = np.append(self.images, [load_image(path, shutter)], axis=0)
            self.original_shape = np.shape(self.images[0])
        else:
            self.images = images

    def __getitem__(self, item):
        """
        Returns the image at a given index
        :param item: The item to fetch, of type integer
        :return: The image at the given index
        """
        return self.images[item]

    def hdr(self, smoothness):
        """
        Generates a hdr graph for a image set
        :param smoothness: The smoothness on the graph
        :return: The hdr graph
        """
        return self.hdr_channels(find_reference_points_for(self), smoothness)

    def hdr_channels(self, pixel_index, smoothness):
        """
        Calculates the HDR-channels for the Image set

        :param pixel_index: The pixels to use as references
        :param smoothness: The amount of smoothing to do on the graph
        :return: The g lookup table and the ln_e.
        This will be an tuple if the images are of one dim and an array with tuples if there is more
        """
        chan = self.channels()
        shape = np.shape(chan)
        if len(shape) == 3:

            chan = chan.reshape((shape[0], shape[1] * shape[2]))
            return hdr_channel(chan[:, pixel_index], self.shutter_speed, smoothness, standard_weighting)
        elif len(shape) == 4:
            chan = chan.reshape((shape[0], shape[1], shape[2] * shape[3]))
            return hdr_color_channels(chan[:, :, pixel_index], self.shutter_speed, smoothness)

    def gray_images(self):
        """
        Converts the image set to grey images
        :return: A new set containing the grey images
        """
        if len(np.shape(self.images)) == 4:
            image_set = ImageSet(self.images.sum(3) / np.shape(self.images)[3])
            image_set.original_shape = self.original_shape
            image_set.shutter_speed = self.shutter_speed
            return image_set
        else:
            return self

    def channels(self):
        """
        Separates the different channels in the images
        :return: The channels
        """
        shape = np.shape(self.images)
        if len(shape) == 3:
            return self.images
        else:
            chan = np.zeros(shape[-1:] + shape[:-1])
            for i in range(0, len(chan)):
                chan[i] = self.images[:, :, :, i]

            return chan


def load_image(path, shutter, image_format=".png"):
    """
    This loads an image

    Note: This expects that the name of the image is on the format
        /path/filename_shutterspeed.format
    Exp:
        /eksempelbilder/Balls/Balls_00032.png

    :param path: The path to the image
    :param shutter: The shutter speed in a string format
    :param image_format: The image format
    :return: a Image object
    """
    return imageio.imread(path + shutter + image_format)
