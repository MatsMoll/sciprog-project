"""
A module simplifying the hdr calculations
"""
import numpy as np
import hdr
from align_image import align_images
from globalHDR import read_image


class ImageSet:
    """
    A class containing a set of images
    This makes it easier to handel the images

    :attr images: The different images in the image set
    :type images: A numpy array

    :attr shutter_speed: The different shutter speeds
    :type shutter_speed: A numpy array

    :attr original_shape: The original image shape
    :type original_shape: A tuple
    """

    images = np.array([])
    shutter_speed = np.array([])
    original_shape = ()

    def __init__(self, images):
        """
        Inits a image set

        :param images: The images
        :type images: A tuple or numpy array
        """
        if isinstance(images, list):
            self.images = np.array([])
            self.shutter_speed = []
            for path, shutter in images:

                self.shutter_speed.append(np.log(float(shutter)))

                if self.images.size == 0:
                    self.images = np.array([read_image(path)])
                else:
                    self.images = np.append(self.images, [read_image(path)], axis=0)

            if images:
                self.original_shape = np.shape(self.images[0])
                self.shutter_speed = np.array(self.shutter_speed)
            #self.images[self.images <= 0] = 1

            #shutter_base = (self.images[1] / self.images[0]).mean()
            #print(shutter_base)

            #for i in range(1, self.images.shape[0]):
            #    self.shutter_speed[i] = np.log(np.exp(self.shutter_speed[0]) * 1.3 ** i)
            #    print(self.shutter_speed[i])
            #print(self.shutter_speed)

        else:
            self.images = images.copy()

    def hdr_image(self, smoothness):
        """
        Generates a hdr image

        :param smoothness: The smoothness on the curve
        :type smoothness: int

        :return: The hdr image
        """
        channels = self.channels()
        curve = self.hdr_curve(smoothness)

        #if len(curve) == 2:
        #    plt.plot(np.arange(0, 256), curve[0])
        #    plt.show()
        #    plt.plot(np.arange(0, 256), np.exp(curve[0]))
        #    plt.show()
        #else:
        #    for i in range(0, len(curve)):
        #        plt.plot(np.arange(0, 256), curve[i][0])
        #        plt.show()
        #        plt.plot(np.arange(0, 256), np.exp(curve[i][0]))
        #        plt.show()

        output_image = np.zeros(self.original_shape[:-1] + (3,))
        if channels.ndim == 3:
            output_image = hdr.reconstruct_image(
                channels, hdr.standard_weighting_vector, curve[0], self.shutter_speed)
        else:
            for i in range(0, 3):
                output_image[:, :, i] = hdr.reconstruct_image(
                    channels[i], hdr.standard_weighting_vector, curve[i][0], self.shutter_speed)
        return output_image

    def hdr_curve(self, smoothness):
        """
        Generates a hdr curve for a image set

        :param smoothness: The smoothness on the curve
        :type smoothness: int

        :return: The hdr curve
        """
        return self.hdr_channels(hdr.find_reference_points_for(self), smoothness)

    def hdr_channels(self, pixel_index, smoothness):
        """
        Calculates the HDR-channels for the Image set

        :param pixel_index: The pixels to use as references
        :type pixel_index: Numpy array

        :param smoothness: The amount of smoothing to do on the graph
        :type smoothness: int

        :return: The g lookup table and the ln_e.
        This will be an tuple if the images are of one dim and an array with tuples if there is more
        """
        chan = self.channels()
        shape = np.shape(chan)
        if len(shape) == 3:
            chan = chan.reshape((shape[0], shape[1] * shape[2]))
            return hdr.hdr_channel(chan[:, pixel_index], self.shutter_speed, smoothness, hdr.standard_weighting)
        else:
            chan = chan.reshape((shape[0], shape[1], shape[2] * shape[3]))
            return hdr.hdr_color_channels(chan[:, :, pixel_index], self.shutter_speed, smoothness)

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
            chan = np.zeros((3,) + shape[:-1])
            for i in range(0, 3):
                chan[i] = self.images[:, :, :, i]

            return chan

    def aligned_image_set(self):
        """
        Aligns tha image set

        :return: The aligned image set
        """
        aligned_images = align_images(self.images)
        aligned_image_set = ImageSet(aligned_images)
        aligned_image_set.shutter_speed = self.shutter_speed.copy()
        aligned_image_set.original_shape = aligned_images.shape[1:]
        return aligned_image_set


def test_image_set_unaligned():
    """
    Creates a image set for testing

    :return: The ImageSet containing all the information
    """
    return ImageSet([
        ("../eksempelbilder/Balls Unaligned/Balls_00001.png", "00001"),
        ("../eksempelbilder/Balls Unaligned/Balls_00004.png", "00004"),
        ("../eksempelbilder/Balls Unaligned/Balls_00016.png", "00016"),
        ("../eksempelbilder/Balls Unaligned/Balls_00032.png", "00032"),
        ("../eksempelbilder/Balls Unaligned/Balls_00064.png", "00064"),
        ("../eksempelbilder/Balls Unaligned/Balls_00128.png", "00128"),
        ("../eksempelbilder/Balls Unaligned/Balls_00256.png", "00256"),
        ("../eksempelbilder/Balls Unaligned/Balls_00512.png", "00512"),
        ("../eksempelbilder/Balls Unaligned/Balls_01024.png", "01024"),
        ("../eksempelbilder/Balls Unaligned/Balls_02048.png", "02048"),
        # ("../eksempelbilder/Balls/Balls_04096.png", "04096"),
        # ("../eksempelbilder/Balls/Balls_08192.png", "08192"),
        # ("../eksempelbilder/Balls/Balls_16384.png", "16384"),
    ])


def test_image_set():
    """
    Creates a image set for testing

    :return: The ImageSet containing all the information
    """
    return ImageSet([
        ("../eksempelbilder/Balls/Balls_00001.png", "00001"),
        ("../eksempelbilder/Balls/Balls_00004.png", "00004"),
        ("../eksempelbilder/Balls/Balls_00016.png", "00016"),
        ("../eksempelbilder/Balls/Balls_00032.png", "00032"),
        ("../eksempelbilder/Balls/Balls_00064.png", "00064"),
        ("../eksempelbilder/Balls/Balls_00128.png", "00128"),
        ("../eksempelbilder/Balls/Balls_00256.png", "00256"),
        ("../eksempelbilder/Balls/Balls_00512.png", "00512"),
        ("../eksempelbilder/Balls/Balls_01024.png", "01024"),
        ("../eksempelbilder/Balls/Balls_02048.png", "02048"),
        # ("../eksempelbilder/Balls/Balls_04096.png", "04096"),
        # ("../eksempelbilder/Balls/Balls_08192.png", "08192"),
        # ("../eksempelbilder/Balls/Balls_16384.png", "16384"),
    ])
