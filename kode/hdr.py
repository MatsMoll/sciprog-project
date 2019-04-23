"""
A module to calculate the HDR-channels in a set of images
"""
import imageio
import numpy as np
import matplotlib.pyplot as plt


def standard_weighting(x):
    """
    A standard weighting function for hdr reconstruction

    :param x: The x value

    :return: The weighted value
    """
    if x >= 128:
        return 255 - x
    else:
        return x


def standard_weighting_vector(x):
    """
    A standard weighting function for hdr reconstruction

    :param x: The x value

    :return: The weighted value
    """
    x[x > 127] = 256 - x[x > 127]
    return x


def hdr_channel(images, shutter, smoothness, weighting):
    """
    This will create a HDR exposure map based on some observed pixels in some images

    :param images: The pixel values in a 2d array where i is the location and j is the image
    :param shutter: log(shutter speed) or log(delta t) for each image j
    :param smoothness: The lambda, or a constant value that sets the smoothness
    :param weighting: The weighting function for a given value Z

    :return: A tuple containing log(exposure) for the pixel value Z and log(irradiance) for pixel i
    """

    if images.max() > 255:
        raise ValueError("Image contain values higher then 255")
    if images.min() < 0:
        raise ValueError("Image contain values lower then 0")

    n = 256
    number_of_pic, number_of_pixels = np.shape(images)

    images = images.astype(int)

    a = np.zeros((number_of_pixels * number_of_pic + n - 1, number_of_pixels + n))
    b = np.zeros(np.shape(a)[0])

    # Setting equations
    pixel = 0
    for j in range(0, number_of_pic):
        for i in range(0, number_of_pixels):
            pixel_value = images[j][i]
            weighted_value = weighting(pixel_value)
            a[pixel][pixel_value] = weighted_value
            a[pixel][n + i] = -weighted_value
            b[pixel] = weighted_value * shutter[j]     # B[j] is the shutter speed
            pixel += 1

    # Setting Z_mid = 0 (radiance unit level)
    a[pixel][128] = 1
    pixel += 1

    # Smoothing out the result
    for i in range(0, n - 2):
        weighted_value = weighting(i + 1)
        a[pixel][i] = smoothness * weighted_value
        a[pixel][i + 1] = -2 * smoothness * weighted_value
        a[pixel][i + 2] = smoothness * weighted_value
        pixel += 1

    result = np.linalg.lstsq(a, b, rcond=None)
    return result[0][0:n], result[0][n:]


def hdr_color_channels(channels, shutter, smoothness):
    """
    This will create a HDR exposure map based on some observed pixels in some images

    :param channels: The pixel values in a 3d array where i is the location and j is the image
    :param shutter: log(shutter speed) or log(delta t) for each image j
    :param smoothness: The lambda, or a constant value that sets the smoothness

    :return: A tuple containing log(exposure) for the pixel value Z and log(irradiance) for pixel i
    """
    result = []   # [[g_r(z), ln(E_r)], ..., [g_b(z), ln(E_b)]]
    shape = np.shape(channels)
    one_dim_channels = channels
    if len(shape) == 4:
        one_dim_channels = channels.reshape((shape[0], shape[1], shape[2] * shape[3]))

    for channel in one_dim_channels:
        print("HDR")
        g, ln_e = hdr_channel(channel, shutter, smoothness, standard_weighting)
        print("Done")
        result.append((g, ln_e))
    return result


def reconstruct_image(channels, weighting, hdr_graph, shutter):
    """
    Reconstruct a image from a hdr graph

    :param channels: The different channels from the different images
    :param weighting: The weighting function
    :param hdr_graph: The HDR graph
    :param shutter: The ln(shutter) speeds for the different images

    :return: The hdr channel
    """
    w_value = weighting(channels.copy() + 1)
    denum_w = w_value.sum(0)
    denum_w[denum_w == 0] = 1
    return np.exp((w_value * (hdr_graph[channels.astype(int)] - shutter[:, None, None])).sum(0) / denum_w)


def find_reference_points_for(images):
    """
    Returns the indexes for a set of images

    :param images: The images to find references for of type images.ImageSet

    :return: The pixel indexes
    """
    channels = images.channels()
    shape = np.shape(channels)
    im_len = shape[-2] * shape[-1]
    spacing = max(int(im_len / 1000), 1)
    return np.arange(0, im_len, spacing)


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
                    self.images = np.array([load_image(path)])
                else:
                    self.images = np.append(self.images, [load_image(path)], axis=0)

            if len(images) != 0:
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
            output_image = reconstruct_image(
                channels, standard_weighting_vector, curve[0], self.shutter_speed)
        else:
            for i in range(0, 3):
                output_image[:, :, i] = reconstruct_image(
                    channels[i], standard_weighting_vector, curve[i][0], self.shutter_speed)
        return output_image

    def hdr_curve(self, smoothness):
        """
        Generates a hdr curve for a image set

        :param smoothness: The smoothness on the curve

        :return: The hdr curve
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
        else:
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
            chan = np.zeros((3,) + shape[:-1])
            for i in range(0, 3):
                chan[i] = self.images[:, :, :, i]

            return chan


def load_image(path):
    """
    This loads an image

    :param path: The path to the image

    :return: a Image object
    """
    return np.array(imageio.imread(path))


def test_image_set():
    """
    Creates a image set for testing

    :return: The ImageSet with the info
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
        # ("../eksempelbilder/Balls/Balls_01024.png", "01024"),
        # ("../eksempelbilder/Balls/Balls_02048.png", "02048"),
        # ("../eksempelbilder/Balls/Balls_04096.png", "04096"),
        # ("../eksempelbilder/Balls/Balls_08192.png", "08192"),
        # ("../eksempelbilder/Balls/Balls_16384.png", "16384"),
        # load_image("../eksempelbilder/Balls/Balls_", "01024"),
        # load_image("../eksempelbilder/Balls/Balls_", "02048"),
    ])


if __name__ == "__main__":
    # Testing

    color_images = test_image_set()

    z_values = np.arange(0, 256)
    color_im = color_images.hdr_image(10)
    current_im = 2

    print("max", color_im.max())
    print("min", color_im.min())

    color_im = (color_im - color_im.min()) / (color_im.max() - color_im.min()) * 255
    color_im = color_im ** 0.2

    for image in color_images.images:
        print(image.min(), image.max())
        print(image.mean())
        plt.imshow(image)
        plt.show()

    print(color_im.min(), color_im.max())
    print(color_im.mean())
    plt.imshow((color_im - color_im.min()) / (color_im.max() - color_im.min()))
    plt.show()
    # plt.imshow(hdrImage.reshape(image_set.images[0].original_shape), plt.cm.gray)
