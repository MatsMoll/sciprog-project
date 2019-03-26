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
    if x > 128:
        return 255 - x
    else:
        return x


def standard_weighting_vector(x):
    """
    A standard weighting function for hdr reconstruction
    :param x: The x value
    :return: The weighted value
    """
    res = x
    res[x > 128] = 255 - x[x > 128]
    return res


def hdr_channel(images, shutter, smoothness, weighting, z_mid=255):
    """
    This will create a HDR exposure map based on some observed pixels in some images

    :param images: The pixel values in a 2d array where i is the location and j is the image
    :param shutter: log(shutter speed) or log(delta t) for each image j
    :param smoothness: The lambda, or a constant value that sets the smoothness
    :param weighting: The weighting function for a given value Z
    :param z_mid: The z value to use a radiance unit (Can be different from channel to channel)
    :return: A tuple containing log(exposure) for the pixel value Z and log(irradiance) for pixel i
    """

    n = 256
    number_of_pic, number_of_pixels = np.shape(images)

    a = np.zeros((number_of_pixels * number_of_pic + n + 1, number_of_pixels + n))
    b = np.zeros(np.shape(a)[0])

    # Setting equations
    pixel = 0
    for j in range(0, number_of_pic):
        for i in range(0, number_of_pixels):
            pixel_value = images[j][i]
            weighted_value = weighting(pixel_value + 1)
            a[pixel][int(round(pixel_value))] = weighted_value
            a[pixel][n + i] = -weighted_value
            b[pixel] = weighted_value * shutter[j]     # B[j] is the shutter speed
            pixel += 1

    # Setting Z_mid = 0 (radiance unit level)
    a[pixel][int(z_mid)] = 1
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
        z_mean = 256 - channel.mean()
        g, ln_e = hdr_channel(channel, shutter, smoothness, standard_weighting, z_mid=z_mean)
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
    w_value = weighting(channels[:, :, :] + 1)
    hdr_image = (w_value * (hdr_graph[channels[:, :, :].astype(int)]
                            - shutter[:, None, None])).sum(0) / w_value.sum(0)
    return hdr_image


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
    """
    def __init__(self, images):
        if isinstance(images, list):
            self.images = np.array([])
            self.shutter_speed = []
            for path, shutter in images:
                self.shutter_speed.append(np.log(float(shutter) * 10 ** (-5)))
                if self.images.size == 0:
                    self.images = np.array([load_image(path)])
                else:
                    self.images = np.append(self.images, [load_image(path)], axis=0)
            self.original_shape = np.shape(self.images[0])
            self.shutter_speed = np.array(self.shutter_speed)
        else:
            self.images = images

    def hdr_image(self, smoothness):
        """
        Generates a hdr image
        :param smoothness: The smoothness on the curve
        :return: The hdr image
        """
        curve = self.hdr_curve(smoothness)
        image = np.zeros(self.original_shape)
        channels = self.channels()
        for i in range(0, channels.ndim - 1):
            image[:, :, i] = reconstruct_image(  # Red
                channels[i], standard_weighting_vector, curve[i][0], self.shutter_speed)
        return image

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
            chan = np.zeros(shape[-1:] + shape[:-1])
            for i in range(0, shape[-1]):
                chan[i] = self.images[:, :, :, i]

            return chan


def load_image(path):
    """
    This loads an image

    :param path: The path to the image
    :return: a Image object
    """
    image = imageio.imread(path)
    image[image == 0] = 1
    return image


if __name__ == "__main__":
    # Testing
    # color_image = load_color_image("../eksempelbilder/Balls/Balls_", "00256")

    color_images = ImageSet([
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
        #("../eksempelbilder/Balls/Balls_", "04096"),
        #("../eksempelbilder/Balls/Balls_", "08192"),
        #("../eksempelbilder/Balls/Balls_", "16384"),
        # load_image("../eksempelbilder/Balls/Balls_", "01024"),
        # load_image("../eksempelbilder/Balls/Balls_", "02048"),
    ])
    #color_hrd_map = color_images.hdr_curve(10)

    z_values = np.arange(0, 256)

    # print(np.shape(ln_e_values))
    # print(np.shape(g))
    # print(np.asarray(first[0]).reshape(-1))

    #im = image_set.channels()
    #print(np.shape(im))


    # hdrImage = transform_image(im, w, g_values, image_set.shutter_speed)

    #color_channels = color_images.channels()

    #gray_im = reconstruct_image(
    #    image_set.channels(), standard_weighting_vector, g_values, image_set.shutter_speed)

    #color_im = np.zeros(color_images.original_shape)

    #color_im[:, :, 0] = reconstruct_image(  # Red
    #    color_channels[0], standard_weighting_vector, color_hrd_map[0][0], color_images.shutter_speed)
    #color_im[:, :, 1] = reconstruct_image(  # Green
    #    color_channels[1], standard_weighting_vector, color_hrd_map[1][0], color_images.shutter_speed)
    #color_im[:, :, 2] = reconstruct_image(  # Blue
    #    color_channels[2], standard_weighting_vector, color_hrd_map[2][0], color_images.shutter_speed)

    color_im = color_images.hdr_image(10)
    current_im = 2

    # im = image_set[current_im].original_image()
    # hdr_im = transform_image_with(hdr_map[0], im, image_set[current_im].original_shape)
    # plt.imshow(hdr_im, plt.cm.gray)
    # plt.imshow(color_image.original_image() / 255)
    # plt.imshow(color_image.original_image() / 255)

    #plt.plot(color_hrd_map[0][0], z_values)
    #plt.plot(color_hrd_map[1][0], z_values)
    #plt.plot(color_hrd_map[2][0], z_values)
    # plt.plot(g_values, z_values)

    # plt.plot(color_hrd_map[2][:255], z_values)

    # print(color_im)

    print("max", color_im.max())
    print("min", color_im.min())
    color_im = np.exp(color_im) ** 0.25
    color_im = color_im + abs(color_im.min())
    print(color_im.min(), color_im.max())
    plt.imshow(color_im / color_im.max())
    # plt.imshow(hdrImage.reshape(image_set.images[0].original_shape), plt.cm.gray)

    plt.show()
