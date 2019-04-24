"""
A module to calculate the HDR-channels in a set of images
"""
import imageio
import numpy as np


def standard_weighting(x):
    """
    A standard weighting function for hdr reconstruction

    :param x: The x value
    :type x: int

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
    :type x: Numpy array

    :return: The weighted value
    """
    x[x > 127] = 256 - x[x > 127]
    return x


def hdr_channel(images, shutter, smoothness, weighting):
    """
    This will create a HDR exposure map based on some observed pixels in some images

    :param images: The pixel values in a 2d array where i is the location and j is the image
    :type images: Numpy array

    :param shutter: log(shutter speed) or log(delta t) for each image j
    :type shutter: Numpy array

    :param smoothness: The lambda, or a constant value that sets the smoothness
    :type smoothness: A number

    :param weighting: The weighting function for a given value Z
    :type weighting: function that takes a int as parameter and returns a int

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
    :type channels: Numpy array

    :param shutter: log(shutter speed) or log(delta t) for each image j
    :type shutter: Numpy array

    :param smoothness: The lambda, or a constant value that sets the smoothness
    :type smoothness: int

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
    :type channels: Numpy array

    :param weighting: The weighting function
    :type weighting: Function that takes a numpy array and returns a numpy array

    :param hdr_graph: The HDR graph
    :type hdr_graph: Numpy array with 256 values

    :param shutter: The ln(shutter) speeds for the different images
    :type shutter: Numpy array

    :return: The hdr channel
    """
    w_value = weighting(channels.copy() + 1)  # + 1 to fix color glitch
    denum_w = w_value.sum(0)
    denum_w[denum_w == 0] = 1
    return np.exp((w_value * (hdr_graph[channels.astype(int)] - shutter[:, None, None])).sum(0) / denum_w)


def find_reference_points_for(images):
    """
    Returns the indexes for a set of images

    :param images: The images to find references for of type images.ImageSet
    :type images: A numpy array

    :return: The pixel indexes
    """
    channels = images.channels()
    shape = np.shape(channels)
    im_len = shape[-2] * shape[-1]
    spacing = max(int(im_len / 1000), 1)
    return np.arange(0, im_len, spacing)


def load_image(path):
    """
    This loads an image

    :param path: The path to the image
    :type path: str

    :return: a Image object
    """
    return np.array(imageio.imread(path))
