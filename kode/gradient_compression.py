"""
A module that compress the gradient of an image
"""
import numpy as np
from cv2 import pyrUp, pyrDown, resize
from globalHDR import luminance


def explicitly(filter_config, initial_values, vector_values):
    """
    Calculates the explicit solution for a partial differential equation

    :param filter_config: The config of the filter
    :type filter_config: GradientFilterConfig

    :param initial_values: The initial values
    :type initial_values: Numpy array

    :param vector_values: The divergence of the equation
    :type vector_values: Numpy array

    :return: A image
    """
    delta_t = 1 / (filter_config.iteration_distance - 1)
    u = initial_values
    for _ in range(0, filter_config.iteration_amount):

        estimate_top_rand = u[0, 1:-1] + u[2, 1:-1] + u[0, 2:] + u[0, :-2] - 4 * u[0, 1:-1]
        estimate_bottom_rand = u[-3, 1:-1] + u[-1, 1:-1] + u[-1, 2:] + u[-1, :-2] - 4 * u[-1, 1:-1]
        estimate_left_rand = u[1:-1, 0] + u[1:-1, 2] + u[2:, 0] + u[:-2, 0] - 4 * u[1:-1, 0]
        estimate_right_rand = u[1:-1, -3] + u[1:-1, -1] + u[2:, -1] + u[:-2, -1] - 4 * u[1:-1, -1]

        u[1:-1, 1:-1] = delta_t * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) \
                        - delta_t * (vector_values[1:-1, 1:-1, 0] + vector_values[1:-1, 1:-1, 1]) + u[1:-1, 1:-1]

        u[0, 1:-1] = delta_t * estimate_top_rand \
                     - delta_t * (vector_values[0, 1:-1, 0] + vector_values[0, 1:-1, 1]) + u[0, 1:-1]
        u[-1, 1:-1] = delta_t * estimate_bottom_rand \
                      - delta_t * (vector_values[-1, 1:-1, 0] + vector_values[-1, 1:-1, 1]) + u[-1, 1:-1]
        u[1:-1, 0] = delta_t * estimate_left_rand \
                     - delta_t * (vector_values[1:-1, 0, 0] + vector_values[1:-1, 0, 1]) + u[1:-1, 0]
        u[1:-1, -1] = delta_t * estimate_right_rand \
                      - delta_t * (vector_values[1:-1, -1, 0] + vector_values[1:-1, -1, 1]) + u[1:-1, -1]

    return u


def image_gradient(u):
    """
    Calculates the gradient of a pixel value in both x and y axis

    :param u: The image to use
    :type u: Numpy array

    :return: A matrix containing the change in x an y directions
    """
    g = np.zeros(u.shape + (2,))

    g[1:-1, :, 0] = u[2:, :] - u[1:-1, :]
    g[:, 1:-1, 1] = u[:, 2:] - u[:, 1:-1]

    g[0, :, 0] = g[1, :, 0]
    g[-1, :, 0] = g[-2, :, 0]
    g[:, 0, 1] = g[:, 1, 1]
    g[:, -1, 1] = g[:, -2, 1]
    return g


def compress_gradient(original, filter_config, initial_value=None):
    """
    Compresses the length of the gradient vector and returns a new image
    based on this new vector

    :param original: The image to compress
    :type original: Numpy array

    :param filter_config: The config of the filter
    :type filter_config: GradientFilterConfig

    :param initial_value: The initial value to use when finding the new image. Will use the original if this is None
    :type initial_value: Numpy array or None

    :return: A new image fitting the compressed vector
    """
    du_0_len, du_0 = gradient_vectors(original)

    f_du_len = filter_config.func(du_0_len)
    du_0_len[du_0_len == 0] = 1
    f_du = f_du_len[:, :, None] * du_0 / du_0_len[:, :, None]
    div_f = divergence_matrix(f_du)

    if initial_value is None:
        return explicitly(filter_config, original, div_f)
    else:
        return explicitly(filter_config, initial_value, div_f)


def compress_gradient_pyr(original, filter_config):
    """
    Compresses the gradient vector in the same way as `compress_gradient`, but using a Gaussian pyramid

    :param original: The luminace of the image to compress
    :type original: Numpy array

    :param filter_config: The config of the filter
    :type filter_config: GradientFilterConfig

    :return: The new image / luminace
    """

    images = [original]
    lowest_res = original

    while images[-1].shape[0] * images[-1].shape[1] > 32 * 4:
        lowest_res = pyrDown(lowest_res)
        images.append(lowest_res)

    initial_value = luminance(images[-1])
    for i in range(1, len(images) + 1):
        lum = images[-i]

        if initial_value.shape != images[-i].shape[0:-1]:
            initial_value = resize(pyrUp(initial_value), tuple(reversed(lum.shape)))

        initial_value = compress_gradient(lum, filter_config, initial_value=initial_value)

    return initial_value


def gradient_compress_color(im, filter_config):
    """
    Compresses the gradient of a color image

    :param im: The image to compress
    :type im: Numpy array

    :param filter_config: The config of the filter
    :type filter_config: GradientFilterConfig

    :return: The new image
    """
    lum = luminance(im).reshape(im.shape[:-1])
    new_lum = lum
    if filter_config.use_pyramid:
        new_lum = compress_gradient_pyr(np.log(lum), filter_config)
    else:
        new_lum = compress_gradient(np.log(lum), filter_config)
    new_lum = np.exp(new_lum)
    if im.ndim == 3:
        return (im[:, :, :] / lum[:, :, None]) ** filter_config.saturation * new_lum[:, :, None]
    else:
        return (im[:, :] / lum[:, :]) ** filter_config.saturation * new_lum[:, :]


def divergence_matrix(matrix):
    """
    Calculates the divergence of a matrix

    :param matrix: The matrix
    :type matrix: Numpy array

    :return: The divergence matrix
    """
    div_f = np.zeros(matrix.shape)
    div_f[1:-1, :, 0] = matrix[1:-1, :, 0] - matrix[:-2, :, 0]
    div_f[:, 1:-1, 1] = matrix[:, 1:-1, 1] - matrix[:, :-2, 1]

    div_f[0, :, 0] = div_f[1, :, 0]
    div_f[-1, :, 0] = div_f[-2, :, 0]
    div_f[:, 0, 1] = div_f[:, 1, 1]
    div_f[:, -1, 1] = div_f[:, -2, 1]
    return div_f


def gradient_vectors(image_matrix):
    """
    Calculates the gradient vector and length of a image

    :param image_matrix: The image to calculate for
    :type image_matrix: Numpy array

    :return: A tuple containing the length and vector
    """
    du_0 = image_gradient(image_matrix)
    du_0_len = np.sqrt(du_0[:, :, 0] ** 2 + du_0[:, :, 1] ** 2)
    return du_0_len, du_0
