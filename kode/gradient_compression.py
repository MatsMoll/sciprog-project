"""
A module that compress the gradient of an image
"""
import numpy as np
from cv2 import pyrUp, pyrDown, resize
from globalHDR import luminance
from hdr import ImageSet


def explicitly(n_points, n_time, initial_values, vector_values):
    """
    Calculates the explicit solution for a partial differential equation

    :param n_points: The number of iterations to calculate
    :param n_time: The length to calculate
    :param initial_values: The initial values
    :param vector_values: The divergence of the equation

    :return: A image
    """
    delta_t = 1 / (n_time - 1)
    u = initial_values
    for n in range(0, n_points):
        u[1:-1, 1:-1] = delta_t * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) \
                        - delta_t * (vector_values[1:-1, 1:-1, 0] + vector_values[1:-1, 1:-1, 1]) + u[1:-1, 1:-1]

    return u


def diff_two(u):
    """
    Calculates the change of a pixel value in both x and y axis

    :param u: The image to use

    :return: A matrix containing the change in x an y directions
    """
    g = np.zeros(u.shape + (2,))
    g[1:-1, :, 0] = u[2:, :] - u[1:-1, :]
    g[:, 1:-1, 1] = u[:, 2:] - u[:, 1:-1]
    return g


def compress_gradient(original, func, initial_value=None):
    """
    Compresses the length of the gradient vector and returns a new image
    based on this new vector

    :param original: The image to compress
    :param func: The function to use when compressing the length
    :param initial_value: The initial value to use when finding the new image. Will use the original if this is None

    :return: A new image fitting the compressed vector
    """
    du_0_len, du_0 = gradient_vectors(original)
    du_0_len[du_0_len == 0] = 0.0001

    f_du_len = func(du_0_len)

    f_du = f_du_len[:, :, None] * du_0 / du_0_len[:, :, None]

    div_f = divergence_matrix(f_du)

    if initial_value is None:
        return explicitly(10, 10, original, div_f)
    else:
        return explicitly(10, 10, initial_value, div_f)


def compress_gradient_pyr(original, func):
    """
    Compresses the gradient vector in the same way as `compress_gradient`, but using a Gaussian pyramid

    :param original: The luminace of the image to compress as a numpy matrix
    :param func: The function to use when compressing the length

    :return: The new image / luminace
    """

    shape = original.shape

    images = [original]
    lowest_res = original

    while images[-1].shape[0] * images[-1].shape[1] > 32 * 4:
        lowest_res = pyrDown(lowest_res)
        shape = np.shape(lowest_res)
        images.append(lowest_res)

    initial_value = luminance(images[-1])
    for i in range(1, len(images) + 1):
        lum = images[-i]

        if initial_value.shape != images[-i].shape[0:-1]:
            initial_value = resize(pyrUp(initial_value), tuple(reversed(lum.shape)))

        initial_value = compress_gradient(lum, func, initial_value=initial_value)

    return initial_value


def divergence_matrix(matrix):
    """
    Calculates the divergence of a matrix

    :param matrix: The matrix

    :return: The divergence matrix
    """
    div_f = np.zeros(matrix.shape)
    div_f[1:-1, :, 0] = matrix[1:-1, :, 0] - matrix[:-2, :, 0]
    div_f[:, 1:-1, 1] = matrix[:, 1:-1, 1] - matrix[:, :-2, 1]
    div_f[:, 0, 1] = matrix[:, 0, 1]
    plt.imshow(np.sqrt(div_f[:, :, 0] ** 2 + div_f[:, :, 1] ** 2), plt.cm.gray)
    plt.show()
    return div_f


def gradient_vectors(image):
    """
    Calculates the gradient vector and length of a image

    :param image: The image to calculate for

    :return: A tuple containing the length and vector
    """
    du_0 = diff_two(image)
    du_0_len = np.sqrt(du_0[:, :, 0] ** 2 + du_0[:, :, 1] ** 2)
    return du_0_len, du_0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import imageio
    from hdr import load_image

    #t = imageio.imread("../eksempelbilder/Balls/Balls.exr")

    test_im_set = color_images = ImageSet([
        ("../eksempelbilder/Tree/Tree_00001.png", "00001"),
        ("../eksempelbilder/Tree/Tree_00004.png", "00004"),
        ("../eksempelbilder/Tree/Tree_00016.png", "00016"),
        ("../eksempelbilder/Tree/Tree_00032.png", "00032"),
        ("../eksempelbilder/Tree/Tree_00064.png", "00064"),
        ("../eksempelbilder/Tree/Tree_00128.png", "00128"),
        ("../eksempelbilder/Tree/Tree_00256.png", "00256"),
        ("../eksempelbilder/Tree/Tree_00512.png", "00512"),
        #("../eksempelbilder/Tree/Tree_01024.png", "01024"),
        #("../eksempelbilder/Tree/Tree_02048.png", "02048"),
        #("../eksempelbilder/Tree/Tree_04096.png", "04096"),
        #("../eksempelbilder/Tree/Tree_08192.png", "08192"),
        #("../eksempelbilder/Tree/Tree_16384.png", "16384"),
    ])
    test_im = test_im_set.hdr_image(10)

    for image in test_im_set.images:
        plt.imshow(image)
        plt.show()

    gamma_im = test_im ** 0.2

    test_im = (test_im - test_im.min()) / (test_im.max() - test_im.min())

    plt.imshow((gamma_im - gamma_im.min()) / (gamma_im.max() - gamma_im.min()))
    plt.show()

    plt.imshow(test_im.sum(2) / 3, plt.cm.gray)
    plt.show()

    lum_im = luminance(test_im)

    print(lum_im.max())
    print(lum_im.min())
    print(lum_im.shape)

    #pyr = np.exp(compress_gradient(np.log(lum_im.copy()), lambda x: x ** 0.8))
    pyr = np.exp(compress_gradient_pyr(np.log(lum_im.copy()), lambda x: x ** 0.8))
    print(test_im.shape)
    print(lum_im.shape)
    print(pyr.shape)
    print(test_im.max())
    print(test_im.min())
    print(pyr.max())
    print(pyr.min())

    test_im_reconstruct = (test_im[:, :, :] / lum_im[:, :, None]) ** 1 * pyr[:, :, None]
    test_im_reconstruct = test_im_reconstruct ** 0.2

    #for image in pyr:
    plt.imshow((pyr - pyr.min())/(pyr.max() - pyr.min()), plt.cm.gray)
    plt.show()

    plt.imshow((test_im_reconstruct - test_im_reconstruct.min())/(test_im_reconstruct.max() - test_im_reconstruct.min()))
    #im = test_im * gradient[:, :, None]
    #edit_im = test_im
    #print(im.max(), im.min())
    #edit_im[:, :, 0] = (test_im[:, :, 0] / lum_im) ** 0.6 * (im.max() - im)
    #edit_im[:, :, 1] = (test_im[:, :, 1] / lum_im) ** 0.6 * (im.max() - im)
    #edit_im[:, :, 2] = (test_im[:, :, 2] / lum_im) ** 0.6 * (im.max() - im)
    #plt.imshow((test_im - test_im.min()) / (test_im.max() - test_im.min()))
    #plt.imshow((lum_im - lum_im.min()) / (lum_im.max() - lum_im.min()), plt.cm.gray)
    #plt.imshow((gradient - gradient.min()) / (gradient.max() - gradient.min()), plt.cm.gray)
    #plt.imshow((im - im.min()) / (im.max() - im.min()), plt.cm.gray)
    #plt.imshow((edit_im - edit_im.min()) / (edit_im.max() - edit_im.min()))
    plt.show()
