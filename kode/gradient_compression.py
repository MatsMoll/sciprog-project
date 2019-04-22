"""
A module that compress the gradient of an image
"""
import numpy as np
from cv2 import pyrUp, pyrDown, resize
from globalHDR import luminance
from hdr import ImageSet


def explicitly(n_points, n_time, initial_values, vector_values):
    """
    :param n_points:
    :param n_time:
    :param initial_values:
    :param vector_values:
    :return: A image
    """
    delta_t = 1 / (n_time - 1)
    u = initial_values
    for n in range(0, n_points):
        #vector_values = gradient_vector(luminance(u), n)
        #print(u.max(), u.min())
        #u = luminance(u) * (0.1 / vector_values) * (vector_values / 0.1) ** 0.85
        u[1:-1, 1:-1] = delta_t * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) - delta_t * (vector_values[1:-1, 1:-1, 0] + vector_values[1:-1, 1:-1, 1]) + u[1:-1, 1:-1]

    return u


def phi(u, b, k):
    """
    Phi function that calculates the importance of a gradient
    :param u: The image
    :param b: A constant that weight large gradients
    :param k: The image level in a gauss pyramid
    :return:
    """
    u_diff = diff(u)
    print(u_diff.min())
    print(u_diff.max())
    a = u_diff.mean()
    return a / np.abs(u_diff) * (np.abs(u_diff) / a) ** b


def phi_two(u, b, k):
    u_diff = diff_two(u, k)
    print(u_diff.min())
    print(u_diff.max())
    a = u_diff.mean()
    return a / np.abs(u_diff) * (np.abs(u_diff) / a) ** b


def diff(u):
    u[1:-1, 1:-1] = u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
    return u


def diff_two(u):
    g = np.zeros(u.shape + (2,))
    print(g.shape)
    g[1:-1, :, 0] = u[2:, :] - u[1:-1, :]
    g[:, 1:-1, 1] = u[:, 2:] - u[:, 1:-1]
    return g


def images(original):

    shape = np.shape(original)

    images = [original]
    lowest_res = original

    while images[-1].shape[0] * images[-1].shape[1] > 32 * 4:
        print("New: ", shape, (int(shape[0] / 2), int(shape[1] / 2)))
        lowest_res = pyrDown(lowest_res)
        shape = np.shape(lowest_res)
        images.append(lowest_res)

    diffs = [phi(images[0], 0.8, len(images) - 1)]

    for i in range(1, len(images) - 1):
        print(i)
        shape = images[len(images) - 2 - i].shape

        up = pyrUp(diffs[i - 1])
        new_im = phi(images[len(images) - 2 - i], 0.8, len(images) - 2 - i) * up[:shape[0], :shape[1]]
        diffs.append(new_im)
        #diffs.append(pyrUp(diffs[i - 1]) * phi(images[len(images) - 1 - i], 0.1, 0.8))

    return diffs
    #return np.sqrt(diffs[-1][:, :, 0] ** 2 + diffs[-1][:, :, 1] ** 2)


def images_two(original):

    shape = original.shape

    images = [original]
    lowest_res = original

    while images[-1].shape[0] * images[-1].shape[1] > 32 * 4:
        print("New: ", shape, (int(shape[0] / 2), int(shape[1] / 2)))
        lowest_res = pyrDown(lowest_res)
        shape = np.shape(lowest_res)
        images.append(lowest_res)

    diffs = [phi_two(images[0], 0.8, len(images) - 1)]
    shape = original.shape

    for i in range(1, len(images) - 1):
        print(i)
        up = images[i]
        for j in range(0, i):
            print("Changeing", up.shape)
            up = pyrUp(up)

        print(up.shape)
        print(diffs[-1].shape)
        new_im = phi_two(up[:shape[0], :shape[1]], 0.8, i) * diffs[-1]
        diffs.append(new_im)
        #diffs.append(pyrUp(diffs[i - 1]) * phi(images[len(images) - 1 - i], 0.1, 0.8))

    #return diffs
    return [np.sqrt(diffs[-1][:, :, 0] ** 2 + diffs[-1][:, :, 1] ** 2)]


def compress(original, func, initial_value=None):
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

    div_f = div_matrix(f_du)

    if initial_value is None:
        return explicitly(10, 10, original, div_f)
    else:
        return explicitly(10, 10, initial_value, div_f)


def compress_pyr(original, func):

    shape = original.shape

    images = [original]
    lowest_res = original

    while images[-1].shape[0] * images[-1].shape[1] > 32 * 4:
        print("New: ", shape, (int(shape[0] / 2), int(shape[1] / 2)))
        lowest_res = pyrDown(lowest_res)
        shape = np.shape(lowest_res)
        images.append(lowest_res)

    initial_value = luminance(images[-1])
    for i in range(1, len(images) + 1):
        lum = images[-i]

        print(images[-i].shape)
        print(initial_value.shape)

        if initial_value.shape != images[-i].shape[0:-1]:
            initial_value = resize(pyrUp(initial_value), tuple(reversed(lum.shape)))
            print(lum.shape)
            print("Upscaled", initial_value.shape)

        initial_value = compress(lum, func, initial_value=initial_value)

        #initial_value = (images[-i] / lum[:, :]) ** 1 * compressed

    return initial_value


def div_matrix(matrix):
    div_f = np.zeros(matrix.shape)
    div_f[1:-1, :, 0] = matrix[1:-1, :, 0] - matrix[:-2, :, 0]
    div_f[:, 1:-1, 1] = matrix[:, 1:-1, 1] - matrix[:, :-2, 1]
    plt.imshow(np.sqrt(div_f[:, :, 0] ** 2 + div_f[:, :, 1] ** 2), plt.cm.gray)
    plt.show()
    return div_f


def gradient_vectors(image):
    du_0 = diff_two(image)
    du_0_len = np.sqrt(du_0[:, :, 0] ** 2 + du_0[:, :, 1] ** 2)
    return du_0_len, du_0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import imageio
    from hdr import load_image

    #t = imageio.imread("../eksempelbilder/MtTamNorth/MtTamNorth.exr")

    test_im_set = color_images = ImageSet([
        ("../eksempelbilder/MtTamNorth/MtTamNorth_00001.png", "00001"),
        ("../eksempelbilder/MtTamNorth/MtTamNorth_00004.png", "00004"),
        ("../eksempelbilder/MtTamNorth/MtTamNorth_00016.png", "00016"),
        ("../eksempelbilder/MtTamNorth/MtTamNorth_00032.png", "00032"),
        ("../eksempelbilder/MtTamNorth/MtTamNorth_00064.png", "00064"),
        ("../eksempelbilder/MtTamNorth/MtTamNorth_00128.png", "00128"),
        ("../eksempelbilder/MtTamNorth/MtTamNorth_00256.png", "00256"),
        ("../eksempelbilder/MtTamNorth/MtTamNorth_00512.png", "00512"),
        #("../eksempelbilder/MtTamNorth/MtTamNorth_01024.png", "01024"),
        #("../eksempelbilder/MtTamNorth/MtTamNorth_02048.png", "02048"),
        #("../eksempelbilder/MtTamNorth/MtTamNorth_04096.png", "04096"),
        #("../eksempelbilder/MtTamNorth/MtTamNorth_08192.png", "08192"),
        #("../eksempelbilder/MtTamNorth/MtTamNorth_16384.png", "16384"),
        # load_image("../eksempelbilder/MtTamNorth/MtTamNorth_", "01024"),
        # load_image("../eksempelbilder/MtTamNorth/MtTamNorth_", "02048"),
    ])
    test_im = test_im_set.hdr_image(10) ** 0.2

    test_im = (test_im - test_im.min()) / (test_im.max() - test_im.min())

    plt.imshow(test_im)
    plt.show()

    plt.imshow(test_im.sum(2) / 3, plt.cm.gray)
    plt.show()

    lum_im = luminance(test_im)

    print(lum_im.max())
    print(lum_im.min())
    print(lum_im.shape)

    #pyr = np.exp(compress(np.log(lum_im.copy()), np.sqrt))
    pyr = np.exp(compress_pyr(np.log(lum_im.copy()), lambda x: x ** 0.8))

    print(test_im.shape)
    print(lum_im.shape)
    print(pyr.shape)
    print(test_im.max())
    print(test_im.min())
    print(pyr.max())
    print(pyr.min())

    test_im_reconstruct = (test_im[:, :, :] / lum_im[:, :, None]) ** 1 * pyr[:, :, None]

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
