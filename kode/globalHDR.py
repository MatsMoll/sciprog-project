"""
This module is rendering the input image with a global function.

"""

import random as rand
import matplotlib.pyplot as plt
import numpy as np
import imageio as io


def edit(im, func):
    """
    Manipulated the given image with the given function and returns it directly.

    Note: Gamma-function is supposed to be a fun addition to this function when you
        get a random value between 0 and 1.

    :param im: Name of the image that is going to be manipulated.
    :param func: Name of the manipulating function.
    :return: Calls the manipulating function.
    """
    if func == "e":
        return np.exp(im)
    elif func == "ln":
        return np.log(im)
    elif func == "squared":
        return im ** 2
    elif func == "sqrt":
        return np.sqrt(im)
    elif func == "gamma":
        return im ** rand.uniform(0.0, 1.0)
    elif func == "bytwo":
        return im / 2
    elif func == "timestwo":
        return im * 2
    else:
        print("Unavailable function:", func, "\n-> Returned original image.")
        return im


def read_image(path="../eksempelbilder/Balls/Balls", image_format=".exr"):
    """
    Reads an image in a given path.

    Note: See example for formatting guidelines (including image format)
    Example: ../eksempelbilder/Balls/Balls.exr

    :param path: The path to the image
    :param image_format: The image format
    :return: The desired image
    """
    im = io.imread(path + image_format)

    im[im > 1] = 1
    im[im <= 0] = 0.1 * (10 ** -10)
    return im


def show(im):
    """
    Shows a given image whether it's monochrome or colorful.

    :param im: Input image
    """
    if im.ndim <= 2:
        plt.imshow(im, plt.cm.gray)
    else:
        plt.imshow(im.astype(float))
    plt.show()


def luminance(im):
    """
    Takes an input image and creates a new luminance channel. (L = R + G + B)

    :param im: Input image.
    :return: Luminance channel.
    """
    shape = (im.shape[0], im.shape[1], 1)
    lum_channel = np.zeros(shape)
    lum_channel[:, :, 0] = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2])
    return lum_channel


def chromasity(im, lum):
    """
    Takes an input image and returns the chromasity with the formula C = [R/L, G/L, B/L]

    :param im: Input image.
    :param lum: Luminance channel from input image.
    :return: The chromasity of the image.
    """
    return im / lum


def split_image(im):
    """
    This function splits the input image into chromasity and luminance.
    Editing happens on the luminance channel alone to reduce saturation.
    Output image is defined by luminance * chromasity and multiplied to give a better end result.

    :param im: Input image.
    :return: Output image.
    """
    lum = luminance(im)
    chroma = chromasity(im, lum)  # Ref. oppg: chrom = [x/L]
    new_lum = edit(lum, "sqrt")  # Ref. oppg: L -> f(L)

    # new_lum = new_lum ** (4)      # RATIO TO LOWER SATURATION
    # chroma = chroma * 1.5         # RATIO TO LOWER SATURATION

    result = new_lum * chroma  # * 2  # * 1.5-2.5 fungerer rimelig bra. høyere = lysere
    result[result > 1] = 1
    result[result <= 0] = 0.1 * (10 ** -10)
    return result


def compare(im1, im2):
    """
    Helping function that takes two input images
        and prints out the difference between their pixel values.

    :param im1: Input image 1
    :param im2: Input image 2
    :return: Prints the difference between the images.
    """
    return print(im2-im1)


image = read_image()
show(image)

edited = edit(image, "sqrt")
show(edited)

split = split_image(image)
show(split)

compare(image, split)
