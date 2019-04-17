"""
This module is rendering the input image with a global function.

"""

import matplotlib.pyplot as plt
import numpy as np
import imageio as io


def edit_globally(im, effect=0.5, func="sqrt"):
    """
    Manipulated the input image with the given function (and effect) and returns it directly.

    :param im: Input image.
    :param func: Function for editing.
    :param effect: Scale the function.
    :return: Calls the manipulating function.
    """
    if func == "e":
        return np.exp(im)
    elif func == "ln":
        return np.log(im)
    elif func == "pow":
        effect = np.clip(effect, 0, 4)
        return im ** effect
    elif func == "sqrt":
        return np.sqrt(im)
    elif func == "gamma":
        effect = np.clip(effect, 0, 1)
        return im ** effect
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
    im[im > 1] = 1
    im[im <= 0] = 0.1 * (10 ** -10)
    if im.ndim <= 2:
        plt.imshow(im, plt.cm.gray)
    else:
        plt.imshow(im.astype(float))
    plt.show()


def luminance(im):
    """
    Takes an input image and creates a new luminance channel. (L = R + G + B)

    Note!
    Monochrome images: Luminance channel = brightness in each pixel.
    Colorful images: Luminance channel = sum of brightness of the color channels.

    :param im: Input image.
    :return: Luminance channel.
    """
    if im.ndim <= 2:
        lum_channel = im
    else:
        lum_channel = im.sum(2)
    return lum_channel


def chromasity(im, lum):
    """
    Takes an input image and returns the chromasity with the formula C = [R/L, G/L, B/L]

    :param im: Input image.
    :param lum: Luminance channel from input image.
    :return: The chromasity of the image.
    """
    return im / lum


def weighted_image(lum, chroma, effect, lum_scale, chrom_scale, func="sqrt"):
    """
    This function is responsible for editing, weighting and putting the image back together.

    Editing happens on the luminance channel alone to reduce saturation.
    This function applies a ratio between luminance and chromasity, customisable through the GUI.
    Output image is defined by luminance * chromasity.

    :param lum: The luminance.
    :param chroma: The chromasity.
    :param effect: Scale of the function.
    :param lum_scale: Weighted ratio for lum * chrom.
    :param chrom_scale: Weighted ratio for lum * chrom.
    :param func: Function for editing the luminance channel.
    :return: The edited picture.
    """
    lum = edit_globally(lum, effect, func)

    lum = lum * lum_scale
    chroma = chroma * chrom_scale

    result = lum * chroma
    return result


def edit_luminance(im, effect=0.5, lum_scale=1, chrom_scale=1, func="sqrt"):
    """
    This function splits the input image into chromasity and luminance.

    :param im: Input image.
    :param effect: Scale of the function.
    :param lum_scale: Weighted ratio for lum * chrom.
    :param chrom_scale: Weighted ratio for lum * chrom.
    :param func: Function for editing the luminance channel.
    :return: Output image.
    """
    lum = luminance(im)
    chroma = chromasity(im, lum)
    return weighted_image(lum, chroma, effect, lum_scale, chrom_scale, func)


def compare(im1, im2):
    """
    Helping function that takes two input images
        and prints out the difference between their pixel values.

    :param im1: Input image 1
    :param im2: Input image 2
    :return: Prints the difference between the images.
    """
    return print(im2-im1)


if __name__ == '__main__':
    image = read_image()
    show(image)

    edited = edit_globally(image, 2, "sqrt")
    show(edited)

    split = edit_luminance(image, 2, 1, 1, "sqrt")
    show(split)
