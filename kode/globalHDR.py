"""
This module is rendering the input image with a global function.

"""

import matplotlib.pyplot as plt
import numpy as np
import imageio as io


def global_edit(im, func="sqrt", effect=0.5):
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


def weighted_image(lum, chroma, effect, func="sqrt"):
    """
    This function is responsible for editing, weighting and putting the image back together.

    Editing happens on the luminance channel alone to reduce saturation.
    This function applies a ratio between luminance and chromasity, customisable through the GUI.
    Output image is defined by luminance * chromasity.

    :param lum: The luminance.
    :param chroma: The chromasity.
    :param effect: Scale of the function.
    :param func: Function for editing the luminance channel.
    :return: The edited picture.
    """
    lum = global_edit(lum, effect, func)

    lum = lum * 1           # *1 (sqrt)
    chroma = chroma ** .65     # **.6 (sqrt)

    result = lum * chroma
    result[result > 1] = 1
    result[result <= 0] = 0.1 * (10 ** -10)
    return result


def split_image(im, effect=0.5, func="sqrt"):
    """
    This function splits the input image into chromasity and luminance.

    :param im: Input image.
    :param func: Function for editing the luminance channel.
    :param effect: Scale of the function.
    :return: Output image.
    """
    lum = luminance(im)
    chroma = chromasity(im, lum)
    return weighted_image(lum, chroma, effect, func)


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

    edited = global_edit(image, "pow", 2)
    edited[edited > 1] = 1
    edited[edited <= 0] = 0
    show(edited)

    split = split_image(image, "pow", 2)
    show(split)
