"""
This module is rendering the input image with a global function.

"""

import matplotlib.pyplot as plt
import numpy as np
import imageio as io


def read_image(path="../eksempelbilder/StillLife/StillLife", image_format=".exr"):
    """
    Reads an image from a given path.

    Note: See example for formatting guidelines (including image format).
    Example: ../eksempelbilder/StillLife/StillLife.exr

    :param path: The path to the image.
    :type path: String.

    :param image_format: The image format.
    :type image_format: String.

    :return: The desired image.
    """
    return io.imread(path + image_format)


def show(im):
    """
    Shows a given image whether it's monochrome or containing color channels.

    :param im: Input image.
    :type im: Numpy array.
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
    Takes an input image and creates a new luminance channel. ( L = R + G + B )

    Note!
    Monochrome images: Luminance channel = brightness in each pixel.
    Colorful images: Luminance channel = sum of brightness in the color channels.

    :param im: Input image.
    :type im: Numpy array.

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
    Takes an input image and returns the chromasity. ( C = [R/L, G/L, B/L] )

    :param im: Input image.
    :type im: Numpy array.

    :param lum: Luminance of input image.
    :type lum: Numpy array.

    :return: The chromasity of the image.
    """
    return im / lum


def edit_globally(im, effect):
    """
    Manipulated the input image with effects from the effect config and returns the image directly.

    :param im: Input image.
    :type im: Numpy array.

    :param effect: Config for the effects.
    :type effect: EffectConfig class.

    :return: The manipulated image.
    """
    if effect.func == "e":
        return np.exp(im)
    elif effect.func == "ln":
        return np.log(im)
    elif effect.func == "pow":
        effect.level = np.clip(effect.level, 0, 4)
        return im ** effect.level
    elif effect.func == "sqrt":
        return np.sqrt(im)
    elif effect.func == "gamma":
        effect.level = np.clip(effect.level, 0, 1)
        return im ** effect.level
    else:
        print("Unavailable function:", effect.func, "\n-> Returned original image.")
        return im


def edit_luminance(lum, chroma, effect):
    """
    This function is responsible for editing, weighting and merging the image.

    Editing happens on the luminance channel alone to reduce saturation.
    Scales, levels and other properties are customisable through the GUI.
    Output image is defined by luminance * chromasity.

    :param lum: The luminance.
    :type lum: Numpy array.

    :param chroma: The chromasity.
    :type chroma: Numpy array.

    :param effect: Config for the effects.
    :type effect: EffectConfig class.

    :return: The edited image.
    """
    lum = edit_globally(lum, effect.level, effect.func)

    lum = lum * effect.lum_scale
    chroma = chroma * effect.chrom_scale

    result = lum * chroma
    return result


def split_image(im, effect):
    """
    This function splits the input image into chromasity and luminance.

    :param im: Input image.
    :type im: Numpy array.

    :param effect: Config for the effects.
    :type effect: EffectConfig class.

    :return: Output image.
    """
    lum = luminance(im)
    chroma = chromasity(im, lum)
    return edit_luminance(lum, chroma, effect)


def compare(im1, im2):
    """
    Help function that takes two input images
        and prints out the difference between their pixel values.

    :param im1: Input image 1.
    :type im1: Numpy array.

    :param im2: Input image 2.
    :type im2: Numpy array.

    :return: Prints the difference between the images.
    """
    return print(im2-im1)


if __name__ == '__main__':
    image = read_image()
    show(image)

    edited = edit_globally(image, 2, "pow")
    show(edited)

    split = split_image(image, 2, 4, .1, "pow")
    show(split)
