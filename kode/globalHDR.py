"""
This module rendering the input image with a global function.
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio as io
import random as rand


def edit(picture, func):
    """
    Manipulated the given picture with the given function and returns it directly.

    Note: Gamma-function is supposed to be a fun addition to this function when you get a random value between 0 and 1.

    :param picture: Name of the picture that is going to be manipulated.
    :param func: Name of the manipulating function.
    :return: Calls the manipulating function.
    """
    if func == "e":
        return np.exp(picture)
    elif func == "ln":
        return np.log(picture)
    elif func == "squared":
        return picture ** 2
    elif func == "sqrt":
        return np.sqrt(picture)
    elif func == "gamma":
        return picture ** rand.uniform(0.0, 1.0)
    else:
        print("Unavailable function:", func, "\n-> Returned original image.")
        return picture


def read_image(path="../eksempelbilder/Balls/Balls", image_format=".exr"):
    """
    Reads an image in a given path.

    Note: See example for formatting guidelines (including image format)
    Example: ../eksempelbilder/Balls/Balls.exr

    :param path: The path to the image
    :param image_format: The image format
    :return: The desired picture
    """
    image = io.imread(path + image_format)

    image[image > 1] = 1
    image[image <= 0] = 0
    return image


def show(image):
    """
    Shows a given image whether it's monochrome or colorful.

    :param image: The image
    """
    if image.ndim <= 2:
        plt.imshow(image.astype(float), plt.cm.gray)
    else:
        plt.imshow(image.astype(float))
    plt.show()


pic = read_image()
edited = edit(pic, "gamma")
show(edited)
