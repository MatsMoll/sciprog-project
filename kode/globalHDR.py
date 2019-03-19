"""
Add docstring


"""

import matplotlib.pyplot as plt
import OpenEXR as exr
import Imath
import numpy as np

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

def ln(u):
    """
    Takes value(s), sends it to a logN function and returns directly in a slick one-liner.

    :param u: Value(s) that is going to be manipulated.
    :return: Manipulated values
    """
    return np.log(u)


def e(u):
    """
    Takes value(s), sends it to a e^(u) function and returns directly in a slick one-liner.

    :param u: Value(s) that is going to be manipulated.
    :return: Manipulated values
    """
    return np.exp(u)


def squared(u):
    """
    Takes value(s), sends it to a u^2 function and returns directly in a slick one-liner.

    :param u: Value(s) that is going to be manipulated.
    :return: Manipulated values
    """
    return u**2


def sqrt(u):
    """
    Takes value(s), sends it to a sqrt(u) function and returns directly in a slick one-liner.

    :param u: Value(s) that is going to be manipulated.
    :return: Manipulated values
    """
    return np.sqrt(u)


def exr_to_array(path):
    """
    This function reads an (.exr) image and reformats its data to a readable and manipulable format.
    Each channel is appended at a new line in the array.

    :param path: The path to the .exr image
    :return: Stack array depth wise with image data
    """
    file = exr.InputFile(path)
    dw = file.header()['dataWindow']

    print(exr.InputFile(path).header())
    channels = file.header()['channels'].keys()
    channels_list = list()
    for c in ('R', 'G', 'B', 'A'):
        if c in channels:
            channels_list.append(c)

    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    color_channels = file.channels(channels_list, FLOAT)
    channels_tuple = [np.frombuffer(channel, dtype='f') for channel in color_channels]

    # print(channels_tuple)  # liste av arrays? (3 stk for Ball-bildet??)
    # print(type(channels_tuple))  # class 'list'
    # print(channels_tuple[0])  # elem 0
    # print(channels_tuple[1])  # elem 1
    # print(channels_tuple[2])  # elem 2

    res = np.dstack(channels_tuple)
    res[res > 1] = 1
    res[res <= 0] = 0 # Kan ikke settes til 0, siden log(0) blir er problem..
    # print(size)
    # print(type(res.reshape(size + (len(channels_tuple),)))) # class 'numpy.ndarray'
    return res.reshape(size + (len(channels_tuple),))


def edit(action):
    """
    Takes an image (in array-form) and sends it to the given function that manipulates the images values.

    :param action: Name of the manipulating function.
    :return: Calls the manipulating function.
    """
    # switch = {
    #    0:  "e",
    #    1:  "ln",
    #    2:  "squared",
    #    3:  "sqrt",
    # }
    # return switch.get(action, "nothing")
    return sqrt(action)


pre_edit = exr_to_array("../eksempelbilder/Tree/Tree.exr")
post_edit = edit(pre_edit)

# print("!!!!!\n!!!!!")
# print(pre_edit) #Verdier på bildet
# print("!!!!!\n!!!!!")
# print(post_edit) #Verdier på bildet
plt.imshow(post_edit)
plt.show()
