"""
Add docstring


"""

import matplotlib.pyplot as plt
import OpenEXR as exr
import Imath
import numpy as np

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

def ln(u):
    return np.log(u)


def e(u):
    return np.exp(u)


def x_Sq(u):
    return (u**2)


def sqrt(u):
    return np.sqrt(u)


def exr_to_array(path):
    """
    This function reads an (.exr) image and reformats its data to a readable and manipulable format.

    :param path: The path to the .exr image
    :return: Stack array depth wise with image data
    """
    file = exr.InputFile(path)
    dw = file.header()['dataWindow']

    print(exr.InputFile(path).header())
    channels = file.header()['channels'].keys()
    channels_list = list()
    for c in ('R', 'G', 'B', 'A'):
        #...
        if c in channels:
            channels_list.append(c)

    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    color_channels = file.channels(channels_list, FLOAT)
    #gå gjennom og send inn i ln(u) for hver channel
    channels_tuple = [np.frombuffer(channel, dtype='f') for channel in color_channels]
    #for channel in color_channels:
    #    ln(channels_tuple[channel])

    print(channels_tuple) #liste av arrays? (3 stk??)
    print(type(channels_tuple)) #class 'list'
    print(channels_tuple[0]) #elem 0
    print(channels_tuple[1]) #elem 1
    print(channels_tuple[2]) #elem 2


    res = np.dstack(channels_tuple)
    res[res > 1] = 1
    res[res <= 0] = 0 #Kan ikke settes til 0, siden log(0) blir er problem..
    #print(size)
    #print(res.reshape(size + (len(channels_tuple),)))
    return res.reshape(size + (len(channels_tuple),))

def edit(action):
    return e(action)

pre_edit = exr_to_array("../eksempelbilder/Balls/Balls.exr")
post_edit = edit(pre_edit)

#print("!!!!!\n!!!!!")
print(pre_edit)
print("!!!!!\n!!!!!")
print(post_edit)
plt.imshow(post_edit)
plt.show()





