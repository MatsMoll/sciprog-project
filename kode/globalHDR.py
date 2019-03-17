import matplotlib.pyplot as plt
import OpenEXR as exr
import Imath
import numpy

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

def exr_to_array(path):
    """
    This function reads an (.exr) image and reformats its data to a readable and manipulable format.

    :param path: The path to the .exr image
    :return: Stack array depth wise with image data
    """
    file = exr.InputFile(path)
    dw = file.header()['dataWindow']

    channels = file.header()['channels'].keys()
    channels_list = list()
    for c in ('R', 'G', 'B', 'A'):
        if c in channels:
            channels_list.append(c)

    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    color_channels = file.channels(channels_list, FLOAT)

    channels_tuple = [numpy.frombuffer(channel, dtype='f') for channel in color_channels]
    res = numpy.dstack(channels_tuple)
    res[res > 1] = 1
    res[res <= 0] = 0
    print(size)
    return res.reshape(size + (len(channels_tuple),))

data = exr_to_array("../eksempelbilder/Balls/Balls.exr")

# print(data)
plt.imshow(data)
plt.show()





