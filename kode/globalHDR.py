"""
This module rendering the input image with a global function.
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio as io
import random as rand


def edit(im, func):
    """
    Manipulated the given image with the given function and returns it directly.

    Note: Gamma-function is supposed to be a fun addition to this function when you get a random value between 0 and 1.

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
    im[im <= 0] = 0
    return im


def show(im):
    """
    Shows a given image whether it's monochrome or colorful.

    :param image: The image
    """
    #plt.imshow(im, plt.cm.Reds_r)
    #plt.imshow(im, plt.cm.Greens_r)
    #plt.imshow(im, plt.cm.Blues_r)
    if im.ndim <= 2:
        plt.imshow(im, plt.cm.gray)
    else:
        plt.imshow(im.astype(float))
    plt.show()


def luminance(im):#, chan):
    #if chan == "all":
    lum = im
    lum[:, :, 0] = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) * .6   #.35 #.6     #* .7
    lum[:, :, 1] = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) * .25  #.2  #.25    #* .25
    lum[:, :, 2] = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) * .05  #.05 #.05    #* .05
    #lum[lum > 1] = 1
    #lum[lum <= 0] = 0
    #lume = edit(lum, "sqrt")
    #show(lume)
    #print(lum)
    #elif chan == "r":
    #    luminance = (im[:, :, 0])
    #elif chan == "g":
    #    luminance = (im[:, :, 1])
    #elif chan == "b":
    #    luminance = (im[:, :, 2])
    #else:
    #    print("Unavailable input:", im, chan, "\n-> Please use the format 'im, chan'")

    #print(lum)
    return lum #* 5                                                  #* 5 (3. kolonne ovenfor)


def chromasity(im):
    #chrom = np.zeros(im)
    #chrom = np.zeros(im.shape)
    #chrom = im
    #print(luminance(im))
    chrom = im / luminance(im)
    ##chrom[:, :, 0] = (im[:, :, 0] / luminance(im))
    ##chrom[:, :, 1] = (im[:, :, 1] / luminance(im))
    ##chrom[:, :, 2] = (im[:, :, 2] / luminance(im))
    #im = luminance(im)
    #bb = (im[:, :, :] / luminance(im[:, :, :]))

    #print(chrom)
    #print(im.shape)
    #print(im)
    #show(im)
    return chrom * 1.95                                             #* 1.95


def split_image(im):
    #show(im)
    lum = im
    chrom = im
    lum = luminance(lum) #* .75
    chrom = chromasity(chrom)
    lum = edit(lum, "sqrt")
    #print("lum", lum.shape)
    #print("ch", chrom.shape)

    return lum * chrom
    #show(image)

    #show(im)
    #show(lum)
    #show(chrom)


def compare(im, im2):
    return (print(im2-im))




image = read_image()
edited = edit(image, "sqrt")
#print(image)
show(image)
show(edited)
#print(edited)                       # hele matrisen
#print(edited.shape, len(edited))    # (784, 1000, 3) # len(edited) 784

#luminance(image)
#chromasity(image)
split = split_image(image)
#print(split)
#split[split <= 0] = 0
#split[split > 1] = 1
show(split)

#compare(edited, split)

#r = image[:, :, 0]             # [0] rød
# g = image[:, :, 1]            # [1] grønn
# b = image[:, :, 2]            # [2] blå


# show(image[:, :, 1])

"""
NOTE!
3.2

Skrive en funksjon som regner ut .. på 6 ukjente
	- en ukjent for luminansen til hver fargekanal
	- en ukjent for den globale chrominanse
	~ kan det skrives som en matrise? idk, jeg ville brukt løkke, men det er ressurskrevende

Per nå har jeg funnet noen kombinasjoner av luminans (per fargekanal) og en global chrominans som ligner på “Balls.exr” sin majoritet mtp farge.

NOTE 2!
Gul = rød + blå
- Prøv å redusere disse kanalenes luminans og se på sammenligningsfunksjonen

"""
