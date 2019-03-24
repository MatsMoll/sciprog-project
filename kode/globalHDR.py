"""
This module is rendering the input image with a global function.

__________

NOTE!
3.2

Skrive en funksjon som regner ut en likning på x ukjente
	- en ukjent for luminansen til hver fargekanal
	- en ukjent for den globale chromasiteten
	~ kan det skrives som en matrise? idk, jeg ville brukt løkke, men det er ressurskrevende
	- når disse verdiene er funnet, kan man justere ned den globale chromasiteten
	    for å endre fargemetningen

Per nå har jeg funnet noen kombinasjoner av luminans (per fargekanal)
og en global chrominasitet som ligner på “Balls.exr” sin majoritet mtp farge.

NOTE 2!
Gul = rød + blå
- Prøv å redusere disse kanalenes luminans og se på sammenligningsfunksjonen

NOTE 3!
Ser ut til at jeg kan kjøre luminans-funksjonen på en kanal.
Trolig vil den beste løsningen være å synkronisere antall kanaler (dimensjoner) for lum og chrom.

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
    """
    Takes an image as an argument and summarizes the different color channels
        to find the luminance. (L = R + G + B)
    A weight is given to each channels luminance to achieve a more realistic image.

    :param im: Input image.
    :return: The luminance of the picture.
    """
    #if chan == "all":
    shape = (im.shape[0], im.shape[1], 1)
    lum_channel = np.zeros(shape)

    #lum = im
    ##lum = np.sum(im, axis=2)
    ##print(lum.shape)
    #lum[:, :, :] = im#(im[:, :, 0] + im[:, :, 1] + im[:, :, 2])
    #####lum = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2])   #.35 #.6     #* .7
    #####print("lum", lum.shape)
    ####lum[:, :, 1] = lum[:, :, 0]
    ####lum[:, :, 2] = lum[:, :, 0]
    lum_channel[:, :, 0] = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2])   #.35 #.6     #* .7
    print("lum_channel", lum_channel.shape)
    ###lum[:, :, 1] = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2])  #.2  #.25    #* .25
    ###lum[:, :, 2] = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2])  #.05 #.05    #* .05
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
    return lum_channel#lum #* 5                                                  #* 5 (3. kolonne ovenfor)


def chromasity(im, lum):
    """
    Takes an input image and calculates the chromasity with the formula C = [R/L, G/L, B/L]
    A weight is given to the global chromasity to achieve a more realistic image.

    :param im: Input image.
    :return: The chromasity of the picture.
    """

    chrom = im / lum

    #im * lum

    #chrom = np.zeros(im.shape)
    #chrom = im
    #print(luminance(im))

    #-abc_r = luminance(im)
    #-print("abc_r", abc_r.shape)     #784 1000 1
    #-chrom = np.zeros(abc_r.shape)
    #-print("ch", chrom.shape)    #784 1000 1

    #-chrom = abc_r / luminance(im)

    ####

    #-abc_g = luminance(im)
    #-print("abc_g", abc_g.shape)  # 784 1000 1

    ####

    #-abc_b = luminance(im)
    #-print("abc_b", abc_b.shape)  # 784 1000 1

    ####

    #-np.append(chrom, [[5, 5, 5], [7, 8, 9]], axis=1)

    ####

    #-print("ch2", chrom.shape)       #784 1000 1
    #-print("im i chrom", im.shape)   #784*1000*3
    #-print("im0", im[:, :, 0].shape) #784*1000
    #--chrom = im
    #--chrom[:, :, 0] = im[:, :, 0] / luminance(im)
    #--chrom[:, :, 1] = im[:, :, 1] / luminance(im)
    #--chrom[:, :, 2] = im[:, :, 2] / luminance(im)
    #chrom[:, :, 0] = (im[:, :, 0] / luminance(im))
    #chrom[:, :, 1] = (im[:, :, 1] / luminance(im))
    #chrom[:, :, 2] = (im[:, :, 2] / luminance(im))

    return chrom #im / (lum * 1)  #im #chrom #* 1.95                                             #* 1.95


def split_image(im):
    """
    This function splits the input image into chromasity and luminance.
    It also manipulates the luminance channel alone to reduce saturation.

    :param im: Input image.
    :return: Output image defined by luminance * chromasity.
    """
    #show(im)
    lum = im
    #chrom = im
    lum = luminance(lum) #* .75         #784*1000*1
    lum = edit(lum, "sqrt")
    chrom = chromasity(im, lum)           #784*1000*3
    #print("lum", lum.shape)
    #print("ch", chrom.shape)
    res = lum * chrom

    return res #chrom#lum * chrom                  # 784*1000*1(?in range 1-3?) * 784*1000*3
    #show(image)

    #show(im)
    #show(lum)
    #show(chrom)


def compare(im1, im2):
    """
    Takes two input images and prints out the difference between their pixel values.

    :param im1: Input image 1
    :param im2: Input image 2
    :return: Prints the difference between the images.
    """
    return print(im2-im1)


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

compare(edited, split)

#r = image[:, :, 0]             # [0] rød
# g = image[:, :, 1]            # [1] grønn
# b = image[:, :, 2]            # [2] blå


# show(image[:, :, 1])
