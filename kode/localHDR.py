"""
...

"""

import numpy as np
import globalHDR
import scipy.ndimage as ndimage


def filter_linear_spatial(im, mode="global"):  # limit=.5):
    """
    Docstring

    :param im: Input image.
    :param mode: Editing mode. (Global | Luminance)
    :return:
    """
    # Laster inn originalbildet
    globalHDR.show(im)
    shape = im.shape

    # Lager et monokromt bilde og et blurry bilde
    grey_im = im.astype(float).sum(2) / (255 * 3)
    #-blurry_im = np.zeros(shape)
    blurry_im = ndimage.uniform_filter(grey_im, size=15)
    #for i in range(0, im.ndim):
    #    grey_im[:, :, i] = im.astype(float).sum(2) / (255 * 3)
    globalHDR.show(blurry_im)

    #Definerer detaljene ved (original - blurry)
    #-detail_im = np.zeros(shape)
    #print("im:", im.shape, "detail_im:", detail_im.shape, "blurry_im:", blurry_im.shape)
    for i in range(0, im.ndim):
        detail_im = im[:, :, i] - blurry_im

    limit = np.percentile(detail_im, 95)  # Dette betyr at kun 10% = detaljer
                                                    #  * 2 # * 3 # 0.5 # detail_im.mean() # Threshold for hva som "er" detaljer
    detail_im[detail_im > limit] = 1
    detail_im[detail_im <= limit] = 0
    globalHDR.show(detail_im)
    
    if mode == "global":
        blurry_im_edited = globalHDR.edit_globally(blurry_im)  # sqrt er default
    else:
        blurry_im_edited = globalHDR.edit_luminance(blurry_im)  # sqrt er default

    filtered_im = np.zeros(shape)
    for i in range(0, detail_im.ndim):
        filtered_im[:, :, i] = detail_im + blurry_im_edited
    globalHDR.show(filtered_im)

    print("\nim", im.shape, "\ngrey_im", grey_im.shape, "\nblurry_im", blurry_im.shape, "\ndetail_im", detail_im.shape, "\nblurry_im_edited", blurry_im_edited.shape, "\nfiltered_im", filtered_im.shape)
    print(blurry_im_edited)
    #globalHDR.compare(blurry_im, blurry_im_edited)


filter_linear_spatial(globalHDR.read_image())
