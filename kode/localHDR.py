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
    #$grey_im = np.zeros(shape)
    #$for i in range(0, im.ndim):
    #$    grey_im[:, :, i] = im.astype(float).sum(2) / (255 * 3)
    # # grey_im = im.astype(float).sum(2) / (255 * 3)
    blurry_im = np.zeros(shape)
        # # for i in range(0, im.ndim):
    ###########blurry_im = ndimage.uniform_filter(im, size=9)
    if im.ndim <= 2:
        blurry_im = ndimage.gaussian_filter(im, sigma=3)
    else:
        for i in range(0, im.shape[-1]): # til im.shape[-1]. tidligere: im.ndim men den funker ikke på svart-hvitt. må ta siste skuffen til dimensjonene
            blurry_im[:, :, i] = ndimage.gaussian_filter(im[:, :, i], sigma=3)
        # # blurry_im[:, :, i] = ndimage.uniform_filter(grey_im, size=15)
    # # blurry_im = ndimage.uniform_filter(grey_im, size=15)
    # for i in range(0, im.ndim):
    #    grey_im[:, :, i] = im.astype(float).sum(2) / (255 * 3)
    globalHDR.show(blurry_im)
    #globalHDR.show(gaus)

    if mode == "global":
        blurry_im_edited = globalHDR.edit_globally(blurry_im)  # sqrt er default
    else:
        blurry_im_edited = globalHDR.edit_luminance(blurry_im, lum_scale=10, chrom_scale=.3)  # sqrt er default
    globalHDR.show(blurry_im_edited)

    # Definerer detaljene ved (original - blurry)
    #-#detail_im = np.zeros(shape)
    # print("im:", im.shape, "detail_im:", detail_im.shape, "blurry_im:", blurry_im.shape)
    detail_im = im - blurry_im # detail = Uh, im = U0, blurry = Ul

    print("detailssss: ", detail_im.mean())

    limit = np.percentile(detail_im, 98)  # Dette betyr at kun 5% = detaljer
    detail_im[detail_im > limit] = 1
    detail_im[detail_im <= limit] = 0
    globalHDR.show(detail_im)

    alpha = 1

    #-#filtered_im = np.zeros(shape)
    filtered_im = detail_im * alpha + blurry_im
    #$#filtered_im = detail_im + alpha * (blurry_im_edited) #- blurry_im)
    #filtered_im = detail_im + 2 * im
    ### - filtered_im = blurry_im_edited + alpha * (blurry_im_edited - detail_im)
    globalHDR.show(filtered_im)

    print("\nim", im.shape, "\nblurry_im", blurry_im.shape, "\ndetail_im", detail_im.shape, "\nblurry_im_edited", blurry_im_edited.shape, "\nfiltered_im", filtered_im.shape)
    #"\ngrey_im", grey_im.shape, "\nblurry_im", blurry_im.shape, "\ndetail_im", detail_im.shape, "\nblurry_im_edited", blurry_im_edited.shape, "\nfiltered_im", filtered_im.shape)


    #sharp_im = np.zeros(shape)
        # sharp_im[:, :, i] = blurry_im + alpha * (blurry_im - blurry_im_edited)
    #-#sharp_im = blurry_im_edited + alpha * (blurry_im_edited - detail_im)
    # sharp_im = np.abs(sharp_im)
    #print(sharp_im)
    #globalHDR.show(sharp_im)


# sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
filter_linear_spatial(globalHDR.read_image("../eksempelbilder/Ocean/Ocean"))
