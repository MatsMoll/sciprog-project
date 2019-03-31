"""
...

"""

import numpy as np
import globalHDR
import scipy.ndimage as ndimage


def blur_image(im, sigma=3):
    """

    :param im:
    :param sigma:
    :return:
    """
    blurry_im = np.zeros(im.shape)
    if im.ndim <= 2:
        blurry_im = ndimage.gaussian_filter(im, sigma)
        # blurry_im = ndimage.uniform_filter(im, size=9)
    else:
        for i in range(0, im.ndim):  # til im.shape[-1]. tidligere: im.ndim men den funker ikke på svart-hvitt. må ta siste skuffen til dimensjonene
            blurry_im[:, :, i] = ndimage.gaussian_filter(im[:, :, i], sigma)
            # blurry_im[:, :, i] = ndimage.uniform_filter(im[:, :, i], size=9)
    # globalHDR.show(blurry_im)
    return blurry_im


def find_details(im, blurry_im, level=98):
    """

    :param im:
    :param level:
    :return:
    """
    detail_im = im - blurry_im  # detail = Uh, im = U0, blurry = Ul

    limit = np.percentile(detail_im, level)  # Dette betyr at kun topp 2% verdier = detaljer
    detail_im[detail_im > limit] = 1
    detail_im[detail_im <= limit] = 0
    # globalHDR.show(detail_im)
    return detail_im


def edit_blurred_image(blurry_im, mode="global", lum_scale=10, chrom_scale=.3):
    """

    :return:
    """
    if mode == "global":
        blurry_im_edited = globalHDR.edit_globally(blurry_im)  # sqrt er default
    else:
        blurry_im_edited = globalHDR.edit_luminance(blurry_im, lum_scale, chrom_scale)  # sqrt er default
    # globalHDR.show(blurry_im_edited)
    return blurry_im_edited


def reconstruct_image(detail_im, blurry_im, alpha=1):
    """

    :param detail_im:
    :param blurry_im:
    :param alpha:
    :return:
    """
    return detail_im * alpha + blurry_im


def filter_linear_spatial(im, mode="global"):
    """
    Docstring

    :param im: Input image.
    :param mode: Editing mode. (Global | Luminance)
    :return:
    """
    # globalHDR.show(im)
    blurry_im = blur_image(im, 3)
    detail_im = find_details(im, blurry_im, 90)
    blurry_im_edited = edit_blurred_image(blurry_im, mode, 1, 1)
    filtered_im = reconstruct_image(detail_im, blurry_im_edited, 1)

    return filtered_im


filtered = filter_linear_spatial(globalHDR.read_image("../eksempelbilder/Ocean/Ocean"))
globalHDR.show(filtered)
