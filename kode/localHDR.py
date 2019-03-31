"""
This module is rendering the input image with a local function.

"""

import numpy as np
import scipy.ndimage as ndimage
import globalHDR

# Add func and effect to filter function?
# Would give a better UI experience


def blur_image(im, sigma=3):
    """
    Blurs an image with the gaussian filter with a set range.

    :param im: Input image.
    :param sigma: Range of gaussian filter.
    :return: Blurred image.
    """
    if im.ndim <= 2:
        blurry_im = ndimage.gaussian_filter(im, sigma)
    else:
        blurry_im = np.zeros(im.shape)
        for i in range(0, im.ndim):
            blurry_im[:, :, i] = ndimage.gaussian_filter(im[:, :, i], sigma)
    # globalHDR.show(blurry_im)
    return blurry_im


def find_details(im, blurry_im, level):
    """
    Extracts the details of the image.
    The details is defined by Uh = U0 - Ul,
        where Uh = details, U0 = original image and Ul = blurred image.

    :param im: Input image.
    :param blurry_im: Input blurred image.
    :param level: Detail level.
    :return: Detailed image.
    """
    detail_im = im - blurry_im

    limit = np.percentile(detail_im, level)
    detail_im[detail_im > limit] = 1
    detail_im[detail_im <= limit] = 0
    # globalHDR.show(detail_im)
    return detail_im


def edit_blurred_image(blurry_im, mode, lum_scale, chrom_scale):
    """
    Edits a blurred image with a given mode and scale. Reuses functions from global rendering.
    Mode decides whether it is edited on the luminance channel or globally.

    :param blurry_im: Input image (blurred).
    :param mode: Editing mode. (Global | Luminance)
    :param lum_scale: Weighting of luminance.
    :param chrom_scale: Weighting of chromasity.
    :return: Edited, blurred image.
    """
    if mode == "global":
        blurry_im_edited = globalHDR.edit_globally(blurry_im)
    else:
        blurry_im_edited = globalHDR.edit_luminance(blurry_im, lum_scale, chrom_scale)
    # globalHDR.show(blurry_im_edited)
    return blurry_im_edited


def reconstruct_image(detail_im, blurry_im, alpha):  # Unsure if the weight is necessary.
    """
    Reconstructs the image with a given weight.

    :param detail_im: Detailed part of the original image.
    :param blurry_im: Blurred part of the original image.
    :param alpha: Weighting of details.
    :return: Reconstructed image.
    """
    return detail_im * alpha + blurry_im


def filter_linear(im, sigma=3, level=90, mode="global", lum_scale=10, chrom_scale=.3, alpha=5):
    """
    Filters the blurred and detailed parts of the image,
        edits the blurred parts and puts it back together.
    Multiple weighting and level options are provided to edit the image the way you want.

    :param im: Input image.
    :param sigma: Range of gaussian filter.
    :param level: Detail level.
    :param mode: Editing mode. (Global | Luminance)
    :param lum_scale: Weighting of luminance.
    :param chrom_scale: Weighting of chromasity.
    :param alpha: Weighting of details.
    :return:
    """
    # globalHDR.show(im)
    blurry_im = blur_image(im, sigma)
    detail_im = find_details(im, blurry_im, level)
    blurry_im_edited = edit_blurred_image(blurry_im, mode, lum_scale, chrom_scale)
    filtered_im = reconstruct_image(detail_im, blurry_im_edited, alpha)
    return filtered_im


input_im = globalHDR.read_image("../eksempelbilder/Balls/Balls")
result_im = filter_linear(input_im, 3, 98, "global", 10, .3, 1)
globalHDR.show(result_im)
