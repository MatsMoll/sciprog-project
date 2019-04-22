"""
This module is rendering the input image with a local function.

"""

import numpy as np
import scipy.ndimage as ndimage
import cv2
import globalHDR


def check_dim(im):
    """
    Bool function that checks if there is a alpha dimension (channel?) present. [RENAME FUNC!]

    :param im: Input image.
    :return: (True | False)
    """
    return True if im.shape[-1] == 4 else False


def extract_alpha(im):
    """
    Extracts the alpha channel from the image.

    :param im: Input image
    :return: Input image and the alpha channel.
    """
    alpha = im[:, :, 3]
    shape = im[:, :, :-1].shape
    im_copy = np.zeros(shape)
    for i in range(0, im.shape[-1] - 1):
        im_copy[:, :, i] = im[:, :, i] * im[:, :, 3]
    im = im_copy
    return im, alpha


def blur_image(im, linear=True, sigma=3):  # Utvid docstring
    """
    Blurs an image with the gaussian filter with a set range (sigma).
    The linear parameter decides wether the operation should be completed linear or non-linear.

    :param im: Input image.
    :param linear: (True | False)
    :param sigma: Range of gaussian filter.
    :return: Blurred image.
    """
    if linear:
        if im.ndim <= 2:
            blurry_im = ndimage.gaussian_filter(im, sigma)
        else:
            blurry_im = np.zeros(im.shape)
            for i in range(0, im.ndim):
                blurry_im[:, :, i] = ndimage.gaussian_filter(im[:, :, i], sigma)
    else:
        im = np.float32(im)
        blurry_im = cv2.bilateralFilter(im, 9, 150, 150)
    # globalHDR.show(blurry_im)
    return blurry_im


def find_details(im, blurry_im):
    """
    Extracts the details of the image.
    The details is defined by Uh = U0 - Ul,
        where Uh = details, U0 = original image and Ul = blurred image.

    :param im: Input image.
    :param blurry_im: Blurred input image.
    :return: Detailed image.
    """
    return im - blurry_im


def edit_blurred_image(blurry_im, lum_scale, chrom_scale, mode="global"):
    """
    Edits a blurred image with a given mode and scale. Reuses functions from global rendering.
    Mode decides whether it is edited on the luminance channel or globally.

    :param blurry_im: Blurred input image.
    :param lum_scale: Weighting of luminance.
    :param chrom_scale: Weighting of chromasity.
    :param mode: Editing mode. (Global | Luminance)
    :return: Edited, blurred image.
    """
    if mode == "global":
        blurry_im_edited = globalHDR.edit_globally(blurry_im)
    else:
        blurry_im_edited = globalHDR.edit_luminance(blurry_im, lum_scale, chrom_scale)
    # globalHDR.show(blurry_im_edited)
    return blurry_im_edited


def reconstruct_image(detail_im, blurry_im, gamma):  # Unsure if the weight (gamma) is necessary.
    """
    Reconstructs the image with a given weight.

    :param detail_im: Detailed part of the original image.
    :param blurry_im: Blurred part of the original image.
    :param gamma: Weighting of details.
    :return: Reconstructed image.
    """
    if detail_im.shape == blurry_im.shape:
        reconstructed = detail_im * gamma + blurry_im
    else:
        reconstructed = np.zeros(blurry_im.shape)
        for i in range(0, blurry_im.ndim):
            reconstructed[:, :, i] = detail_im * gamma + blurry_im[:, :, i]
    return reconstructed


def append_alpha(im, alpha):  # Should this function be renamed to "append_channel" to give a general purpose?
    """
    Appends a channel to the input image.

    :param im: Input image.
    :param alpha: Image with alpha channel. # Rename to new_channel??
    :return: Output image with the extra channel alpha ("new_channel")?
    """
    return np.dstack((im, alpha))  # im


def filter_image(im, linear=True, sigma=3, level=90, mode="global", lum_scale=10, chrom_scale=.3, gamma=5):  # Skriv ekstra i docstring om den ekstra kanalen
    """
    Filters the blurred and detailed parts of the image,
        edits the blurred parts and puts it back together.
    Multiple weighting and level options are provided to edit the image the way you want.

    :param im: Input image.
    :param linear: Linear filtering (True | False)
    :param sigma: Range of gaussian filter.
    :param level: Detail level.
    :param mode: Editing mode. (Global | Luminance)
    :param lum_scale: Weighting of luminance.
    :param chrom_scale: Weighting of chromasity.
    :param gamma: Weighting of details.
    :return:
    """
    # globalHDR.show(im)
    alpha_exist = check_dim(im)
    if alpha_exist:
        im, alpha = extract_alpha(im)

    blurry_im = blur_image(im, linear, sigma)
    detail_im = find_details(im, blurry_im)  # , level)
    blurry_im_edited = edit_blurred_image(blurry_im, mode, lum_scale, chrom_scale)
    filtered_im = reconstruct_image(detail_im, blurry_im_edited, gamma)

    if alpha_exist:
        filtered_im = append_alpha(filtered_im, alpha)

    return filtered_im


if __name__ == '__main__':
    input_im = globalHDR.read_image("../eksempelbilder/Ocean/Ocean")
    globalHDR.show(input_im)

    linear_im = filter_image(input_im, True, 3, 95, "global", 1, 1, 1)
    globalHDR.show(linear_im)

    nonlinear_im = filter_image(input_im, False, 3, 95, "global", 1, 1, 1)
    globalHDR.show(nonlinear_im)
