"""
This module is rendering the input image with a local function.

"""

import numpy as np
import scipy.ndimage as ndimage
import cv2
import globalHDR


def has_alpha(im):
    """
    Bool function that checks if there is a alpha channel present.

    :param im: Input image.
    :type im: Numpy array.

    :return: (True | False)
    """
    return True if im.shape[-1] == 4 else False


def extract_alpha(im):
    """
    Extracts the alpha channel from the image.

    :param im: Input image
    :type im: Numpy array.

    :return: Input image and the alpha channel.
    """
    alpha = im[:, :, 3]
    shape = im[:, :, :-1].shape
    im_copy = np.zeros(shape)
    for i in range(0, im.shape[-1] - 1):
        im_copy[:, :, i] = im[:, :, i]
    im = im_copy
    return im, alpha


def blur_image(im, blur):
    """
    Blurs an image based on the parameters in the blur class.
    These parameters are customizable through the GUI.

    :param im: Input image.
    :type im: Numpy array.

    :param blur: Config for the blur.
    :type blur: BlurImageConfig class.

    :return: Blurred image.
    """
    if blur.linear:
        if im.ndim <= 2:
            blurry_im = ndimage.gaussian_filter(im, blur.sigma)
        else:
            blurry_im = np.zeros(im.shape)
            for i in range(0, im.ndim):
                blurry_im[:, :, i] = ndimage.gaussian_filter(im[:, :, i], blur.sigma)
    else:
        im = np.float32(im)
        blurry_im = cv2.bilateralFilter(im, blur.diameter, blur.sigma_space, blur.sigma_color)
    # globalHDR.show(blurry_im)
    return blurry_im


def find_details(im, blurry_im):
    """
    Extracts the details of the image.
    The details is defined by Uh = U0 - Ul,
        where Uh = details, U0 = original image and Ul = blurred image.

    :param im: Input image.
    :type im: Numpy array.

    :param blurry_im: Blurred input image.
    :type blurry_im: Numpy array.

    :return: Detailed image.
    """
    return im - blurry_im


def edit_blurred_image(blurry_im, effect):
    """
    Edits a blurred image according to the effect class' parameters.
    These parameters are customizable through the GUI.

    :param blurry_im: Blurred input image.
    :type blurry_im: Numpy array.

    :param effect: Config for the effects.
    :type effect: EffectConfig class.

    :return: Edited, blurred image.
    """
    if effect.mode == "global":
        return globalHDR.edit_globally(blurry_im, effect)
    elif effect.mode == "luminance":
        return globalHDR.split_image(blurry_im, effect)
    else:
        return blurry_im


def reconstruct_image(detail_im, blurry_im, filters):
    """
    Reconstructs the image with a given weight from filter config.
    These parameters are customizable through the GUI.

    :param detail_im: Detailed part of the original image.
    :type detail_im: Numpy array.

    :param blurry_im: Blurred part of the original image.
    :type blurry_im: Numpy array.

    :param filters: Config for the filtering.
    :type filters: FilterImageConfig class.

    :return: Reconstructed image.
    """
    if detail_im.shape == blurry_im.shape:
        return detail_im * filters.gamma + blurry_im
    else:
        reconstructed = np.zeros(blurry_im.shape)
        for i in range(0, blurry_im.ndim):
            reconstructed[:, :, i] = detail_im * filters.gamma + blurry_im[:, :, i]
        return reconstructed


def append_channel(im, channel):
    """
    Appends a channel at the end of an input image.

    :param im: Input image.
    :type im: Numpy array.

    :param channel: Input image containing the extra channel.
    :type channel: Numpy array.

    :return: Output image with the extra channel appended.
    """
    return np.dstack((im, channel))


def filter_image(im, filters):
    """
    Filters the blurred and detailed parts of the image,
        edits the blurred parts and puts it back together.

    Multiple weighting and level options are customizable through the GUI.
    These options are provided through the following classes:
     - FilterImageConfig
     - EffectConfig
     - BlurImageConfig

    If there is a fourth (alpha) channel, it is extracted before the editing happens.
    When the editing is completed it is appended back to the image.

    :param im: Input image.
    :type im: Numpy array.

    :param filters: Config for the filtering.
    :type filters: FilterImageConfig class.

    :return:
    """
    # globalHDR.show(im)
    alpha_exist = has_alpha(im)
    if alpha_exist:
        im, alpha = extract_alpha(im)

    blurry_im = blur_image(im, filters.blur)
    detail_im = find_details(im, blurry_im)
    blurry_im_edited = edit_blurred_image(blurry_im, filters.effect)
    filtered_im = reconstruct_image(detail_im, blurry_im_edited, filters)

    if alpha_exist:
        filtered_im = append_channel(filtered_im, alpha)

    return filtered_im


if __name__ == '__main__':
    input_im = globalHDR.read_image("../eksempelbilder/Ocean/Ocean")
    globalHDR.show(input_im)
