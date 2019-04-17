"""
This module is rendering the input image with a local function.

"""

import numpy as np
import scipy.ndimage as ndimage
import globalHDR
import cv2
from skimage import morphology

# Add func and effect to filter function?
# Would give a better UI experience

# # Test with only the non-linear part of the find_details

# # # Implement one of the following algorithms for 4.2
# # # 1) Canny edge detection   2) Bilateral filtering  3) Anisotropic filtering


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
    #globalHDR.show(blurry_im)
    return blurry_im


def find_details(im, blurry_im):#, level):
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

    # limit = np.percentile(detail_im, level) ### IKKE LINEÆRE OPERASJONER
    # detail_im[detail_im > limit] = 1
    # detail_im[detail_im <= limit] = 0
    #globalHDR.show(detail_im)
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
    detail_im = find_details(im, blurry_im)  # , level)
    blurry_im_edited = edit_blurred_image(blurry_im, mode, lum_scale, chrom_scale)
    filtered_im = reconstruct_image(detail_im, blurry_im_edited, alpha)
    return filtered_im


input_im = globalHDR.read_image("../eksempelbilder/Ocean/Ocean")
#result_im = filter_linear(input_im, 3, 95, "global", 1, 1, 1)
#globalHDR.show(input_im)
#globalHDR.show(result_im)


def detect_edges(im):
    """
    ...

    :param im:
    :return:
    """


## CANNY
# Greyscale
# Gaussisk blur
# Determine the intensity gradient
# Non maximum suppression
# Double thresholding
# Edge tracking by hysteris
# Cleaning up

canny = input_im

# Greyscale
grey_canny = canny.astype(float).sum(2) / (255 * 3)
globalHDR.show(grey_canny)

# Gaussisk blur
blur_canny = ndimage.gaussian_filter(grey_canny, sigma=1)
globalHDR.show(blur_canny)

# Determine the intensity gradient and angle
sx = ndimage.sobel(blur_canny, axis=0, mode="constant")
sy = ndimage.sobel(blur_canny, axis=1, mode="constant")

print("før convolve")
Ix = ndimage.convolve(blur_canny, sx)
Iy = ndimage.convolve(blur_canny, sy)


print("etter convolve")
G = np.hypot(Ix, Iy)
G = G / G.max() * 255
theta = np.arctan2(Iy, Ix)
# sobel = np.hypot(sx, sy)    # Gradient, samme som     np.sqrt(sx**2 + sy**2)
# theta = np.arctan2(sy, sx)  # Angle
# globalHDR.show(sobel)


def round_angle(angle):
    """ Input angle must be \in [0,180) """
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle


def suppression(img, D):
    """ Step 3: Non-maximum suppression

    Args:
        img: Numpy ndarray of image to be processed (gradient-intensed image)
        D: Numpy ndarray of gradient directions for each pixel in img

    Returns:
        ...
    """

    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)

    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            where = round_angle(D[i, j])
            if where == 0:
                if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                    Z[i, j] = img[i, j]
            elif where == 90:
                if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                    Z[i, j] = img[i, j]
            elif where == 135:
                if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                    Z[i, j] = img[i, j]
            elif where == 45:
                if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                    Z[i, j] = img[i, j]
    return Z

print("før suppression")
test = suppression(grey_canny, theta)
print("etter suppression")
print(test)
globalHDR.show(test)


"""
# Gjør om til binært
high_limit = sobel
max = np.percentile(sobel, 70)
high_limit[sobel <= max] = 0
high_limit[sobel > max] = 1
globalHDR.show(high_limit)

# low_limit = sobel
abc = np.hypot(sx, sy)
min = np.percentile(abc, 30)
abc[abc <= min] = 0
abc[abc > min] = 1
globalHDR.show(abc)


test = ndimage.grey_erosion(grey_canny, structure=np.ones((1, 1)))
globalHDR.show(test)
"""




"""
sx = ndimage.sobel(input_im, axis=0, mode="constant")
sy = ndimage.sobel(input_im, axis=1, mode="constant")
sob = np.hypot(sx, sy)
globalHDR.show(sob)

new = input_im - sob
globalHDR.show(new)

new_blurred = blur_image(new, 3)
globalHDR.show(new_blurred)
# Setter sammen Sobel med blurred "bakgrunn"
res = sob + new_blurred
globalHDR.show(res)

# Prøver å ta detaljene fra Sobel og legge de detaljene på input_im
blur_res = blur_image(res)
detail_res = find_details(res, blur_res)
globalHDR.show(detail_res)
new_res = detail_res + input_im
globalHDR.show(new_res)
"""