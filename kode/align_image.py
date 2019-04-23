"""
A module that aligns images
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from hdr import ImageSet


def align_image_set(image_set):
    """
    Aligns the images in the image set a crops the image to remove unwanted borders
    :param image_set: The ImageSet to align
    :return: A aligned ImageSets
    """
    aligned_set = image_set.images.copy()

    for i in range(1, image_set.images.shape[0]):
        if np.std(image_set.images[i]) > 20:
            aligned_set[i] = align_images(image_set.images[i], aligned_set[i - 1])[0]
        else:
            print("To low detail in order to align images")
            break

    cropped_image = crop_image_set(aligned_set)

    aligned_image_set = ImageSet(cropped_image)
    aligned_image_set.shutter_speed = image_set.shutter_speed.copy()
    aligned_image_set.original_shape = cropped_image.shape[1:]

    return aligned_image_set


def align_images(im_one, im_two):
    """
    Align two images to each other

    :param im_one: The image to align
    :param im_two: The image to use as a reference point "the correct image"
    :return: A aligned version of the im_one
    """

    # Convert images to gray-scale
    max_features = 500
    good_match_percent = 0.2

    im_one = im_one.astype(np.uint8)
    im_two = im_two.astype(np.uint8)

    im1Gray = None
    im2Gray = None

    if im_one.ndim > 2:
        im1Gray = cv2.cvtColor(im_one, cv2.COLOR_RGB2GRAY)
        im2Gray = cv2.cvtColor(im_two, cv2.COLOR_RGB2GRAY)
    else:
        im1Gray = cv2.cvtColor(np.dstack((im_one, im_one, im_one)), cv2.COLOR_RGB2GRAY)
        im2Gray = cv2.cvtColor(np.dstack((im_two, im_two, im_two)), cv2.COLOR_RGB2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    key_points1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    key_points2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    # Draw top matches
    im_matches = cv2.drawMatches(im_one, key_points1, im_two,
                                 key_points2, matches, None)
    cv2.imwrite("matches.jpg", im_matches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = key_points1[match.queryIdx].pt
        points2[i, :] = key_points2[match.trainIdx].pt

    # Find homography
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    im1Reg = cv2.warpPerspective(im_one, h, (im_one.shape[1], im_one.shape[0]))

    return im1Reg, h


def crop_image_set(images):
    """
    Crops a image based on the alpha channel

    This uses the assumption that the alpha is only in the corner or as a border

    :param images: The image set to crop
    :return: The cropped image
    """
    x_1 = 0
    y_1 = 0
    x_2 = images.shape[2] - 1
    y_2 = images.shape[1] - 1

    alpha_channel = images

    if images.shape[-1] < 4:  # no alpha
        return images
    elif images.ndim > 3:
        alpha_channel = images[:, :, :, -1]

    while not np.all(alpha_channel[:, y_1, x_1] > 0):
        y_1 += 1
        x_1 += 1

    while not np.all(alpha_channel[:, y_2, x_2] > 0):
        x_2 -= 1
        y_2 -= 1

    while not np.all(alpha_channel[:, y_2, x_1] > 0):
        y_2 -= 1
        x_1 += 1

    while not np.all(alpha_channel[:, y_1, x_2] > 0):
        x_2 -= 1
        y_1 += 1

    return images[:, y_1:y_2 + 1, x_1:x_2 + 1]


if __name__ == '__main__':
    from hdr import test_image_set

    test_image_set = test_image_set()
    im_set = align_image_set(test_image_set)

    im = im_set.hdr_image(10)[:, :, 0:3] ** 0.2
    print("Im Done")
    print((im - im.min()) / (im.max() - im.min()))
    plt.imshow((im - im.min()) / (im.max() - im.min()))
    plt.show()
