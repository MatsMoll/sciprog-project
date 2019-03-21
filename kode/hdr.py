"""
A module to calculate the HDR-channels in a set of images
"""
import numpy as np
import matplotlib.pyplot as plt


def standard_weighting(x):
    """
    A standard weighting function for hdr reconstruction
    :param x: The x value
    :return: The weighted value
    """
    if x > 128:
        return 255 - x
    else:
        return x - 0


def hdr_channel(images, shutter, smoothness, weighting):
    """
    This will create a HDR exposure map based on some observed pixels in some images

    :param images: The pixel values in a 2d array where i is the location and j is the image
    :param shutter: log(shutter speed) or log(delta t) for each image j
    :param smoothness: The lambda, or a constant value that sets the smoothness
    :param weighting: The weighting function for a given value Z
    :return: A tuple containing log(exposure) for the pixel value Z and log(irradiance) for pixel i
    """

    n = 255
    number_of_pic, number_of_pixels = np.shape(images)

    a = np.zeros((number_of_pixels * number_of_pic + n + 1, number_of_pixels + n))
    b = np.zeros(np.shape(a)[0])

    # Setting equations
    pixel = 0
    for j in range(0, number_of_pic):
        for i in range(0, number_of_pixels):
            pixel_value = images[j][i]
            weighted_value = weighting(pixel_value + 1)
            a[pixel][int(round(pixel_value))] = weighted_value
            a[pixel][n + i] = -weighted_value
            b[pixel] = weighted_value * shutter[j]     # B[j] is the shutter speed
            pixel += 1

    # Setting Z_mid = 0
    a[pixel][128] = 1
    pixel += 1

    # Smoothing out the result
    for i in range(0, n - 1):
        weighted_value = weighting(i + 1)
        a[pixel][i] = smoothness * weighted_value
        a[pixel][i + 1] = -2 * smoothness * weighted_value
        a[pixel][i + 2] = smoothness * weighted_value
        pixel += 1

    result = np.linalg.lstsq(a, b, rcond=None)
    return result[0][0:n + 1], result[0][n:]


def hdr_color_channels(channels, shutter, smoothness):
    """
    This will create a HDR exposure map based on some observed pixels in some images

    :param channels: The pixel values in a 3d array where i is the location and j is the image
    :param shutter: log(shutter speed) or log(delta t) for each image j
    :param smoothness: The lambda, or a constant value that sets the smoothness
    :return: A tuple containing log(exposure) for the pixel value Z and log(irradiance) for pixel i
    """
    result = []   # [[g_r(z), ln(E_r)], ..., [g_b(z), ln(E_b)]]
    shape = np.shape(channels)
    one_dim_channels = channels
    if len(shape) == 4:
        one_dim_channels = channels.reshape((shape[0], shape[1], shape[2] * shape[3]))

    for channel in one_dim_channels:
        print("HDR")
        g, ln_e = hdr_channel(channel, shutter, smoothness)
        print("Done")
        result.append((g, ln_e))
    return result


def reconstruct_image(channels, weighting, hdr_graph, shutter):
    """
    Reconstruct a image from a hdr graph
    :param channels: The different channels from the different images
    :param weighting: The weighting function
    :param hdr_graph: The HDR graph
    :param shutter: The ln(shutter )speeds for the different images
    :return: The hdr channel
    """
    print("Transform")
    shape = np.shape(channels)
    hdr_image = np.zeros((shape[-2], shape[-1]))

    for x in range(0, shape[-2]):
        for y in range(0, shape[-1]):
            weighted_sum = 0
            g_value_sum = 0

            for j in range(0, shape[0]):
                weighted_sum += weighting(channels[j][x][y] + 1)
                z_value = int(channels[j][x][y])
                g_value_sum += weighted_sum * (hdr_graph[z_value] - shutter[j])
            hdr_image[x][y] = g_value_sum / weighted_sum

    return hdr_image


if __name__ == "__main__":
    from images import ImageSet
    # Testing

    # color_image = load_color_image("../eksempelbilder/Balls/Balls_", "00256")

    color_images = ImageSet([
        ("../eksempelbilder/Balls/Balls_", "00001"),
        ("../eksempelbilder/Balls/Balls_", "00004"),
        ("../eksempelbilder/Balls/Balls_", "00016"),
        ("../eksempelbilder/Balls/Balls_", "00032"),
        ("../eksempelbilder/Balls/Balls_", "00256"),
        ("../eksempelbilder/Balls/Balls_", "01024"),
        ("../eksempelbilder/Balls/Balls_", "02048"),
        # load_image("../eksempelbilder/Balls/Balls_", "01024"),
        # load_image("../eksempelbilder/Balls/Balls_", "02048"),
    ])
    color_hrd_map = color_images.hdr(10)
    #print("HDR Map")
    
    image_set = ImageSet([
        ("../eksempelbilder/Balls/Balls_", "00001"),
        ("../eksempelbilder/Balls/Balls_", "00004"),
        ("../eksempelbilder/Balls/Balls_", "00016"),
        ("../eksempelbilder/Balls/Balls_", "00032"),
        ("../eksempelbilder/Balls/Balls_", "00256"),
    ]).gray_images()

    z_values = np.arange(0, 255)

    g_values, ln_e_values = image_set.hdr(10)

    # print(np.shape(ln_e_values))
    # print(np.shape(g))
    # print(np.asarray(first[0]).reshape(-1))

    #im = image_set.channels()
    #print(np.shape(im))


    # hdrImage = transform_image(im, w, g_values, image_set.shutter_speed)

    color_channels = color_images.channels()

    color_im = np.zeros(color_images.original_shape)

    color_im[:, :, 0] = reconstruct_image(  # Red
        color_channels[0], standard_weighting, color_hrd_map[0][0], color_images.shutter_speed)
    color_im[:, :, 1] = reconstruct_image(  # Green
        color_channels[1], standard_weighting, color_hrd_map[1][0], color_images.shutter_speed)
    color_im[:, :, 2] = reconstruct_image(  # Blue
        color_channels[2], standard_weighting, color_hrd_map[2][0], color_images.shutter_speed)

    current_im = 2

    # im = image_set[current_im].original_image()
    # hdr_im = transform_image_with(hdr_map[0], im, image_set[current_im].original_shape)
    # plt.imshow(hdr_im, plt.cm.gray)
    # plt.imshow(color_image.original_image() / 255)
    # plt.imshow(color_image.original_image() / 255)

    # plt.plot(color_hrd_map[0][0], z_values)
    # plt.plot(color_hrd_map[1][0], z_values)
    # plt.plot(color_hrd_map[2][0], z_values)
    # plt.plot(g_values, z_values)

    # plt.plot(color_hrd_map[2][:255], z_values)

    color_im = color_im + abs(color_im.min())
    print(color_im.max())
    print(color_im.min())
    plt.imshow(color_im / 255)
    # plt.imshow(hdrImage.reshape(image_set.images[0].original_shape), plt.cm.gray)

    plt.show()
