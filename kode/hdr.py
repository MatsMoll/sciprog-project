"""
A module to calculate the HDR-channels in a set of images
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio


class ImageSet:
    """
    A class containing a set of images
    This makes it easier to handel the images
    """
    def __init__(self, images):
        self.images = []
        for path, shutter in images:
            self.images.append(load_image(path, shutter))

    def __getitem__(self, item):
        """
        Returns the image at a given index
        :param item: The item to fetch, of type integer
        :return: The image at the given index
        """
        return self.images[item]

    def hdr_channels(self, pixel_index, smoothness):
        """
        Calculates the HDR-channels for the Image set

        :param pixel_index: The pixels to use as references
        :param smoothness: The amount of smoothing to do on the graph
        :return: The g lookup table and the ln_e.
        This will be an tuple if the images are of one dim and an array with tuples if there is more
        """

        pixel_values, shutter_values = pixel_matrix(self.images, pixel_index)

        print(np.shape(pixel_values))

        if len(np.shape(pixel_values)) == 2:

            z_max = pixel_values.max()
            z_min = pixel_values.min()
            z_mid = (z_max + z_min) / 2

            def w(x):
                if x > z_mid:
                    return z_max - x
                else:
                    return x - z_min

            return hdr_channel(pixel_values, shutter_values, smoothness, w)
        else:
            return hdr_color_channels(pixel_values, shutter_values, smoothness)

    def gray_images(self):
        """
        Converts the image set to grey images
        :return: A new set containing the grey images
        """
        new_images = []
        for image in self.images:
            new_images.append(image.grey_image())
        im_set = ImageSet([])
        im_set.images = new_images
        return im_set


class Image:
    """
    A class containing the image information
    """
    def __init__(self, image, shutter, original_shape):
        self.image = image
        self.shutter = shutter
        self.original_shape = original_shape

    def original_image(self):
        """
        :return: The original image, in its original shape
        """
        if len(self.original_shape) == 3:
            return self.image.transpose().reshape(self.original_shape)
        else:
            return self.image.reshape(self.original_shape)

    def grey_image(self):
        """
        Converts one image to a grey image if possible
        :return: The grey image
        """
        if len(self.original_shape) == 3:
            return Image(
                self.image.sum(0) / self.original_shape[2],
                self.shutter,
                (self.original_shape[0], self.original_shape[1])
            )
        else:
            return self


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

    print(np.shape(a), np.shape(b))

    # Setting equations
    pixel = 0
    for i in range(0, number_of_pixels):
        for j in range(0, number_of_pic):
            pixel_value = images[j][i]
            weighted_value = weighting(pixel_value)
            a[pixel][int(round(pixel_value))] = weighted_value
            a[pixel][n + i] = -weighted_value
            b[pixel] = weighted_value * shutter[j]     # B[j] is the shutter speed
            pixel += 1

    # Setting Z_mid = 0
    a[pixel][128] = 0
    pixel += 1

    # Smoothing out the result
    for i in range(0, n - 2):
        a[pixel][i] = smoothness * weighting(i)
        a[pixel][i + 1] = -2 * smoothness * weighting(i)
        a[pixel][i + 2] = smoothness * weighting(i)
        pixel += 1

    result = np.linalg.lstsq(a, b, rcond=None)
    return result[0][0:n], result[0][n:]


def hdr_color_channels(channels, shutter, smoothness):
    """
    This will create a HDR exposure map based on some observed pixels in some images

    :param channels: The pixel values in a 3d array where i is the location and j is the image
    :param shutter: log(shutter speed) or log(delta t) for each image j
    :param smoothness: The lambda, or a constant value that sets the smoothness
    :return: A tuple containing log(exposure) for the pixel value Z and log(irradiance) for pixel i
    """
    result = []   # [[g_r(z), ln(E_r)], ..., [g_b(z), ln(E_b)]]
    for c_i in range(0, len(channels)):
        z_max = channels[c_i].max()
        z_min = channels[c_i].min()
        z_mid = (z_max + z_min) / 2

        print(z_max)
        print(z_min)

        def w(x):
            if x > z_mid:
                return z_max - x
            else:
                return x - z_min

        g, ln_e = hdr_channel(channels[c_i], shutter, smoothness, w)

        result.append((g, ln_e))
    return result


def load_image(path, shutter, image_format=".png"):
    """
        This loads an image

        Note: This expects that the name of the image is on the format
            /path/filename_shutterspeed.format
        Exp:
            /eksempelbilder/Balls/Balls_00032.png

        :param path: The path to the image
        :param shutter: The shutter speed in a string format
        :param image_format: The image format
        :return: a Image object
        """
    image = imageio.imread(path + shutter + image_format)
    image = image.astype(float)
    shutter_value = float(shutter) * (10 ** -5)
    if len(image.shape) > 2:
        color_dims = image.reshape(image.shape[0] * image.shape[1], image.shape[2]).transpose()
        return Image(color_dims, shutter_value, np.shape(image))
    else:
        return Image(image, shutter_value, np.shape(image))


def pixel_matrix(images, pixel_index):
    """
    Creates a Z image matrix and a B shutter vector
    This will filter out the pixels NOT specified in the pixel_index array

    :param images: The images to convert on the format tuple(image, shutter speed)
    :param pixel_index: The pixel indexes that should be used in the matrix
    :return: A tuple in the format (z, b)
    """
    image_count = len(images)
    if len(images[0].original_shape) == 3:

        shape = (3, image_count, len(pixel_index))
        z = np.zeros(shape)
        b = np.zeros(image_count)

        for j in range(0, 3):
            for i in range(0, image_count):
                b[i] = np.log(images[i].shutter)
                for k in range(0, len(pixel_index)):
                    z[j][i][k] = images[i].image[j][k]

        print("Pixel M:", z)
        return z, b
    else:
        z = np.zeros((image_count, len(pixel_index)))
        b = np.zeros(image_count)

        for i in range(0, image_count):
            b[i] = np.log(images[i].shutter)

            for j in range(0, len(pixel_index)):
                z[i][j] = images[i].image[j]

        return z, b


def transform_image_with(hdr_map, image, shutter):
    """
    Transforms a image to a HDR-image, given the images HDR-map

    :param hdr_map: The HDR-map / g(z) function
    :param image: The image to transform
    :param shutter: The shutter time for the image
    :return: The new image
    """
    shape = np.shape(image)
    im = np.zeros(shape)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            im[i][j] = hdr_map[int(image[i][j]) - 1] - np.log(shutter)
    return im

# Testing

# color_image = load_color_image("../eksempelbilder/Balls/Balls_", "00256")


color_hrd_map = ImageSet([
    ("../eksempelbilder/Balls/Balls_", "00001"),
    ("../eksempelbilder/Balls/Balls_", "00004"),
    ("../eksempelbilder/Balls/Balls_", "00016"),
    ("../eksempelbilder/Balls/Balls_", "00032"),
    ("../eksempelbilder/Balls/Balls_", "00256"),
    # load_image("../eksempelbilder/Balls/Balls_", "01024"),
    # load_image("../eksempelbilder/Balls/Balls_", "02048"),
]).hdr_channels(np.linspace(0, 7000, 700), 1)

image_set = ImageSet([
    ("../eksempelbilder/Balls/Balls_", "00001"),
    ("../eksempelbilder/Balls/Balls_", "00004"),
    ("../eksempelbilder/Balls/Balls_", "00016"),
    ("../eksempelbilder/Balls/Balls_", "00032"),
    ("../eksempelbilder/Balls/Balls_", "00256"),
]).gray_images()

# print(values[-1])
z_values = np.arange(0, 255)

g_values, ln_e_values = image_set.hdr_channels(np.linspace(0, 7000, 700), 10)

# print(np.shape(ln_e_values))
# print(np.shape(g))
# print(np.asarray(first[0]).reshape(-1))

current_im = 2

# im = image_set[current_im].original_image()
# hdr_im = transform_image_with(hdr_map[0], im, image_set[current_im].original_shape)
# plt.imshow(hdr_im, plt.cm.gray)
# plt.imshow(color_image.original_image() / 255)
# plt.imshow(color_image.original_image() / 255)

plt.plot(color_hrd_map[0][0], z_values)
plt.plot(color_hrd_map[1][0], z_values)
plt.plot(color_hrd_map[2][0], z_values)
plt.plot(g_values, z_values)

# plt.plot(color_hrd_map[2][:255], z_values)

plt.show()
