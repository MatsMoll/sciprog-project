import numpy as np
import matplotlib.pyplot as plt
import imageio


def hdr(Z, B, l, w):
    """
    This will create a HDR exposure map based on some observed pixels in some images

    :param Z: The pixel values in a 2d array where i is the location and j is the image
    :param B: log(shutter speed) or log(delta t) for each image j
    :param l: The lambda, or a constant value that sets the smoothness
    :param w: The weighting function for a given value Z
    :return: A tuple containing log(exposure) for the pixel value Z and log(irradiance) for pixel i
    """
    N = 255
    number_of_pic, number_of_pixels = np.shape(Z)

    a = np.zeros((number_of_pixels * number_of_pic + N + 1, number_of_pixels + N))
    b = np.zeros(np.shape(a)[0])

    print(np.shape(b), np.shape(a))
    print(np.shape(B), np.shape(Z))

    # Setting equations
    pixel = 0
    for i in range(0, number_of_pixels):
        for j in range(0, number_of_pic):
            pixel_value = Z[j][i]
            weighted_value = w(pixel_value)
            a[pixel][int(round(pixel_value))] = weighted_value
            a[pixel][N + i] = -weighted_value
            b[pixel] = weighted_value * B[j]     # B[j] is the shutter speed
            pixel += 1

    # Setting Z_mid = 0
    a[pixel][128] = 0
    pixel += 1

    # Smoothing out the result
    for i in range(0, N - 2):
        a[pixel][i] = l * w(i)
        a[pixel][i + 1] = -2 * l * w(i)
        a[pixel][i + 2] = l * w(i)
        pixel += 1

    return np.linalg.lstsq(a, b, rcond=None)[0] # (g(Z_ij), ln(E_i))


def load_gray_image(path, shutter, format =".png"):
    """
    This loads an image and converts it to a gray image

    Note: This expects that the name of the image is on the format
        /path/filename_shutterspeed.format
    Exp:
        /eksempelbilder/Balls/Balls_00032.png

    :param path: The path to the image
    :param shutter: The shutter speed in a string format
    :param format: The image format
    :return: a tuple containing the image array and the shutter speed
    """
    image = imageio.imread(path + shutter + format)
    image = image.astype(float).sum(2) / (3)
    shutter_value = float(shutter) * (10 ** -5)
    return np.asarray(image).reshape(-1), shutter_value


def pixel_matrix(images, pixel_index):
    """
    Creates a Z image matrix and a B shutter vector
    This will filter out the pixels NOT specified in the pixel_index array

    :param images: The images to convert on the format tuple(image, shutter speed)
    :param pixel_index: The pixel indexes that should be used in the matrix
    :return: A tuple in the format (z, b)
    """
    image_count = len(images)
    z = np.zeros((image_count, len(pixel_index)))
    b = np.zeros(image_count)
    for i in range(0, image_count):
        for j in range(0, len(pixel_index)):
            z[i][j] = images[i][0][j]
            b[i] = np.log(images[i][1])

    return z, b


# Testing
images = [
    load_gray_image("../eksempelbilder/Balls/Balls_", "00001"),
    load_gray_image("../eksempelbilder/Balls/Balls_", "00032"),
    load_gray_image("../eksempelbilder/Balls/Balls_", "00256"),
    load_gray_image("../eksempelbilder/Balls/Balls_", "01024"),
    load_gray_image("../eksempelbilder/Balls/Balls_", "02048"),
]

z, b = pixel_matrix(
    images,
    np.linspace(0, 7000, 700)
)

z_max = z.max()
z_min = z.min()
z_mid = (z_max + z_min) / 2

print(z_min, z_max, z_mid)

def w(x):
    if x > z_mid:
        return z_max - x
    else:
        return x - z_min


values = hdr(z, b, 10, w)
#print(values[-1])
z_values = np.arange(0, 255)
#estimate_values = values[0] + values[1] * z_values + values[2] * z_values



#print(np.asarray(first[0]).reshape(-1))
#plt.imshow(first[0], plt.cm.gray)
plt.plot(values[0:255], z_values)
plt.show()
