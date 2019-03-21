import numpy as np


def find_reference_points_for(images):
    """
    :param images: The images to find references for of type images.ImageSet
    :return: The pixel indexes
    """
    channels = images.channels()
    shape = np.shape(channels)
    im_len = shape[-2] * shape[-1]
    spacing = max(int(im_len / 1000), 1)
    return np.arange(0, im_len, spacing)


if __name__ == "__main__":
    from images import ImageSet
    ims = ImageSet([
        ("../eksempelbilder/Balls/Balls_", "00002"),
        # load_image("../eksempelbilder/Balls/Balls_", "01024"),
        # load_image("../eksempelbilder/Balls/Balls_", "02048"),
    ])

    print(find_reference_points_for(ims))
