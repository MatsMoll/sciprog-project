"""
A module that provides attributes to simplifies function calls in 'globalHDR.py' and 'localHDR.py'.
"""


class BlurImageConfig:
    """
    A class providing the blur configuration.

    :attr linear: (True | False)
    :type linear: Bool.

    :attr sigma: Range of gaussian filter.
    :type sigma: Int.

    :attr diameter: Diameter of pixel neighborhood. (Bilateral)
    :type diameter: Int.

    :attr sigma_space: Range of pixel distance values for mixing in the pixel neighborhood. (Bilateral)
    :type sigma_space: Double.

    :attr sigma_color: Range of pixel color values for color mixing in the pixel neighborhood. (Bilateral)
    :type sigma_color: Double.
    """

    linear = True
    sigma = 3
    diameter = 9
    sigma_space = 150
    sigma_color = 150


class EffectConfig:
    """
    A class providing the effect options.

    :attr lum_scale: Weighting of luminance.
    :type lum_scale: Int.

    :attr chrom_scale: Weighting of chromasity.
    :type chrom_scale: Int.

    :attr level: Scale of the editing function.
    :type level: Float.

    :attr func: Editing function. (e | ln | pow | sqrt | gamma)
    :type func: String.

    :attr mode: Editing mode. (Global | Luminance)
    :type mode: String.
    """

    lum_scale = 1
    chrom_scale = 1
    level = .1
    func = "sqrt"
    mode = "global"


class FilterImageConfig:
    """
    A class providing the filtering options.

    :attr blur: Config for the blur.
    :type blur: BlurImageConfig class.

    :attr effect: Config for the effect.
    :type effect: EffectConfig class.

    :attr gamma: Weighting of details.
    :type gamma: Int.
    """

    blur = BlurImageConfig()
    effect = EffectConfig()
    gamma = 1


class GradientFilterConfig:
    """
    A class providing the filtering for gradient compression.

    :attr saturation: The saturation
    :type saturation: Float

    :attr iteration_amount: The number of iterations in the gradient descent
    :type iteration_amount: Int

    :attr iteration_distance: The the length to travel when descending
    :type iteration_distance: Int

    :attr use_pyramid: Should use a gaussian pyramid or not
    :type use_pyramid: Bool

    :attr func: Should use a gaussian pyramid or not
    :type func: Function
    """

    saturation = 1
    iteration_amount = 5
    iteration_distance = 5
    use_pyramid = False
    func = lambda x: x
