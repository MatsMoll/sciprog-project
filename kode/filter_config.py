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

    :attr diameter: Diameter of pixel neighborhood.
    :type diameter: Int.

    :attr sigma_space: Range of pixel distance values for mixing in the pixel neighborhood.
    :type sigma_space: Double.

    :attr sigma_color: Range of pixel color values for color mixing in the pixel neighborhood.
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
