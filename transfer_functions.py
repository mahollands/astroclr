"""
Gamma correction / electro-optical tranfser functions

Non-gamma corrected rgb values, also called non-linear rgb values,
correspond to palette register entries [although here they are kept
in the range 0.0 to 1.0.]  The numerical values are not proportional
to the amount of light energy present.

Gamma corrected rgb values, also called linear rgb values,
do not correspond to palette entries.  The numerical values are
proportional to the amount of light energy present.
"""

import numpy as np
from functools import partial


def gamma_EOTF(gamma: float = 2.2):
    """
    Simple power law for gamma correction.
    """

    def EOTF(x, gamma):
        return x**gamma

    def OETF(x, gamma):
        return x ** (1 / gamma)

    EOTF = partial(EOTF, gamma=gamma)
    EOTF.inv = partial(OETF, gamma=gamma)
    return EOTF


def sRGB_EOTF():
    """
    sRGB standard for gamma correction. The effect of the equations is to
    closely fit a straightforward gamma 2.2 curve with an slight offset to
    allow for invertability in integer math. Therefore, we are maintaining
    consistency with the gamma 2.2 legacy images and the video industry.
    https://www.color.org/chardata/rgb/srgb.xalter
    """

    @np.vectorize
    def EOTF(x):
        # break_point = 0.03928 #old value
        break_point = 0.04045
        linear = x / 12.92
        gamma = ((x + 0.055) / 1.055) ** 2.4
        return linear if x <= break_point else gamma

    @np.vectorize
    def OETF(x):
        # break_point = 0.00304 #old value
        break_point = 0.0031308
        linear = 12.92 * x
        gamma = 1.055 * x ** (1 / 2.4) - 0.055
        return linear if x <= break_point else gamma

    EOTF.inv = OETF
    return EOTF


def BT1886_EOTF(gamma: float = 2.4, Lb_Lw: float = 0):
    """
    BTU 1886 is a essentially a gamma function, with specificed black luminance
    relative to white. This EOTF is done on a scale of 0-1 rather than in
    physical luminance.
    """
    igamma = 1 / gamma
    Lb_igamma = Lb_Lw**igamma
    a = (1 - Lb_igamma) ** gamma
    b = Lb_igamma / (1 - Lb_igamma)

    def EOTF(x, gamma, a, b):
        return a * max(x + b, 0) ** gamma

    # TODO check this
    def OETF(x, igamma, a, b):
        scaled = (x / a) ** igamma
        return max(scaled - b, 0)

    EOTF = partial(EOTF, gamma=gamma, a=a, b=b)
    EOTF.inv = partial(OETF, igamma=igamma, a=a, b=b)
    return EOTF


def PQ_EOTF():
    """
    Perceptual Quantizer transfer function. Unlike the SDR EOTFs which map to
    an arbitrary luminance range, the unit PQ output scale (input scale to the
    inv method), maps linearly to a 10000 cd/m^2 (nit) range in physical
    luminance.
    https://www.color.org/chardata/rgb/BT2100.xalter
    """
    ms = 1305 / 8192, 2523 / 32
    cs = 107 / 128, 2413 / 128, 2392 / 128

    @np.vectorize
    def EOTF(x, ms, cs):
        (m1, m2), (c1, c2, c3) = ms, cs
        Em2 = x ** (1 / m2)
        numerator = max(Em2 - c1, 0)
        denominator = c2 - c3 * Em2
        return (numerator / denominator) ** (1 / m1)

    @np.vectorize
    def OETF(x, ms, cs):
        (m1, m2), (c1, c2, c3) = ms, cs
        Ym1 = x**m1
        numerator = c1 + c2 * Ym1
        denominator = 1 + c3 * Ym1
        return (numerator / denominator) ** m2

    EOTF = partial(EOTF, ms=ms, cs=cs)
    EOTF.inv = partial(OETF, ms=ms, cs=cs)
    return EOTF
