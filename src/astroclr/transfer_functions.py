"""
Gamma correction / electro-optical tranfser functions

In astroclr, RGB_Color is used to represent linear RGB on a scale of 0-1, where
RGB_Color(1, 1, 1) represents peak white. However these do not directly correspond
to the 8-bit display values used in most display colour spaces. Instead a transfer
function must be applied to each component, before quantising to the 0-255 display
scale. In other words the display scale is a non-linear encoding of intensity.
"""

import numpy as np
from functools import partial


def gamma_EOTF(gamma: float):
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


def piecewise_EOTF(break_point: float, gamma: float, slope: float, offset: float):
    """
    Colour spaces such as sRGB use a piecewise EOTF/OETF, which approximates a
    gamma shifted curve curve over most of the range, but features a small
    linear section near zero. This avoids a vertical slope in the OETF at zero,
    allowing higher precision gamma correction for near-black values.
    """

    def EOTF(x, break_point, gamma, slope, offset):
        linear_section = x / slope
        gamma_section = ((x + offset) / (1 + offset)) ** gamma
        return np.where(x <= break_point * slope, linear_section, gamma_section)

    def OETF(x, break_point, gamma, slope, offset):
        linear_section = slope * x
        gamma_section = (1 + offset) * x ** (1 / gamma) - offset
        return np.where(x <= break_point, linear_section, gamma_section)

    EOTF = partial(
        EOTF, break_point=break_point, gamma=gamma, slope=slope, offset=offset
    )
    EOTF.inv = partial(
        OETF, break_point=break_point, gamma=gamma, slope=slope, offset=offset
    )
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

    def EOTF(x, ms, cs):
        (m1, m2), (c1, c2, c3) = ms, cs
        Em2 = x ** (1 / m2)
        numerator = np.clip(Em2 - c1, 0, np.inf)
        denominator = c2 - c3 * Em2
        return (numerator / denominator) ** (1 / m1)

    def OETF(x, ms, cs):
        (m1, m2), (c1, c2, c3) = ms, cs
        Ym1 = x**m1
        numerator = c1 + c2 * Ym1
        denominator = 1 + c3 * Ym1
        return (numerator / denominator) ** m2

    ms = 1305 / 8192, 2523 / 32
    cs = 107 / 128, 2413 / 128, 2392 / 128

    EOTF = partial(EOTF, ms=ms, cs=cs)
    EOTF.inv = partial(OETF, ms=ms, cs=cs)
    return EOTF
