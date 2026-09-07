"""
Sub-module for calculating color differences such as ΔE and
the correlated color temperature.
"""

import math
from scipy.optimize import leastsq
from astropy import units as u
from .spectral import Spectrum
from .colormodels import XYZ_Color, RGB_Color, UCS_Color, LUV_Color, LAB_Color

__all__ = [
    "CCT",
    "delta_E",
]

type RealColor = XYZ_Color | RGB_Color | UCS_Color | LUV_Color | LAB_Color


def delta_uv(XYZ1: XYZ_Color, XYZ2: XYZ_Color) -> float:
    """
    vector difference in UCS 1960 uv terms between a pair of XYZ colors
    """
    UCS1, UCS2 = UCS_Color.from_XYZ(XYZ1), UCS_Color.from_XYZ(XYZ2)
    du, dv = UCS1.u - UCS2.u, UCS1.v - UCS2.v
    return [du, dv]


def delta_uv_iT(invT: float, XYZ: XYZ_Color) -> float:
    """
    vector difference in UCS 1960 uv terms between a blackbody temperature (in
    mireds) and another XYZ color.
    """
    T = 1e6 / invT[0]
    BB = Spectrum.BlackBody(T)
    return delta_uv(BB.to_XYZ(), XYZ)


def CCT(color: RealColor, return_uv_vec: bool = False) -> u.Quantity:
    """
    Correlated Color Temperature from euclidean distance in the uv plane
    """
    if not isinstance(color, XYZ_Color):
        color = color.to_XYZ()

    result = leastsq(
        delta_uv_iT,
        1e6 / 6500,  # initialise at D65 in mireds
        args=(color,),
    )
    T = 1e6 / result[0][0] * u.K

    if not return_uv_vec:
        return T

    BB = Spectrum.BlackBody(T)
    duv = delta_uv(BB.to_XYZ(), color)
    return T, duv, math.hypot(du, dv)


def delta_E(color1: RealColor, color2: RealColor) -> float:
    """
    CIE 1976 method to calculate colour difference ΔE. Most color types
    are accepted where they are internaly converted to CIE LAB space.
    """
    if not isinstance(color1, LAB_Color):
        if not isinstance(color1, XYZ_Color):
            color1 = color1.to_XYZ()
        color1 = LAB_Color.from_XYZ(color1)
    if not isinstance(color2, LAB_Color):
        if not isinstance(color2, XYZ_Color):
            color2 = color2.to_XYZ()
        color2 = LAB_Color.from_XYZ(color2)

    dL = color1.L - color2.L
    da = color1.a - color2.a
    db = color1.b - color2.b
    return math.sqrt(dL**2 + da**2 + db**2)
