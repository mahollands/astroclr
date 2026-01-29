"""
Sub-module for calculating color differences such as ΔE and
the correlated color temperature.
"""
import math
from scipy.optimize import minimize
from astropy import units as u
from .ciexyz import Spectrum
from .colormodels import XYZ_Color, RGB_Color, LUV_Color, LAB_Color

__all__ = [
    "CCT",
    "delta_E",
]

type RealColor = XYZ_Color | RGB_Color | LUV_Color | LAB_Color

def delta_uv(XYZ1: XYZ_Color, XYZ2: XYZ_Color) -> float:
    """
    Euclidean distance in UCS 1960 uv terms between a pair of XYZ colors
    """
    u1, v1 = LUV_Color.uv_primes(XYZ1) #uv' 1972
    u2, v2 = LUV_Color.uv_primes(XYZ2)
    du, dv = u1-u2, v1-v2
    dv *= 2/3 #convert from uv' 1972 to UCS 1960 uv
    return du**2 + dv**2


def theta_to_T(theta: float) -> float:
    """
    Maps temperature to a Reimann sphere where 0K -> 0, 6500K -> π/2 and
    inf -> π
    """
    return 6500 * math.tan(abs(theta)/2)


def CCT(color: RealColor) -> u.Quantity:
    """
    Correlated Color Temperature from euclidean distance in the uv plane
    """
    if not isinstance(color, XYZ_Color):
        color = color.to_XYZ()

    result = minimize(
        lambda theta, xyz: delta_uv(Spectrum.BlackBody(theta_to_T(theta)).to_XYZ(), xyz),
        math.pi/2, # initialise at D65
        args=(color,),
    )
    return theta_to_T(result['x'][0]) * u.K


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
