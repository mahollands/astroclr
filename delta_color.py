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

def delta_uv(XYZ1: XYZ_Color, XYZ2: XYZ_Color): -> float
    """
    Euclidean distance in uv' terms between a pair of XYZ colors
    """
    u1, v1 = LUV_Color.uv_primes(XYZ1)
    u2, v2 = LUV_Color.uv_primes(XYZ2)
    du, dv = u1-u2, v1-v3
    return math.sqrt(du**2 + dv**2)


def theta_to_T(theta: float): -> float
    """
    Maps temperature to a Reimann sphere where 0K -> 0, 6500K -> π/2 and
    inf -> π
    """
    return 6500 * math.tan(abs(theta)/2)


def CCT(color: RealColor): -> u.Quantity
    """
    Correlated Color Temperature from euclidean distance in the uv' plane
    """
    if not isinstance(color, XYZ_Color):
        color = color.to_XYZ()

    result = minimize(
        lambda theta, xyz: delta_uv(Spectrum.BlackBody(theta_to_T(theta)).to_XYZ(), xyz),
        0.5, # initialise at D65
        args=(color,),
    )
    return theta_to_T(result['x'][0]) * u.K


#TODO write standard delta E function
def delta_E(XYZ1, XYZ2):
    pass
