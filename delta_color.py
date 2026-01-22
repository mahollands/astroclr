import numpy as np
from scipy.optimize import minimize
from astropy import units as u
from .ciexyz import Spectrum
from .colormodels import XYZ_Color, RGB_Color, LUV_Color, LAB_Color

def delta_uv(XYZ1, XYZ2):
    """
    Euclidean distance in uv' terms between a pair of XYZ colors
    """
    u1, v1 = LUV_Color.uv_primes(XYZ1)
    u2, v2 = LUV_Color.uv_primes(XYZ2)
    du, dv = u1-u2, v1-v2
    return np.sqrt(du**2 + dv**2)

def theta_to_T(theta):
    """
    Maps temperature to a scale where 0K -> 0, 6500K -> 0.5 and inf -> 1.
    """
    atanT = np.pi/2 * abs(theta)
    return 6500 * float(np.tan(atanT))


def CCT(color: XYZ_Color | RGB_Color | LUV_Color | LAB_Color):
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
