from scipy.optimize import minimize
from .ciexyz import Spectrum
from .colormodels import LUV_Color

def delta_uv(XYZ1, XYZ2):
    """
    Euclidean distance in uv' terms between a pair of XYZ colors
    """
    u1, v1 = LUV_Color.uv_primes(XYZ1)
    u2, v2 = LUV_Color.uv_primes(XYZ2)
    du, dv = u1-u2, v1-v2
    return np.sqrt(du**2 + dv**2)


def CCT(XYZ):
    """
    Correlated Colour Temperature from euclidean distance in the uv' plane
    """
    result = minimize(
        lambda T, xyz: delta_uv(Spectrum.BlackBody(T).to_XYZ(), XYZ),
        6500, # initialise at D65
        args=(XYZ,),
        tol=1e-8, #tol>1e-8 does not converge accuately enough
    )
    return float(result['x'][0])


#TODO write standard delta E function
def delta_E(XYZ1, XYZ2):
    pass
