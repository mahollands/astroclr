"""
Functions defined here generate various standard illuminants normalised to Y=1
in the CIE XYZ colour space. Arguably the most important of these is CIE
standard illuminant D65, which is used to define the white point chromaticity
of several RGB colour spaces, including sRGB.
"""

import os
import functools
import numpy as np
from astropy import units as u
from .spectral import Spectrum, CMFs
from . import plots

__all__ = [
    "get_blackbody_illuminant",
    "get_illuminant_A",
    "get_illuminant_DXX",
    "get_illuminant_D65",
    "get_illuminant_E",
]

_wavelengths = CMFs['1931'].wavelength


def scale_illuminant(get_illuminant):
    """
    wrapper function to scale an illuminant to Y=1.0
    """

    @functools.wraps(get_illuminant)
    def _wrapper(*args, **kwargs):
        illuminant = get_illuminant(*args, **kwargs)
        XYZ = illuminant.to_XYZ()
        illuminant.flux /= XYZ.Y
        return illuminant

    return _wrapper


@scale_illuminant
def get_blackbody_illuminant(T_K: float | u.Quantity) -> Spectrum:
    """
    Get the spectrum of a blackbody at the given temperature.
    """
    return Spectrum.BlackBody(T_K, _wavelengths)


def get_illuminant_A() -> Spectrum:
    """
    Get CIE Illuminant A which is just a blackbody for T = 2856 K.
    """
    return get_blackbody_illuminant(2856.0 * u.K)


@functools.cache
def _get_illuminant_D_data() -> tuple:
    wavelength, *S_eig = np.loadtxt(
        f"{os.path.dirname(__file__)}/data/CIE_illum_Dxx_comp.csv",
        delimiter=",",
        unpack=True,
    )
    xd_cool = np.polynomial.Polynomial([0.244063, 0.09911, 2.9678, -4.6070])
    xd_hot = np.polynomial.Polynomial([0.237040, 0.24748, 1.9018, -2.0064])
    yd_poly = np.polynomial.Polynomial([-0.275, +2.870, -3.000])
    d_polys = xd_cool, xd_hot, yd_poly
    M_matrix = np.array(
        [
            [-1.3515, -1.7703, +5.9114],
            [+0.0300, -31.4424, +30.0717],
            [+0.0241, +0.2562, -0.7341],
        ]
    )
    return wavelength, S_eig, d_polys, M_matrix


@scale_illuminant
def get_illuminant_DXX(T: float | u.Quantity) -> Spectrum:
    """
    Generate CIE Illuminant D series. The temperature, T [K], is required as an
    input. For D65 a pre-caculated table from the CIE is loaded instead.
    """
    T *= 1.438776877 / 1.4380  # CCT correcetion
    T_inv = 1000 / T
    wavelength, S_eig, d_polys, M_matrix = _get_illuminant_D_data()
    xd_cool, xd_hot, yd_poly = d_polys
    xd = np.select(
        condlist=[4000 <= T <= 7000, 7000 < T <= 25000],
        choicelist=[xd_cool(T_inv), xd_hot(T_inv)],
        default=np.nan,
    )
    yd = yd_poly(xd)
    M1, M2, M = M_matrix @ (1, xd, yd)
    M1 /= M
    M2 /= M
    S0, S1, S2 = S_eig
    S = S0 + M1 * S1 + M2 * S2
    return Spectrum(wavelength, S)


@scale_illuminant
def get_illuminant_D65() -> Spectrum:
    """
    Get CIE Illuminant D65. CIE standard illuminant D65 represents a phase of
    natural daylight with a correlated color temperature of 6504 K. For other D
    illuminants, use get_illuminant_DXX.
    """
    # table of CIE Illuminant D65 spectrum.
    # data from: http://cvrl.ioo.ucl.ac.uk/database/data/cie/Illuminantd65.txt

    wavelength, flux = np.loadtxt(
        f"{os.path.dirname(__file__)}/data/Illuminantd65.txt",
        delimiter=",",
        unpack=True,
    )
    return Spectrum(wavelength, flux)


@scale_illuminant
def get_illuminant_E() -> Spectrum:
    """
    Get CIE Illuminant E. This is a spectrum with constant flux as a function
    of wavelength.
    """
    return Spectrum(_wavelengths, 1)


def figures():
    """
    Plot spectra for several illuminants.
    """
    import matplotlib.pyplot as plt

    data = [
        (get_illuminant_A(), "CIE Illuminant A"),
        (get_illuminant_D65(), "CIE Illuminant D65"),
        (get_illuminant_DXX(5500), "CIE Illuminant D55"),
        (get_illuminant_DXX(7500), "CIE Illuminant D75"),
        (get_illuminant_E(), "CIE Illuminant E"),
        (get_blackbody_illuminant(10000.0), "10000 K Illuminant"),
    ]
    for illum, title in data:
        plt.figure()
        plots.spectrum_plot(illum, title, ylabel='Flux [nm$^{-1}$]')
        plt.show()
