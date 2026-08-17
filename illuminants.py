"""
illuminants.py - Definitions of some standard illuminants.

Description:

Illuminants are spectrums, normalized so that Y = 1.0.

Spectrums are 2D numpy arrays, with one row for each wavelength,
with the first column holding the wavelength in nm, and the
second column the intensity.

The spectrums have a wavelength increment of 1 nm.

Functions:

init () -
    Initialize CIE Illuminant D65.  This runs on module startup.

get_illuminant_D65 () -
    Get CIE Illuminant D65, as a spectrum, normalized to Y = 1.0.

    CIE standard illuminant D65 represents a phase of natural daylight
    with a correlated color temperature of approximately 6504 K.  (Wyszecki, p. 144)

    In the interest of standardization the CIE recommends that D65 be used
    whenever possible.  Otherwise, D55 or D75 are recommended.  (Wyszecki, p. 145)

    (ColorPy does not currently provide D55 or D75, however.)

get_illuminant_A () -
    Get CIE Illuminant A, as a spectrum, normalized to Y = 1.0.
    This is actually a blackbody illuminant for T = 2856 K.  (Wyszecki, p. 143)

get_blackbody_illuminant (T_K) -
    Get the spectrum of a blackbody at the given temperature, normalized to Y = 1.0.

get_constant_illuminant () -
    Get an illuminant, with spectrum constant over wavelength, normalized to Y = 1.0.

scale_illuminant (illuminant, scaling) -
    Scale the illuminant intensity by the specfied factor.

References:

Wyszecki and Stiles, Color Science: Concepts and Methods, Quantitative Data and Formulae,
    2nd edition, John Wiley, 1982. Wiley Classics Library Edition 2000. ISBN 0-471-39918-3.

CVRL Color and Vision Database - http://cvrl.ioo.ucl.ac.uk/index.htm - (accessed 17 Sep 2008)
    Color and Vision Research Laboratories.
    Provides a set of data sets related to color vision.
    ColorPy uses the tables from this site for the 1931 CIE XYZ matching functions,
    and for Illuminant D65, both at 1 nm wavelength increments.

CIE Standards - http://cvrl.ioo.ucl.ac.uk/cie.htm - (accessed 17 Sep 2008)
    CIE standards as maintained by CVRL.
    The 1931 CIE XYZ and D65 tables that ColorPy uses were obtained from the following files, linked here:
        http://cvrl.ioo.ucl.ac.uk/database/data/cmfs/ciexyz31_1.txt
        http://cvrl.ioo.ucl.ac.uk/database/data/cie/Illuminantd65.txt

CIE International Commission on Illumination - http://www.cie.co.at/ - (accessed 17 Sep 2008)
    Official website of the CIE.
    There are tables of the standard functions (matching functions, illuminants) here:
        http://www.cie.co.at/main/freepubs.html
        http://www.cie.co.at/publ/abst/datatables15_2004/x2.txt
        http://www.cie.co.at/publ/abst/datatables15_2004/y2.txt
        http://www.cie.co.at/publ/abst/datatables15_2004/z2.txt
        http://www.cie.co.at/publ/abst/datatables15_2004/sid65.txt
    ColorPy does not use these specific files.
"""

import os
import functools
import numpy as np
from astropy import units as u
from .spectral import Spectrum
from . import plots

__all__ = [
    "get_blackbody_illuminant",
    "get_illuminant_A",
    "get_illuminant_DXX",
    "get_illuminant_D65",
    "get_illuminant_E",
]

#
# Get any of the available illuminants - D65, A, any blackbody, or a constant spectrum.
# ColorPy does not currently provide D55 or D75.
#

_wavelengths = np.arange(360, 830 + 0.5, 1.0) * u.nm


def scale_illuminant(get_illuminant):
    """
    Scale the illuminant intensity to Y=1.0
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
    Get the spectrum of a blackbody at the given temperature, normalized to Y = 1.0.o
    """
    return Spectrum.BlackBody(T_K, _wavelengths)


def get_illuminant_A() -> Spectrum:
    """
    Get CIE Illuminant A, as a spectrum, normalized to Y = 1.0.
    This is actually a blackbody illuminant for T = 2856 K.  (Wyszecki, p. 143)
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
    Generate CIE Illuminant D series, as a spectrum, normalized to Y = 1.0.

    The temperature, T [K], is required as an input

    For D65 a separate pre-caculated table from the CIE is loaded instead
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
    Get CIE Illuminant D65, as a spectrum, normalized to Y = 1.0.

    CIE standard illuminant D65 represents a phase of natural daylight
    with a correlated color temperature of approximately 6504 K.  (Wyszecki, p. 144)

    In the interest of standardization the CIE recommends that D65 be used
    whenever possible.  Otherwise, D55 or D75 are recommended.  (Wyszecki, p. 145)

    For other D illuminants, use get_illuminant_DXX
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
    Get CIE Illuminant E, as a spectrum, normalized to Y = 1.0.
    This is a spectrum constant in wavelength
    """
    return Spectrum(_wavelengths, 1)


def figures():
    """
    Plot spectra for several illuminants.
    """
    plots.spectrum_plot(get_illuminant_A(), "CIE Illuminant A", "Illuminant-A")
    plots.spectrum_plot(get_illuminant_D65(), "CIE Illuminant D65", "Illuminant-D65")
    plots.spectrum_plot(
        get_illuminant_DXX(5500), "CIE Illuminant D55", "Illuminant-D55"
    )
    plots.spectrum_plot(
        get_illuminant_DXX(7500), "CIE Illuminant D75", "Illuminant-D75"
    )
    plots.spectrum_plot(get_illuminant_E(), "Constant Illuminant", "Illuminant-Const")
    plots.spectrum_plot(
        get_blackbody_illuminant(5778.0), "5778 K Illuminant", "Illuminant-5778"
    )
