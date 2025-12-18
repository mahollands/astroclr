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

License:

Copyright (C) 2008 Mark Kness

Author - Mark Kness - mkness@alumni.utexas.net

This file is part of ColorPy.

ColorPy is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

ColorPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with ColorPy.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import functools
import numpy as np
from astropy import units as u
from .ciexyz import Spectrum
from . import plots

__all__ = [
    "get_illuminant_D65",
    "get_blackbody_illuminant",
    "get_illuminant_A",
    "get_constant_illuminant",
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
        illuminant.flux /= XYZ.y
        return illuminant

    return _wrapper


@scale_illuminant
def get_illuminant_D65():
    """
    Get CIE Illuminant D65, as a spectrum, normalized to Y = 1.0.

    CIE standard illuminant D65 represents a phase of natural daylight
    with a correlated color temperature of approximately 6504 K.  (Wyszecki, p. 144)

    In the interest of standardization the CIE recommends that D65 be used
    whenever possible.  Otherwise, D55 or D75 are recommended.  (Wyszecki, p. 145)

    (ColorPy does not currently provide D55 or D75, however.)
    """
    # table of CIE Illuminant D65 spectrum.
    # data from: http://cvrl.ioo.ucl.ac.uk/database/data/cie/Illuminantd65.txt
    # massaged into this format.

    wavelength, flux = np.loadtxt(
        f"{os.path.dirname(__file__)}/data/Illuminantd65.txt",
        delimiter=",",
        unpack=True,
    )
    return Spectrum(wavelength, flux)


@scale_illuminant
def get_blackbody_illuminant(T_K):
    """
    Get the spectrum of a blackbody at the given temperature, normalized to Y = 1.0.o
    """
    return Spectrum.BlackBody(T_K, _wavelengths)


def get_illuminant_A():
    """
    Get CIE Illuminant A, as a spectrum, normalized to Y = 1.0.
    This is actually a blackbody illuminant for T = 2856 K.  (Wyszecki, p. 143)
    """
    return get_blackbody_illuminant(2856.0)


@scale_illuminant
def get_constant_illuminant():
    """
    Get an illuminant, with spectrum constant over wavelength, normalized to Y = 1.0.
    """
    return Spectrum(_wavelengths, 1)


def figures():
    """
    Plot spectra for several illuminants.
    """
    plots.spectrum_plot(get_illuminant_D65(), "CIE Illuminant D65", "Illuminant-D65")
    plots.spectrum_plot(get_illuminant_A(), "CIE Illuminant A", "Illuminant-A")
    plots.spectrum_plot(get_constant_illuminant(), "Constant Illuminant", "Illuminant-Const")
    plots.spectrum_plot(get_blackbody_illuminant(5778.0), "5778 K Illuminant", "Illuminant-5778")
