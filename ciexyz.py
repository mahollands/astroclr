"""
ciexyz.py - Spectral response curves for 1931 CIE XYZ 2 degree field of view
matching functions.

Description:

This module provides the CIE standard XYZ color matching functions. The 1931
tabulation, for a 2 degree field of view, is used in preference to the 10
degree 1964 set, as is conventional in computer graphics.

The matching functions are stored internally at 1 nm increments, and linear
interpolation is used for any wavelength in between.

ColorPy attempts to scale the matching functions so that: A spectrum, constant
with wavelength, over the range 360 nm to 830 nm, with a total intensity equal
to the (assumed) physical intensity of the monitor, will sample with Y = 1.0.

This scaling corresponds with that in colormodels.py, which assumes Y = 1.0 at
full white.

NOTE - I suspect that the scaling is not quite correct.  I think it is at least
close.

Ideally, we would like the spectrum of the actual monitor display, at full
white, which is not independent of wavelength, to sample to Y = 1.0.

Constants and Functions:

start_wl_nm, end_wl_nm - Default starting and ending range of wavelengths, in nm, as integers.
delta_wl_nm            - Default wavelength spacing, in nm, as a float.

def xyz_from_wavelength (wl_nm) -
    Given a wavelength (nm), return the corresponding xyz color, for unit
    intensity.

def xyz_from_spectrum (spectrum) -
    Determine the xyz color of the spectrum.

    The spectrum is assumed to be a 2D numpy array, with a row for each
    wavelength, and two columns.  The first column should hold the wavelength
    (nm), and the second should hold the light intensity.  The set of
    wavelengths can be arbitrary, it does not have to be the set that
    empty_spectrum() returns.

def get_normalized_spectral_line_colors (
    brightness = 1.0,
    num_purples = 0,
    dwl_angstroms = 10):
    Get an array of xyz colors covering the visible spectrum. Optionally add a
    number of 'purples', which are colors interpolated between the color of the
    lowest wavelength (violet) and the highest (red).

    brightness - Desired maximum rgb component of each color.  Default 1.0.
    (Maxiumum displayable brightness)
    num_purples - Number of colors to interpolate in the 'purple' range.
    Default 0.  (No purples)
    dwl_angstroms - Wavelength separation, in angstroms (0.1 nm).  Default 10
    A. (1 nm spacing)

References:

Wyszecki and Stiles, Color Science: Concepts and Methods, Quantitative Data and Formulae,
    2nd edition, John Wiley, 1982. Wiley Classics Library Edition 2000. ISBN 0-471-39918-3.

CVRL Color and Vision Database - http://cvrl.ioo.ucl.ac.uk/index.htm - (accessed 17 Sep 2008)
    Color and Vision Research Laboratories.
    Provides a set of data sets related to color vision.
    ColorPy uses the tables from this site for the 1931 CIE XYZ matching functions,
    and for Illuminant D65, both at 1 nm wavelength increments.

CIE Standards - http://cvrl.ioo.ucl.ac.uk/cie.htm - (accessed 17 Sep 2008)
    CIE standards as maintained by CVRL. The 1931 CIE XYZ and D65 tables that
    ColorPy uses were obtained from the following files, linked here:
        http://cvrl.ioo.ucl.ac.uk/database/data/cmfs/ciexyz31_1.txt
        http://cvrl.ioo.ucl.ac.uk/database/data/cie/Illuminantd65.txt

CIE International Commission on Illumination - http://www.cie.co.at/ - (accessed 17 Sep 2008)
    Official website of the CIE. There are tables of the standard functions
    (matching functions, illuminants) here:
        http://www.cie.co.at/main/freepubs.html
        http://www.cie.co.at/publ/abst/datatables15_2004/x2.txt
        http://www.cie.co.at/publ/abst/datatables15_2004/y2.txt
        http://www.cie.co.at/publ/abst/datatables15_2004/z2.txt
        http://www.cie.co.at/publ/abst/datatables15_2004/sid65.txt
    ColorPy does not use these specific files.

Charles Poynton - Frequently asked questions about Gamma and Color,
    posted to comp.graphics.algorithms, 25 Jan 1995.

License:

Copyright (C) 2008 Mark Kness

Author - Mark Kness - mkness@alumni.utexas.net

This file is part of ColorPy.

ColorPy is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

ColorPy is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with ColorPy.  If not, see <http://www.gnu.org/licenses/>.
"""

import os.path
import numpy as np
from astropy import units as u
from astropy.modeling.physical_models import BlackBody
from .colormodels import XYZ_Color, RGB_Color

# table of 1931 CIE XYZ matching functions.
# data from: http://cvrl.ioo.ucl.ac.uk/database/data/cmfs/ciexyz31_1.txt
# massaged into this format.

_CIEXYZ_1931_table = np.loadtxt(
    f"{os.path.dirname(__file__)}/data/ciexyz31_1.txt",
    delimiter=",",
    unpack=True,
)

# Private - tables of spectral curves
_wavelengths, *_xyz_table = _CIEXYZ_1931_table
_wavelengths <<= u.nm

_flux_unit1 = u.W/u.m**2/u.sr/u.nm
_flux_unit2 = _flux_unit1 * u.sr
_valid_flux_units = _flux_unit1, _flux_unit2


def interpolate_xyz_curves(wavelength):
    xyz_curves_interpolated = [
        np.interp(wavelength, _wavelengths, xyz_curve, left=0, right=0)
        for xyz_curve in _xyz_table
    ]
    return xyz_curves_interpolated


class Spectrum:
    """
    Simple spectrum class for calculating XYZ values. 'wavelength' and 'flux'
    can be given as numpy arrays or astropy quantities. If these are arrays,
    default units of nm and W/m2/sr/nm are applied, respectively. If given as
    quantities, wavelengths must be of type length, and fluxes must be in units
    equivalent to W/m2/sr/nm or W/m2/nm.

    wavelengths -
        wavelength array/Quantity. If an array is given nm is assumed for the
        units. If a Quantity is given, it must have dimensions of length.
    flux -
        flux array/Quantity. If an array is given W/m^2/sr/nm is assumed for
        the units. If a Quanitity is given, it must have dimensions of flux per
        unit wavelength or flux per unit wavelength per solid_angle.
    """

    __slots__ = ["_x", "_y"]

    def __init__(self, wavelength, flux):
        self.wavelength = wavelength
        self.flux = flux

    @property
    def wavelength(self):
        return self._x
    @wavelength.setter
    def wavelength(self, value):
        if not isinstance(value, u.Quantity):
            value <<= u.nm
        if value.isscalar:
            raise ValueError("wavelengths must be array-like")
        if not value.unit.is_equivalent(u.nm):
            raise ValueError("wavelengths must be a kind of length")
        self._x = value

    @property
    def flux(self):
        return self._y
    @flux.setter
    def flux(self, value):
        if not isinstance(value, u.Quantity):
            value <<= _flux_unit1
        if value.isscalar:
            value = value * np.ones(self.wavelength.size)
        elif len(value) != len(self.wavelength):
            raise ValueError("Fluxes must be the same length as wavelengths")
        if not value.unit.is_equivalent(_valid_flux_units):
            raise ValueError("Flux units must be equivalent to W/m2/sr/nm or W/m2/nm")
        self._y = value

    def __len__(self):
        return len(self._x)

    def __repr__(self):
        n_spec =  len(self)
        x_unit = self.wavelength.unit.to_string()
        y_unit = self.flux.unit.to_string()
        wavelength_str = f"wavelength[{n_spec}] * {x_unit}"
        flux_str = f"flux[{n_spec}] * {y_unit}"
        return f"Spectrum({wavelength_str}, {flux_str})"

    @classmethod
    def BlackBody(cls, temperature, wavelength):
        """
        Class method for generating a black body spectrum. For
        a given temperature and wavelength array/Quantity.
        >>> BB = Spectrum.BlackBody(temperature, wavelengths)
        """

        if not isinstance(temperature, u.Quantity):
            temperature <<= u.K
        if not temperature.isscalar:
            raise ValueError("temperature should be a scalar")
        BB = BlackBody(temperature, scale=1*_flux_unit1)
        flux = BB(wavelength)
        return cls(wavelength, flux)

    def _XYZ_integrals(self):
        """
        Integrate spectrum over CIE1931 X/Y/Z color-matching functions.
        The result is a 3-valued astropy quantity with units included.
        """
        xyz_intp = interpolate_xyz_curves(self.wavelength)
        integrand = self.flux[np.newaxis,:] * xyz_intp
        I_XYZ = np.trapz(integrand, self.wavelength, axis=1)
        return I_XYZ

    def to_XYZ(self):
        """
        Calculate the XYZ color of the spectrum. Units are dropped but
        converted to SI  W/m^2 or W/m^2/sr first (depending on the flux units).
        """
        XYZ = self._XYZ_integrals()
        return XYZ_Color(*XYZ.si.value)

    def to_luminance(self):
        """
        If flux units are equivalent to W/m2/sr/nm calculate the luminance in
        cd/m^2 (nits). If flux units are equivalent to W/m2/nm calculate the
        illuminance (luminous flux density arriving at a surface from a distant
        light source) in lm/m^2 (lux), or the luminous exitance (luminous flux
        density radiating from a surface) also in lm/m^2 (lux not used).
        """
        X, Y, Z = self._XYZ_integrals()
        L = 683 * u.lm/u.W * Y

        if self.flux.unit.is_equivalent(_flux_unit2):
            return L.to(u.lm/u.m**2)
        return L.to(u.cd/u.m**2)


def XYZ_from_wavelength(wavelength):
    """
    Given a wavelength (nm), return the corresponding XYZ_Color, for unit
    intensity.
    """
    if not isinstance(wavelength, u.Quantity):
        wavelength <<= u.nm
    if not wavelength.isscalar:
        raise ValueError("Wavlength must be scalar")
    xyz = interpolate_xyz_curves(wavelength)
    return XYZ_Color(*xyz)


def get_normalized_spectral_line_colors(
    brightness=1.0, num_purples=0, dwl=1.0 * u.nm, annotate=False
):
    """
    Get a list of XYZ colors covering the visible spectrum. Optionally add
    a number of 'purples', which are colors interpolated between the color of
    the lowest wavelength (violet) and the highest (red).

    brightness - Desired maximum rgb component of each color.  Default 1.0.
    (Maximum displayable brightness)
    num_purples - Number of colors to interpolate in the 'purple' range.
    Default 0. (No purples)
    dwl - wavelength separation, Default 1nm
    annotate - additionally return string labels for each color
    """
    wavelengths = np.arange(_wavelengths[0], _wavelengths[-1] + dwl / 2, dwl)

    # build list of normalized color x,y values proceeding along each wavelength
    XYZs = [XYZ_from_wavelength(wavelength) for wavelength in wavelengths]
    xyzs = [XYZ.normalize() for XYZ in XYZs]

    # interpolate from start point to end point (filling in the purples)
    first_xyz, last_xyz = xyzs[0], xyzs[-1]
    fs = np.linspace(0, 1, num_purples)
    xyz_purples = [first_xyz * f + last_xyz * (1 - f) for f in fs]
    xyzs.extend(xyz_purples)

    # scale each color to have the max rgb component equal to the desired
    # brightness
    RGBs = [brightness * RGB_Color.from_XYZ(xyz, brightest=True) for xyz in xyzs]
    XYZs = [rgb.to_XYZ() for rgb in RGBs]

    if not annotate:
        return XYZs

    annotation_monochromatic = [lam.to_string() for lam in wavelengths]
    annotation_purples = [f"{f:.3f} purples" for f in fs]
    annotation = annotation_monochromatic + annotation_purples

    return XYZs, annotation
