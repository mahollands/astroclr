"""
blackbody.py - Color of thermal blackbodies.

Description:

Calculate the spectrum of a thermal blackbody at an arbitrary temperature.

Constants:

SUN_TEMPERATURE   - Surface temperature of the Sun, in K

Functions:

blackbody_specific_intensity (wl_nm, T_K) -
    Get the monochromatic specific intensity for a blackbody -
        wl_nm = wavelength [nm]
        T_K   = temperature [K]
    This is the energy radiated per second per unit wavelength per unit solid angle.
    Reference - Shu, eq. 4.6, p. 78.

blackbody_spectrum (T_K) -
    Get the spectrum of a blackbody, as a numpy array.

blackbody_color (T_K) -
    Given a temperature (K), return the xyz color of a thermal blackbody.

Plots:

blackbody_patch_plot (T_list, title, filename) -
    Draw a patch plot of blackbody colors for the given temperature range.

blackbody_color_vs_temperature_plot (T_list, title, filename) -
    Draw a color vs temperature plot for the given temperature range.

blackbody_spectrum_plot (T_K) -
    Draw the spectrum of a blackbody at the given temperature.

References:

Frank H. Shu, The Physical Universe. An Introduction to Astronomy,
University Science Books, Mill Valley, California. 1982. ISBN 0-935702-05-9.

Charles Kittel and Herbert Kroemer, Thermal Physics, 2nd edition,
W. H. Freeman, New York, 1980. ISBN 0-7167-1088-9.

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

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from .ciexyz import Spectrum, CMFs
from .colormodels import RGB_Color
from . import plots

# Physical constants in mks units
SUN_TEMPERATURE = 5778.0 * u.K
_wavelengths = CMFs['1931'].wavelength

# Figures

def blackbody_patch_plot(T_list, title, filename):
    """Draw a patch plot of blackbody colors for the given temperature range."""
    xyz_colors = []
    color_names = []
    for Ti in T_list:
        xyz = blackbody_color(Ti)
        xyz_colors.append(xyz)
        name = "%g K" % (Ti)
        color_names.append(name)
    plots.xyz_patch_plot(xyz_colors, color_names, title, filename)


def blackbody_color_vs_temperature_plot(T_list, title, filename):
    """Draw a color vs temperature plot for the given temperature range."""
    BBs = (Spectrum.BlackBody(T, _wavelengths) for T in T_list)
    XYZs = (BB.to_XYZ() for BB in BBs)
    RGBs = [RGB_Color.from_XYZ(XYZ) for XYZ in XYZs]
    # Note that b and g become negative for low T.
    # MatPlotLib skips those on the semilog plot.
    plots.color_vs_param_plot(
        T_list,
        RGBs,
        title,
        filename,
        plotfunc=plt.semilogy,
        tight=True,
        xlabel=r"Temperature (K)",
        ylabel=r"RGB Color",
    )


def blackbody_spectrum_plot(T_K):
    """
    Draw the spectrum of a blackbody at the given temperature.
    """
    S = Spectrum.BlackBody(T_K, _wavelengths)
    title = f"Blackbody Spectrum - T {round(T_K)} K"
    filename = f"BlackbodySpectrum-{T_K}dK"
    x_unit = S.wavelength.unit.to_string('latex_inline')
    y_unit = S.flux.unit.to_string('latex_inline')
    plots.spectrum_plot(
        S,
        title,
        filename,
        xlabel="Wavelength [{x_unit}]",
        ylabel=r"Specific Intensity [{y_unit}]",
    )

# Create sample figures


def figures():
    """Create some blackbody plots."""
    # Some patch plots.
    T_norm = np.geomspace(1200.0, 20000.0, 48)
    T_hot = np.geomspace(10000.0, 40000.0, 24)
    T_cool = np.geomspace(950.0, 1200.0, 24)
    blackbody_patch_plot(T_norm, "Blackbody Colors", "Blackbody-Patch")
    blackbody_patch_plot(T_hot, "Hot Blackbody Colors", "Blackbody-HotPatch")
    blackbody_patch_plot(T_cool, "Cool Blackbody Colors", "Blackbody-CoolPatch")

    # Color vs temperature.
    T_norm = np.linspace(1200.0, 16000.0, 300)
    T_hot = np.linspace(10000.0, 40000.0, 300)
    T_cool = np.linspace(950.0, 1200.0, 300)
    blackbody_color_vs_temperature_plot(T_norm, "Blackbody Colors", "Blackbody-Colors")
    blackbody_color_vs_temperature_plot(
        T_hot, "Hot Blackbody Colors", "Blackbody-HotColors"
    )
    blackbody_color_vs_temperature_plot(
        T_cool, "Cool Blackbody Colors", "Blackbody-CoolColors"
    )

    # Spectrum for some specific temperatures.
    blackbody_spectrum_plot(2000.0)
    blackbody_spectrum_plot(3000.0)  # Proxima Centauri.
    blackbody_spectrum_plot(SUN_TEMPERATURE)  # Sun.
    blackbody_spectrum_plot(11000.0)  # Rigel.
    blackbody_spectrum_plot(15000.0)


if __name__ == "__main__":
    figures()
