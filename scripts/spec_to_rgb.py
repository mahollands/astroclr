#!/usr/bin/env python
"""
Reads in a spectrum (or spectra) from the command line and plots the
corresponding spectrum.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from colorpy2.ciexyz import Spectrum, RGB_from_wavelength
from colorpy2.colormodels import ColorList, RGB_Color

if len(sys.argv) == 0:
    print("need at least one filename for a spectrum")
    sys.exit()

rect_x = np.array([-0.5, 0.5, 0.5, -0.5])
rect_y = np.array([1.1, 1.1, 1.2, 1.2])


def spectrum_figure(fname, units=(u.AA, u.erg / u.s / u.cm**2 / u.AA)):
    S = Spectrum.from_file(fname, units=units)
    S.wavelength = S.wavelength.to(u.nm)
    S.flux = S.flux.to(u.W / u.m**2 / u.nm)
    S.flux /= np.percentile(S.flux, 99).value
    S_XYZ = S.to_XYZ()
    S_hex = S.to_IRGB_string()

    RGBs = RGB_from_wavelength(S.wavelength)
    RGBs = ColorList([RGB * f for RGB, f in zip(RGBs, S.flux.value)])
    offset = RGBs.min_component()
    grey = RGB_Color(offset, offset, offset)
    RGBs = ColorList([RGB - grey for RGB in RGBs])
    norm = RGBs.max_component()
    RGBs = RGBs.rescale(1 / norm)

    print(fname, S_XYZ.normalize(), S_hex)

    plt.figure(fname)
    for lam, RGB in zip(S.wavelength, RGBs):
        plt.fill(rect_x + lam.value, rect_y, c=RGB.to_IRGB().to_string(), zorder=10)
    plt.fill_between(S.wavelength.value, 0, S.flux.value, facecolor=S_hex)
    plt.plot(S.wavelength, S.flux, c="k", lw=2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(r"$F_\lambda$")
    plt.ylim(0, 1.2)
    plt.xlim(360, 740)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for fname in sys.argv[1:]:
        figure(fname)
