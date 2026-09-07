"""
AstroClr is a package for converting spectra into perceived colours
(chromaticities) as well as sRGB that can be shown on a computer display. It
provides various tools for conversions between different colour spaces, and
provides a few useful scripts for things you might want to do with this module.
AstroClr designed primarily for astronomers who may want to determine the
visual appearance of stars from astrophysical spectra, though will work
more generally for all kinds of optical spectra.
"""

# This file only needs to exist to indicate that this is a package.

from .spectral import *
from .colormodels import *
from .RGB_color_spaces import sRGB
from .delta_color import *
