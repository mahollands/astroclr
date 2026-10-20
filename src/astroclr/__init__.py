""" AstroClr is a package for converting spectra into perceived colours
(chromaticities) as well as RGB colour spaces (such as sRGB) that can be shown
on a computer display. It provides the tools needed for converting between
different colour spaces, and provides a few useful scripts for things you might
want to do with this module.

AstroClr is designed primarily for astronomers who may want to determine the
visual appearance of stars (or other astrophyisical objects) from astrophysical
spectra, though will work more generally for all kinds of optical spectra.

Astroclr is heavily inspired by ColorPy by Mark Kness, but has been rewritten
from the ground up, providing a modern pythonic interface. For example, a
Spectrum class is provided which is compatable with astropy units, and provides
methods for various colour transformations. Furthermore, colour classes can
accept arrays of colours, allowing efficient calculations when working with
lists of objects, or images.


References for colour science
=============================

CIE, Technical Note: Brussels Session of the International Commision on
Illumination, J. Opt. Soc. Am. (50), 89 (1960)

Foley et al., Computer Graphics Principles and Practice, Addison Wesley, Boston
(1982)

Hall, Illuminaiton and Color in Computer Generated Imagery, Springer, New York
(1988)

IEC, Multimedia systema dn equipment - color measurement and management -
part2-1: Colour management - default rgb colour space - srgb, International
Electrotechnical Comission, 61966-2-1 (1999)

Judd & Wyszecki, Color in Business, Science, and Industry, Wiley, New York
(1975)

Judd, Repost of U.S. secretariat committee on colorimetry and artificial
daylight, Proceedings of the Twelfth Session of the CIE, Stockholm, Bureau
Central de la CIE (1951)

MacAdam, Projective Transormations of I. C. I. Color Specifications, J. Opt.
Soc. Am., 27, 294 (1937)

MacLeod et al., Funadamental chromaticity diagram with physiological axes -
part 1, Tech Rep. 170-1, International Comission on Illumination (2006)

Smith & Guild, The CIE colorimetric standards and their use, Transactions of
the Optical Society 33, 73 (1931)

Stockman & Sharpe, The spectral sensitivities of the middle and long-wavelength
sensitive cones derived from measurements in observers of known genotype,
Vision Res. 40, 1711 (2000)

Viénot et al., Fundamental chromacticity diagram with physiological axes - part
2: Spectral luminous efficiency functions and chromaticity diagrams, Tech Rep.
170-2, International Comission on Illumination (2015)

Vos, Colorimetric and photometric properties of a 2° fundamanetalobserver,
Color Research and Application 3, 125 (1978)

Weszecki & Stiles, Color science: concepts and methods, quantitative data and
formulae, Wiley, New York, Chichester (1982)


Astronomy in colour
===================

Burrows, Alkali Metals and the Colour of Browind Dwarfs, Ultracool Dwarfs: New
Spectral Types L and T (2021)

Burrows et al., Beyond the T Dwarfs: Theoretical Spectra, Colors, and
Detectability of the Coolest Brown Dwarfs, ApJ 596, 587 (2003)

Cardiel et al: Synthetic RGB photometry of bright stars: the definition of the
standard photometric system and UCM library of spectrophotometric spectra,
MNRAS 504, 3730 (2021)

Gao et al., Aerosols in Exoplanet Atmospheres, Journal of Geophysical Research
(Planets), 126, e06655 (2021)

Harre & Heller, Digital color codes of Stars, Astronomische Nachrichten 342,
578 (2021)

Hollands et al., Any colour you like: the diverse chromaticities of white
dwarfs, RASTI 5, rzag069 (2026)

Irwin et al., Modelling the seasonal cycle of Uranus's colour and magnitude,
and comparison with Neptune, MNRAS 527, 11521 (2024)

Kohman, Visual Star Colours from Instrumental Photometry, J. R. Astron Soc.
Can. 99, 98 (2005)

Rering & Charnay, New Calculations of Human-eye Colors of Cloudy Brown Dwarfs
and Exoplanets, Research Notes of the American Astronomical Socitey, 9, 168
(2025)
"""

from .colormodels import *
from .delta_color import *
from .rgb_color_spaces import *
from .spectral import *

__version__ = "1.0.3"
