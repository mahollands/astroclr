"""
Definitions of some standard values for colors and conversions
Chromaticities of various standard phosphors and white points.
"""

import functools
from typing import Callable
from dataclasses import dataclass
import numpy as np
from astropy import units as u
from ._colormodels_base import XYZ_Color
from .transfer_functions import gamma_EOTF, sRGB_EOTF, PQ_EOTF

# White points [all are for CIE 1931 for small field of view]
#   These are from Judd/Wyszecki
Whites = {
    "A": XYZ_Color(0.4476, 0.4074),  # approx 2856 K
    "B": XYZ_Color(0.3484, 0.3516),  # approx 4874 K
    "C": XYZ_Color(0.3101, 0.3162),  # approx 6774 K
    "D55": XYZ_Color(0.3324, 0.3475),  # approx 5500 K
    "D65": XYZ_Color(0.3127, 0.3290),  # approx 6500 K
    "D75": XYZ_Color(0.2990, 0.3150),  # approx 7500 K
    # MacBeth Color Checker white patch
    #   Using this as white point will force MacBeth chart entry to equal machine RGB
    "MacBethWhite": XYZ_Color(0.30995, 0.31596, 0.37409),
}

@dataclass(frozen=True)
class RGB_ColorSpace:
    """
    RGB Colour space for performing transforms. This stores the
    chromacities in CIE1931 (x,y) values for the 3 RGB primaries
    and white point. Furthermore, a transform function is supplied
    to convert to display values.

    R: chromacity of the red channel as an XYZ_Color
    G: chromacity of the green channel as an XYZ_Color
    B: chromacity of the blue channel as an XYZ_Color
    W: chromacity of the white point as an XYZ_Color
    EOTF: Electro-Optical Transfer Function
    bit depth: integer bit depth (default 8)
    peak_luminance: Physical luminance in cd/m^2 (nits)
    """

    name: str
    R: XYZ_Color
    G: XYZ_Color
    B: XYZ_Color
    W: XYZ_Color = Whites['D65']
    EOTF: Callable = gamma_EOTF(gamma=2.2),
    bit_depth: int = 8
    peak_luminance: u.Quantity = 80 * u.cd/u.m**2

    @property
    def primaries(self):
        return self.R, self.G, self.B

    @functools.cached_property
    def xyz_from_rgb_matrix(self):
        """
        Determine intensities of each phosphor by solving:
        phosphor_matrix * intensity_vector = white_point
        """
        # See [Foley/Van Dam, p.587, eqn 13.27, 13.29] and [Hall, p. 239].
        phosphor_matrix = np.column_stack([x._values for x in self.primaries])
        I = np.linalg.solve(phosphor_matrix, self.W.normalize_Y1()._values)
        return phosphor_matrix * I[np.newaxis, :]

    @functools.cached_property
    def rgb_from_xyz_matrix(self):
        return np.linalg.inv(self.xyz_from_rgb_matrix)

    def in_gamut(self, XYZ_Color):
        """
        Check whether an XYZ_Color falls within the gamut of a colour space
        """
        rgb = self.rgb_from_xyz_matrix @ XYZ_Color._values
        return np.all(rgb >= 0)


# https://www.color.org/chardata/rgb/srgb.xalter
sRGB = RGB_ColorSpace(
    name="sRGB",
    R=XYZ_Color(0.640, 0.330),
    G=XYZ_Color(0.300, 0.600),
    B=XYZ_Color(0.150, 0.060),
    W=Whites["D65"],
    EOTF=sRGB_EOTF(),
    bit_depth = 8,
    peak_luminance = 80 * u.cd/u.m**2
)

# HDTV standard phosphors, from Poynton [Color FAQ] p. 9
# These are claimed to be similar to typical computer monitors
HDTV = RGB_ColorSpace(
    name="HDTV",
    R=XYZ_Color(0.640, 0.330),
    G=XYZ_Color(0.300, 0.600),
    B=XYZ_Color(0.150, 0.060),
    W=Whites["D65"],
    EOTF=gamma_EOTF(gamma=2.2),
)

# https://www.color.org/chardata/rgb/BT709.xalter
Rec709 = RGB_ColorSpace(
    name="Rec. 709",
    R=XYZ_Color(0.640, 0.330),
    G=XYZ_Color(0.300, 0.600),
    B=XYZ_Color(0.150, 0.060),
    W=Whites["D65"],
    EOTF=gamma_EOTF(gamma=2.4),
)

# SMPTE phosphors
# However, Hall [p. 188] notes that TV expects values calibrated for NTSC
# even though actual phosphors are as below.
# From Hall p. 118, and Kasson p. 400
SMPTE = RGB_ColorSpace(
    name="SMPTE",
    R=XYZ_Color(0.630, 0.340),
    G=XYZ_Color(0.310, 0.595),
    B=XYZ_Color(0.155, 0.070),
    W=Whites["D65"],
)

# NTSC phosphors [original standard for TV, but no longer used in TV sets]
# From Hall p. 119 and Foley/Van Dam p. 589
NTSC = RGB_ColorSpace(
    name="NTSC",
    R=XYZ_Color(0.670, 0.330),
    G=XYZ_Color(0.210, 0.710),
    B=XYZ_Color(0.140, 0.080),
    W=Whites["D65"],
)

# Typical short persistence phosphors from Foley/Van Dam p. 583
FoleyShort = RGB_ColorSpace(
    name="FoleyShort",
    R=XYZ_Color(0.61, 0.35),
    G=XYZ_Color(0.29, 0.59),
    B=XYZ_Color(0.15, 0.063),
)

# Typical long persistence phosphors from Foley/Van Dam p. 583
FoleyLong = RGB_ColorSpace(
    name="FoleyLong",
    R=XYZ_Color(0.62, 0.33),
    G=XYZ_Color(0.21, 0.685),
    B=XYZ_Color(0.15, 0.063),
)

# Typical TV phosphors from Judd/Wyszecki p. 239
Judd = RGB_ColorSpace(
    name="Judd",
    R=XYZ_Color(0.68, 0.32),  # Europium Yttrium Vanadate
    G=XYZ_Color(0.28, 0.60),  # Zinc Cadmium Sulfide
    B=XYZ_Color(0.15, 0.07),  # Zinc Sulfide
)


# https://www.itu.int/rec/R-REC-BT.2020-2-201510-I/en
# https://www.color.org/chardata/rgb/BT2020.xalter
Rec2020 = RGB_ColorSpace(
    name="Rec. 2020",
    R=XYZ_Color(0.708, 0.292),
    G=XYZ_Color(0.170, 0.797),
    B=XYZ_Color(0.131, 0.046),
    W=Whites['D65'],
    EOTF=gamma_EOTF(gamma=2.4)
)

# https://www.color.org/chardata/rgb/BT2100.xalter
Rec2100 = RGB_ColorSpace(
    name="Rec. 2100",
    R=XYZ_Color(0.708, 0.292),
    G=XYZ_Color(0.170, 0.797),
    B=XYZ_Color(0.131, 0.046),
    W=Whites['D65'],
    bit_depth = 10,
    EOTF=PQ_EOTF(),
    peak_luminance=10000*u.cd/u.m**2,
)

DCI_P3_SDR = RGB_ColorSpace(
    name="DCI-P3 D65 SDR",
    R=XYZ_Color(0.680, 0.320),
    G=XYZ_Color(0.265, 0.690),
    B=XYZ_Color(0.150, 0.060),
    W=Whites['D65'],
    EOTF=sRGB_EOTF(),
    bit_depth=10,
)

DCI_P3_HDR = RGB_ColorSpace(
    name="DCI-P3 D65 HDR",
    R=XYZ_Color(0.680, 0.320),
    G=XYZ_Color(0.265, 0.690),
    B=XYZ_Color(0.150, 0.060),
    W=Whites['D65'],
    EOTF=PQ_EOTF(),
    bit_depth=10,
    peak_luminance=10000*u.cd/u.m**2,
)
