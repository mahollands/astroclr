"""
Module containing various RGB colour spaces. Each RGB colour space is defined
by a set of primary (R, G, B) and white point chromaticities, as well as a
transfer function for conversion between linear and display RGB.
"""

import functools
from typing import Callable
from dataclasses import dataclass
import numpy as np
from astropy import units as u
from ._colormodels_core import XYZ_Color
from .transfer_functions import gamma_EOTF, piecewise_EOTF, PQ_EOTF


__all__ = [
    "sRGB",
    "Rec709",
    "Rec2020",
    "Rec2100",
    "DCI_P3_D65",
    "DCI_P3_DCI",
    "Display_P3",
]


Whites = {
    "A": XYZ_Color(0.4476, 0.4074),  # approx 2856 K
    "B": XYZ_Color(0.3484, 0.3516),  # approx 4874 K
    "C": XYZ_Color(0.3101, 0.3162),  # approx 6774 K
    "D55": XYZ_Color(0.3324, 0.3475),  # approx 5500 K
    "D65": XYZ_Color(0.3127, 0.3290),  # approx 6500 K
    "D75": XYZ_Color(0.2990, 0.3150),  # approx 7500 K
}


@dataclass(frozen=True)
class RGB_ColorSpace:
    """
    RGB Colour space for performing transforms. This stores the
    chromaticities in CIE1931 (x,y) values for the 3 RGB primaries
    and white point. Furthermore, a transform function is supplied
    to convert to display values.

    R: chromaticity of the red channel as an XYZ_Color
    G: chromaticity of the green channel as an XYZ_Color
    B: chromaticity of the blue channel as an XYZ_Color
    W: chromaticity of the white point as an XYZ_Color
    EOTF: Electro-Optical Transfer Function
    bit depth: integer bit depth (default 8)
    peak_luminance: Physical luminance in cd/m^2 (nits)
    """

    name: str
    R: XYZ_Color
    G: XYZ_Color
    B: XYZ_Color
    W: XYZ_Color
    EOTF: Callable
    bit_depth: int = 8
    peak_luminance: u.Quantity = 80 * u.cd / u.m**2

    @property
    def primaries(self) -> tuple:
        return self.R, self.G, self.B

    @functools.cached_property
    def xyz_from_rgb_matrix(self) -> np.ndarray:
        """
        Determine the transformation from XYZ to RGB. This requires the
        chromaticities of the RGB primaries and white point, and the constraint
        (R,G,B)=(1,1,1) corresponds to Y=1.
        """
        phosphor_matrix = np.column_stack([p.values for p in self.primaries])
        full_white = self.W.normalize_Y1().values
        phosphor_intensities = np.linalg.solve(phosphor_matrix, full_white)
        return phosphor_matrix * phosphor_intensities[np.newaxis, :]

    @functools.cached_property
    def rgb_from_xyz_matrix(self) -> np.ndarray:
        """
        Determine the transformation from XYZ to RGB. This requires the
        chromaticities of the RGB primaries and white point and the constraint
        (R,G,B)=(1,1,1) corresponds to Y=1.
        """
        return np.linalg.inv(self.xyz_from_rgb_matrix)

    def in_gamut(self, XYZ: XYZ_Color) -> bool:
        """
        Check whether an XYZ_Color falls within the gamut of a colour space
        """
        rgb = self.rgb_from_xyz_matrix @ XYZ.values
        return np.all(rgb >= 0, axis=0)
    
    @functools.cached_property
    def max_int(self):
        """
        The maximium integer for a given bit depth
        """
        return 2**self.bit_depth - 1
    
    @property
    def OETF(self):
        return self.EOTF.inv


# https://registry.color.org/rgb-registry/srgb
sRGB = RGB_ColorSpace(
    name="sRGB",
    R=XYZ_Color(0.640, 0.330),
    G=XYZ_Color(0.300, 0.600),
    B=XYZ_Color(0.150, 0.060),
    W=Whites["D65"],
    EOTF=piecewise_EOTF(break_point=0.0031308, gamma=2.4, slope=12.92, offset=0.055),
    bit_depth=8,
    peak_luminance=80 * u.cd / u.m**2,
)

# https://registry.color.org/rgb-registry/bt709
Rec709 = RGB_ColorSpace(
    name="Rec. 709",
    R=XYZ_Color(0.640, 0.330),
    G=XYZ_Color(0.300, 0.600),
    B=XYZ_Color(0.150, 0.060),
    W=Whites["D65"],
    EOTF=piecewise_EOTF(break_point=0.018, gamma=1/0.45, slope=4.5, offset=0.099),
    peak_luminance=100 *u.cd/u.m**2
)

# https://www.itu.int/rec/R-REC-BT.2020-2-201510-I/en
# https://registry.color.org/rgb-registry/bt2020
Rec2020 = RGB_ColorSpace(
    name="Rec. 2020",
    R=XYZ_Color(0.708, 0.292),
    G=XYZ_Color(0.170, 0.797),
    B=XYZ_Color(0.131, 0.046),
    W=Whites["D65"],
    EOTF=gamma_EOTF(gamma=2.4),
    peak_luminance= 100 *u.cd / u.m**2
)

# https://registry.color.org/rgb-registry/bt2100
Rec2100 = RGB_ColorSpace(
    name="Rec. 2100",
    R=XYZ_Color(0.708, 0.292),
    G=XYZ_Color(0.170, 0.797),
    B=XYZ_Color(0.131, 0.046),
    W=Whites["D65"],
    bit_depth=10,
    EOTF=PQ_EOTF(),
    peak_luminance=10000 * u.cd / u.m**2,
)

# https://registry.color.org/rgb-registry/dcip3
DCI_P3_D65 = RGB_ColorSpace(
    name="DCI-P3 D65",
    R=XYZ_Color(0.680, 0.320),
    G=XYZ_Color(0.265, 0.690),
    B=XYZ_Color(0.150, 0.060),
    W=Whites["D65"],
    EOTF=gamma_EOTF(gamma=2.6),
    peak_luminance= 48 *u.cd / u.m**2
)

# https://registry.color.org/rgb-registry/dcip3
DCI_P3_DCI = RGB_ColorSpace(
    name="DCI-P3 DCI",
    R=XYZ_Color(0.680, 0.320),
    G=XYZ_Color(0.265, 0.690),
    B=XYZ_Color(0.150, 0.060),
    W=XYZ_Color(0.314, 0.351),
    EOTF=gamma_EOTF(gamma=2.6),
    peak_luminance= 48 *u.cd / u.m**2
)

# https://registry.color.org/rgb-registry/displayp3
Display_P3 = RGB_ColorSpace(
    name="Display-P3",
    R=XYZ_Color(0.680, 0.320),
    G=XYZ_Color(0.265, 0.690),
    B=XYZ_Color(0.150, 0.060),
    W=Whites["D65"],
    EOTF=piecewise_EOTF(break_point=0.0031308, gamma=2.4, slope=12.92, offset=0.055),
    peak_luminance= 80 *u.cd / u.m**2
)
