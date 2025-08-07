"""
colormodels.py - Conversions between color models

Defines several color models, and conversions between them. The models are:

xyz - CIE XYZ color space, based on the 1931 matching functions for a 2 degree
field of view. Spectra are converted to xyz color values by integrating with
the matching functions in ciexyz.py.

xyz colors are often handled as absolute values, conventionally written
with uppercase letters XYZ, or as scaled values (so that X+Y+Z = 1.0),
conventionally written with lowercase letters xyz.

This is the fundamental color model around which all others are based.

rgb - Colors expressed as red, green and blue values, in the nominal range 0.0
- 1.0. These are linear color values, meaning that doubling the number implies
a doubling of the light intensity. rgb color values may be out of range
(greater than 1.0, or negative), and do not account for gamma correction.
They should not be drawn directly.

irgb - Displayable color values expressed as red, green and blue values, in the
range 0 - 255. These have been adjusted for gamma correction, and have been
clipped into the displayable range 0 - 255. These color values can be drawn
directly.

Luv - A nearly perceptually uniform color space.

Lab - Another nearly perceptually uniform color space.


Color model classes:

XYZ_Color(x, y, z = None) -
    Construct an xyz color.  If z is omitted, set it so that x+y+z = 1.0.

RGB_Color(r, g, b, color_space=sRGB) -
    Construct a linear rgb color from components.

IRGB_Color(ir, ig, ib, color_space=sRGB) -
    Construct a displayable integer irgb color from components.

LUV_Color(L, u, v, white_point=sRGB.W) -
    Construct a Luv color from components.

LAB_Color(L, a, b, white_point=sRGB.W) -
    Construct a Lab color from components.


By design, the XYZ color space is ignorant of the other spaces. However, the
others, such as RGB, know about XYZ. So conversions from XYZ are of the form:
>>> RGB = RGB_Color.from_XYZ(XYZ)

But conversions back are of the simpler form
>>> XYZ = RGB.to_XYZ()

This is because the various RGB color spaces and white points are defined in
terms of XYZ colors. While sRGB is used by default, this allows flexibility in
working in other RGB color spaces, or with other white points. If we were using
a different color space, we would instead do something like
>>> RGB = RGB_Color.from_XYZ(XYZ, color_space=Rec2020)

Color Spaces:

sRGB - Chromaticity values for sRGB standard display monitors.
HDTV - Same as sRGB
SMPTE
NSTC
FoleyShort
FoleyLong
Judd


References:

Foley, van Dam, Feiner and Hughes. Computer Graphics: Principles and Practice, 2nd edition,
    Addison Wesley Systems Programming Series, 1990. ISBN 0-201-12110-7.

Roy Hall, Illumination and Color in Computer Generated Imagery. Monographs in Visual Communication,
    Springer-Verlag, New York, 1989. ISBN 0-387-96774-5.

Wyszecki and Stiles, Color Science: Concepts and Methods, Quantitative Data and Formulae, 2nd edition,
    John Wiley, 1982. Wiley Classics Library Edition 2000. ISBN 0-471-39918-3.

Judd and Wyszecki, Color in Business, Science and Industry, 1975.

Kasson and Plouffe, An Analysis of Selected Computer Interchange Color Spaces,
    ACM Transactions on Graphics, Vol. 11, No. 4, October 1992.

Charles Poynton - Frequently asked questions about Gamma and Color,
    posted to comp.graphics.algorithms, 25 Jan 1995.

sRGB - http://www.color.org/sRGB.xalter - (accessed 15 Sep 2008)
    A Standard Default Color Space for the Internet: sRGB,
    Michael Stokes (Hewlett-Packard), Matthew Anderson (Microsoft), Srinivasan Chandrasekar (Microsoft),
    Ricardo Motta (Hewlett-Packard), Version 1.10, November 5, 1996.

"""

from enum import StrEnum
import numpy as np
from astropy import units as u
from ._colormodels_base import LinearColor, XYZ_Color, Lpq_Color
from .RGB_color_spaces import RGB_ColorSpace, sRGB

__all__ = [
    "XYZ_Color",
    "RGB_Color",
    "IRGB_Color",
    "LUV_Color",
    "LAB_Color",
    "ClipMethod",
]


# Utility function for Lab
#     See [Kasson p.399] for details.
#     The linear range coefficient has more digits than in the paper,
#     this makes the function more continuous over the boundary.

L_LUM = {"A": 116, "B": 16, "C": (29 / 3) ** 3, "CUTOFF": (6 / 29) ** 3}
LAB_F = {"A": (29 / 6) ** 2 / 3, "B": 4 / 29}

# Some standard xyz/rgb conversion matricies, which assume particular phosphors.
# These are useful for testing.


class ClipMethod(StrEnum):
    CLAMP = "clamp"
    ADD_WHITE = "add white"


def Lab_f(t):
    """
    Lab utility function.
    """
    if t > L_LUM["CUTOFF"]:
        return np.cbrt(t)
    # linear range
    return LAB_F["A"] * t + LAB_F["B"]


def Lab_f_inverse(F):
    """
    Inverse of Lab_f().
    """
    if F > (LAB_F["A"] * L_LUM["CUTOFF"] + LAB_F["B"]):
        return F * F * F
    # linear range
    return (F - LAB_F["B"]) / LAB_F["A"]


def L_luminance(y):
    """
    L coefficient for Luv and Lab models.
    """
    if y > L_LUM["CUTOFF"]:
        return L_LUM["A"] * np.cbrt(y) - L_LUM["B"]
    return L_LUM["C"] * y  # linear range


def L_luminance_inverse(L):
    """Inverse of L_luminance()."""
    # Luminance function [of Y value of an XYZ color] used in Luv and Lab.
    # See [Kasson p.399] for details. The linear range coefficient L_LUM['C']
    # has more digits than in the paper, this makes the function more
    # continuous over the boundary.

    if L > 8:
        t = (L + L_LUM["B"]) / L_LUM["A"]
        y = t * t * t
        return y
    # linear range
    y = L / L_LUM["C"]
    return y


###############################################################
# Various RGB color spaces are defined in terms of XYZ_Color and
# a Colorspace or White Point.


class RGB_Color(LinearColor):
    def __init__(self, r, g, b, color_space=sRGB):
        super().__init__((r, g, b))
        if not isinstance(color_space, RGB_ColorSpace):
            raise TypeError("color_space should be an RGB_ColorSpace")
        self.color_space = color_space

    r = property(fget=LinearColor._1.fget, fset=LinearColor._1.fset)
    g = property(fget=LinearColor._2.fget, fset=LinearColor._2.fset)
    b = property(fget=LinearColor._3.fget, fset=LinearColor._3.fset)

    @property
    def _kwargs(self):
        return {'color_space': self.color_space} 

    def normalize_brightest(self):
        return self / self._values.max()

    @classmethod
    def from_XYZ(cls, XYZ, color_space=sRGB, brightest=False):
        """
        Create an RGB color from an XYZ color and color space (sRGB by
        default). 'brightest' keyword argument scales this so the brightest RGB
        channel is at 1.0.
        """
        rgb = color_space.rgb_from_xyz_matrix @ XYZ._values
        RGB = cls(*rgb, color_space=color_space)
        if brightest:
            RGB = RGB.normalize_brightest()
        return RGB

    def to_XYZ(self):
        """Convert an rgb color to xyz."""
        xyz = self.color_space.xyz_from_rgb_matrix @ self._values
        return XYZ_Color(*xyz)

    def clip_to_gamut(self, clip_method=ClipMethod.ADD_WHITE):
        """
        Clip an RGB colour to within its colour space, either by mixing with
        the white point, or clamp any negative channels to zero. Both methods
        move the colour to the edge of the RGB gamut.
        """
        if clip_method == ClipMethod.ADD_WHITE:
            rgb = self._values
            rgb_min = min(0.0, rgb.min())
            rgb_max = rgb.max()
            scaling = rgb_max / (rgb_max - rgb_min) if rgb_max > 0.0 else 1.0
            # add enough white to cancel this out, maintaining the maximum of rgb
            if rgb_min < 0.0:
                rgb = scaling * (rgb - rgb_min)
        elif clip_method == ClipMethod.CLAMP:
            rgb = np.clip(self._values, 0, np.inf)
        else:
            raise ValueError(f"Invalid color clipping method {clip_method}")
        return RGB_Color(*rgb)

    def to_IRGB(self, clip_method=ClipMethod.ADD_WHITE):
        """
        Convert a (linear) rgb value (range 0.0 - 1.0) into a 0-255
        displayable integer irgb value (range 0 - 255), clipping as
        necessary
        """
        rgb = self.clip_to_gamut(clip_method)._values
        # clip intensity if needed (rgb values > 1.0) by scaling
        rgb_max = max(rgb)
        # we actually don't overflow until 255.0 * intensity > 255.5, so
        # instead of 1.0 use ...
        intensity_cutoff = 1.0 + (0.5 / 255.0)
        if rgb_max > intensity_cutoff:
            # must scale intensity, so max value is intensity_cutoff
            scaling = intensity_cutoff / rgb_max
            rgb *= scaling

        # apply inverse EOTF
        irgb_unit = [self.color_space.EOTF.inv(rgb[i]) for i in range(3)]

        # scale to 0--255
        irgb = np.round(255 * np.array(irgb_unit))
        # ensure that values are in the range 0--255
        ir, ig, ib = np.clip(irgb, 0, 255).astype(int)
        return IRGB_Color(ir, ig, ib, self.color_space)


class IRGB_Color(LinearColor):
    def __init__(self, r, g, b, color_space=sRGB):
        if not isinstance(color_space, RGB_ColorSpace):
            raise TypeError("color_space should be an RGB_ColorSpace")
        if not all(x in range(2**color_space.bit_depth) for x in (r, g, b)):
            raise ValueError("RGB integral values out of range")
        super().__init__((r, g, b), dtype=int)
        self.color_space = color_space

    r = property(fget=LinearColor._1.fget, fset=LinearColor._1.fset)
    g = property(fget=LinearColor._2.fget, fset=LinearColor._2.fset)
    b = property(fget=LinearColor._3.fget, fset=LinearColor._3.fset)

    def to_string(self):
        """
        Convert a displayable 8-bit irgb color (0-255) into a hex string.
        """
        if self.color_space.bit_depth != 8:
            raise ValueError("Can only make hexcodes for bit depths of 8")
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}"

    @classmethod
    def from_hex_string(cls, irgb_string, colorspace=sRGB):
        """
        Create a IRGB value (range 0 - 255) from a corresponding hex_string
        """
        if not isinstance(irgb_string, str):
            raise TypeError
        if len(irgb_string) != 7 or not irgb_string.startswith("#"):
            raise ValueError("Expecting 7 character string like #AB13D2")
        parts = irgb_string[1:3], irgb_string[3:5], irgb_string[5:7]
        ir, ig, ib = [int(part, 16) for part in parts]
        return cls(ir, ig, ib, colorspace)

    def to_RGB(self, to_luminance=False):
        """
        Convert a displayable (gamma corrected) IRGB value (range 0 - 255)
        into a linear rgb value (range 0.0 - 1.0).
        """
        max_val = 2**self.color_space.bit_depth
        rgb = [self.color_space.EOTF(val / max_val) for val in self._values]
        RGB = RGB_Color(*rgb, color_space=self.color_space)

        if to_luminance:
            RGB *= self.color_space.peak_luminance.to_value(u.cd/u.m**2)
        return RGB


class LUV_Color(Lpq_Color):
    def __init__(self, L, u, v, white_point=sRGB.W):
        super().__init__(L, u, v, white_point)

    u = property(fget=Lpq_Color._2.fget, fset=Lpq_Color._2.fset)
    v = property(fget=Lpq_Color._3.fget, fset=Lpq_Color._3.fset)

    @staticmethod
    def uv_primes(XYZ):
        """
        Converts an XYZ color to the (u', v') form of chromacity associated with
        the LUV color space.
        """
        w_denom = XYZ.x + 15.0 * XYZ.y + 3.0 * XYZ.z
        if w_denom == 0.0:  # x=y=z=0 [i.e. black]
            return (0.0, 0.0)
        u_prime = 4.0 * XYZ.x / w_denom
        v_prime = 9.0 * XYZ.y / w_denom
        return (u_prime, v_prime)

    @staticmethod
    def uv_primes_inverse(u_prime, v_prime, y):
        """
        Converts a luminance Y and (u', v') chromacity to an XYZ colour"""
        if v_prime == 0.0:
            return XYZ_Color(0.0, 0.0, 0.0)
        w_denom = (9.0 * y) / v_prime
        x = 0.25 * u_prime * w_denom
        z = (w_denom - x - 15.0 * y) / 3.0
        return XYZ_Color(x, y, z)

    @classmethod
    def from_XYZ(cls, XYZ, white_point=sRGB.W):
        """
        Create an LUV color from an XYZ color and white point (D65 by default).
        """
        y_p = XYZ.y / white_point.y
        L = L_luminance(y_p)
        u_prime, v_prime = XYZ.to_uv_primes()
        reference_u_prime, reference_v_prime = white_point.to_uv_primes()
        u = 13.0 * L * (u_prime - reference_u_prime)
        v = 13.0 * L * (v_prime - reference_v_prime)
        return LUV_Color(L, u, v, white_point=white_point)

    def to_XYZ(self):
        """
        Converts an LUV color to an XYZ color.
        """
        if self.L == 0.0:
            return XYZ_Color(0.0, 0.0, 0.0)
        y = L_luminance_inverse(self.L)
        L13 = 13.0 * self.L

        white_point = self.color_space.W
        reference_u_prime, reference_v_prime = white_point.to_uv_primes()
        u_prime = reference_u_prime + (self.u / L13)
        v_prime = reference_v_prime + (self.v / L13)
        return self.uv_primes_inverse(u_prime, v_prime, y)


class LAB_Color(Lpq_Color):
    def __init__(self, L, a, b, white_point=sRGB):
        super().__init__(L, a, b, white_point)

    a = property(fget=Lpq_Color._2.fget, fset=Lpq_Color._2.fset)
    b = property(fget=Lpq_Color._3.fget, fset=Lpq_Color._3.fset)

    @classmethod
    def from_XYZ(cls, XYZ, white_point=sRGB.W):
        """
        Create an LAB color from an XYZ color and white point (D65 by default).
        """
        x_p, y_p, z_p = xyz_p = XYZ._values / white_point._values
        f_x, f_y, f_z = map(Lab_f, xyz_p)

        L = L_luminance(y_p)
        a = 500.0 * (f_x - f_y)
        b = 200.0 * (f_y - f_z)
        return cls(L, a, b)

    def to_XYZ(self):
        """
        Converts an LUV color to an XYZ color.
        """
        y_p = L_luminance_inverse(self.L)
        f_y = Lab_f(y_p)
        f_x = f_y + (self.a / 500.0)
        f_z = f_y - (self.b / 200.0)
        x_p = Lab_f_inverse(f_x)
        z_p = Lab_f_inverse(f_z)
        xyz_p = XYZ_Color(x_p, y_p, z_p)
        return xyz_p * self.white_point


class ColorList(list):
    """
    Class for a list of colours (all of the same type). This allows attributes
    and methods to be mapped across elements.
    """
    color_types = {
        XYZ_Color,
        RGB_Color,
        IRGB_Color,
        LUV_Color,
        LAB_Color,
    }

    def __init__(self, values):
        if (first_type := type(values[0])) not in ColorList.color_types:
            raise TypeError("Not one of the supported colors")
        if not all(isinstance(element, first_type) for element in values):
            raise TypeError("All Colors must be of the same type")
        super().__init__(values) 

    def __getattr__(self, attr):
        if not callable(getattr(self[0], attr)):
            return [getattr(item, attr) for item in self]
        # Return a function that maps this method over all items
        def method(*args, **kwargs):
            ret = [getattr(item, attr)(*args, **kwargs) for item in self]
            if type(ret[0]) in ColorList.color_types:
                ret = ColorList(ret)
            return ret
        return method
