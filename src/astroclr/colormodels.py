"""
Defines several color models, and conversions between them. The models are:

XYZ - CIE XYZ color space, based on the 1931 matching functions for a 2 degree
field of view. Spectra are converted to xyz color values by integrating with
the matching functions in ciexyz.py.

XYZ colors are often handled as absolute values, conventionally written
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
    Construct a linear RGB color from components.

IRGB_Color(ir, ig, ib, color_space=sRGB) -
    Construct a displayable "intensity" RGB color from components.

LUV_Color(L, u, v, white_point=sRGB.W) -
    Construct a Luv color from components.

LAB_Color(L, a, b, white_point=sRGB.W) -
    Construct a Lab color from components.


By design, the XYZ color space is ignorant of the other spaces. However, the
others (e.g. RGB), know about XYZ. So conversions from XYZ are of the form:
>>> RGB = RGB_Color.from_XYZ(XYZ)

But conversions back are of the simpler form
>>> XYZ = RGB.to_XYZ()

This is because the various RGB color spaces and white points are defined in
terms of XYZ colors. While sRGB is used by default, this allows flexibility in
working in other RGB color spaces, or with other white points. If we were using
a different color space, we would instead do something like
>>> RGB = RGB_Color.from_XYZ(XYZ, color_space=Rec2020)
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
    "UCS_Color",
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


def Lab_f(t: float) -> float:
    """
    Lab utility function.
    """
    if t > L_LUM["CUTOFF"]:
        return np.cbrt(t)
    # linear range
    return LAB_F["A"] * t + LAB_F["B"]


def Lab_f_inverse(F: float) -> float:
    """
    Inverse of Lab_f().
    """
    if F > (LAB_F["A"] * L_LUM["CUTOFF"] + LAB_F["B"]):
        return F * F * F
    # linear range
    return (F - LAB_F["B"]) / LAB_F["A"]


def L_luminance(y: float) -> float:
    """
    L coefficient for Luv and Lab models.
    """
    if y > L_LUM["CUTOFF"]:
        return L_LUM["A"] * np.cbrt(y) - L_LUM["B"]
    return L_LUM["C"] * y  # linear range


def L_luminance_inverse(L: float) -> float:
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
    channels = "RGB"

    def __init__(self, R, G, B, color_space=sRGB):
        super().__init__(R, G, B)
        if not isinstance(color_space, RGB_ColorSpace):
            raise TypeError("color_space should be an RGB_ColorSpace")
        self.color_space = color_space

    R = property(fget=LinearColor._1.fget, fset=LinearColor._1.fset)
    G = property(fget=LinearColor._2.fget, fset=LinearColor._2.fset)
    B = property(fget=LinearColor._3.fget, fset=LinearColor._3.fset)

    @property
    def r(self):
        return self.R / (self.R + self.G + self.B)

    @property
    def g(self):
        return self.G / (self.R + self.G + self.B)

    @property
    def b(self):
        return self.B / (self.R + self.G + self.B)

    @property
    def _kwargs(self):
        return {"color_space": self.color_space}

    def normalize_brightest(self):
        return self / self.values.max()

    @classmethod
    def from_XYZ(cls, XYZ, color_space=sRGB, brightest=False):
        """
        Create an RGB color from an XYZ color and color space (sRGB by
        default). 'brightest' keyword argument scales this so the brightest RGB
        channel is at 1.0.
        """
        transform_matrix = color_space.rgb_from_xyz_matrix
        rgb = np.einsum("ij,j...->i...", transform_matrix, XYZ.values)
        RGB = cls(*rgb, color_space=color_space)
        if brightest:
            RGB = RGB.normalize_brightest()
        return RGB

    @classmethod
    def from_grey(cls, grey):
        """
        Create an RGB color where all three colour channels are identical.
        Equivalent to RGB_Color(R=grey, G=grey, B=grey).
        """
        return cls(R=grey, G=grey, B=grey)

    def to_XYZ(self):
        """Convert an rgb color to xyz."""
        transform_matrix = self.color_space.xyz_from_rgb_matrix
        xyz = np.einsum("ij,j...->i...", transform_matrix, self.values)
        return XYZ_Color(*xyz)

    def clip_to_gamut(self, clip_method=ClipMethod.ADD_WHITE):
        """
        Clip an RGB colour to within its colour space, either by mixing with
        the white point, or clamp any negative channels to zero. Both methods
        move the colour to the edge of the RGB gamut.
        """
        if np.all((rgb := self.values.copy()) >= 0):
            return self
        if clip_method is ClipMethod.ADD_WHITE:
            rgb_min = np.clip(rgb.min(axis=0), -np.inf, 0.0)
            rgb_max = rgb.max(axis=0)
            rgb -= rgb_min
            #tiny offset fixes issues with black values
            rgb *= rgb_max / (1e-300 + rgb.max(axis=0))
        elif clip_method is ClipMethod.CLAMP:
            rgb = np.clip(rgb, 0, np.inf)
        else:
            raise ValueError(f"Invalid color clipping method {clip_method}")
        return RGB_Color(*rgb)

    def to_IRGB(self, clip_method=ClipMethod.ADD_WHITE):
        """
        Convert a (linear) rgb value (range 0.0--1.0) into a displayable
        integer irgb value (0--255 for bit depth of 8), clipping as necessary.
        """
        rgb = self.clip_to_gamut(clip_method).values
        rgb = np.clip(rgb, 0, 1)
        irgb_unit = self.color_space.OETF(rgb)
        irgb = np.round(self.color_space.max_int * irgb_unit).astype(int)
        return IRGB_Color(*irgb, self.color_space)


class IRGB_Color(LinearColor):
    """
    Color class for display RGB values. These have the same primaries and white
    point as RGB_Color, but the values are on a non-linear scale (passed through
    a transfer function and quantised (typically 0--255 for a bit depth of 8).
    """
    channels = "RGB"

    def __init__(self, R, G, B, color_space=sRGB):
        if not isinstance(color_space, RGB_ColorSpace):
            raise TypeError("color_space should be an RGB_ColorSpace")
        if not np.all(np.array([R, G, B]) < 2**color_space.bit_depth):
            raise ValueError("RGB integral values out of range")
        super().__init__(R, G, B, dtype=int)
        self.color_space = color_space

    R = property(fget=LinearColor._1.fget, fset=LinearColor._1.fset)
    G = property(fget=LinearColor._2.fget, fset=LinearColor._2.fset)
    B = property(fget=LinearColor._3.fget, fset=LinearColor._3.fset)


    @staticmethod
    def _to_hex(R, G, B, fs):
        hex_vals = [format(v, fs) for v in (R, G, B)]
        return "#{}{}{}".format(*hex_vals)

    def to_string(self):
        """
        Convert a displayable irgb color into a hex string. Will also work for
        arrays of colours.
        """
        n_hex = (self.color_space.bit_depth + 3) // 4
        fs = f"{n_hex:02}X"
        values = self.values

        if values.ndim == 1:
            return self._to_hex(*values, fs)
            
        f = np.vectorize(self._to_hex)
        return f(*self.values, fs)

    @classmethod
    def from_grey(cls, grey):
        """
        Create an RGB color where all three colour channels are identical.
        Equivalent to RGB_Color(R=grey, G=grey, B=grey).
        """
        return cls(R=grey, G=grey, B=grey)

    @classmethod
    def from_hex_string(cls, irgb_string, colorspace=sRGB):
        """
        Create a IRGB value (typical range 0 - 255) from a corresponding hex_string
        """
        if not isinstance(irgb_string, str):
            raise TypeError
        if len(irgb_string) != 7 or not irgb_string.startswith("#"):
            raise ValueError("Expecting 7 character string like #AB13D2")
        parts = irgb_string[1:3], irgb_string[3:5], irgb_string[5:7]
        ir, ig, ib = [int(part, 16) for part in parts]
        return cls(ir, ig, ib, colorspace)

    @classmethod
    def from_numpy_image(cls, img, colorspace=sRGB):
        """
        Create an IRGB_Color object from a numpy image. For an image with
        spatial dimensions NxM, img should have shape NxMx3, with data values
        in the range usually in the range 0--255 for 8-bit per channel data.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("img should be a numpy array")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("img should be have dimensions NxMx3")
        if not np.issubdtype(img.dtype, np.integer):
            raise ValueError("img should be integer type")
        if not np.all((img >= 0) & (img < 2**colorspace.bit_depth)):
            raise ValueError("img values should be in the range 0--255")
        R, G, B = np.moveaxis(img, 2, 0)
        return cls(R, G, B, colorspace)

    def to_RGB(self, to_luminance=False):
        """
        Convert a displayable (gamma corrected) IRGB value (typical range
        0--255) into a linear rgb value (range 0.0--1.0).
        """
        rgb = self.color_space.EOTF(self.values / self.color_space.max_int)
        RGB = RGB_Color(*rgb, color_space=self.color_space)

        if to_luminance:
            RGB *= self.color_space.peak_luminance.to_value(u.cd / u.m**2)
        return RGB

    @property
    def values_image(self):
        """
        In astroclr, colour arrays are of shape (3,...), i.e. (3,m,n) for
        images. For displaying images with plt.imshow we instead need a shape
        of (m,n,3) which is accessed with this property
        """
        if self.values.ndim != 3:
            raise ValueError("values_image is for working with 2D colour arrays")
        return np.moveaxis(self.values, 0, 2)


class LUV_Color(Lpq_Color):
    channels="Luv"

    def __init__(self, L, u, v, white_point=sRGB.W):
        super().__init__(L, u, v, white_point)

    u = property(fget=Lpq_Color._2.fget, fset=Lpq_Color._2.fset)
    v = property(fget=Lpq_Color._3.fget, fset=Lpq_Color._3.fset)

    @staticmethod
    def uv_primes(XYZ):
        """
        Converts an XYZ color to the (u', v') form of chromaticity associated with
        the LUV color space.
        """
        w_denom = XYZ.X + 15.0 * XYZ.Y + 3.0 * XYZ.Z
        if w_denom == 0.0:  # x=y=z=0 [i.e. black]
            return (0.0, 0.0)
        u_prime = 4.0 * XYZ.X / w_denom
        v_prime = 9.0 * XYZ.Y / w_denom
        return (float(u_prime), float(v_prime))

    @staticmethod
    def uv_primes_inverse(u_prime, v_prime, y):
        """
        Converts a luminance Y and (u', v') chromaticity to an XYZ colour
        """
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
        y_p = XYZ.Y / white_point.Y
        L = L_luminance(y_p)
        u_prime, v_prime = cls.uv_primes(XYZ)
        reference_u_prime, reference_v_prime = cls.uv_primes(white_point)
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
        reference_u_prime, reference_v_prime = LUV_Color.uv_primes(white_point)
        u_prime = reference_u_prime + (self.u / L13)
        v_prime = reference_v_prime + (self.v / L13)
        return self.uv_primes_inverse(u_prime, v_prime, y)


class LAB_Color(Lpq_Color):
    channels="Lab"
    
    def __init__(self, L, a, b, white_point=sRGB.W):
        super().__init__(L, a, b, white_point)

    a = property(fget=Lpq_Color._2.fget, fset=Lpq_Color._2.fset)
    b = property(fget=Lpq_Color._3.fget, fset=Lpq_Color._3.fset)

    @classmethod
    def from_XYZ(cls, XYZ, white_point=sRGB.W):
        """
        Create an LAB color from an XYZ color and white point (D65 by default).
        """
        x_p, y_p, z_p = xyz_p = XYZ.values / white_point.values
        f_x, f_y, f_z = map(Lab_f, xyz_p)

        L = L_luminance(y_p)
        a = 500.0 * (f_x - f_y)
        b = 200.0 * (f_y - f_z)
        return cls(L, a, b, white_point=white_point)

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
        xyz_p = np.array([x_p, y_p, z_p]) * self.white_point.values
        return XYZ_Color(*xyz_p)


class UCS_Color(LinearColor):
    """
    UCS 1960 Color space with components UVW.
    """
    channels="UVW"

    def __init__(self, u, v, w=None):
        if w is None:
            w = 1 - u - v
        if w < 0:
            w = 0
        super().__init__(u, v, w)

    def __repr__(self):
        rpr = super().__repr__()
        if np.isclose(self.values.sum(), 1):
            parts = rpr.split(", ")
            rpr = ", ".join(parts[:2]) + ")"
            rpr = rpr.replace("U=", "u=")
            rpr = rpr.replace("V=", "v=")
        return rpr

    def __format__(self, format_spec):
        fmt = super().__format__(format_spec)
        if np.isclose(self.values.sum(), 1):
            parts = fmt.split(", ")
            fmt = ", ".join(parts[:2]) + ")"
            fmt = fmt.replace('U=', 'u=')
            fmt = fmt.replace('V=', 'v=')
        return fmt

    U = property(fget=LinearColor._1.fget, fset=LinearColor._1.fset)
    V = property(fget=LinearColor._2.fget, fset=LinearColor._2.fset)
    W = property(fget=LinearColor._3.fget, fset=LinearColor._3.fset)

    def normalize(self):
        return self / self.values.sum()

    @property
    def u(self):
        return self.normalize().U

    @property
    def v(self):
        return self.normalize().V

    @classmethod
    def from_XYZ(cls, XYZ):
        U = 2 / 3 * XYZ.X
        V = XYZ.Y
        W = (-XYZ.X + 3 * XYZ.Y + XYZ.Z) / 2
        return cls(U, V, W)

    def to_XYZ(self):
        X = 3 / 2 * self.X
        Y = self.V
        Z = 3 / 2 * self.U - 3 * self.V + 2 * self.W
        return XYZ_Color(X, Y, Z)
