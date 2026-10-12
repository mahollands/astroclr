"""
Defines several colour models, and conversions between them. The models are:

XYZ_Color - CIE XYZ colour space, based on the 1931 colour matching functions
for a 2 degree field of view. Spectra are converted to XYZ colour values by
integrating with the matching functions within the astroclr.spectral module.
This is the fundamental colour model around which all others are based.

RGB_Color - Linear light representation of RGB colours expressed as Red, Green
and Blue values in the range 0-1. By default the sRGB colour space is used.

IRGB_Color - Displayable RGB colour values in the range 0-255 (for sRGB). These
have been converted from the linear RGB_Color space to the display scale using
an Opto-Electrical transfer function, and can be drawn to the display directly.

Luv_Color, Lab_Color, UCS_Color - A set of nearly perceptually uniform colour
spaces. See individual documentation.


By design, the XYZ colour space is ignorant of the other spaces. However, the
others (e.g. linear RGB), know about XYZ. So conversions from XYZ are of the
form:
>>> RGB = RGB_Color.from_XYZ(XYZ)

But conversions back are of the simpler form
>>> XYZ = RGB.to_XYZ()

This is because the various RGB colour spaces and white points are defined in
terms of XYZ colours. While sRGB is used by default, this allows flexibility in
working in other RGB colour spaces, or with other white points. If we were
using a different colour space, we would instead do something like
>>> RGB = RGB_Color.from_XYZ(XYZ, color_space=Rec2020)
"""

from enum import StrEnum
import numpy as np
from astropy import units as u
from ._colormodels_core import Linear_Color, Lpq_Color, XYZ_Color
from .rgb_color_spaces import RGB_ColorSpace, sRGB

__all__ = [
    "ClipMethod",
    "XYZ_Color",
    "RGB_Color",
    "IRGB_Color",
    "UCS_Color",
    "LUV_Color",
    "LAB_Color",
]


class ClipMethod(StrEnum):
    CLAMP = "clamp"
    ADD_WHITE = "add white"


###############################################################
# Various RGB color spaces are defined in terms of XYZ_Color and
# a Colorspace or White Point.


class RGB_Color(Linear_Color):
    channels = "RGB"

    def __init__(self, R, G, B, color_space=sRGB):
        super().__init__(R, G, B)
        if not isinstance(color_space, RGB_ColorSpace):
            raise TypeError("color_space should be an RGB_ColorSpace")
        self.color_space = color_space

    R = property(fget=Linear_Color._1.fget, fset=Linear_Color._1.fset)
    G = property(fget=Linear_Color._2.fget, fset=Linear_Color._2.fset)
    B = property(fget=Linear_Color._3.fget, fset=Linear_Color._3.fset)

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

    def in_gamut(self):
        """
        Check if an RGB colour (or array of colours) falls within the RGB
        gamut. Specifically this means that none of the R, G, B values are
        negative.
        """
        return np.all(self.values >= 0, axis=0)

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


class IRGB_Color(Linear_Color):
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
        super().__init__(R, G, B, dtype=np.int64)
        self.color_space = color_space

    R = property(fget=Linear_Color._1.fget, fset=Linear_Color._1.fset)
    G = property(fget=Linear_Color._2.fget, fset=Linear_Color._2.fset)
    B = property(fget=Linear_Color._3.fget, fset=Linear_Color._3.fset)


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
        if self.values.ndim == 2:
            img = self.values.T
            return img.reshape((1, len(self), 3))
        elif self.values.ndim == 3:
            return np.moveaxis(self.values, 0, 2)
        else:
            raise ValueError("values_image is for working with 2D colour arrays")


class LUV_Color(Lpq_Color):
    """
    CIE LUV 1976 L* u* v* colour space.
    """
    channels="Luv"

    def __init__(self, L, u, v, white_point=sRGB.W):
        super().__init__(L, u, v, white_point)

    u = property(fget=Lpq_Color._2.fget, fset=Lpq_Color._2.fset)
    v = property(fget=Lpq_Color._3.fget, fset=Lpq_Color._3.fset)

    @property
    def uv_prime(self):
        white_point = self.white_point
        u_prime_wp, v_prime_wp = LUV_Color.uv_primes_from_XYZ(white_point)
        L13 = 13.0 * self.L
        u_prime = u_prime_wp + (self.u / L13)
        v_prime = v_prime_wp + (self.v / L13)
        return u_prime, v_prime
        

    @staticmethod
    def uv_primes_from_XYZ(XYZ):
        """
        Converts an XYZ color to the (u', v') form of chromaticity associated with
        the LUV color space.
        """
        xyz_denom = XYZ.X + 15.0 * XYZ.Y + 3.0 * XYZ.Z
        u_prime = 4.0 * XYZ.X / xyz_denom
        v_prime = 9.0 * XYZ.Y / xyz_denom
        return u_prime, v_prime

    @staticmethod
    def XYZ_from_uv_primes_Y(u_prime, v_prime, Y):
        """
        Converts a luminance Y and (u', v') chromaticity to an XYZ colour
        """
        uv_denom = 6 * u_prime - 16 * v_prime + 12
        x = 9 * u_prime / uv_denom
        y = 4 * v_prime / uv_denom
        return XYZ_Color.from_xyY(x, y, Y)

    @classmethod
    def from_XYZ(cls, XYZ, white_point=sRGB.W):
        """
        Create an LUV color from an XYZ color and white point (D65 by default).
        LUV colour space assumes luminance on scale 0--1.
        """
        wp = white_point.normalize_Y1()
        L = 116*cls.f(XYZ.Y / wp.Y)-16
        u_prime, v_prime = cls.uv_primes_from_XYZ(XYZ)
        u_prime_wp, v_prime_wp = cls.uv_primes_from_XYZ(white_point)
        u = 13.0 * L * (u_prime - u_prime_wp)
        v = 13.0 * L * (v_prime - v_prime_wp)
        return cls(L, u, v, white_point=white_point)

    def to_XYZ(self):
        """
        Converts an LUV color to an XYZ color.
        """
        f_y = (self.L+16)/116
        Y = self.f_inverse(f_y)
        u_prime, v_prime = self.uv_prime
        return self.XYZ_from_uv_primes_Y(u_prime, v_prime, Y)


class LAB_Color(Lpq_Color):
    """
    CIE LAB 1976 L* a* b* colour space.
    """
    channels="Lab"
    
    def __init__(self, L, a, b, white_point=sRGB.W):
        super().__init__(L, a, b, white_point)

    a = property(fget=Lpq_Color._2.fget, fset=Lpq_Color._2.fset)
    b = property(fget=Lpq_Color._3.fget, fset=Lpq_Color._3.fset)

    @classmethod
    def from_XYZ(cls, XYZ, white_point=sRGB.W):
        """
        Create an LAB color from an XYZ color and white point (D65 by default).
        LUV colour space assumes luminance on scale 0--100.
        """
        wp = 100 * white_point.normalize_Y1()
        f_x = cls.f(XYZ.X/wp.X)
        f_y = cls.f(XYZ.Y/wp.Y)
        f_z = cls.f(XYZ.Z/wp.Z)

        L = 116 * f_y - 16
        a = 500 * (f_x - f_y)
        b = 200 * (f_y - f_z)
        return cls(L, a, b, white_point=white_point)

    def to_XYZ(self):
        """
        Converts an LUV color to an XYZ color.
        """
        f_y = (self.L+16)/116
        f_x = f_y + self.a / 500
        f_z = f_y - self.b / 200
        x_p, y_p, z_p = map(self.f_inverse, (f_x, f_y, f_z))
        wp = 100 * self.white_point.normalize_Y1()
        return XYZ_Color(x_p*wp.X, y_p*wp.Y, z_p*wp.Z)


class UCS_Color(Linear_Color):
    """
    UCS 1960 Colour space with components UVW.
    """
    channels="UVW"

    def __init__(self, u, v, w=None):
        if w is None:
            w = 1 - u - v
        super().__init__(u, v, w)

    def __format__(self, format_spec):
        fmt = super().__format__(format_spec)
        if np.isclose(self.values.sum(), 1):
            parts = fmt.split(", ")
            fmt = ", ".join(parts[:2]) + ")"
            fmt = fmt.replace('U=', 'u=')
            fmt = fmt.replace('V=', 'v=')
        return fmt

    U = property(fget=Linear_Color._1.fget, fset=Linear_Color._1.fset)
    V = property(fget=Linear_Color._2.fget, fset=Linear_Color._2.fset)
    W = property(fget=Linear_Color._3.fget, fset=Linear_Color._3.fset)

    def normalize(self):
        return self / self.values.sum(axis=0)

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
        X = 3 / 2 * self.U
        Y = self.V
        Z = 3 / 2 * self.U - 3 * self.V + 2 * self.W
        return XYZ_Color(X, Y, Z)
