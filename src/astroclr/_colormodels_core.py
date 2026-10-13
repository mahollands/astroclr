"""
Basic classes for working with colours:

Color - This is the base class for all colour types, storing a set of three
values.

Linear_Color - For colours where each component represents some linear value.
Linear colours of the same type can be added together, subtracted, or scaled by
a constant.

Lpq_Color - Base class for LAB and LUV color spaces which both include a
lightness component, and depend on a fixed white point.

XYZ_Color - The fundamental colour model from which all other colour models and
RGB colour spaces are defined. XYZ_Color is defined here as RGB colour spaces
like sRGB are defined in terms of XYZ chromaticities, and RGB colour objects
are defined in terms of RGB colour spaces.
"""

import numpy as np


class Color:
    """
    Tristimulus color space
    """

    __slots__ = ["_values"]
    channels = "_1", "_2", "_3"

    def __init__(self, _1, _2, _3, dtype):
        self.values = dtype(_1), dtype(_2), dtype(_3)

    @classmethod
    def from_list(cls, values):
        """
        Convert list of colours to a color with array attributes
        """
        if not isinstance(values, list):
            raise TypeError("Values must be a list of colours of this type")
        for value in values:
            if not isinstance(value, cls):
                raise TypeError("Values must be a list of colours of this type")
            if isinstance(value._1, np.ndarray):
                raise TypeError("Colours must be scalars")
        values = np.array([value.values for value in values]).T
        return cls(*values)

    def __format__(self, format_spec):
        values_formatted = (
            f"{c}={format(v, format_spec)}" for c, v in zip(self.channels, self.values)
        )
        return "{}({}, {}, {})".format(self.__class__.__name__, *values_formatted)

    def __repr__(self):
        return format(self, "")

    def __rmul__(self, other):
        return self * other

    def __len__(self):
        """
        Length of the colour array if a scalar or vector of colours.
        """
        if self.values.ndim == 1:
            return 1
        elif self.values.ndim == 2:
            return self.values.shape[1]
        else:
            raise ValueError("Length isn't defined for colour arrays with ndim > 1")

    def __getitem__(self, key):
        return type(self)(*[val[key] for val in self.values])

    @staticmethod
    def _validate_values(_1, _2, _3):
        if not isinstance(_1, type(_2)) or not isinstance(_1, type(_3)):
            raise TypeError("Inputs must be the same type")
        if isinstance(_1, np.ndarray) and not (_1.shape == _2.shape == _3.shape):
            raise ValueError("Arrays must have the same shape")

    @property
    def _1(self):
        return self._values[0]

    @_1.setter
    def _1(self, value):
        self._validate_values(value, self._2, self._3)
        self._values[0] = value

    @property
    def _2(self):
        return self._values[1]

    @_2.setter
    def _2(self, value):
        self._validate_values(self._1, value, self._3)
        self._values[1] = value

    @property
    def _3(self):
        return self._values[2]

    @_3.setter
    def _3(self, value):
        self._validate_values(self._1, self._2, value)
        self._values[2] = value

    @property
    def values(self):
        """
        An array of all colour values. For scalar colours this is a length 3
        array. For a length n vector of colours, this is a 3xn array. For a nxm
        image, a 3xnxm array, etc.
        """
        return self._values

    @values.setter
    def values(self, values):
        self._validate_values(*values)
        self._values = np.array(values)


class Linear_Color(Color):
    """
    Template class for Linear colors which can be added together or multiplied by
    a scalar
    """

    def __init__(self, _1, _2, _3, dtype=np.float64):
        super().__init__(_1, _2, _3, dtype=dtype)

    @property
    def _kwargs(self):
        return {}

    def __add__(self, other):
        """
        Add two colours of the same type.
        """
        if not isinstance(other, type(self)):
            raise TypeError
        new = [x + y for x, y in zip(self.values, other.values)]
        return self.__class__(*new, **self._kwargs)

    def __sub__(self, other):
        """
        Subtract two colours of the same type.
        """
        if not isinstance(other, type(self)):
            raise TypeError
        new = [x - y for x, y in zip(self.values, other.values)]
        return self.__class__(*new, **self._kwargs)

    def __mul__(self, other):
        """
        Multiply colours by a constant or array of constants.
        """
        if not isinstance(other, (int, float, np.ndarray)):
            raise TypeError
        new = [x * other for x in self.values]
        return self.__class__(*new, **self._kwargs)

    def __truediv__(self, other):
        """
        Divide colours by a constant or array of constants.
        """
        if not isinstance(other, (int, float, np.ndarray)):
            raise TypeError
        new = [x / other for x in self.values]
        return self.__class__(*new, **self._kwargs)


class Lpq_Color(Color):
    """
    Template class for LAB and LUB color spaces, which have a shared L property
    """
    _delta = (6 / 29)
    _delta2 = _delta*_delta
    _delta3 = _delta*_delta2

    def __init__(self, L, p, q, white_point):
        super().__init__(L, p, q, dtype=np.float64)
        if not isinstance(white_point, XYZ_Color):
            raise TypeError("white_point should be an XYZ_Color")
        self.white_point = white_point

    L = property(fget=Linear_Color._1.fget, fset=Linear_Color._1.fset)

    @classmethod
    def f(cls, t):
        """
        Lab/Luv f(t) utility function.
        """
        linear = (t + 2*cls._delta3)/(3*cls._delta2)
        cbrt = np.cbrt(t)
        return np.where(t > cls._delta3, cbrt, linear)

    @classmethod
    def f_inverse(cls, F):
        """
        Inverse of Lab/Luv f(t).
        """
        linear = (3*F - 2*cls._delta) * cls._delta2
        cubic = F * F * F
        return np.where(F > cls._delta, cubic, linear)


class XYZ_Color(Linear_Color):
    """
    The fundamental color model from which all others (such as RGB) are
    defined. This typically refers to the 1931 XYZ space though is flexible
    enough to accommodate more recent XYZ models.
    """

    channels = "XYZ"

    def __init__(self, x, y, z=None):
        if z is None:
            z = 1 - x - y
        super().__init__(x, y, z)

    def __format__(self, format_spec):
        fmt = super().__format__(format_spec)
        if np.isclose(self.values.sum(), 1):
            parts = fmt.split(", ")
            fmt = ", ".join(parts[:2]) + ")"
            fmt = fmt.replace("X=", "x=")
            fmt = fmt.replace("Y=", "y=")
        return fmt

    X = property(fget=Linear_Color._1.fget, fset=Linear_Color._1.fset)
    Y = property(fget=Linear_Color._2.fget, fset=Linear_Color._2.fset)
    Z = property(fget=Linear_Color._3.fget, fset=Linear_Color._3.fset)

    def normalize(self):
        """
        Normalize so that each the tristimulus components add to 1, i.e.
        the X, Y, Z values become equivalent to the chromaticity.
        """
        return self / self.values.sum(axis=0)

    def normalize_Y1(self):
        """
        Normalize colour to a luminance of Y=1.
        """
        return self / self.Y

    @property
    def x(self):
        """
        The x component of the chromaticity.
        """
        return self.normalize().X

    @property
    def y(self):
        """
        The y component of the chromaticity.
        """
        return self.normalize().Y

    @property
    def z(self):
        """
        The z component of the chromaticity.
        """
        return self.normalize().Z

    @classmethod
    def from_xyY(cls, x, y, Y):
        """
        Given the 'little' x,y chromaticity, and the intensity Y, construct an
        XYZ color.
        """
        return cls(x, y) * (Y / y)
