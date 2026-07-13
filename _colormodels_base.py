"""
Basic classes for working with colours:

Color - This is the most basic colour class, storing a set of three values.

LinearColor - For colours where each component represents some linear value.
Colours of this type can be added together, subtracted, or scaled by a
constant.

Lpq_Color - Base class for LAB and LUV color spaces which both include a
lightness component.

XYZ_Color - The fundamental colour model from which all RGB colour spaces as
well as Luv and Lab are defined. It is defined here for dependency reasons.
"""

import numpy as np


class Color:
    """
    Tristimulus color space
    """

    __slots__ = ["_values"]

    def __init__(self, _1, _2, _3, dtype):
        self.values = _1, _2, _3

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

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, *self.values)

    def __rmul__(self, other):
        return self * other

    def __len__(self):
        if self.values.ndim == 1:
            return 1
        elif self.values.ndim == 2:
            return self.values.shape[1]
        else:
            raise ValueError("Length isn't defined for colour arrays with ndim > 1")

    def __getitem__(self, key):
        return type(self)(*[val[key] for val in self.values])
        
    @staticmethod
    def validate_values(_1, _2, _3):
        if not isinstance(_1, type(_2)) or not isinstance(_1, type(_3)):
            raise TypeError("Inputs must be the same type")
        if isinstance(_1, np.ndarray) and not (_1.shape == _2.shape == _3.shape):
            raise ValueError("Arrays must have the same shape")

    @property
    def _1(self):
        return self._values[0]

    @_1.setter
    def _1(self, value):
        self.validate_values(value, self._2, self._3)
        self._values[0] = value

    @property
    def _2(self):
        return self._values[1]

    @_2.setter
    def _2(self, value):
        self.validate_values(self._1, value, self._3)
        self._values[1] = value

    @property
    def _3(self):
        return self._values[2]

    @_3.setter
    def _3(self, value):
        self.validate_values(self._1, self._2, value)
        self._values[2] = value

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self.validate_values(*values)
        self._values = np.array(values)



class LinearColor(Color):
    """
    Template class for Linear colors which can be added together or multiplied by
    a scalar
    """

    def __init__(self, _1, _2, _3, dtype=float):
        super().__init__(_1, _2, _3, dtype=dtype)

    @property
    def _kwargs(self):
        return {}

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError
        new = [x+y for x, y in zip(self.values, other.values)]
        return self.__class__(*new, **self._kwargs)

    def __sub__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError
        new = [x-y for x, y in zip(self.values, other.values)]
        return self.__class__(*new, **self._kwargs)

    def __mul__(self, other):
        if not isinstance(other, (int, float, np.ndarray)):
            raise TypeError
        new = [x*other for x in self.values]
        return self.__class__(*new, **self._kwargs)

    def __truediv__(self, other):
        if not isinstance(other, (int, float, np.ndarray)):
            raise TypeError
        new = [x/other for x in self.values]
        return self.__class__(*new, **self._kwargs)


class Lpq_Color(Color):
    """
    Template class for LAB and LUB color spaces, which have a shared L property
    """

    def __init__(self, L, p, q, white_point):
        super().__init__(L, p, q, dtype=float)
        if not isinstance(white_point, XYZ_Color):
            raise TypeError("white_point should be an XYZ_Color")
        self.white_point = white_point

    L = property(fget=LinearColor._1.fget, fset=LinearColor._1.fset)


class XYZ_Color(LinearColor):
    def __init__(self, x, y, z=None):
        if z is None:
            z = 1 - x - y
        super().__init__(x, y, z)

    def __repr__(self):
        rpr = super().__repr__()
        if np.isclose(self.values.sum(), 1):
            parts = rpr.split(", ")
            rpr = ", ".join(parts[:2]) + ")"
        return rpr

    X = property(fget=LinearColor._1.fget, fset=LinearColor._1.fset)
    Y = property(fget=LinearColor._2.fget, fset=LinearColor._2.fset)
    Z = property(fget=LinearColor._3.fget, fset=LinearColor._3.fset)

    def normalize(self):
        return self / self.values.sum(axis=0)

    def normalize_Y1(self):
        return self / self.Y

    @property
    def x(self):
        return self.normalize().X

    @property
    def y(self):
        return self.normalize().Y

    @property
    def z(self):
        return self.normalize().Z

    @classmethod
    def from_xyY(cls, x, y, Y):
        """
        Given the 'little' x,y chromaticity, and the intensity Y, construct an
        XYZ color.
        """
        return cls(x, y) * (Y / y)
