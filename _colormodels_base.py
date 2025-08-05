import numpy as np


class Color:
    """
    Tristimulus color space
    """

    def __init__(self, values, dtype):
        if len(values) != 3:
            raise ValueError
        self._values = np.array(values, dtype=dtype)

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, *self._values)

    def __rmul__(self, other):
        return self * other

    @property
    def _1(self):
        return self._values[0]
    @_1.setter
    def _1(self, value):
        self._values[0] = value

    @property
    def _2(self):
        return self._values[1]
    @_2.setter
    def _2(self, value):
        self._values[1] = value

    @property
    def _3(self):
        return self._values[2]
    @_3.setter
    def _3(self, value):
        self._values[2] = value

class LinearColor(Color):
    """
    Template class for Linear colors which can be added together or multiplied by
    a scalar
    """

    def __init__(self, values, dtype=float):
        if np.isnan(values).any():
            values = np.zeros(3) 
        super().__init__(values, dtype=dtype)

    @property
    def _kwargs(self):
        return {}

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError
        new = self._values + other._values
        return self.__class__(*new, **self._kwargs)

    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError
        new = self._values - other._values
        return self.__class__(*new, **self._kwargs)

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError
        new = self._values * other
        return self.__class__(*new, **self._kwargs)

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError
        new = self._values / other
        return self.__class__(*new, **self._kwargs)


class XYZ_Color(LinearColor):
    def __init__(self, x, y, z=None):
        if z is None:
            z = 1 - x - y
            if np.isclose(z, 0):
                z = 0
        if any(val < 0 for val in (x, y, z)):
            raise ValueError("XYZ values must be positive")
        super().__init__((x, y, z))
    
    def __repr__(self):
        rpr = super().__repr__()
        if np.isclose(self._values.sum(), 1):
            parts = rpr.split(', ')
            rpr = ', '.join(parts[:2]) + ')'
        return rpr

    x = property(fget=LinearColor._1.fget, fset=LinearColor._1.fset)
    y = property(fget=LinearColor._2.fget, fset=LinearColor._2.fset)
    z = property(fget=LinearColor._3.fget, fset=LinearColor._3.fset)

    def normalize(self):
        return self / self._values.sum()

    def normalize_Y1(self):
        return self / self.y

    @classmethod
    def from_xyY(cls, x, y, Y):
        """Given the 'little' x,y chromaticity, and the intensity Y, construct
        an xyz color.  See Foley/Van Dam p. 581, eq. 13.21."""
        return Y / y * cls(x, y)


class Lpq_Color(Color):
    """
    Template class for LAB and LUB color spaces, which have a shared L property
    """

    def __init__(self, L, p, q, white_point):
        super().__init__((L, p, q), dtype=float)
        if not isinstance(white_point, XYZ_Color):
            raise TypeError("white_point should be an XYZ_Color")
        self.white_point = white_point

    L = property(fget=LinearColor._1.fget, fset=LinearColor._1.fset)
