"""
Gamma correction / electro-optical tranfser functions

In astroclr, RGB_Color is used to represent linear RGB on a scale of 0-1, where
RGB_Color(1, 1, 1) represents peak white. However these do not directly correspond
to the 8-bit display values used in most display colour spaces. Instead a transfer
function must be applied to each component, before quantising to the 0-255 display
scale. In other words the display scale is a non-linear encoding of intensity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
import numpy as np


class TransferFunction(ABC):
    """
    Base class: __call__() is the EOTF, inv() is its inverse.
    """

    @abstractmethod
    def __call__(self, V): ...

    @abstractmethod
    def inv(self, L): ...


@dataclass(frozen=True)
class gamma_EOTF(TransferFunction):
    """
    Simple power law for gamma correction.
    """

    gamma: float

    def __call__(self, V):
        return np.asarray(V) ** self.gamma

    def inv(self, L):
        return np.asarray(L) ** (1 / self.gamma)


@dataclass(frozen=True)
class piecewise_EOTF(TransferFunction):
    """
    Colour spaces such as sRGB use a piecewise EOTF and inverse, which approximates a
    gamma shifted curve curve over most of the range, but features a small
    linear section near zero. This avoids a vertical slope in the inverse EOTF at zero,
    allowing higher precision gamma correction for near-black values.
    """

    break_point: float
    gamma: float
    slope: float
    offset: float

    def __call__(self, V):
        V = np.asarray(V)
        linear = V / self.slope
        power = ((V + self.offset) / (1 + self.offset)) ** self.gamma
        return np.where(V <= self.break_point * self.slope, linear, power)

    def inv(self, L):
        L = np.asarray(L)
        linear = self.slope * L
        power = (1 + self.offset) * L ** (1 / self.gamma) - self.offset
        return np.where(L <= self.break_point, linear, power)


@dataclass(frozen=True)
class BT1886_EOTF(TransferFunction):
    """
    BTU 1886 is a essentially a gamma function, with specificed black luminance
    relative to white. This EOTF is done on a scale of 0-1 rather than in
    physical luminance.
    """

    gamma: float = 2.4
    Lb_Lw: float = 0

    @property
    def _a(self):
        return (1 - self.Lb_Lw ** (1 / self.gamma)) ** self.gamma

    @property
    def _b(self):
        Lb = self.Lb_Lw ** (1 / self.gamma)
        return Lb / (1 - Lb)

    def __call__(self, V):
        return self._a * np.maximum(np.asarray(V) + self._b, 0) ** self.gamma

    def inv(self, L):
        scaled = (np.asarray(L) / self._a) ** (1 / self.gamma)
        return np.maximum(scaled - self._b, 0)


@dataclass(frozen=True)
class PQ_EOTF(TransferFunction):
    """
    Perceptual Quantizer transfer function. Unlike the SDR EOTFs which map to
    an arbitrary luminance range, the unit PQ output scale (input scale to the
    inv method), maps linearly to a 10000 cd/m^2 (nit) range in physical
    luminance.
    https://www.color.org/chardata/rgb/BT2100.xalter
    """

    m1: ClassVar[float] = 1305 / 8192
    m2: ClassVar[float] = 2523 / 32
    c1: ClassVar[float] = 107 / 128
    c2: ClassVar[float] = 2413 / 128
    c3: ClassVar[float] = 2392 / 128

    def __call__(self, V):
        Vm2 = np.asarray(V) ** (1 / self.m2)
        numerator = np.maximum(Vm2 - self.c1, 0)
        denominator = self.c2 - self.c3 * Vm2
        return (numerator / denominator) ** (1 / self.m1)

    def inv(self, L):
        Lm1 = np.asarray(L) ** self.m1
        numerator = self.c1 + self.c2 * Lm1
        denominator = 1 + self.c3 * Lm1
        return (numerator / denominator) ** self.m2
