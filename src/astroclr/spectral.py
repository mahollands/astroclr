"""
A collection of classes and functions for working with spectra and converting
to directly to perceived colours. This module also imports CMFs from the CMFs
submodule.

Basic usage:
>>> S = Spectrum.BlackBody(6000 * u.K)
>>> XYZ = S.to_XYZ()
>>> RGB = RGB_Color.from_XYZ(XYZ)
>>> IRGB = RGB.to_IRGB()
>>> print(IRGB.to_string())
'#FFF3F1'

Even faster usage:
>>> S = Spectrum.from_file("my_spec.dat")
>>> print(S.to_IRGB_string())
'#DDFFB8'

Monochromatic colours:
>>> XYZ = XYZ_from_wavelength(555 * u.nm)
>>> print(XYZ)
XYZ_Color(0.5120501, 1.0, 0.005749999)
"""

import functools
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid as integrate
from astropy import units as u
from astropy.modeling.physical_models import BlackBody
from astropy.io import fits
from .cmfs import CMFs, DATA_DIR
from .colormodels import XYZ_Color, RGB_Color, IRGB_Color


__all__ = [
    "CMFs",
    "Spectrum",
    "XYZ_from_wavelength",
]

UNIT_SPEC_RADIANCE = u.W / u.m**2 / u.sr / u.nm
UNIT_SPEC_IRRADIANCE = u.W / u.m**2 / u.nm
VALID_FLUX_UNITS = UNIT_SPEC_RADIANCE, UNIT_SPEC_IRRADIANCE
K_cd = 683 * u.lm/u.W # luminous efficacy constant defined at 540THz


def XYZ_from_wavelength(
    wavelength: None | float | np.ndarray | u.Quantity = None,
    CMFname: str = "1931",
) -> XYZ_Color:
    """
    Given a wavelength (in nm if no units supplied), return the corresponding
    XYZ_Color, for unit intensity. Wavelengths can be given as arrays (or
    Quantity arrays) returning an array. If no wavelength is given, the CMF
    wavelength table is used, giving an XYZ_Color array for all wavelengths.
    """
    CMF = CMFs[CMFname]
    if wavelength is None:
        wavelength = CMF.wavelength
    elif isinstance(wavelength, u.Quantity):
        wavelength = wavelength.to(u.nm)
    else:
        wavelength <<= u.nm
    xyz = CMF.interp(wavelength)
    return XYZ_Color(*xyz)


class Spectrum:
    """
    Simple spectrum class for calculating XYZ values. 'wavelength' and 'flux'
    can be given as numpy arrays or astropy quantities. If these are arrays,
    default units of nm and W/m2/sr/nm are applied, respectively. If given as
    quantities, wavelengths must be of type length, and fluxes must be in units
    equivalent to W/m2/sr/nm or W/m2/nm.

    wavelengths -
        wavelength array/Quantity. If an array is given nm is assumed for the
        units. If a Quantity is given, it must have dimensions of length.
    flux -
        flux array/Quantity. If an array is given W/m^2/sr/nm is assumed for
        the units. If a Quanitity is given, it must have dimensions of flux per
        unit wavelength or flux per unit wavelength per solid_angle.
    """

    __slots__ = ["_x", "_y", "_y_unit_option", "interp_flux"]

    def __init__(
        self,
        wavelength: np.ndarray | u.Quantity,
        flux: np.ndarray | u.Quantity,
    ):
        self.wavelength = wavelength
        self.flux = flux

    @property
    def wavelength(self) -> u.Quantity:
        return self._x

    @wavelength.setter
    def wavelength(self, value: np.ndarray | u.Quantity):
        if not isinstance(value, u.Quantity):
            value <<= u.nm
        if value.isscalar:
            raise ValueError("wavelengths must be array-like")
        if not value.unit.is_equivalent(u.nm):
            raise ValueError("wavelengths must be a kind of length")
        self._x = value

    @property
    def flux(self) -> u.Quantity:
        return self._y

    @flux.setter
    def flux(self, value: np.ndarray | u.Quantity):
        if not isinstance(value, u.Quantity):
            value <<= UNIT_SPEC_RADIANCE
        if value.isscalar:
            value = value * np.ones(self.wavelength.size)
        elif len(value) != len(self.wavelength):
            raise ValueError("Fluxes must be the same length as wavelengths")
        for flux_unit in VALID_FLUX_UNITS:
            if value.unit.is_equivalent(flux_unit):
                self._y_unit_option = flux_unit
                break
        else:
            raise ValueError("Flux units must be equivalent to W/m2/sr/nm or W/m2/nm")
        self._y = value
        self.interp_flux = interp1d(
            self.wavelength,
            self.flux,
            kind="cubic",
            copy=False,
            bounds_error=False,
            fill_value=0,
            assume_sorted=True,
        )

    def __len__(self) -> int:
        return len(self._x)

    def __repr__(self) -> str:
        n_spec = len(self)
        x_unit = self.wavelength.unit.to_string()
        y_unit = self.flux.unit.to_string()
        wavelength_str = f"wavelength[{n_spec}] * {x_unit}"
        flux_str = f"flux[{n_spec}] * {y_unit}"
        return f"Spectrum({wavelength_str}, {flux_str})"

    def __getitem__(self, key):
        """
        Return a slice from a spectrum using numpy style indexing.
        """
        wavelength_new, flux_new = self.wavelength[key], self.flux[key]
        return Spectrum(wavelength_new, flux_new)

    def clip(self, lam0, lam1):
        """
        Clip spectrum between wavelength values of lam0 and lam1
        """
        if not isinstance(lam0, u.Quantity):
            lam0 *= self.wavelength.unit
        if not isinstance(lam1, u.Quantity):
            lam1 *= self.wavelength.unit
        key = (self.wavelength > lam0) & (self.wavelength < lam1)
        return self[key]

    @classmethod
    def BlackBody(
        cls,
        temperature: float | u.Quantity,
        wavelength: np.ndarray | u.Quantity = CMFs["1931"].wavelength,
    ):
        """
        Class method for generating a black body spectrum. For
        a given temperature and wavelength array/Quantity.
        >>> BB = Spectrum.BlackBody(temperature, wavelengths)
        """

        if not isinstance(temperature, u.Quantity):
            temperature <<= u.K
        if not temperature.isscalar:
            raise ValueError("temperature should be a scalar")
        if not isinstance(wavelength, u.Quantity):
            wavelength <<= u.nm
        if wavelength.isscalar:
            raise ValueError("wavelengths must be array-like")
        if not wavelength.unit.is_equivalent(u.nm):
            raise ValueError("wavelengths must be a kind of length")
        BB = BlackBody(temperature, scale=1 * UNIT_SPEC_RADIANCE)
        flux = BB(wavelength)
        return cls(wavelength, flux)

    @staticmethod
    @functools.cache
    def _load_Vega():
        """
        Loads the alpha_lyr_stis_011 model of Vega from Calspec, and caches
        for repeated calls.
        """
        hdulist = fits.open(f"{DATA_DIR}/alpha_lyr_stis_011.fits")
        data = hdulist[1].data
        wavelength = data["WAVELENGTH"] * u.AA
        flux = data["FLUX"] * u.erg / u.s / u.cm**2 / u.AA
        return wavelength, flux

    @classmethod
    def load_Vega(cls):
        """
        Loads the alpha_lyr_stis_011 model of Vega from Calspec as a Spectrum
        """
        wavelength, flux = cls._load_Vega()
        Vega = cls(wavelength, flux)
        return Vega

    @classmethod
    def from_file(cls, fname: str, units: tuple[u.Unit, u.Unit], **kwargs):
        """
        Simple reader for column text files
        """
        wavelength, flux = np.loadtxt(fname, unpack=True, usecols=(0, 1), **kwargs)
        wavelength_unit, flux_unit = units
        wavelength <<= wavelength_unit
        flux <<= flux_unit
        return cls(wavelength, flux)

    def XYZ_integrals(self, CMFname: str = "1931") -> u.Quantity:
        """
        Integrate spectrum over CIE1931 X/Y/Z color-matching functions.
        The result is a 3-valued astropy quantity with units of W/m^2
        or W/m^2/sr.
        """
        CMF = CMFs[CMFname]
        # merge spectrum and CMF wavelength arrays to interpolate for integrals
        wunit, funit = self.wavelength.unit, self.flux.unit
        iwavelength = np.hstack([self.wavelength, CMF.wavelength.to(wunit)])
        iwavelength = np.sort(np.unique(iwavelength))
        iflux = self.interp_flux(iwavelength) * funit
        iwavelength, iflux = iwavelength.to(u.nm), iflux.to(self._y_unit_option)
        integrand = iflux[np.newaxis, :] * CMF.interp(iwavelength)
        I_XYZ = integrate(integrand, iwavelength, axis=1)
        return I_XYZ

    def to_XYZ(self, CMFname: str = "1931") -> XYZ_Color:
        """
        Calculate the XYZ color of the spectrum. Units are dropped but
        converted to SI W/m^2 or W/m^2/sr first (depending on the flux units).
        """
        XYZ = self.XYZ_integrals(CMFname=CMFname)
        return XYZ_Color(*XYZ.value)

    def to_RGB(self, CMFname: str = "1931", **kwargs) -> RGB_Color:
        """
        Calculate the RGB color of the spectrum. Units are dropped but
        converted to SI W/m^2 or W/m^2/sr first (depending on the flux units).
        Keyword arguments to be passed to RGB_Color.
        """
        XYZ = self.to_XYZ(CMFname=CMFname)
        RGB = RGB_Color.from_XYZ(XYZ, **kwargs)
        return RGB

    def to_IRGB(self, CMFname: str = "1931", **kwargs) -> IRGB_Color:
        """
        Calculate the displayable RGB color (IRGB) of the spectrum. Keyword
        arguments passed to RGB_Color and RGB_Color.to_IRGB.
        """
        irgb_kw = {}
        if "clip_method" in kwargs:
            irgb_kw["clip_method"] = kwargs.pop("clip_method")
        RGB = self.to_RGB(CMFname=CMFname, **kwargs)
        IRGB = RGB.to_IRGB(**irgb_kw)
        return IRGB

    def to_IRGB_string(self, CMFname: str = "1931", **kwargs) -> str:
        """
        Calculate the displayable RGB color (IRGB) of the spectrum as a
        hexcode. Keyword arguments passed to RGB_Color, and RGB_Color.to_IRGB.
        """
        IRGB = self.to_IRGB(CMFname=CMFname, **kwargs)
        return IRGB.to_string()

    def to_luminance(self, CMFname: str = "1931") -> u.Quantity:
        """
        If flux units are equivalent to W/m2/sr/nm calculate the luminance in
        cd/m^2 (nits). If flux units are equivalent to W/m2/nm calculate the
        illuminance (luminous flux density arriving at a surface from a distant
        light source) in lm/m^2 (lux), or the luminous exitance (luminous flux
        density radiating from a surface) also in lm/m^2 (lux not used).
        """
        # The candela definition specifies a luminous efficacy of 683 lm/W at
        # 540 THz, with a corresponding air wavelength of 555.0171 nm. However,
        # the CIE 1931 photopic luminous efficiency function peaks at 555 nm.
        # This gives a peak luminous efficacy of about 683.002 lm/W.
        correct_540THz = XYZ_from_wavelength(555.0171*u.nm, CMFname=CMFname).Y

        Y = self.XYZ_integrals(CMFname=CMFname)[1]
        L = K_cd / correct_540THz * Y

        if self.flux.unit.is_equivalent(UNIT_SPEC_IRRADIANCE):
            return L.to(u.lm / u.m**2)
        return L.to(u.cd / u.m**2)

    def visual_magnitude(self, CMFname: str = "1931") -> u.Quantity:
        """
        Calculates the visual magnitude of a spectrum relative to Vega
        """
        if not self.flux.unit.is_equivalent(UNIT_SPEC_IRRADIANCE):
            raise ValueError("Flux should be in terms of spectral irradiance")
        Vega = Spectrum.load_Vega()
        L_Vega = Vega.to_luminance(CMFname=CMFname)  # illuminances
        L_self = self.to_luminance(CMFname=CMFname)
        return -2.5 * np.log10(L_self / L_Vega) * u.mag
