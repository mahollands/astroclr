"""
Tests for astroclr.spectral and astroclr.cmfs: CMF loading, the Spectrum
class, and conversions from spectra to colours and photometric quantities.
"""

import numpy as np
import pytest
from astropy import units as u

from astroclr.cmfs import CMFs, LMS_XYZ_MATRIX_2000
from astroclr.colormodels import XYZ_Color, RGB_Color, IRGB_Color
from astroclr.spectral import (
    Spectrum,
    XYZ_from_wavelength,
    UNIT_SPEC_RADIANCE,
    UNIT_SPEC_IRRADIANCE,
)

CMF_NAMES = ["1931", "1931J", "1931JV", "1964_10deg", "2015_2deg", "2015_10deg"]


# --- CMFs ---


class TestCMFs:
    @pytest.mark.parametrize("name", CMF_NAMES)
    def test_cmf_structure(self, name):
        wavelength, xyz_table, interp = CMFs[name]
        assert isinstance(wavelength, u.Quantity)
        assert wavelength.unit.is_equivalent(u.nm)
        assert xyz_table.shape == (3, len(wavelength))
        assert not np.any(np.isnan(xyz_table))
        assert np.all(np.diff(wavelength) > 0)

    @pytest.mark.parametrize("name", CMF_NAMES)
    def test_cmf_zero_outside_range(self, name):
        wavelength, _, interp = CMFs[name]
        outside = [wavelength[0].value - 50, wavelength[-1].value + 50]
        assert np.all(interp(outside) == 0)

    def test_lms_matrix_shape(self):
        assert LMS_XYZ_MATRIX_2000.shape == (3, 3)


# --- XYZ_from_wavelength ---


class TestXYZFromWavelength:
    def test_scalar_peak(self):
        # The 1931 ybar function peaks at 1.0 at 555 nm
        XYZ = XYZ_from_wavelength(555.0 * u.nm)
        assert isinstance(XYZ, XYZ_Color)
        assert XYZ.Y == pytest.approx(1.0, abs=1e-3)

    def test_plain_number_is_nm(self):
        assert XYZ_from_wavelength(555).Y == XYZ_from_wavelength(555 * u.nm).Y

    def test_unit_conversion(self):
        np.testing.assert_allclose(
            XYZ_from_wavelength(5550 * u.AA).values,
            XYZ_from_wavelength(555 * u.nm).values,
        )

    def test_array(self):
        XYZ = XYZ_from_wavelength(np.array([500.0, 600.0]) * u.nm)
        assert len(XYZ) == 2

    def test_none_gives_full_table(self):
        XYZ = XYZ_from_wavelength()
        assert len(XYZ) == len(CMFs["1931"].wavelength)


# --- Spectrum construction ---


class TestSpectrumInit:
    def test_with_quantities(self, sample_wavelengths, sample_fluxes):
        spec = Spectrum(sample_wavelengths, sample_fluxes)
        assert len(spec) == len(sample_wavelengths)
        assert spec.wavelength.unit.is_equivalent(u.nm)
        assert spec.flux.unit.is_equivalent(UNIT_SPEC_RADIANCE)

    def test_with_arrays_default_units(self, sample_wavelengths, sample_fluxes):
        spec = Spectrum(sample_wavelengths.value, sample_fluxes.value)
        assert spec.wavelength.unit == u.nm
        assert spec.flux.unit == UNIT_SPEC_RADIANCE

    def test_with_lists(self):
        spec = Spectrum([400.0, 500.0, 600.0], [1.0, 2.0, 3.0])
        assert len(spec) == 3

    def test_scalar_flux_broadcasts(self):
        wl = np.linspace(380, 780, 50)
        spec = Spectrum(wl, 1.0)
        assert spec.flux.shape == wl.shape
        assert np.all(spec.flux.value == 1.0)

    def test_irradiance_units_accepted(self, sample_wavelengths, sample_fluxes):
        spec = Spectrum(sample_wavelengths, sample_fluxes * u.sr)
        assert spec.flux.unit.is_equivalent(UNIT_SPEC_IRRADIANCE)

    def test_mismatched_lengths(self):
        wl = np.array([400, 500]) * u.nm
        flux = np.array([0.1, 0.2, 0.3]) * UNIT_SPEC_RADIANCE
        with pytest.raises(ValueError, match="same length"):
            Spectrum(wl, flux)

    def test_invalid_wavelength_unit(self):
        with pytest.raises(ValueError, match="kind of length"):
            Spectrum(np.array([400, 500]) * u.kg, [0.1, 0.2])

    def test_scalar_wavelength(self):
        with pytest.raises(ValueError, match="array-like"):
            Spectrum(500 * u.nm, 1.0)

    def test_invalid_flux_unit(self):
        with pytest.raises(ValueError, match="Flux units must be equivalent"):
            Spectrum(np.array([400, 500]) * u.nm, np.array([0.1, 0.2]) * u.kg)

    @pytest.mark.parametrize("wl", [[600.0, 500.0, 400.0], [400.0, 500.0, 500.0]])
    def test_unsorted_or_repeated_wavelengths(self, wl):
        with pytest.raises(ValueError, match="sorted and unique"):
            Spectrum(wl, [1.0, 1.0, 1.0])

    def test_does_not_share_caller_arrays(self, sample_wavelengths, sample_fluxes):
        wl, flux = sample_wavelengths.value.copy(), sample_fluxes.value.copy()
        spec = Spectrum(wl, flux)
        assert not np.shares_memory(spec.wavelength, wl)
        assert not np.shares_memory(spec.flux, flux)

    def test_does_not_share_caller_quantities(self, sample_wavelengths, sample_fluxes):
        spec = Spectrum(sample_wavelengths, sample_fluxes)
        assert not np.shares_memory(spec.wavelength, sample_wavelengths)
        assert not np.shares_memory(spec.flux, sample_fluxes)

    def test_default_blackbody_does_not_alias_cmf(self):
        w0 = CMFs["1931"].wavelength.copy()
        bb = Spectrum.BlackBody(5000 * u.K)
        bb.wavelength *= 1.001
        np.testing.assert_array_equal(CMFs["1931"].wavelength, w0)


# --- Spectrum methods ---


class TestSpectrumMethods:
    def test_repr(self, sample_spectrum):
        text = repr(sample_spectrum)
        assert text.startswith("Spectrum(")
        assert f"[{len(sample_spectrum)}]" in text

    def test_getitem_slice(self, sample_spectrum):
        sub = sample_spectrum[2:5]
        assert isinstance(sub, Spectrum)
        assert len(sub) == 3

    def test_clip(self, sample_spectrum):
        clipped = sample_spectrum.clip(400, 700)
        assert np.all(clipped.wavelength > 400 * u.nm)
        assert np.all(clipped.wavelength < 700 * u.nm)

    def test_clip_with_quantities(self, sample_spectrum):
        clipped = sample_spectrum.clip(4000 * u.AA, 7000 * u.AA)
        assert len(clipped) == len(sample_spectrum.clip(400, 700))

    def test_from_file(self, tmp_path):
        wl = np.linspace(4000, 7000, 31)
        flux = np.ones_like(wl)
        fname = tmp_path / "spec.dat"
        np.savetxt(fname, np.column_stack([wl, flux]))
        units = (u.AA, u.erg / u.s / u.cm**2 / u.AA)
        spec = Spectrum.from_file(fname, units=units)
        assert len(spec) == 31
        assert spec.wavelength.unit == u.AA
        assert spec.flux.unit.is_equivalent(UNIT_SPEC_IRRADIANCE)


# --- Spectrum.BlackBody ---


class TestBlackBody:
    def test_creation(self):
        wl = np.linspace(400, 700, 100) * u.nm
        bb = Spectrum.BlackBody(5000 * u.K, wl)
        assert isinstance(bb, Spectrum)
        assert len(bb) == 100
        assert np.all(bb.flux.value > 0)

    def test_plain_number_temperature(self):
        np.testing.assert_allclose(
            Spectrum.BlackBody(5000).flux.value,
            Spectrum.BlackBody(5000 * u.K).flux.value,
        )

    def test_non_scalar_temperature(self):
        with pytest.raises(ValueError, match="scalar"):
            Spectrum.BlackBody(np.array([5000, 6000]) * u.K)

    def test_planckian_locus_6500K(self):
        # Planckian locus at 6500 K (CIE 1931): x=0.3135, y=0.3236
        xy = Spectrum.BlackBody(6500 * u.K).to_XYZ().normalize()
        assert xy.X == pytest.approx(0.3135, abs=2e-4)
        assert xy.Y == pytest.approx(0.3236, abs=2e-4)

    def test_hotter_is_bluer(self):
        cool = Spectrum.BlackBody(3000).to_RGB()
        hot = Spectrum.BlackBody(10000).to_RGB()
        assert cool.R / cool.B > hot.R / hot.B


# --- Colour conversions ---


class TestSpectrumToColor:
    def test_xyz_integrals(self, sample_spectrum):
        I_XYZ = sample_spectrum.XYZ_integrals(CMFname="1931")
        assert isinstance(I_XYZ, u.Quantity)
        assert I_XYZ.shape == (3,)
        assert I_XYZ.unit.is_equivalent(u.W / u.m**2 / u.sr)
        assert np.all(I_XYZ.value > 0)

    @pytest.mark.parametrize("name", CMF_NAMES)
    def test_to_xyz_all_cmfs(self, sample_spectrum, name):
        XYZ = sample_spectrum.to_XYZ(CMFname=name)
        assert isinstance(XYZ, XYZ_Color)
        assert np.all(XYZ.values > 0)

    def test_unit_invariance(self):
        # The same physical spectrum expressed in different units must give
        # the same XYZ (which is in SI units)
        wl = np.linspace(380, 780, 81) * u.nm
        flux = np.linspace(1, 2, 81) * UNIT_SPEC_IRRADIANCE
        spec_si = Spectrum(wl, flux)
        spec_cgs = Spectrum(wl.to(u.AA), flux.to(u.erg / u.s / u.cm**2 / u.AA))
        np.testing.assert_allclose(
            spec_si.to_XYZ().values, spec_cgs.to_XYZ().values, rtol=1e-10
        )

    def test_flux_scaling_is_linear(self, sample_spectrum):
        doubled = Spectrum(sample_spectrum.wavelength, 2 * sample_spectrum.flux)
        np.testing.assert_allclose(
            doubled.to_XYZ().values, 2 * sample_spectrum.to_XYZ().values
        )

    def test_no_overlap_with_cmf_is_black(self):
        spec = Spectrum(np.linspace(1000, 2000, 11), 1.0)
        np.testing.assert_array_equal(spec.to_XYZ().values, 0)

    def test_to_rgb(self, sample_spectrum):
        assert isinstance(sample_spectrum.to_RGB(), RGB_Color)

    def test_to_irgb(self, sample_spectrum):
        irgb = sample_spectrum.to_IRGB()
        assert isinstance(irgb, IRGB_Color)
        assert np.all((irgb.values >= 0) & (irgb.values <= 255))

    def test_to_irgb_kwargs(self, sample_spectrum):
        irgb = sample_spectrum.to_IRGB(brightest=True, clip_method="clamp")
        assert irgb.values.max() == 255

    def test_to_irgb_string(self, sample_spectrum):
        hex_code = sample_spectrum.to_IRGB_string()
        assert len(hex_code) == 7
        assert hex_code.startswith("#")


# --- Photometry ---


class TestPhotometry:
    def test_luminance_units_radiance(self, sample_spectrum):
        assert sample_spectrum.to_luminance().unit.is_equivalent(u.cd / u.m**2)

    def test_luminance_units_irradiance(self, sample_wavelengths, sample_fluxes):
        spec = Spectrum(sample_wavelengths, sample_fluxes * u.sr)
        assert spec.to_luminance().unit.is_equivalent(u.lm / u.m**2)

    def test_luminous_efficacy_at_540THz(self):
        # A narrow line at 555.0171 nm (540 THz) carrying 1 W/m^2/sr has a
        # luminance of 683 cd/m^2 by definition of the candela.
        lam0, width = 555.0171, 0.01
        wl = np.array([lam0 - width, lam0, lam0 + width])
        flux = np.array([0.0, 1.0, 0.0]) / width  # integrates to 1 W/m^2/sr
        L = Spectrum(wl, flux).to_luminance()
        assert L.to_value(u.cd / u.m**2) == pytest.approx(683, rel=1e-4)

    def test_vega_is_magnitude_zero(self):
        V = Spectrum.load_Vega().visual_magnitude()
        assert V.unit == u.mag
        assert V.value == pytest.approx(0.0, abs=1e-10)

    def test_magnitude_scaling(self):
        vega = Spectrum.load_Vega()
        fainter = Spectrum(vega.wavelength, vega.flux / 100)
        assert fainter.visual_magnitude().value == pytest.approx(5.0)

    def test_magnitude_requires_irradiance(self, sample_spectrum):
        with pytest.raises(ValueError, match="irradiance"):
            sample_spectrum.visual_magnitude()

    def test_vega_cache_not_corrupted(self):
        vega = Spectrum.load_Vega()
        vega.flux *= 2
        assert Spectrum.load_Vega().visual_magnitude().value == pytest.approx(0.0)
