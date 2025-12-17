import numpy as np
import pytest
from astropy import units as u
from colorpy2.ciexyz import (
    CMFs,
    Spectrum,
    XYZ_from_wavelength,
    get_normalized_spectral_line_colors,
    _unit_spec_radiance,
    _unit_spec_irradiance,
)
from colorpy2.colormodels import XYZ_Color, RGB_Color, IRGB_Color, ColorList

# --- Fixtures ---

@pytest.fixture
def sample_wavelengths():
    """Wavelengths in nm as an Astropy Quantity."""
    return np.arange(300.0, 901.0, 50.0) * u.nm

@pytest.fixture
def sample_fluxes(sample_wavelengths):
    """Fluxes with default units as an Astropy Quantity."""
    # Arbitrary values for testing
    flux = np.exp(-0.5*((sample_wavelengths-550*u.nm)/(300*u.nm))**2)
    return  flux * _unit_spec_radiance

@pytest.fixture
def sample_spectrum(sample_wavelengths, sample_fluxes):
    """A basic Spectrum instance."""
    return Spectrum(sample_wavelengths, sample_fluxes)

# --- Tests for CMFs data loading and access ---

class TestGroup:
    def test_cmfs_loading(self):
        """Test that CMFs dict is populated and contains standard data."""
        assert "1931" in CMFs
        assert "1964_10deg" in CMFs
        
        # Check 1931 data structure
        wl_1931, xyz_1931, xyz_1931_interp = CMFs["1931"]
        assert isinstance(wl_1931, u.Quantity)
        assert wl_1931.unit.is_equivalent(u.nm)
        assert isinstance(xyz_1931, np.ndarray)
        assert xyz_1931.shape[0] == 3 # x, y, z

        # Check 1964 data structure
        wl_1964, xyz_1964, xyz_1931_interp = CMFs["1964_10deg"]
        assert isinstance(wl_1964, u.Quantity)
        assert wl_1964.unit.is_equivalent(u.nm)
        assert isinstance(xyz_1964, np.ndarray)
        assert xyz_1964.shape[0] == 3 # x, y, z

    # --- Tests for Spectrum Class ---

    def test_spectrum_init_with_quantities(self, sample_wavelengths, sample_fluxes):
        """Test successful initialization with Astropy Quantities."""
        spec = Spectrum(sample_wavelengths, sample_fluxes)
        assert len(spec) == len(sample_wavelengths)
        assert spec.wavelength.unit.is_equivalent(u.nm)
        assert spec.flux.unit.is_equivalent(_unit_spec_radiance)

    def test_spectrum_init_with_arrays(self, sample_wavelengths, sample_fluxes):
        """Test successful initialization with numpy arrays (default units applied)."""
        wl_array = sample_wavelengths.value
        flux_array = sample_fluxes.value
        spec = Spectrum(wl_array, flux_array)
        assert len(spec) == len(wl_array)
        assert spec.wavelength.unit == u.nm
        assert spec.flux.unit == _unit_spec_radiance

    def test_spectrum_init_mismatched_lengths(self):
        """Test ValueError for mismatched wavelength and flux lengths."""
        wl = np.array([400, 500]) * u.nm
        flux = np.array([0.1, 0.2, 0.3]) * _unit_spec_radiance
        with pytest.raises(ValueError, match="Fluxes must be the same length as wavelengths"):
            Spectrum(wl, flux)

    def test_spectrum_init_invalid_wavelength_unit(self):
        """Test ValueError for invalid wavelength units."""
        wl = np.array([400, 500]) * u.kg
        flux = np.array([0.1, 0.2]) * _unit_spec_radiance
        with pytest.raises(ValueError, match="wavelengths must be a kind of length"):
            Spectrum(wl, flux)

    def test_spectrum_init_invalid_flux_unit(self):
        """Test ValueError for invalid flux units."""
        wl = np.array([400, 500]) * u.nm
        flux = np.array([0.1, 0.2]) * u.kg
        with pytest.raises(ValueError, match="Flux units must be equivalent to"):
            Spectrum(wl, flux)


    # --- Tests for Spectrum.BlackBody Class Method ---

    def test_blackbody_classmethod(self):
        """Test BlackBody creation."""
        temp = 5000 * u.K
        wl = np.linspace(400, 700, 100) * u.nm
        bb_spec = Spectrum.BlackBody(temp, wl)
        
        assert isinstance(bb_spec, Spectrum)
        assert len(bb_spec) == 100
        assert np.all(bb_spec.flux.value > 0)

    def test_blackbody_classmethod_invalid_temp(self):
        """Test BlackBody creation with non-scalar temperature."""
        temp = np.array([5000, 6000]) * u.K
        wl = np.linspace(400, 700, 100) * u.nm
        with pytest.raises(ValueError, match="temperature should be a scalar"):
            Spectrum.BlackBody(temp, wl)

    # --- Tests for Spectrum.XYZ_integrals ---

    def test_spectrum_xyz_integrals(self, sample_spectrum):
        """Test the XYZ_integrals calculation."""
        I_XYZ = sample_spectrum.XYZ_integrals(CMFname="1931")
        assert isinstance(I_XYZ, u.Quantity)
        assert I_XYZ.shape == (3,)
        # The result is an integrated quantity, should have units of I * d_lambda
        # Default flux units are W/m2/sr/nm, integrating over nm gives W/m2/sr
        expected_unit = u.W / u.m**2 / u.sr
        assert I_XYZ.unit.is_equivalent(expected_unit)
        # Check that Y is the largest integral for a typical spectrum
        assert np.all(I_XYZ.value >= 0)
        # (Cannot check absolute values without knowing CMF data points)

    def test_spectrum_to_xyz(self, sample_spectrum):
        """Test conversion to XYZ_Color."""
        xyz = sample_spectrum.to_XYZ()
        assert isinstance(xyz, XYZ_Color)
        # Check if the values are positive
        assert all(val >= 0 for val in [xyz.x, xyz.y, xyz.z])

    def test_spectrum_to_rgb(self, sample_spectrum):
        """Test conversion to RGB_Color."""
        rgb = sample_spectrum.to_RGB()
        assert isinstance(rgb, RGB_Color)

    def test_spectrum_to_irgb(self, sample_spectrum):
        """Test conversion to IRGB_Color."""
        irgb = sample_spectrum.to_IRGB()
        assert isinstance(irgb, IRGB_Color)
        # Assuming IRGB_Color is just RGB_Color with integer components or a standard tuple
        assert all(val in range(256) for val in [irgb.r, irgb.g, irgb.b])

    def test_spectrum_to_irgb_string(self, sample_spectrum):
        """Test conversion to hexcode string."""
        hex_code = sample_spectrum.to_IRGB_string()
        assert isinstance(hex_code, str)
        assert len(hex_code) == 7
        assert hex_code.startswith("#")

    def test_spectrum_to_luminance_candela(self, sample_spectrum):
        """Test luminance calculation for W/m2/sr/nm flux units (cd/m^2)."""
        # Sample spectrum uses _flux_unit1 (W/m2/sr/nm)
        luminance = sample_spectrum.to_luminance()
        assert luminance.unit.is_equivalent(u.cd / u.m**2)

    def test_spectrum_to_luminance_lux(self, sample_wavelengths, sample_fluxes):
        """Test luminance calculation for W/m2/nm flux units (lm/m^2)."""
        flux = sample_fluxes * u.sr # W/m2/nm (equivalent to flux_unit2)
        spec_lux = Spectrum(sample_wavelengths, flux)
        illuminance = spec_lux.to_luminance()
        assert illuminance.unit.is_equivalent(u.lm / u.m**2)

    # --- Tests for XYZ_from_wavelength ---

    def test_xyz_from_wavelength_scalar(self):
        """Test XYZ_from_wavelength for a single wavelength."""
        wl = 555.0 * u.nm
        xyz_color = XYZ_from_wavelength(wl)
        assert isinstance(xyz_color, XYZ_Color)
        # Max Y value for 1931 CMFs is at 555nm, which should be 1.0 (approximately)
        # Since the input is unit intensity, the output should be roughly normalized to CMF scale.
        # The Y value should be close to the maximum value of the y_bar function (1.0)
        # The interpolated value at 555nm for the 1931 CMFs is 1.0
        assert np.isclose(xyz_color.y, 1.0, atol=1e-3)

    def test_xyz_from_wavelength_array(self):
        """Test XYZ_from_wavelength for an array of wavelengths."""
        wl_array = np.array([500.0, 600.0]) * u.nm
        color_list = XYZ_from_wavelength(wl_array)
        assert isinstance(color_list, ColorList)
        assert len(color_list) == 2
        assert isinstance(color_list[0], XYZ_Color)

    # --- Tests for get_normalized_spectral_line_colors ---

    def test_get_normalized_spectral_line_colors_no_purples(self):
        """Test color list generation without purples."""
        XYZs = get_normalized_spectral_line_colors(brightness=1.0, num_purples=0)
        assert isinstance(XYZs, ColorList)
        # Length should match the number of steps in the CMF table
        # (471 steps from 360nm to 830nm at 1nm increment)
        assert len(XYZs) == len(CMFs["1931"][0])

    def test_get_normalized_spectral_line_colors_with_purples(self):
        """Test color list generation with purples."""
        num_purples = 5
        XYZs = get_normalized_spectral_line_colors(brightness=1.0, num_purples=num_purples)
        assert isinstance(XYZs, ColorList)
        # Total length = Spectral points + Purples
        expected_length = len(CMFs["1931"][0]) + num_purples
        assert len(XYZs) == expected_length

    def test_get_normalized_spectral_line_colors_annotated(self):
        """Test color list generation with annotations."""
        XYZs, annotations = get_normalized_spectral_line_colors(
            brightness=1.0, num_purples=2, annotate=True
        )
        expected_length = len(CMFs["1931"][0]) + 2
        assert isinstance(XYZs, ColorList)
        assert isinstance(annotations, list)
        assert len(XYZs) == expected_length
        assert len(annotations) == expected_length
        # Check annotation content
        assert annotations[-1].endswith("purples")
