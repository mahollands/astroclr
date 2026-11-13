"""
Tests for astroclr.illuminants.
"""

import numpy as np
import pytest
from astropy import units as u

from astroclr import illuminants
from astroclr.delta_color import CCT
from astroclr.rgb_color_spaces import Whites
from astroclr.spectral import Spectrum

ILLUMINANTS = {
    "A": illuminants.get_illuminant_A,
    "D65": illuminants.get_illuminant_D65,
    "D55": lambda: illuminants.get_illuminant_DXX(5500),
    "E": illuminants.get_illuminant_E,
    "BB": lambda: illuminants.get_blackbody_illuminant(5000 * u.K),
}


@pytest.mark.parametrize("get", ILLUMINANTS.values(), ids=ILLUMINANTS.keys())
def test_normalised_to_Y1(get):
    illum = get()
    assert isinstance(illum, Spectrum)
    assert illum.to_XYZ().Y == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["A", "D65"])
def test_chromaticities_match_whites(name):
    xy = ILLUMINANTS[name]().to_XYZ().normalize()
    np.testing.assert_allclose(xy.values, Whites[name].values, atol=2e-4)


def test_illuminant_E_is_flat():
    flux = illuminants.get_illuminant_E().flux.value
    np.testing.assert_allclose(flux, flux[0])


def test_illuminant_E_is_equal_energy_white():
    xy = illuminants.get_illuminant_E().to_XYZ().normalize()
    np.testing.assert_allclose(xy.values, 1 / 3, atol=1e-3)


def test_illuminant_A_is_2856K_blackbody():
    A = illuminants.get_illuminant_A()
    BB = illuminants.get_blackbody_illuminant(2856)
    np.testing.assert_allclose(A.flux, BB.flux)


class TestDSeries:
    def test_dxx_matches_d65_table(self):
        xy_calc = illuminants.get_illuminant_DXX(6504).to_XYZ().normalize()
        xy_table = illuminants.get_illuminant_D65().to_XYZ().normalize()
        np.testing.assert_allclose(xy_calc.values, xy_table.values, atol=2e-4)

    @pytest.mark.parametrize("T, name", [(5500, "D55"), (7500, "D75")])
    def test_dxx_chromaticities(self, T, name):
        xy = illuminants.get_illuminant_DXX(T).to_XYZ().normalize()
        np.testing.assert_allclose(xy.values, Whites[name].values, atol=5e-4)

    @pytest.mark.parametrize("T", [4500, 6500, 9000])
    def test_dxx_cct(self, T):
        cct = CCT(illuminants.get_illuminant_DXX(T).to_XYZ())
        assert cct.to_value(u.K) == pytest.approx(T, rel=5e-3)

    def test_accepts_quantity(self):
        a = illuminants.get_illuminant_DXX(5500)
        b = illuminants.get_illuminant_DXX(5500 * u.K)
        np.testing.assert_allclose(a.flux, b.flux)

    @pytest.mark.parametrize("T", [3000, 30000])
    def test_out_of_range(self, T):
        with pytest.raises(ValueError, match="out of range"):
            illuminants.get_illuminant_DXX(T)

    def test_cache_not_corrupted(self):
        before = illuminants.get_illuminant_DXX(5000).flux.copy()
        illuminants.get_illuminant_DXX(5000).flux *= 3
        np.testing.assert_allclose(illuminants.get_illuminant_DXX(5000).flux, before)
