"""
Tests for astroclr.delta_color: colour differences and correlated colour
temperature.
"""

import numpy as np
import pytest
from astropy import units as u

from astroclr.colormodels import XYZ_Color, RGB_Color, LAB_Color, LUV_Color, UCS_Color
from astroclr.delta_color import CCT, delta_E, delta_uv
from astroclr.spectral import Spectrum

WHITE = RGB_Color(1.0, 1.0, 1.0)
BLACK = RGB_Color(0.0, 0.0, 0.0)


class TestDeltaE:
    def test_white_black(self):
        assert delta_E(WHITE, BLACK) == pytest.approx(100.0)

    def test_identical_is_zero(self):
        assert delta_E(WHITE, WHITE) == pytest.approx(0.0)

    def test_symmetric(self):
        c1, c2 = RGB_Color(0.2, 0.5, 0.7), RGB_Color(0.6, 0.1, 0.3)
        assert delta_E(c1, c2) == pytest.approx(delta_E(c2, c1))

    def test_lab_euclidean(self):
        c1 = LAB_Color(50.0, 10.0, 20.0)
        c2 = LAB_Color(53.0, 14.0, 20.0)
        assert delta_E(c1, c2) == pytest.approx(5.0)

    @pytest.mark.parametrize(
        "convert",
        [
            lambda c: c,
            lambda c: c.to_XYZ(),
            lambda c: LAB_Color.from_XYZ(c.to_XYZ()),
            lambda c: LUV_Color.from_XYZ(c.to_XYZ()),
            lambda c: UCS_Color.from_XYZ(c.to_XYZ()),
        ],
        ids=["RGB", "XYZ", "LAB", "LUV", "UCS"],
    )
    def test_accepts_colour_types(self, convert):
        c1, c2 = RGB_Color(0.2, 0.5, 0.7), RGB_Color(0.6, 0.1, 0.3)
        expected = delta_E(c1, c2)
        assert delta_E(convert(c1), convert(c2)) == pytest.approx(expected)

    def test_arrays(self):
        greys = RGB_Color.from_grey(np.array([0.0, 1.0]))
        np.testing.assert_allclose(delta_E(greys, BLACK), [0.0, 100.0])


class TestCCT:
    @pytest.mark.parametrize("T", [2000, 3000, 6500, 10000])
    def test_blackbody(self, T):
        XYZ = Spectrum.BlackBody(T * u.K).to_XYZ()
        assert CCT(XYZ).to_value(u.K) == pytest.approx(T, rel=1e-4)

    def test_returns_quantity(self):
        assert CCT(WHITE).unit == u.K

    def test_srgb_white_is_d65(self):
        # D65 has a CCT of ~6504 K
        assert CCT(WHITE).to_value(u.K) == pytest.approx(6504, abs=5)

    def test_return_uv_vec(self):
        XYZ = Spectrum.BlackBody(4000 * u.K).to_XYZ()
        T, duv, dist = CCT(XYZ, return_uv_vec=True)
        assert T.to_value(u.K) == pytest.approx(4000, rel=1e-4)
        assert dist == pytest.approx(0, abs=1e-6)
        assert np.hypot(*duv) == pytest.approx(dist)

    def test_off_locus_distance(self):
        # Green-tinted white sits above the Planckian locus
        _, _, dist = CCT(RGB_Color(0.9, 1.0, 0.9), return_uv_vec=True)
        assert dist > 1e-3


def test_delta_uv_zero_for_same_colour():
    XYZ = XYZ_Color(0.3, 0.4, 0.5)
    np.testing.assert_allclose(delta_uv(XYZ, XYZ), [0, 0])
