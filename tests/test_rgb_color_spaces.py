"""
Tests for astroclr.rgb_color_spaces and astroclr.transfer_functions.
"""

import dataclasses
import functools
import pickle

import numpy as np
import pytest

from astroclr.colormodels import XYZ_Color
from astroclr.rgb_color_spaces import (
    RGB_ColorSpace,
    sRGB,
    Rec709,
    Rec2020,
    Rec2100,
    DCI_P3_D65,
    DCI_P3_DCI,
    Display_P3,
)
from astroclr.transfer_functions import (
    TransferFunction,
    gamma_EOTF,
    piecewise_EOTF,
    BT1886_EOTF,
    PQ_EOTF,
)

ALL_SPACES = [sRGB, Rec709, Rec2020, Rec2100, DCI_P3_D65, DCI_P3_DCI, Display_P3]
space_ids = [cs.name for cs in ALL_SPACES]

ALL_EOTFS = [
    gamma_EOTF(gamma=2.2),
    piecewise_EOTF(break_point=0.0031308, gamma=2.4, slope=12.92, offset=0.055),
    BT1886_EOTF(),
    BT1886_EOTF(gamma=2.4, Lb_Lw=0.001),
    PQ_EOTF(),
]
eotf_ids = ["gamma", "piecewise", "BT1886", "BT1886_black", "PQ"]


# --- Colour spaces ---


class TestColorSpaces:
    @pytest.mark.parametrize("cs", ALL_SPACES, ids=space_ids)
    def test_matrices_are_inverse(self, cs):
        np.testing.assert_allclose(
            cs.rgb_from_xyz_matrix @ cs.xyz_from_rgb_matrix, np.eye(3), atol=1e-12
        )

    @pytest.mark.parametrize("cs", ALL_SPACES, ids=space_ids)
    def test_white_maps_to_unit_rgb(self, cs):
        rgb = cs.rgb_from_xyz_matrix @ cs.W.normalize_Y1().values
        np.testing.assert_allclose(rgb, 1)

    @pytest.mark.parametrize("cs", ALL_SPACES, ids=space_ids)
    def test_primaries_have_correct_chromaticity(self, cs):
        for column, primary in zip(cs.xyz_from_rgb_matrix.T, cs.primaries):
            np.testing.assert_allclose(column / column.sum(), primary.values)

    def test_srgb_matrix_reference(self):
        # IEC 61966-2-1 XYZ -> linear sRGB matrix
        expected = [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
        np.testing.assert_allclose(sRGB.rgb_from_xyz_matrix, expected, atol=1e-3)

    def test_in_gamut(self):
        # D65 white is in both; (0.2, 0.7) is inside Rec2020 but outside sRGB
        XYZ = XYZ_Color(np.array([0.3127, 0.2]), np.array([0.3290, 0.7]))
        np.testing.assert_array_equal(sRGB.in_gamut(XYZ), [True, False])
        np.testing.assert_array_equal(Rec2020.in_gamut(XYZ), [True, True])

    def test_in_gamut_image(self):
        XYZ = XYZ_Color(np.full((4, 5), 0.3127), np.full((4, 5), 0.3290))
        assert sRGB.in_gamut(XYZ).shape == (4, 5)

    @pytest.mark.parametrize(
        "cs, max_int", [(sRGB, 255), (Rec2100, 1023), (DCI_P3_DCI, 4095)]
    )
    def test_max_int(self, cs, max_int):
        assert cs.max_int == max_int

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            sRGB.bit_depth = 10

    @pytest.mark.parametrize("cs", ALL_SPACES, ids=space_ids)
    def test_pickle(self, cs):
        assert pickle.loads(pickle.dumps(cs)) == cs

    def test_value_equality(self):
        rebuilt = RGB_ColorSpace(
            name="sRGB",
            R=sRGB.R, G=sRGB.G, B=sRGB.B, W=sRGB.W,
            EOTF=piecewise_EOTF(break_point=0.0031308, gamma=2.4, slope=12.92, offset=0.055),
        )
        assert rebuilt == sRGB
        assert sRGB != Rec2020


class TestColorSpaceHash:
    @pytest.mark.parametrize("cs", ALL_SPACES, ids=space_ids)
    def test_hashable(self, cs):
        assert isinstance(hash(cs), int)

    @pytest.mark.parametrize("cs", ALL_SPACES, ids=space_ids)
    def test_hash_survives_pickle(self, cs):
        assert hash(pickle.loads(pickle.dumps(cs))) == hash(cs)

    def test_equal_spaces_hash_equal(self):
        # Rebuilt from new objects with the same values
        rebuilt = RGB_ColorSpace(
            name="sRGB",
            R=XYZ_Color(*sRGB.R.values),
            G=XYZ_Color(*sRGB.G.values),
            B=XYZ_Color(*sRGB.B.values),
            W=XYZ_Color(*sRGB.W.values),
            EOTF=piecewise_EOTF(break_point=0.0031308, gamma=2.4, slope=12.92, offset=0.055),
        )
        assert rebuilt == sRGB
        assert hash(rebuilt) == hash(sRGB)

    def test_all_spaces_distinct(self):
        assert len({hash(cs) for cs in ALL_SPACES}) == len(ALL_SPACES)

    def test_set_and_dict(self):
        restored = pickle.loads(pickle.dumps(sRGB))
        assert len({sRGB, restored, Rec2020}) == 2
        lookup = {cs: cs.name for cs in ALL_SPACES}
        assert lookup[restored] == "sRGB"

    def test_lru_cache(self):
        @functools.lru_cache
        def name_of(cs):
            return cs.name

        name_of(sRGB)
        name_of(pickle.loads(pickle.dumps(sRGB)))
        assert name_of.cache_info().hits == 1


# --- Transfer functions ---


class TestTransferFunctions:
    @pytest.mark.parametrize("eotf", ALL_EOTFS, ids=eotf_ids)
    def test_is_transfer_function(self, eotf):
        assert isinstance(eotf, TransferFunction)

    @pytest.mark.parametrize("eotf", ALL_EOTFS, ids=eotf_ids)
    def test_round_trip(self, eotf):
        # PQ's inverse maps L=0 to c1**m2 ~ 7e-7 rather than exactly 0, so
        # allow for that at the black end.
        V = np.linspace(0, 1, 101)
        np.testing.assert_allclose(eotf.inv(eotf(V)), V, atol=1e-6)
        # Luminances below the display black level (BT.1886 with Lb > 0)
        # can't be reproduced, so only check L in [EOTF(0), 1].
        L = np.linspace(eotf(0.0), 1, 101)
        np.testing.assert_allclose(eotf(eotf.inv(L)), L, atol=1e-10)

    @pytest.mark.parametrize("eotf", ALL_EOTFS, ids=eotf_ids)
    def test_endpoints(self, eotf):
        assert eotf(1.0) == pytest.approx(1.0)
        assert eotf.inv(1.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("eotf", ALL_EOTFS, ids=eotf_ids)
    def test_monotonic(self, eotf):
        V = np.linspace(0, 1, 101)
        assert np.all(np.diff(eotf(V)) > 0)

    @pytest.mark.parametrize("eotf", ALL_EOTFS, ids=eotf_ids)
    def test_array_shapes(self, eotf):
        V = np.full((3, 4, 5), 0.5)
        assert eotf(V).shape == V.shape
        assert eotf.inv(V).shape == V.shape

    def test_srgb_reference(self):
        # 18% linear grey encodes to 118/255 in sRGB
        assert round(255 * sRGB.EOTF.inv(0.18)) == 118
        # continuity at the break point between linear and power segments
        bp = 0.0031308
        assert sRGB.EOTF.inv(bp - 1e-12) == pytest.approx(sRGB.EOTF.inv(bp + 1e-12), abs=1e-6)

    def test_bt1886_zero_black_is_gamma(self):
        V = np.linspace(0, 1, 11)
        np.testing.assert_allclose(BT1886_EOTF(gamma=2.4)(V), V**2.4)

    def test_bt1886_black_level(self):
        assert BT1886_EOTF(Lb_Lw=0.001)(0.0) == pytest.approx(0.001)

    def test_pq_reference(self):
        # 100 cd/m^2 out of 10000 encodes to ~0.508 in PQ
        assert PQ_EOTF().inv(0.01) == pytest.approx(0.5081, abs=1e-4)
        assert PQ_EOTF()(0.0) == 0.0

    def test_display_curves(self):
        assert Rec709.EOTF == BT1886_EOTF(gamma=2.4)
        assert Rec2020.EOTF == BT1886_EOTF(gamma=2.4)
        assert Display_P3.EOTF == sRGB.EOTF

    @pytest.mark.parametrize("eotf", ALL_EOTFS, ids=eotf_ids)
    def test_equality_pickle_repr(self, eotf):
        clone = dataclasses.replace(eotf)
        assert clone == eotf
        assert hash(clone) == hash(eotf)
        assert pickle.loads(pickle.dumps(eotf)) == eotf
        assert repr(eotf).startswith(type(eotf).__name__ + "(")

    def test_abstract_base(self):
        with pytest.raises(TypeError):
            TransferFunction()
