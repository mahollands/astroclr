"""
Tests for the colour classes in astroclr._colormodels_core and
astroclr.colormodels: construction, indexing, arithmetic, and conversions
between colour models.
"""

import pickle

import numpy as np
import pytest
from astropy import units as u

from astroclr.colormodels import (
    ClipMethod,
    XYZ_Color,
    RGB_Color,
    IRGB_Color,
    UCS_Color,
    LUV_Color,
    LAB_Color,
)
from astroclr.rgb_color_spaces import sRGB, Rec2020, Rec2100, DCI_P3_DCI, Whites

rng = np.random.default_rng(42)


@pytest.fixture
def xyz_scalar():
    return XYZ_Color(0.3, 0.4, 0.5)


@pytest.fixture
def xyz_vector():
    return XYZ_Color(*rng.uniform(0.05, 1.0, size=(3, 10)))


@pytest.fixture
def xyz_image():
    return XYZ_Color(*rng.uniform(0.05, 1.0, size=(3, 4, 5)))


# --- Core Color behaviour ---


class TestColorCore:
    def test_values_shape_scalar(self, xyz_scalar):
        assert xyz_scalar.values.shape == (3,)
        assert len(xyz_scalar) == 1

    def test_values_shape_vector(self, xyz_vector):
        assert xyz_vector.values.shape == (3, 10)
        assert len(xyz_vector) == 10

    def test_len_undefined_for_images(self, xyz_image):
        with pytest.raises(ValueError):
            len(xyz_image)

    def test_channel_properties(self, xyz_scalar):
        assert (xyz_scalar.X, xyz_scalar.Y, xyz_scalar.Z) == (0.3, 0.4, 0.5)

    def test_channel_setter(self, xyz_scalar):
        xyz_scalar.X = 0.9
        assert xyz_scalar.X == 0.9
        assert xyz_scalar.values[0] == 0.9

    def test_mismatched_types(self):
        with pytest.raises(TypeError):
            XYZ_Color(np.array([0.1, 0.2]), 0.3, 0.4)

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError):
            XYZ_Color(np.ones(2), np.ones(3), np.ones(2))

    def test_getitem(self, xyz_vector):
        first = xyz_vector[0]
        assert isinstance(first, XYZ_Color)
        np.testing.assert_array_equal(first.values, xyz_vector.values[:, 0])
        assert len(xyz_vector[2:5]) == 3

    def test_iteration(self, xyz_vector):
        assert len(list(xyz_vector)) == 10

    def test_from_list(self, xyz_scalar):
        combined = XYZ_Color.from_list([xyz_scalar, 2 * xyz_scalar])
        assert len(combined) == 2
        np.testing.assert_allclose(combined[1].values, 2 * xyz_scalar.values)

    def test_from_list_keeps_kwargs(self):
        colours = [RGB_Color(0.1, 0.2, 0.3, color_space=Rec2020)] * 3
        assert RGB_Color.from_list(colours).color_space is Rec2020

    def test_from_list_rejects_mixed_kwargs(self):
        colours = [RGB_Color(0.1, 0.2, 0.3), RGB_Color(0.1, 0.2, 0.3, Rec2020)]
        with pytest.raises(ValueError):
            RGB_Color.from_list(colours)

    @pytest.mark.parametrize(
        "bad", [(0.1, 0.2, 0.3), [RGB_Color(0.1, 0.2, 0.3)]], ids=["tuple", "wrong type"]
    )
    def test_from_list_type_errors(self, bad):
        with pytest.raises(TypeError):
            XYZ_Color.from_list(bad)

    def test_from_list_rejects_arrays(self, xyz_vector):
        with pytest.raises(TypeError):
            XYZ_Color.from_list([xyz_vector, xyz_vector])

    def test_repr_and_format(self, xyz_scalar):
        assert repr(xyz_scalar) == "XYZ_Color(X=0.3, Y=0.4, Z=0.5)"
        assert format(xyz_scalar, ".2f") == "XYZ_Color(X=0.30, Y=0.40, Z=0.50)"

    def test_repr_chromaticity(self):
        assert repr(XYZ_Color(0.25, 0.5)) == "XYZ_Color(x=0.25, y=0.5)"

    def test_pickle(self):
        rgb = RGB_Color(0.1, 0.2, 0.3, color_space=Rec2100)
        restored = pickle.loads(pickle.dumps(rgb))
        assert restored == rgb
        assert restored.color_space == Rec2100

    def test_unpickled_combines_with_fresh(self):
        # e.g. results returned from multiprocessing workers
        restored = pickle.loads(pickle.dumps(RGB_Color(0.1, 0.2, 0.3)))
        total = restored + RGB_Color(0.1, 0.2, 0.3)
        np.testing.assert_allclose(total.values, [0.2, 0.4, 0.6])
        assert RGB_Color.from_list([restored, RGB_Color(0.1, 0.2, 0.3)])


class TestEquality:
    def test_equal_values(self):
        assert XYZ_Color(0.3, 0.4, 0.5) == XYZ_Color(0.3, 0.4, 0.5)

    def test_different_values(self):
        assert XYZ_Color(0.3, 0.4, 0.5) != XYZ_Color(0.3, 0.4, 0.6)

    def test_arrays_compare_as_a_whole(self, xyz_vector):
        assert xyz_vector == xyz_vector[:]
        assert xyz_vector != xyz_vector * 2
        assert xyz_vector != xyz_vector[:5]

    def test_different_types(self):
        assert XYZ_Color(1.0, 1.0, 1.0) != RGB_Color(1.0, 1.0, 1.0)
        assert XYZ_Color(1.0, 1.0, 1.0) != UCS_Color(1.0, 1.0, 1.0)

    def test_non_colours(self):
        assert XYZ_Color(1.0, 1.0, 1.0) != 1
        assert XYZ_Color(1.0, 1.0, 1.0) != (1.0, 1.0, 1.0)
        # Defers to the other operand rather than deciding itself
        assert XYZ_Color(1.0, 1.0, 1.0).__eq__(1) is NotImplemented

    def test_symmetric(self):
        a, b = XYZ_Color(1.0, 1.0, 1.0), RGB_Color(1.0, 1.0, 1.0)
        assert (a == b) == (b == a)

    @pytest.mark.parametrize(
        "colour",
        [
            XYZ_Color(0.3, 0.4, 0.5),
            RGB_Color(0.1, 0.2, 0.3),
            IRGB_Color(1, 2, 3),
            UCS_Color(0.1, 0.2, 0.3),
            LAB_Color(50.0, 0.0, 0.0),
            LUV_Color(50.0, 0.0, 0.0),
        ],
        ids=lambda c: type(c).__name__,
    )
    def test_colours_unhashable(self, colour):
        # Colours are mutable, so must not be hashable
        with pytest.raises(TypeError, match="unhashable"):
            hash(colour)
        with pytest.raises(TypeError):
            {colour}

    def test_different_color_space(self):
        assert RGB_Color(0.1, 0.2, 0.3) != RGB_Color(0.1, 0.2, 0.3, color_space=Rec2020)

    def test_different_white_point(self):
        assert LAB_Color(50.0, 0.0, 0.0) != LAB_Color(50.0, 0.0, 0.0, Whites["A"])

    def test_equal_white_point_by_value(self):
        wp = XYZ_Color(*Whites["A"].values)  # same values, different object
        assert LAB_Color(50.0, 0.0, 0.0, Whites["A"]) == LAB_Color(50.0, 0.0, 0.0, wp)

    def test_irgb(self):
        assert IRGB_Color(1, 2, 3) == IRGB_Color.from_hex_string("#010203")
        assert IRGB_Color(1, 2, 3) != IRGB_Color(1, 2, 3, color_space=Rec2020)


# --- Linear colour arithmetic ---


class TestLinearArithmetic:
    def test_add_sub(self, xyz_scalar):
        np.testing.assert_allclose((xyz_scalar + xyz_scalar).values, [0.6, 0.8, 1.0])
        np.testing.assert_allclose((xyz_scalar - xyz_scalar).values, 0)

    @pytest.mark.parametrize("scale", [2, 2.0, np.int64(2), np.float32(2)])
    def test_scalar_multiplication(self, xyz_scalar, scale):
        np.testing.assert_allclose((xyz_scalar * scale).values, [0.6, 0.8, 1.0])
        np.testing.assert_allclose((scale * xyz_scalar).values, [0.6, 0.8, 1.0])

    def test_division(self, xyz_scalar):
        np.testing.assert_allclose((xyz_scalar / 2).values, [0.15, 0.2, 0.25])

    def test_array_multiplication_broadcasts(self, xyz_scalar):
        scales = np.array([1.0, 2.0, 3.0])
        for result in (xyz_scalar * scales, scales * xyz_scalar):
            assert isinstance(result, XYZ_Color)
            assert len(result) == 3
            np.testing.assert_allclose(result[2].values, 3 * xyz_scalar.values)

    def test_image_scaling(self, xyz_image):
        scaled = xyz_image / np.full((4, 5), 2.0)
        np.testing.assert_allclose(scaled.values, xyz_image.values / 2)

    def test_add_different_types(self, xyz_scalar):
        with pytest.raises(TypeError):
            xyz_scalar + UCS_Color(0.1, 0.2, 0.3)

    def test_add_different_color_spaces(self):
        with pytest.raises(ValueError):
            RGB_Color(1.0, 1.0, 1.0) + RGB_Color(1.0, 1.0, 1.0, color_space=Rec2020)

    def test_multiply_non_numeric(self, xyz_scalar):
        with pytest.raises(TypeError):
            xyz_scalar * "a"
        with pytest.raises(TypeError):
            xyz_scalar * xyz_scalar

    def test_arithmetic_keeps_color_space(self):
        rgb = RGB_Color(0.1, 0.2, 0.3, color_space=Rec2020)
        assert (rgb * 2).color_space is Rec2020
        assert (rgb + rgb).color_space is Rec2020
        assert (rgb / 2).color_space is Rec2020


# --- XYZ ---


class TestXYZ:
    def test_chromaticity_constructor(self):
        xyz = XYZ_Color(0.3, 0.4)
        assert xyz.Z == pytest.approx(0.3)

    def test_normalize(self, xyz_vector):
        np.testing.assert_allclose(xyz_vector.normalize().values.sum(axis=0), 1)

    def test_chromaticity_properties(self, xyz_vector):
        np.testing.assert_allclose(xyz_vector.x + xyz_vector.y + xyz_vector.z, 1)

    def test_normalize_Y1(self, xyz_vector):
        np.testing.assert_allclose(xyz_vector.normalize_Y1().Y, 1)

    def test_from_xyY_round_trip(self, xyz_vector):
        rebuilt = XYZ_Color.from_xyY(xyz_vector.x, xyz_vector.y, xyz_vector.Y)
        np.testing.assert_allclose(rebuilt.values, xyz_vector.values)


# --- RGB ---


class TestRGB:
    def test_white_to_d65(self):
        # sRGB (1,1,1) is D65 normalised to Y=1
        XYZ = RGB_Color(1.0, 1.0, 1.0).to_XYZ()
        np.testing.assert_allclose(XYZ.values, [0.9505, 1.0, 1.0891], atol=2e-4)

    @pytest.mark.parametrize("color_space", [sRGB, Rec2020, DCI_P3_DCI])
    def test_xyz_round_trip(self, xyz_image, color_space):
        rgb = RGB_Color.from_XYZ(xyz_image, color_space=color_space)
        assert rgb.values.shape == xyz_image.values.shape
        np.testing.assert_allclose(rgb.to_XYZ().values, xyz_image.values)

    def test_from_xyz_brightest(self, xyz_vector):
        rgb = RGB_Color.from_XYZ(xyz_vector, brightest=True)
        assert rgb.values.max() == pytest.approx(1.0)

    def test_from_grey(self):
        np.testing.assert_array_equal(RGB_Color.from_grey(0.5).values, 0.5)

    def test_chromaticity(self):
        rgb = RGB_Color(1.0, 2.0, 1.0)
        assert (rgb.r, rgb.g, rgb.b) == pytest.approx((0.25, 0.5, 0.25))

    def test_in_gamut(self):
        rgb = RGB_Color(np.array([0.5, -0.1]), np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        np.testing.assert_array_equal(rgb.in_gamut(), [True, False])

    @pytest.mark.parametrize("method", list(ClipMethod))
    def test_clip_to_gamut(self, method):
        rgb = RGB_Color(
            np.array([-0.2, 0.5, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.8, 0.5, 0.0]),
            color_space=Rec2020,
        )
        clipped = rgb.clip_to_gamut(method)
        assert np.all(clipped.in_gamut())
        assert clipped.color_space is Rec2020
        np.testing.assert_allclose(clipped[1].values, 0.5)  # in-gamut unchanged
        np.testing.assert_allclose(clipped[2].values, 0.0)  # black stays black

    def test_clip_add_white_preserves_max(self):
        clipped = RGB_Color(-0.2, 0.5, 0.8).clip_to_gamut(ClipMethod.ADD_WHITE)
        assert clipped.values.min() == pytest.approx(0)
        assert clipped.values.max() == pytest.approx(0.8)

    def test_clip_invalid_method(self):
        with pytest.raises(ValueError):
            RGB_Color(-0.2, 0.5, 0.8).clip_to_gamut("bogus")

    def test_clip_in_gamut_is_noop(self):
        rgb = RGB_Color(0.1, 0.2, 0.3)
        assert rgb.clip_to_gamut() is rgb

    def test_getitem_keeps_color_space(self):
        rgb = RGB_Color(np.ones(3), np.ones(3), np.ones(3), color_space=Rec2020)
        assert rgb[0].color_space is Rec2020

    def test_invalid_color_space(self):
        with pytest.raises(TypeError):
            RGB_Color(0.1, 0.2, 0.3, color_space="sRGB")


# --- IRGB ---


class TestIRGB:
    @pytest.mark.parametrize(
        "rgb, expected",
        [((1.0, 1.0, 1.0), "#FFFFFF"), ((0.0, 0.0, 0.0), "#000000"),
         ((0.18, 0.18, 0.18), "#767676"), ((1.0, 0.0, 0.0), "#FF0000")],
    )
    def test_known_srgb_values(self, rgb, expected):
        assert RGB_Color(*rgb).to_IRGB().to_string() == expected

    def test_out_of_range_rgb_is_clipped(self):
        assert RGB_Color(2.0, 0.5, 0.5).to_IRGB().R == 255

    def test_to_string_array(self):
        irgb = IRGB_Color(np.array([255, 0]), np.array([0, 0]), np.array([0, 255]))
        np.testing.assert_array_equal(irgb.to_string(), ["#FF0000", "#0000FF"])

    @pytest.mark.parametrize(
        "hex_string, color_space",
        [("#12AB3F", sRGB), ("#3FF2001A0", Rec2100), ("#123ABCFFF", DCI_P3_DCI)],
    )
    def test_hex_round_trip(self, hex_string, color_space):
        irgb = IRGB_Color.from_hex_string(hex_string, color_space=color_space)
        assert irgb.to_string() == hex_string

    @pytest.mark.parametrize("bad", ["12AB3F", "#12AB3", "#12AB3F0"])
    def test_hex_wrong_format(self, bad):
        with pytest.raises(ValueError):
            IRGB_Color.from_hex_string(bad)

    def test_hex_not_string(self):
        with pytest.raises(TypeError):
            IRGB_Color.from_hex_string(0x12AB3F)

    @pytest.mark.parametrize("values", [(256, 0, 0), (-1, 0, 0)])
    def test_out_of_range(self, values):
        with pytest.raises(ValueError):
            IRGB_Color(*values)

    def test_higher_bit_depth_range(self):
        IRGB_Color(1023, 0, 0, color_space=Rec2100)
        with pytest.raises(ValueError):
            IRGB_Color(1024, 0, 0, color_space=Rec2100)

    def test_no_linear_arithmetic(self):
        irgb = IRGB_Color(100, 100, 100)
        with pytest.raises(TypeError):
            irgb * 2
        with pytest.raises(TypeError):
            irgb + irgb

    @pytest.mark.parametrize("color_space", [sRGB, Rec2020, Rec2100, DCI_P3_DCI])
    def test_rgb_round_trip(self, color_space):
        irgb = IRGB_Color(*rng.integers(0, color_space.max_int + 1, size=(3, 20)),
                          color_space=color_space)
        back = irgb.to_RGB().to_IRGB()
        np.testing.assert_array_equal(back.values, irgb.values)
        assert back.color_space is color_space

    def test_to_rgb_luminance(self):
        rgb = IRGB_Color(255, 255, 255).to_RGB(to_luminance=True)
        np.testing.assert_allclose(rgb.values, sRGB.peak_luminance.to_value(u.cd / u.m**2))

    def test_from_numpy_image(self):
        img = rng.integers(0, 256, size=(4, 5, 3))
        irgb = IRGB_Color.from_numpy_image(img)
        assert irgb.values.shape == (3, 4, 5)
        np.testing.assert_array_equal(irgb.values_image, img)

    @pytest.mark.parametrize(
        "img, error",
        [
            (np.zeros((4, 5, 3)).tolist(), TypeError),
            (np.zeros((4, 5), dtype=int), ValueError),
            (np.zeros((4, 5, 3), dtype=float), ValueError),
            (np.full((4, 5, 3), 256), ValueError),
        ],
        ids=["list", "2d", "float", "range"],
    )
    def test_from_numpy_image_validation(self, img, error):
        with pytest.raises(error):
            IRGB_Color.from_numpy_image(img)

    def test_values_image_vector(self):
        irgb = IRGB_Color(np.arange(5), np.arange(5), np.arange(5))
        assert irgb.values_image.shape == (1, 5, 3)

    def test_from_grey(self):
        np.testing.assert_array_equal(IRGB_Color.from_grey(128).values, 128)


# --- Perceptually uniform spaces ---


class TestLabLuvUcs:
    def test_lab_white_and_black(self):
        white = LAB_Color.from_XYZ(RGB_Color(1.0, 1.0, 1.0).to_XYZ())
        np.testing.assert_allclose(white.values, [100, 0, 0], atol=1e-6)
        black = LAB_Color.from_XYZ(XYZ_Color(0.0, 0.0, 0.0))
        np.testing.assert_allclose(black.values, 0, atol=1e-10)

    def test_lab_srgb_red(self):
        # Reference value for sRGB red (D65): L*=53.24, a*=80.09, b*=67.20
        lab = LAB_Color.from_XYZ(RGB_Color(1.0, 0.0, 0.0).to_XYZ())
        np.testing.assert_allclose(lab.values, [53.24, 80.09, 67.20], atol=0.01)

    def test_luv_white(self):
        luv = LUV_Color.from_XYZ(RGB_Color(1.0, 1.0, 1.0).to_XYZ())
        np.testing.assert_allclose(luv.values, [100, 0, 0], atol=1e-6)

    @pytest.mark.parametrize("model", [LAB_Color, LUV_Color])
    def test_round_trip(self, model, xyz_image):
        back = model.from_XYZ(xyz_image).to_XYZ()
        np.testing.assert_allclose(back.values, xyz_image.values, rtol=1e-10)

    @pytest.mark.parametrize("model", [LAB_Color, LUV_Color])
    def test_round_trip_dark(self, model):
        # Exercise the linear segment of f(t) near black
        dark = XYZ_Color(0.001, 0.002, 0.003)
        np.testing.assert_allclose(model.from_XYZ(dark).to_XYZ().values, dark.values)

    @pytest.mark.parametrize("model", [LAB_Color, LUV_Color])
    def test_white_point_kept(self, model, xyz_vector):
        colour = model.from_XYZ(xyz_vector, white_point=Whites["A"])
        assert colour.white_point is Whites["A"]
        assert colour[0].white_point is Whites["A"]
        np.testing.assert_allclose(colour.to_XYZ().values, xyz_vector.values)

    def test_white_point_type_checked(self):
        with pytest.raises(TypeError):
            LAB_Color(50.0, 0.0, 0.0, white_point=(0.3, 0.3))

    def test_ucs_round_trip(self, xyz_image):
        back = UCS_Color.from_XYZ(xyz_image).to_XYZ()
        np.testing.assert_allclose(back.values, xyz_image.values)

    def test_ucs_uv_matches_luv_prime(self, xyz_vector):
        # CIE 1960 (u, v) relates to 1976 (u', v') by u = u', v = 2v'/3
        ucs = UCS_Color.from_XYZ(xyz_vector)
        u_prime, v_prime = LUV_Color.uv_primes_from_XYZ(xyz_vector)
        np.testing.assert_allclose(ucs.u, u_prime)
        np.testing.assert_allclose(ucs.v, 2 * v_prime / 3)
