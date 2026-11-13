"""
Smoke tests for astroclr.plots: each plotting function should run without
error and draw something. Figures use the headless Agg backend (conftest.py).
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u

from astroclr import plots, illuminants
from astroclr.colormodels import ClipMethod, RGB_Color
from astroclr.spectral import Spectrum
from astroclr.rgb_color_spaces import sRGB, Rec2020


def _has_content():
    fig = plt.gcf()
    return any(ax.has_data() for ax in fig.axes)


def test_color_vs_param_plot():
    params = np.linspace(0, 1, 6)
    RGBs = RGB_Color.from_grey(params)
    plots.color_vs_param_plot(params, RGBs, "test")
    assert len(plt.gcf().axes) == 2
    assert _has_content()


def test_visible_spectrum_plot():
    plots.visible_spectrum_plot()
    assert _has_content()


@pytest.mark.parametrize("CMF", ["1931", "2015_2deg"])
def test_color_matching_functions_plot(CMF):
    plots.color_matching_functions_plot(CMF=CMF)
    assert len(plt.gcf().axes) == 3
    assert CMF in plt.gcf().axes[0].get_title()


@pytest.mark.parametrize("color_space", [sRGB, Rec2020])
@pytest.mark.parametrize("clip_method", list(ClipMethod))
@pytest.mark.parametrize("brightest", [True, False])
def test_chromaticity_diagram(color_space, clip_method, brightest):
    plots.chromaticity_diagram(
        color_space=color_space,
        clip_method=clip_method,
        brightest=brightest,
        N_pixels=40,
    )
    assert _has_content()


def test_chromaticity_diagram_minimal():
    plots.chromaticity_diagram(
        draw_cie_space=False, draw_bb_locus=False, draw_rgb_triangle=False
    )
    assert _has_content()


@pytest.fixture
def d65():
    return illuminants.get_illuminant_D65()


def test_spectrum_subplot(d65):
    plots.spectrum_subplot(d65)
    ax = plt.gca()
    assert ax.has_data()
    # one coloured patch per wavelength interval, plus the black outline
    assert len(ax.patches) == len(d65) - 1
    assert len(ax.lines) == 1


def test_spectrum_plot(d65):
    plots.spectrum_plot(d65, "D65", ylabel="Flux")
    fig = plt.gcf()
    top, bottom = fig.axes
    assert top.get_title() == "D65"
    # the top patch is filled with the spectrum's own colour
    patch_colour = top.patches[0].get_facecolor()
    expected = tuple(int(c, 16) / 255 for c in ("FF", "FF", "FF"))
    assert patch_colour[:3] == pytest.approx(expected, abs=1 / 255)
    assert bottom.get_ylabel() == "Flux"
    assert bottom.get_xlim() == pytest.approx(
        (d65.wavelength[0].value, d65.wavelength[-1].value)
    )


def test_spectrum_plot_non_nm_units():
    # Vega is in Angstrom; this range extends past the CMFs at both ends.
    # (Clipped because plotting all ~9000 points takes several seconds.)
    S = Spectrum.load_Vega().clip(3000 * u.AA, 9000 * u.AA)
    plots.spectrum_plot(S, "Vega")
    assert _has_content()


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_spectrum_plot_invisible_spectrum():
    # Entirely outside the CMFs: every wavelength maps to black, which must
    # not trigger a 0/0 normalisation.
    S = Spectrum(np.linspace(1000, 2000, 50), 1.0)
    plots.spectrum_plot(S, "infrared")
    assert _has_content()
    top, bottom = plt.gcf().axes
    assert top.patches[0].get_facecolor()[:3] == (0, 0, 0)
    assert all(p.get_facecolor()[:3] == (0, 0, 0) for p in bottom.patches)


def test_illuminant_figures(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    illuminants.figures()
    assert plt.get_fignums()
