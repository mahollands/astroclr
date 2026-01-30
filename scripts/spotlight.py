"""
A spotlight undergoing chromatic abberation.

Since the sRGB primaries are not monochromatic, no monochromatic colour can be
correctly displayed in the sRGB colour space. This results in monochromatic
colours having at least one negative RGB value that must be compensated for.

I mix all colours with the minimum amount of grey needed to just move all colours
within the sRGB gamut. The results can be thought of as a rainbow projected onto
an otherwise dimly illuminated white wall.
"""

import numpy as np
import numba
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from colorpy2.ciexyz import XYZ_from_wavelength, CMFs
from colorpy2.colormodels import RGB_Color

# from colorpy2.transfer_functions import sRGB_EOTF
from mh.spectra import Spectrum

CMF = CMFs["1931"]

N_PX = 200
SPOT_EXPONENT = 20
SPOT_RADIUS = 30
PX_PER_LAM = 0.10
USE_SPEC = False
SPEC_NAME = "/home/mark/GD492/J1637/models/J1637-fit-2024.dk"

XYZs = XYZ_from_wavelength(CMF.wavelength)
RGBs = [RGB_Color.from_XYZ(XYZ)._values for XYZ in XYZs]
lams = 360 - 555 + np.arange(len(RGBs))


# OETF = numba.vectorize(sRGB_EOTF().inv)
@numba.vectorize(cache=True)
def OETF(x):
    break_point = 0.0031308
    linear = 12.92 * x
    gamma = 1.055 * x ** (1 / 2.4) - 0.055
    return linear if x <= break_point else gamma


if USE_SPEC:
    S = Spectrum.from_dk(SPEC_NAME)
    S.x_unit_to("nm")
    S = S.convolve_gaussian(2.0)
    S = S.interp(CMF.wavelength)
    XYZs = [XYZ * f for XYZ, f in zip(XYZs, S.y)]

ax = np.arange(N_PX) - N_PX / 2
x, y = np.meshgrid(ax, ax)


@numba.njit(cache=True)
def mk_img(disp, RGBs, lams):
    img_RGB = np.zeros((N_PX, N_PX, 3))
    # precompute spotlight pattern
    r2 = (x * x + y * y) / (SPOT_RADIUS * SPOT_RADIUS)
    spotlight = np.exp(-(r2 ** (SPOT_EXPONENT / 2)))
    for RGBarr, lam in zip(RGBs, lams):
        spot_shifted = np.roll(spotlight, int(lam * disp))
        img_RGB += spot_shifted[:, :, None] * RGBarr[None, None, :]
    img_RGB -= img_RGB.min()
    img_RGB /= img_RGB.max()
    img_IRGB = OETF(img_RGB)
    return img_IRGB


def mk_figure():
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.15)
    ax_sl = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    slider = Slider(
        ax=ax_sl,
        label="dispersion",
        valmin=0.0,
        valmax=5 * PX_PER_LAM,
        valinit=PX_PER_LAM,
    )

    img_IRGB = mk_img(PX_PER_LAM, RGBs, lams)
    img = ax.imshow(img_IRGB, aspect="auto")

    def update_plot(val):
        img_IRGB = mk_img(val, RGBs, lams)
        img.set_array(img_IRGB)

    slider.on_changed(update_plot)
    plt.show()


mk_figure()
