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
from astroclr.spectral import XYZ_from_wavelength, CMFs
from astroclr.colormodels import RGB_Color
from astroclr.RGB_color_spaces import sRGB as cspace

CMF = CMFs["1931"]
OETF = cspace.EOTF.inv

N_PX = 200
SPOT_EXPONENT = 20
SPOT_RADIUS = 30
PX_PER_LAM = 0.10

XYZs = XYZ_from_wavelength(CMF.wavelength)
RGBs = RGB_Color.from_XYZ(XYZs, color_space=cspace)
lams = 360 - 555 + np.arange(len(RGBs))

x = np.arange(N_PX) - N_PX // 2
y = np.arange(N_PX//2) - N_PX // 4
x, y = np.meshgrid(x, y)
r2 = (x * x + y * y) / (SPOT_RADIUS * SPOT_RADIUS)
spotlight = np.exp(-(r2 ** (SPOT_EXPONENT / 2)))


@numba.njit(cache=True)
def mk_img(spotlight, disp, RGBs, lams):
    img_RGB = np.zeros((N_PX//2, N_PX, 3))
    for RGBarr, lam in zip(RGBs, lams):
        spot_shifted = np.roll(spotlight, int(lam * disp))
        img_RGB += spot_shifted[:, :, None] * RGBarr[None, None, :]
    img_RGB -= img_RGB.min()
    img_RGB /= img_RGB.max()
    img_IRGB = OETF(img_RGB)
    return img_IRGB


def mk_figure():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(bottom=0.15)
    ax_sl = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    slider = Slider(
        ax=ax_sl,
        label="dispersion",
        valmin=0.0,
        valmax=5 * PX_PER_LAM,
        valinit=0,
    )

    img_IRGB = mk_img(spotlight, 0, RGBs.values.T, lams)
    img = ax.imshow(img_IRGB)

    def update_plot(val):
        img_IRGB = mk_img(spotlight, val, RGBs.values.T, lams)
        img.set_array(img_IRGB)

    slider.on_changed(update_plot)
    plt.show()


mk_figure()
