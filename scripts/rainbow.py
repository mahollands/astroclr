"""
Plots a rainbow of colour as a function of wavelength, showing the wavelength
sensitivity of the eye.

Since the sRGB primaries are not monochromatic, no monochromatic colour can be
correctly displayed in the sRGB colour space. This results in monochromatic
colours having at least one negative RGB value that must be compensated for.

I mix all colours with the minimum amount of grey needed to just move all colours
within the sRGB gamut. The results can be thought of as a rainbow projected onto
an otherwise dimly illuminated white wall.
"""

import numpy as np
import matplotlib.pyplot as plt
from colorpy2.ciexyz import CMFs, XYZ_from_wavelength
from colorpy2.colormodels import RGB_Color, IRGB_Color, ColorList
from colorpy2.RGB_color_spaces import sRGB


WRITE = False
CSPACE = sRGB

wavelengths = CMFs["1931"].wavelength.value
XYZs = XYZ_from_wavelength(wavelengths)
RGBs = [RGB_Color.from_XYZ(XYZ, color_space=CSPACE) for XYZ in XYZs]

min_flux = min(RGB._values.min() for RGB in RGBs)
for RGB in RGBs:
    RGB._values -= min_flux

norm = max(RGB._values.max() for RGB in RGBs)
RGBs = [RGB / norm for RGB in RGBs]
IRGBs = [RGB.to_IRGB() for RGB in RGBs]
IRGBs = [IRGB_Color(*IRGB._values) for IRGB in IRGBs]
print(IRGBs[-1])

rect_x = np.array([-0.5, 0.5, 0.5, -0.5])
rect_y = np.array([0, 0, 1, 1])
for lam, IRGB in zip(wavelengths, IRGBs):
    plt.fill(rect_x + lam, rect_y, c=IRGB.to_string())
plt.xlim(380, 720)
plt.ylim(0, 1)
plt.savefig("rainbow.png")
plt.show()

plt.plot(wavelengths, [RGB.r for RGB in RGBs], "r-")
plt.plot(wavelengths, [RGB.g for RGB in RGBs], "g-")
plt.plot(wavelengths, [RGB.b for RGB in RGBs], "b-")
plt.xlim(380, 720)
plt.show()

plt.plot(wavelengths, [IRGB.r for IRGB in IRGBs], "r-")
plt.plot(wavelengths, [IRGB.g for IRGB in IRGBs], "g-")
plt.plot(wavelengths, [IRGB.b for IRGB in IRGBs], "b-")
plt.xlim(380, 720)
plt.show()

XYZs = XYZs.normalize()

plt.plot(XYZs.x, XYZs.y)
XYZs = ColorList([RGB.to_XYZ() for RGB in RGBs]).normalize()
plt.plot(XYZs.x, XYZs.y)
plt.plot(
    [CSPACE.R.x, CSPACE.G.x, CSPACE.B.x, CSPACE.R.x],
    [CSPACE.R.y, CSPACE.G.y, CSPACE.B.y, CSPACE.R.y],
    "k:",
)
plt.plot(CSPACE.R.x, CSPACE.R.y, "ro")
plt.plot(CSPACE.G.x, CSPACE.G.y, "go")
plt.plot(CSPACE.B.x, CSPACE.B.y, "bo")
plt.show()


if WRITE:
    with open("rainbow.dat", "w") as F:
        for lam, IRGB in zip(wavelengths, IRGBs):
            F.write(f"{lam} {IRGB.to_string()}\n")
