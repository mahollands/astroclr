"""
Plots a rainbow of colour as a function of wavelength, showing the wavelength
sensitivity of the eye.

Since the sRGB primaries are not monochromatic, no monochromatic colour can be
correctly displayed in the sRGB colour space. This results in monochromatic
colours having at least one negative RGB value that must be compensated for.

Here we mix all colours with the minimum amount of grey needed to just move all
colours within the sRGB gamut. The results can be thought of as a rainbow
projected onto an otherwise dimly illuminated white wall.
"""

import numpy as np
import matplotlib.pyplot as plt
from astroclr.spectral import CMFs, XYZ_from_wavelength
from astroclr.colormodels import RGB_Color, IRGB_Color
from astroclr.RGB_color_spaces import sRGB, Rec2020
from astroclr.plots import chromaticity_diagram


WRITE = False
CSPACE = sRGB
#CSPACE = Rec2020 #more vibrant but less realistic
CMF = "1931"

wavelengths = CMFs[CMF].wavelength.value
XYZs = XYZ_from_wavelength(wavelengths, CMFname=CMF)
RGBs = RGB_Color.from_XYZ(XYZs, color_space=CSPACE)

RGBs.values -= RGBs.values.min()
RGBs /= RGBs.values.max()
IRGBs = RGBs.to_IRGB()

rect_x = np.array([-0.5, 0.5, 0.5, -0.5])
rect_y = np.array([0, 0, 1, 1])
for lam, IRGB in zip(wavelengths, IRGBs):
    plt.fill(rect_x + lam, rect_y, c=IRGB.to_string())
plt.xlim(380, 720)
plt.ylim(0, 1)
plt.xlabel("Wavelength [nm]")
plt.savefig("rainbow.png")
plt.show()

plt.plot(wavelengths, RGBs.R, "r-")
plt.plot(wavelengths, RGBs.G, "g-")
plt.plot(wavelengths, RGBs.B, "b-")
plt.title("Linear RGB")
plt.xlim(380, 720)
plt.xlabel("Wavelength [nm]")
plt.show()

plt.title("Display RGB")
plt.plot(wavelengths, IRGBs.R, "r-")
plt.plot(wavelengths, IRGBs.G, "g-")
plt.plot(wavelengths, IRGBs.B, "b-")
plt.xlim(380, 720)
plt.xlabel("Wavelength [nm]")
plt.show()

chromaticity_diagram(color_space=CSPACE, draw_bb_locus=False)
XYZs = RGBs.to_XYZ()
plt.plot(XYZs.x, XYZs.y, 'k-')
plt.show()


if WRITE:
    with open("rainbow.dat", "w") as F:
        for lam, IRGB in zip(wavelengths, IRGBs):
            F.write(f"{lam} {IRGB.to_string()}\n")
