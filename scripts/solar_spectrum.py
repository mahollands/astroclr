import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astroclr import *

cs = sRGB
#cs = Rec2020 # More vibrant/less accurate on most monitors

xi = np.arange(350, 750, 0.01) * u.nm
Sun = Spectrum.from_file("data/solar_model.dat", units=(u.AA, u.erg/u.s/u.cm**2/u.AA))
yi = Sun.interp_flux(xi.to(u.AA))
Sun = Spectrum(xi, yi)

XYZs = XYZ_from_wavelength(xi) * Sun.flux / Sun.flux.max()
RGBs = RGB_Color.from_XYZ(XYZs, color_space=cs)
RGBs -= RGB_Color.from_grey(RGBs.values.min())
RGBs /= RGBs.values.max()
IRGBs = RGBs.to_IRGB()

plt.imshow(IRGBs.values_image, extent=(xi[0].value, xi[-1].value, 0, 1), aspect='auto')
plt.show()
