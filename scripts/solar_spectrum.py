import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.convolution import Gaussian1DKernel, convolve
from astroclr import *

DLAM = 0.02 * u.AA
RES = 0.5 * u.AA

cs = sRGB
#cs = Rec2020 # More vibrant/less accurate on most monitors

xi = np.arange(3500, 7500, DLAM.value) * DLAM.unit
Sun = Spectrum.from_file("data/solar_model.dat", units=(u.AA, u.erg/u.s/u.cm**2/u.AA))
yi = Sun.interp_flux(xi.to(u.AA)) * Sun.flux.unit
Sun = Spectrum(xi, yi)
print("Solar colour temperature:", CCT(Sun))

XYZs = XYZ_from_wavelength(xi) * Sun.flux / Sun.flux.max()
RGBs = RGB_Color.from_XYZ(XYZs, color_space=cs)
RGBs -= RGB_Color.from_grey(RGBs.values.min())
RGBs /= RGBs.values.max()
IRGBs = RGBs.to_IRGB()

kernel = Gaussian1DKernel(stddev=RES/DLAM)
flux_conv = convolve(Sun.flux.value, kernel=kernel, boundary="extend")
sunhex = Sun.to_IRGB_string(brightest=True)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.fill_between(Sun.wavelength.value, 0, flux_conv, fc=sunhex)
ax1.plot(Sun.wavelength, flux_conv, 'k-', lw=0.5)
ax1.set_ylabel("Flux [erg/s/cm2/AA]")
ax1.set_ylim(0, 1.1*Sun.flux.max().value)

ax2.imshow(IRGBs.values_image, extent=(xi[0].value, xi[-1].value, 0, 1), aspect='auto')
ax2.set_xlabel("Wavelength [AA]")
ax2.set_xlim(Sun.wavelength[0].value, Sun.wavelength[-1].value)

plt.tight_layout()
plt.show()
