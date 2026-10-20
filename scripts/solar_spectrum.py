import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.convolution import Gaussian1DKernel, convolve
from astroclr import *

DLAM = 0.02 * u.AA
RES = 2.0 * u.AA

cs = sRGB
#cs = Rec2020 # More vibrant/less accurate on most monitors

xi = np.arange(3500, 7500, DLAM.value) * DLAM.unit
Sun = Spectrum.from_file("data/solar_model.dat", units=(u.AA, u.erg/u.s/u.cm**2/u.AA))
yi = Sun.interp_flux(xi.to(u.AA)) * Sun.flux.unit
Sun = Spectrum(xi, yi)
CCT_Sun = CCT(Sun.to_XYZ())

# Blackbody spectrum with same colour temperature as the Sun
BB_Sun = Spectrum.BlackBody(temperature=CCT_Sun, wavelength=xi)
BB_Sun.flux /= BB_Sun.flux.max().value

print(f"Solar colour temperature: {CCT_Sun:.0f}")

#Rainbow
XYZs = XYZ_from_wavelength(xi) * Sun.flux / Sun.flux.max()
RGBs = RGB_Color.from_XYZ(XYZs, color_space=cs)
RGBs -= RGB_Color.from_grey(RGBs.values.min())
RGBs /= RGBs.values.max()
IRGBs = RGBs.to_IRGB()

#Convolved spectrum
kernel = Gaussian1DKernel(stddev=RES/DLAM)
flux_conv = convolve(Sun.flux.value, kernel=kernel, boundary="extend")
flux_conv /= flux_conv.max()
sunhex = Sun.to_IRGB_string(brightest=True)

# peak wavelengths
Sun_peak_lam = 4650 * u.AA
XYZ = XYZ_from_wavelength(Sun_peak_lam)
RGB = RGB_Color.from_XYZ(XYZ, brightest=True)
hex_Sun_peak = RGB.to_IRGB().to_string()
BB_peak_lam = xi[np.argmax(BB_Sun.flux)]
XYZ = XYZ_from_wavelength(BB_peak_lam)
RGB = RGB_Color.from_XYZ(XYZ, brightest=True)
hex_BB_peak = RGB.to_IRGB().to_string()

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.imshow(IRGBs.values_image, extent=(xi[0].value, xi[-1].value, 0, 1), aspect='auto')
ax1.set_xlabel("Wavelength [AA]")
ax1.set_xlim(xi[0].value, xi[-1].value)

ax2.fill_between(xi.value, 0, flux_conv, fc=sunhex)
ax2.plot(xi, BB_Sun.flux, c='grey', ls=':', lw=0.5)
ax2.plot(xi, flux_conv, 'k-', lw=0.5)
ax2.axvline(Sun_peak_lam.value, c=hex_Sun_peak, ls='--')
ax2.axvline(BB_peak_lam.value, c=hex_BB_peak, ls='--')
ax2.set_ylabel("Flux [erg/s/cm2/AA]")
ax2.set_ylim(0, 1.1)


plt.tight_layout()
plt.show()
