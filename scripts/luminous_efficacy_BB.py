"""
The luminous efficacy determines how much luminous flux is emitted for a given
amount of radiated power, measured in lm/W. Equivalently this can be thought
of as the ratio between the luminance and radiance of a source. For an ideal
monochromatic source at 555nm, this reaches a maximum of 683 lm/W, via
the definition of the lumen.

This script plots the luminous efficacy as a function of black body temperature.
At low temperatures, much of radiation is emitted in the infrared, and so the
efficacy is low. At temperatures similar to the Sun, most of the flux is in the
optical range, peaking at roughly 95 lm/W at 6600 K. For hotter temperatures
flux moves into the ultraviolet range, and so the luminous efficacy drops off.

Since LEDs are not Blackbody emitters, they can generate "white" light at
higher luminous efficacy than even Sunlight, though at the expense of colour
rendering index.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import sigma_sb
from astroclr import Spectrum

Ts = np.arange(1000, 20001, 100) * u.K
BBs = [Spectrum.BlackBody(T) for T in Ts]

luminance = u.Quantity([BB.to_luminance() for BB in BBs])
radiance = sigma_sb * Ts**4 / (np.pi * u.sr)

lum_eff = luminance / radiance


plt.figure("Blackbody luminous efficacies")
plt.plot(Ts, lum_eff)
plt.xlabel("Temperature [K]")
plt.ylabel("Luminous efficacy [lm/W]")
plt.tight_layout()
plt.show()
