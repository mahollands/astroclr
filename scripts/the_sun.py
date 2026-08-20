"""
Luminous calculations for the Sun, assuming a 5900 K black body.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import sigma_sb
from astroclr import Spectrum

T_sun = 5900 * u.K
K_cd_max = 683.002 * u.lm/u.W
S = Spectrum.BlackBody(T_sun)

# Light radiated over 2pi sr, with constant instensity as a function
# of mu (for a black body -- really there should be limb darkening).
# There is also a cosine factor averaging to half to consider, so
# we need to multiply by pi sr.
solid_angle_factor = 0.5 * 2*np.pi * u.sr
solar_surface_area = 4*np.pi*u.Rsun**2

################################################################################
print("Luminous quantities at the Sun's surface/independent of distance:")
luminance = S.to_luminance()
exitance = luminance * solid_angle_factor
luminous_flux = exitance * solar_surface_area
luminous_intensity = luminous_flux / (4*np.pi * u.sr)
power_density = sigma_sb * T_sun**4
luminous_efficacy = exitance / power_density
luminous_efficiency = luminous_efficacy / K_cd_max

print("Solar luminance:", luminance)
print("Solar exitance:", exitance.to(u.lm/u.m**2))
print("Solar luminous flux:", luminous_flux.to(u.lm))
print("Solar luminous intensity:", luminous_intensity.to(u.cd))
print("Solar luminous efficacy:", luminous_efficacy.to(u.lm/u.W))
print("Solar luminous efficiency:", luminous_efficiency.to(u.percent))


################################################################################
print("\nLuminous quantities at 1 au")
D = 1 * u.au
S.flux *= np.pi * u.sr * (u.Rsun/D)**2
# The illuminance is given in lm/m**2 like the exitance, however here lux can
# be used instead. The to_luminance method can be used if flux units have no
# directional dependence (no steradian**-1).
illuminance = S.to_luminance() 

print("Solar exitance:", illuminance.to(u.lux))
print("Solar visual magnitude:", S.visual_magnitude())


################################################################################
print("\nLuminous quantities at 10 pc")
D = 10* u.pc
S.flux *= (u.au/D)**2
illuminance = S.to_luminance()

print("Solar exitance:", illuminance.to(u.lux))
print("Solar visual magnitude:", S.visual_magnitude())
