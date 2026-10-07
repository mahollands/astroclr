"""
Luminous calculations for the Sun, assuming a 5900 K black body.
"""
import numpy as np
from astropy import units as u
from astropy.constants import sigma_sb
from astroclr import Spectrum, CCT

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
print("Chromatic quantities of the Sun:")
XYZ = S.to_XYZ()
rgb_string = S.to_IRGB_string(brightest=True)
colour_temp = CCT(XYZ) #We're using a black body, so we better get 5900 K again

print(f"Solar chromaticity: {XYZ.normalize():.3f}")
print(f"Solar RGB hex: {rgb_string}")
print(f"Solar colour temperature: {colour_temp:.0f}")


################################################################################
print("\nLuminous quantities at the Sun's surface/independent of distance:")
luminance = S.to_luminance()
exitance = luminance * solid_angle_factor
luminous_flux = exitance * solar_surface_area
luminous_intensity = luminous_flux / (4*np.pi * u.sr)
power_density = sigma_sb * T_sun**4
luminous_efficacy = exitance / power_density
luminous_efficiency = luminous_efficacy / K_cd_max

print(f"Solar luminance: {luminance.to(u.cd/u.m**2):.3e}")
print(f"Solar exitance: {exitance.to(u.lm/u.m**2):.3e}")
print(f"Solar luminous flux: {luminous_flux.to(u.lm):.2e}")
print(f"Solar luminous intensity: {luminous_intensity.to(u.cd):.2e}")
print(f"Solar luminous efficacy: {luminous_efficacy.to(u.lm/u.W):.1f}")
print(f"Solar luminous efficiency: {luminous_efficiency.to(u.percent):.1f}")


################################################################################
print("\nLuminous quantities at 1 au")
D = 1 * u.au
S.flux *= np.pi * u.sr * (u.Rsun/D)**2
# The illuminance is given in lm/m**2 like the exitance, however here lux can
# be used instead. The to_luminance method can be used if flux units have no
# directional dependence (no steradian**-1).
illuminance = S.to_luminance() 

print(f"Solar exitance: {illuminance.to(u.lux):.2e}")
print(f"Solar visual magnitude: {S.visual_magnitude():.2f}")


################################################################################
print("\nLuminous quantities at 10 pc")
D = 10* u.pc
S.flux *= (u.au/D)**2
illuminance = S.to_luminance()

print(f"Solar exitance: {illuminance.to(u.lux):.2e}")
print(f"Solar visual magnitude: {S.visual_magnitude():.2f}")
