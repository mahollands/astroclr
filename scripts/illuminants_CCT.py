"""
Script that calculates the correlated colour-temperature for
several of the standard illuminants:

A - Black body spectrum of T=2856K
E - flat spectrum in F_lambda
D65 - standard daylight spectrum with colour temperature tuned to 6504K
"""
from astroclr.delta_color import CCT
from astroclr.illuminants import *

illums = {
    "A" : get_illuminant_A(),
    "E" : get_illuminant_E(),
    "D65" : get_illuminant_D65(),
}

print("Colour temperatures of standard illuminants:")
for name, illum in illums.items():
    cct = CCT(illum.to_XYZ())
    print(f"{name}: {cct:.1f}")
