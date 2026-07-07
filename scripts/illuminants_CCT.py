from astroclr.delta_color import CCT
from astroclr.illuminants import *

A = get_illuminant_A()
E = get_illuminant_E()
D65 = get_illuminant_D65()

print('A', CCT(A.to_XYZ()))
print('E', CCT(E.to_XYZ()))
print('D65', CCT(D65.to_XYZ()))
