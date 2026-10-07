"""
Plots a blackbody scale from 800K (roughly the Draper point), up
to 30000K. For low temperatures, where a blackbody may be less luminous
than a typical monitor luminance of 200 cd/m2, the displayed intensity
is approximately correct. To avoid hard clipping at 200 cd/m2, a
basic form of tone mapping adds some amount of roll-off, giving
a smoother luminance scale.
"""

import numpy as np
import matplotlib.pyplot as plt
from astroclr import Spectrum, RGB_Color

REF_LUM = 200 #Display peak-luminance in cd/m2
T_LIMITS = 800, 30000

Ts = np.geomspace(*T_LIMITS, 256)
dlogT = np.diff(np.log(Ts)).mean()

y = [0, 0, 1, 1]
dlogx = np.array([-1, 1, 1, -1])
dx = np.exp(dlogT * dlogx/2)

for T in Ts:
    BB = Spectrum.BlackBody(T)
    XYZ = BB.to_XYZ() * 683 / REF_LUM
    RGB = RGB_Color.from_XYZ(XYZ)
    # tonemapping to avoid hard edge from luminance clipping
    RGB /= 1 + np.max(RGB.values)

    clr = RGB.to_IRGB().to_string()
    plt.fill(T * dx, y, c=clr)

plt.semilogx()
plt.xlim(*T_LIMITS)
plt.ylim(0, 1)
plt.xticks([1000, 2000, 5000, 10000, 20000], [1, 2, 5, 10, 20])
plt.yticks(visible=False)
plt.xlabel("$T$ [K]")
plt.title("Blackbody scale")
plt.show()
