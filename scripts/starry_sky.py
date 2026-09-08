"""
Generate a colour accurate image of a starry sky, with a few assumptions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astroclr import XYZ_Color, RGB_Color

N_stars = 1000
N_im = 1000
D_scl = 300.0
SIGMA = 2.0
L_scale = 100.0

# Empirically determined polynomials to convert between BpRp and CIE 1931
# chromaticity for black bodies.
Px = np.polynomial.Polynomial(
    coef=[0.43582307, 0.19917189, -0.02513542, -0.03552102, 0.01025754],
    domain=[-0.25968806, 4.67501896],
)
Py = np.polynomial.Polynomial(
    coef=[0.4036012, 0.07160398, -0.1083138, -0.00348626, 0.03068216],
    domain=[-0.25968806, 4.67501896],
)

# In the Gaia HRD, the main sequence very roughly spans 0-4 in BpRp which very
# roughly corresponds to 0-14 in G_abs. The distance distribution is roughly
# correct for the thin disc.
BpRp = np.random.uniform(0, 4, size=N_stars)
G_abs = 14 / 4 * BpRp
D = np.random.gamma(3, scale=D_scl, size=N_stars)
G = G_abs + 5 * np.log10(D) - 5

#Chromaticities from BpRp, and luminance from G
x, y = Px(BpRp), Py(BpRp)
Y = 10 ** (-0.4 * G)

XYZs = XYZ_Color.from_xyY(x, y, Y)
RGBs = RGB_Color.from_XYZ(XYZs)

# Generate image where each star fills a pixel
x_pos, y_pos = np.random.randint(0, N_im, (2, N_stars))
R_im, G_im, B_im = np.zeros((3, N_im, N_im))
for i, rgb in enumerate(RGBs):
    R_im[x_pos[i], y_pos[i]] = rgb.R
    G_im[x_pos[i], y_pos[i]] = rgb.G
    B_im[x_pos[i], y_pos[i]] = rgb.B

# Simulate instrumental resolution
R_im = gaussian_filter(R_im, SIGMA)
G_im = gaussian_filter(G_im, SIGMA)
B_im = gaussian_filter(B_im, SIGMA)

# scale to simulate overexposure and convert to display RGB
RGB_im = RGB_Color(R_im, G_im, B_im)
RGB_im *= L_scale / RGB_im.values.max()
IRGB_im = RGB_im.to_IRGB()

plt.figure(figsize=(N_im / 100, N_im / 100))
plt.imshow(IRGB_im.values_image)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
