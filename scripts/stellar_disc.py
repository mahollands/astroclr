from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astroclr import Spectrum, RGB_Color

N = 1000
N_AA = 2 # antialiasing factor
DPI = 100
xylim = 1.2
DATA = "data/WDJ1330+6435_mu.dat"
#DATA = "data/WDJ1011+2845_mu.dat"
LIMB_MODEL = "claret"

# See Claret et al., A&A 634, 93 (2020)
limb_basis = {
    "linear": lambda mu: [1 - mu],
    "quad": lambda mu: [(1 - mu) ** n for n in (1, 2)],
    "sqrt": lambda mu: [(1 - mu), (1 - np.sqrt(mu))],
    "log": lambda mu: [(1 - mu), (mu * np.log(mu))],
    "claret": lambda mu: [(1 - mu ** (k / 2)) for k in range(1, 5)],
}


def limb_interpolator(mu_i, vec, scale):
    basis = limb_basis[LIMB_MODEL](np.abs(mu_i))
    I_mu = 1 - sum(a * f for a, f in zip(vec, basis))
    I_mu = np.clip(scale * I_mu, 0, 1)
    I_mu = np.where(mu_i < 0, 0, I_mu)
    return I_mu


def fit_limb(mu_i, C):
    basis = np.array(limb_basis[LIMB_MODEL](mu_i))
    scale = max(C)
    y_i = C / scale
    matrix, y_sums = basis @ basis.T, basis @ (1 - y_i)
    vec = np.linalg.solve(matrix, y_sums)
    return partial(limb_interpolator, vec=vec, scale=scale)


# Read in mu from file header
with open(DATA) as F:
    for i in range(3):
        line = next(F)
mu = np.array([float(word) for word in line.split()[1:]])

# Read in spectral fluxes as a function of mu
flux_unit = u.erg / u.s / u.cm**2 / u.sr / u.AA
wavelength, *fluxes = np.loadtxt(DATA, unpack=True)
MM = [
    Spectrum(wavelength * u.AA, flux * u.erg / u.s / u.cm**2 / u.sr / u.AA)
    for flux in fluxes
]

# Determine RGB interpolating functions
RGB = RGB_Color.from_list([M.to_RGB() for M in MM])
RGB /= RGB.values.max()
f_R, f_G, f_B = [fit_limb(mu, C) for C in (RGB.R, RGB.G, RGB.B)]

# Calculate RGB image and show
axis = np.linspace(-xylim, xylim, N*N_AA)
x, y = np.meshgrid(axis, axis, sparse=True)
r2 = x*x + y*y
mu_i = np.where(r2 < 1, np.sqrt(abs(1 - r2)), -1)
im_RGB = RGB_Color(R=f_R(mu_i), G=f_G(mu_i), B=f_B(mu_i))
im_RGB.values = im_RGB.values.reshape((3, N, N_AA, N, N_AA)).mean(axis=(2, 4)) #AA
im_IRGB = im_RGB.to_IRGB()

plt.figure(figsize=(N / DPI, N / DPI))
plt.imshow(im_IRGB.values_image)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.show()
