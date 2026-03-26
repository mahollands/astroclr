import os.path
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy import units as u

DATA_DIR = f"{os.path.dirname(__file__)}/data"

CIE_CMF = namedtuple("CIE_CMF", "wavelength xyz_table interp")


def load_CMF(fname: str) -> CIE_CMF:
    CIEXYZ_table = np.loadtxt(
        f"{DATA_DIR}/{fname}",
        delimiter=",",
        unpack=True,
    )
    wavelength, xyz_table = np.split(CIEXYZ_table, [1])
    wavelength = wavelength.ravel() << u.nm
    xyz_table[np.isnan(xyz_table)] = 0.0
    xyz_interpolator = interp1d(wavelength, xyz_table, bounds_error=False, fill_value=0)
    return CIE_CMF(wavelength, xyz_table, xyz_interpolator)


def CMF_plot(name, CMF):
    wavelength, (x, y, z), *_ = CMF
    plt.figure(name)
    plt.plot(wavelength, x, 'r')
    plt.plot(wavelength, y, 'g')
    plt.plot(wavelength, z, 'b')
    plt.show()


CMFs = {
    "1931": load_CMF("ciexyz31_1.txt"),
    "1931J": load_CMF("ciexyzj.csv"),
    "1931JV": load_CMF("ciexyzjv.csv"),
    "1964_10deg": load_CMF("CIE_xyz_1964_10deg.csv"),
}



if __name__ == "__main__":
    for name, CMF in CMFs.items():
        CMF_plot(name, CMF) 
