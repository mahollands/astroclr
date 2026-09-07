"""
Data for colour matching functions, used to convert spectra to XYZ colour
spaces. These are stored in a dictionary called CMFs.

Available CMFs:
    "1931" - The standard CIE 1931 CMFs (2 degree observer). This is the
             default set of CMFs used unless otherwise specified. 

    "1931J" - CIE 1931 CMFs with Judd correction.

    "1931JV" - CIE 1931 CMFs with Judd-Vos correction.

    "1964_10deg" - CIE 1964 CMFs (10 degree observer)

    "2015_2deg" - CIE 2015 CMFs (2 degree observer).
                  These use the Stockman & Sharp (2000) data.

    "2015_10deg" - CIE 2015 CMFs (10 degree observer).

LMS_XYZ_MATRIX_2000 -
    The Stockman & Sharp (2000) data are specified in terms of cone fundamentals
    (long/medium/short), and have been transformed into their CMF form. This
    matrix can be used to convert the CMFs back to LMS form, or transform
    sets of XYZ values to LMS values.
"""

import os.path
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy import units as u

__all__ =   [
    "CMFs",
    "LMS_XYZ_MATRIX_2000",
]

DATA_DIR = f"{os.path.dirname(__file__)}/data"

CIE_CMF = namedtuple("CIE_CMF", "wavelength xyz_table interp")

# The Stockman & Sharp (2000) data come from LMS cone fundametals and are
# transformed to XYZ using the following matrix. The CIE data have already been
# transformed, but this matrix can be used to recover the original LMS curves.
LMS_XYZ_MATRIX_2000 = np.array([
    [1.94735469, -1.41445123, 0.36476327],
    [0.68990272,  0.34832189, 0         ],
    [0         ,  0         , 1.93485343],
])

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
    plt.plot(wavelength, x, "r")
    plt.plot(wavelength, y, "g")
    plt.plot(wavelength, z, "b")
    plt.show()


CMFs = {
    "1931": load_CMF("ciexyz31_1.txt"),
    "1931J": load_CMF("ciexyzj.csv"),
    "1931JV": load_CMF("ciexyzjv.csv"),
    "1964_10deg": load_CMF("CIE_xyz_1964_10deg.csv"),
    "2015_2deg": load_CMF("CIE_cfb_stv_2deg.csv"),
    "2015_10deg": load_CMF("CIE_cfb_stv_10deg.csv"),
}


if __name__ == "__main__":
    for name, CMF in CMFs.items():
        CMF_plot(name, CMF)
