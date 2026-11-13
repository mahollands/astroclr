"""
Shared pytest configuration for the astroclr test suite.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend so plotting tests never open windows

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u

from astroclr.spectral import Spectrum, UNIT_SPEC_RADIANCE


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figures a test created."""
    yield
    plt.close("all")


@pytest.fixture
def sample_wavelengths():
    """Wavelengths in nm as an Astropy Quantity."""
    return np.arange(300.0, 901.0, 50.0) * u.nm


@pytest.fixture
def sample_fluxes(sample_wavelengths):
    """Smooth, positive fluxes with default units."""
    flux = np.exp(-0.5 * ((sample_wavelengths - 550 * u.nm) / (300 * u.nm)) ** 2)
    return flux * UNIT_SPEC_RADIANCE


@pytest.fixture
def sample_spectrum(sample_wavelengths, sample_fluxes):
    return Spectrum(sample_wavelengths, sample_fluxes)
