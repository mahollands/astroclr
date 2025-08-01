# Colorpy2

Colorpy2 allows perceived colours of light to be calculated from the physical
spectra that emit them. This allows spectra radiated from (for example)
astrophyiscal objects to be converted into colours that can be accurately be
reproduced on a computer display.

Colorpy2 is an almost complete rewrite of the original
[colorpy](https://github.com/markkness/ColorPy) by Mark Kness, further
documentation for the original package can be found
[here](https://github.com/markkness/ColorPy). Mark Kness' original colorpy has
been an invaluable resource for me for calculating colours of stars from their
spectra. The original is exceptionally well documented and referenced, and is
largely responsible for developing my own interest in colour science. The
original, however, is no longer actively maintained, is still designed for python2
(though only requires a few tweaks to get it working in python3), and the module
interface is far from pythonic by modern standards.

## Features
Colorpy2 aims to solve several issues with the original:
* A modern interface with a variety of colour classes and methods to convert
  between them
    - XYZ_Color
    - RGB_Color
    - IRGB_Color 
    - LUV_Color
    - LAB_Color
* A bespoke Spectrum type with easy conversion to XYZ colours
    - XYZ integrations are performed trapezoidally, so wavelengths don't need
      to be uniformly spaced.
    - Integration with astropy units for calculating luminance
    - Easy generation of black body spectra
* A class for RGB colour spaces, including RGB and white point chromacities as
  well as  electro-optical transfer functions.
    - sRGB
    - Rec709
    - Rec2020
    - DCI_P3 SDR/HDR
    - and many more

## Installation:

At the moment installation is simply cloning this module and pointing your
PYTHONPATH environment variable at it.

## Usage

If you want to convert the luminance to luminous flux / exitance /
illuminance you must take care to account for factors π, 2, 4π etc.
To calculate the luminance of the Sun assuming a perfect radiator
(black body) you can do
```python
>>> Sun = Spectrum.BlackBody(5773*u.K, wavelength)
>>> L = Sun.to_luminance()
>>> L
<Quantity 1.8451589e+09 cd / m2> #luminance
```

But to get the luminous flux $\Phi$ in lm, you must do some integration:
$\mathrm{d}\Phi = L \cos\theta \mathrm{d}\Omega \mathrm{d}A$
where $\cos\theta$ accounts for the projection of a small area element $\mathrm{d}A$.
This is equivalent to 
$\mathrm{d}\Phi = L \cos\theta \sin\theta \mathrm{d}\theta \mathrm{d}\phi \mathrm{d}A$
The integral over $\theta$ runs from $0$ to $\pi/2$, since we only care about the
outward radiated flux. So
$\Phi = L \int \mathrm{d}A \int_0^{2\pi} \mathrm{d}\phi \int_0^{\pi/2} \cos\theta \sin\theta \mathrm{d}\theta
  = L (4\pi R_\odot^2)(2\pi \mathrm{sr})(1/2)
  = 4\pi^2 \mathrm{sr} R_\odot^2 L$
```python
>>> A_sun = 4 * np.pi * u.Rsun**2
>>> Ω = 2 * np.pi * u.sr
>>> Φ = 0.5 * Ω * A_sun * L
>>> Φ.to(u.lm)
<Quantity 3.52563635e+28 lm> #luminous flux
```

To calculate the illuminance at Earth, then
```python
>>> A_au = 4*np.pi * u.au**2
>>> E = Φ / A_au
>>> E.to(u.lx)
<Quantity 125365.15060712 lx> #illuminance
>>> M = Φ / A_sun
>>> M.to(u.lm/u.m**2)
<Quantity 5.79673763e+09 lm / m2> #luminous exitance
```
