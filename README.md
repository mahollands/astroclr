# astroclr

Astroclr allows perceived colours of light sources to be calculated from the
physical spectra that emit them, and conversions between various colour spaces.
This allows spectra radiated from (for example) stars to be converted into
colours that can be replicated on a computer display.

Astroclr is heavily inspired by [Colorpy](https://github.com/markkness/ColorPy)
by Mark Kness, which was an invaluable resource for me when calculating colours
of white dwarfs, and developing my own interest in colour science. While
Colorpy is well documented and referenced, it is no longer actively maintained,
and requires several tweaks for compatibility with python3. Astroclr was
written with the express aim of making colour calculations from spectra as
simple as possible, usng a modern python interface, and with a particular focus
on observational astronomy.

## Features
* A modern interface with a variety of colour classes and methods to convert
  between them (all supporting numpy arrays for multiple objects or images)
    - XYZ_Color
    - RGB_Color
    - IRGB_Color 
    - UCS_Color
    - LUV_Color
    - LAB_Color
* A bespoke Spectrum class with easy conversion to XYZ colours
    - XYZ integrations are performed trapezoidally, so wavelengths can have
      arbitrary spacing.
    - Works with astropy units for calculating luminance, illuminance, and
      visual magnitudes
    - Easy generation of black body spectra
* Support for a variety of RGB colour spaces
    - sRGB
    - Rec709
    - Rec2020/2100
    - DCI P3
* A suite of example scripts for interesting things an astronomer might want to
  do, and demonstrations of various colour science concepts

Note: All documentation uses the British spelling of "colour" (since I am
British). For code, however, I have opted to use "color", for consistency with
Colorpy, and other similar software (and to save on typing). The module name
was chosen to be neutral towards spelling.

## Installation:

At the moment, installation is simply cloning this module and pointing your
PYTHONPATH environment variable at it. This module will be made available for
installation via pip asap.

### Dependencies
* numpy
* scipy
* matplotlib
* astropy


## Examples

Calculating colours with astroclr has been made as easy as possible, with the
basic recipe looking like:
* create a Spectrum object
* convert to the CIE 1931 XYZ colour space
* convert from XYZ to a linear RGB colour space (sRGB by default)
* convert linear RGB to the non-linear display RGB (IRGB) space by applying a
  transfer function

Lets say we want to determine the sRGB colour of a 10,000 K black body
(something like the A-type star Sirius A) as a hexstring. In Astroclr we can
follow the above recipe as follows:

```python
>>> from astropy import units as u
>>> from astroclr import Spectrum, RGB_Color

>>> lam = np.arange(300, 900, 0.1) * u.nm
>>> S = Spectrum.BlackBody(temperature=10000*u.K, wavelength=lam)
>>> XYZ = S.to_XYZ()
>>> RGB = RGB_Color.from_XYZ(XYZ, brightest=True)
>>> IRGB = RGB.to_IRGB()
>>> print(IRGB.to_string())
'#CDD9FF'
```

The `brightest` keyword, ensures the brightest RGB channel is scaled to 1.0, so
observed spectra (typically faint) can be shown as bright as possible, and
bright spectra (such as models) can be displayed without colour clipping. The
above code can be actually be performed even more succinctly:

```python
>>> S = Spectrum.BlackBody(temperature=10000*u.K, wavelength=lam)
>>> print(S.to_IRGB_string(brightest=True))
'#CDD9FF'
```

You may have noticed that most conversion methods are prefixed with 'to', but
to convert to RGB we need to use a 'from' method. This is because RGB colour
spaces (the default is sRGB) are defined in terms of XYZ chromaticities (see
below), and so all colour spaces are defined in terms of the more fundamental
`XYZ_Color`. This can be seen by inspecting the sRGB colour space:

```python
>>> from astroclr import sRGB
>>> print(sRGB)
RGB_ColorSpace(name='sRGB', R=XYZ_Color(0.64, 0.33), G=XYZ_Color(0.3, 0.6), ...)
```

If you have your own spectrum, calculating colours is just as simple:

```python

>>> wavelength, flux = np.loadtxt("my_spectrum.dat", unpack=True)
>>> wavelength *= u.AA
>>> flux *= u.erg / u.s / u.cm**2 / u.AA
>>> S = Spectrum(wavelength, flux)
>>> print(S.to_IRGB_string(brightest=True))
'#DDFFB8'
```

Note that fluxes must be in terms of spectral radiance/irradiance (per unit
wavelength), i.e. equivalent units to W/m2/nm/sr or W/m2/nm, which is necessary
for calculating the luminance/illuminance using the `to_luminance()` method. If
no units are provided, default units of nm and W/m2/nm/sr are assumed for
wavelengths and fluxes.

## Colour science basics
If we have a spectrum $F_\lambda(\lambda)$ and want to calculate its displayable RGB colour.
Mathematically we do something akin to synthetic photometry in astronomy:
```math
X = \int F_\lambda(\lambda) \bar{x}(\lambda) \,\mathrm{d}\lambda \qquad
Y = \int F_\lambda(\lambda) \bar{y}(\lambda) \,\mathrm{d}\lambda \qquad
Z = \int F_\lambda(\lambda) \bar{z}(\lambda) \,\mathrm{d}\lambda
```

where $\bar{x}(\lambda)$ etc, are the CIE 1931 colour matching functions, and
$X,Y,Z$ are the integrated fluxes in the XYZ colour space. But what if we want
to consider the colour independent of its brightness/intensity/luminance? Then
we can normalise the $X, Y, Z$ by scaling by their sum: 

```math
x = \frac{X}{X+Y+Z} \qquad y = \frac{Y}{X+Y+Z} \qquad z = \frac{Z}{X+Y+Z}
```

Having lost a degree of freedom (the intensity), $x$ and $y$ now uniquely
define the brightness-independent component of colour and is called the
chromaticity. The CIE 1931 chromaticity diagram is shown below, where the
shark-fin shape corresponds to monochromactic colours and bounds all perceivable
colours, the triangle shows the sRGB gamut (colours outside the triangle are
approximated by clipping negative values), and the black curve starting in the red
corner is the blackbody locus (ticks from 2000K to 10000K in 1000K steps).
![image](https://github.com/mahollands/astroclr/blob/main/src/astroclr/images/CIE_xy.png
"CIE 1931 chromaticity diagram")

The sRGB colour space specifies the chromaticities of the red/green/blue
primaries (and also the white point), which uniquely define a linear
transformation between XYZ and sRGB.

```math
    \left(\begin{array}{c}
    R \\
    G \\
    B
    \end{array}\right) =
    \left(\begin{array}{rrr}
     3.2410 & -1.5374 & -0.4986 \\
    -0.9692 &  1.8760 &  0.0416 \\
     0.0556 & -0.2040 &  1.0570
    \end{array}\right) \cdot
    \left(\begin{array}{c}
    X \\
    Y \\
    Z
    \end{array}\right)
```

These give linear intensities in the R/G/B channels in the range 0 to 1, where
(R,G,B)=(1,1,1) represents full white. To convert these to display RGB values
(IRGB in astroclr), we cannot simply multiply by 255 and round. As the display
scale is non-linear. We therefore have to apply an opto-electrical transfer
function (OETF). In some colour-spaces this is just a power-law (gamma curve), though
in sRGB a piecewise function is used that is linear at low-brightness and a
shifted plus scaled power-law at higher brightness, but closely resembles a
power-law with exponent 1/2.2. After applying a OETF, RGB values can be multiplied
by 255 and rounded to determine the 8-bit per channel display values.
