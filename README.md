# astroclr

Astroclr allows perceived colours of light sources to be calculated from the
physical spectra that emit them, and conversions between various colour spaces.
This allows spectra radiated from (for example) astrophyiscal objects to be
converted into colours that can be replicated on a computer display.

Astroclr is an almost complete rewrite of
[Colorpy](https://github.com/markkness/ColorPy) by Mark Kness, which has been
an invaluable resource for me for calculating colours of white dwarfs. While
Colorpy is well documented and referenced (and helped develop my own interest
in colour science), it is no longer actively maintained, is still designed for
python2 (can be made to work in python3 with a few tweaks), and by modern
standards the module interface is far from pythonic.

## Features
Astroclr aims to solve several issues with the original:
* A modern interface with a variety of colour classes and methods to convert
  between them (all supporting numpy arrays for multiple objects or images)
    - XYZ_Color
    - RGB_Color
    - IRGB_Color 
    - UCS_Color
    - LUV_Color
    - LAB_Color
* A bespoke Spectrum type with easy conversion to XYZ colours
    - XYZ integrations are performed trapezoidally, so wavelengths can have
      arbitraty spacing.
    - Works with astropy units for calculating luminance/illuminance
    - Easy generation of black body spectra
* Support for a variety of RGB colour spaces
    - sRGB
    - Rec709
    - Rec2020
    - DCI P3 SDR/HDR
    - and many more
* Scripts for interesting things an astronomer might want to do
    - Approximated rainbow plot of monochromtic colours
    - Interactive simulation of chromatic abberation

## Installation:

At the moment installation is simply cloning this module and pointing your
PYTHONPATH environment variable at it.

## Colour Science
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

Having lost a degree of freedom (the intensity), $x$ and $y$ now uniquely define
the brightness-indepedent component of colour and is called the chromaticity.
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
function. In some colour-spaces this is just a power-law (gamma curve), though
in sRGB a piecewise function is used that is linear at low-brightness and a
shifted plus scaled power-law at higher brightness, but closely resembles a
power-law with exponent 1/2.2.


## Usage

Calculating colours with astroclr has been made as easy as possible. The recipe
for doing so looks basically like:
* create a Spectrum object
* convert to the CIE 1931 XYZ colour space
* convert from XYZ to a linear RGB colour space (sRGB by default)
* convert linear RGB to the non-linear display RGB (IRGB) space by applying a transfer function

Lets say we want to determine the sRGB colour of a 10,000 K black body (something
like the A-type star Sirius A) as a hexstring. In Colorpy2 we can do

```python
>>> from astropy import units as u
>>> from colorpy2 import Spectrum

>>> lam = np.arange(300, 900, 0.1) * u.nm
>>> S = Spectrum.BlackBody(temperature=10000*u.K, wavelength=lam)
>>> print(S.to_IRGB_string())
'#CDD9FF'
```

It's that simple! You may have noticed that most of the conversion methods are
prefixed with 'to', but to convert to RGB we need to use a 'from' method. There
is good reason for this: By default RGB colours are in the sRGB space, but
optionally can use another space e.g. `RGB.from_XYZ(XYZ, color_space=Rec2020)`.
Lets inspect the sRGB object:

```python
>>> from colorpy2 import sRGB
>>> print(sRGB)
RGB_ColorSpace(name='sRGB', R=XYZ_Color(0.64, 0.33), G=XYZ_Color(0.3, 0.6), ...)
```

Crucially we can see that these colour spaces are defined in terms of RGB
primaries and a white point, which are all of type `XYZ_Color`. This means that
`RGB_Color` (and all other color models), are built on the more fundamental
`XYZ_Color`. Going back to our previous example, lets print all the
intermediate steps

```python
>>> print(S)
Spectrum(wavelength[6000] * nm, flux[6000] * W / (nm sr m2))
>>> print(XYZ)
XYZ_Color(18899521.886203796, 19415029.182752594, 29031295.252976917)
>>> print(RGB)
RGB_Color(16929126.8935794, 19310119.61155575, 27776415.403297085)
>>> print(IRGB)
IRGB_Color(205, 217, 255)
```

Firstly, we can see that the spectrum wavelengths and fluxes have units. The
XYZ and linear RGB values drop their units but the numbers are still very large
(a 10000K black body is bright!). However when we convert to IRGB, these overly
large values are rescaled so that the brightest channel is 255. If the RGB_values
were all below 1.0, then no rescaling would be performed. We have a few different
options if we want to rescale the linear colours. Going back to example

```python
>>> XYZ.normalize()
XYZ_Color(0.2806338166107294, 0.28828844306065954)
```

This method scales an `XYZ_Color` so all components add to 1. In this form the
Z component (now written z) is redundant, and the x and y components
define the chromaticity.

```python
>>> XYZ.normalize_Y1()
XYZ_Color(0.9734480287566731, 1.0, 1.4953001089880908)
```

This method scales the Y component (the luminance) 1.


### Luminance

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
```math
\mathrm{d}\Phi = L \cos\theta \,\mathrm{d}\Omega \,\mathrm{d}A
```
where $\cos\theta$ accounts for the projection of a small area element $\mathrm{d}A$.
This is equivalent to 
```math
\mathrm{d}\Phi = L \cos\theta \sin\theta \,\mathrm{d}\theta \,\mathrm{d}\phi
\,\mathrm{d}A
```
The integral over $\theta$ runs from $0$ to $\pi/2$, since we only care about the
outward radiated flux. So
```math
\Phi = L \oiint \mathrm{d}A \int_0^{2\pi} \mathrm{d}\phi \int_0^{\pi/2} \cos\theta \sin\theta \,\mathrm{d}\theta
  = L (4\pi R_\odot^2)(2\pi \mathrm{sr})(1/2)
  = 4\pi^2 \mathrm{sr} R_\odot^2 L
```
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
