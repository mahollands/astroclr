"""
Gamma correction / electro-optical tranfser functions

Non-gamma corrected rgb values, also called non-linear rgb values,
correspond to palette register entries [although here they are kept
in the range 0.0 to 1.0.]  The numerical values are not proportional
to the amount of light energy present.

Gamma corrected rgb values, also called linear rgb values,
do not correspond to palette entries.  The numerical values are
proportional to the amount of light energy present.
"""

# sRGB standard effective gamma. This exponent is not applied explicitly.
STANDARD_GAMMA = 2.2

# Although NTSC specifies a gamma of 2.2 as standard, this is designed to
# account for the dim viewing environments typical of TV, but not computers.
# Well-adjusted CRT displays have a true gamma in the range 2.35 through 2.55.
# We use the physical gamma value here, not 2.2, thus not correcting for a dim
# viewing environment.
# [Poynton, Gamma FAQ p.5, p.9, Hall, p. 121]
POYNTON_GAMMA = 2.45

class gamma_EOTF:
    """
    Simple power law for gamma correction.
    """
    def __init__(self, gamma):
        self.gamma = gamma
    
    def __repr__(self):
        return f"gamma_EOTF(gamma={self.gamma})"

    def __call__(self, x):
        return x**self.gamma

    def inv(self, x):
        return x ** (1/self.gamma)

class sRGB_EOTF:
    """
    sRGB standard for gamma correction. The effect of the equations is to
    closely fit a straightforward gamma 2.2 curve with an slight offset to
    allow for invertability in integer math. Therefore, we are maintaining
    consistency with the gamma 2.2 legacy images and the video industry.
    https://www.color.org/chardata/rgb/srgb.xalter
    """
    def __init__(self):
        pass

    def __repr__(self):
        return "sRGB_EOTF()"

    def __call__(self, x):
        #break_point = 0.03928 #old value
        break_point = 0.04045
        linear = x / 12.92 
        gamma = ((x + 0.055) / 1.055) ** 2.4
        return linear if x <= break_point else gamma

    def inv(self, x):
        #break_point = 0.0304 #old value
        break_point = 0.031308
        linear = 12.92 * x 
        gamma = 1.055 * x ** (1 / 2.4) - 0.055
        return linear if x <= break_point else gamma

class PQ_EOTF:
    """
    Perceptual Quantizer transfer function. Unlike the SDR EOTFs which map to
    an arbitrary luminance range, the unit PQ output scale (input scale to the
    inv method), maps linearly to a 10000 cd/m^2 (nit) range in physical
    luminance.
    https://www.color.org/chardata/rgb/BT2100.xalter
    """
    def __init__(self):
        self.m1, self.m2 = 1305/8192, 2523/32
        self.c1, self.c2, self.c3 = 107/128, 2413/128, 2392/128

    def __repr__(self):
        return "PQ_EOTF()"

    def __call__(self, x):
        Em2 = x ** (1/self.m2)
        numerator = max(Em2-self.c1, 0)
        denominator = self.c2 - self.c3 * Em2
        return (numerator/denominator)**(1/self.m1)

    def inv(self, x):
        Ym1 = x ** self.m1
        numerator = self.c1 + self.c2*Ym1
        denominator = 1 + self.c3 * Ym1
        return (numerator/denominator)**self.m2
