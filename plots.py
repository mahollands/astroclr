"""
plots.py - Various types of plots.

Description:

Functions to draw various types of plots for light spectra.
"""

import itertools
import math
import matplotlib.pyplot as plt
from astropy import units as u

from .colormodels import RGB_Color, ColorList
from .ciexyz import *
from .RGB_color_spaces import sRGB


def rgb_patch_plot(RGBs, color_names, title, patch_gap=0.05, num_across=6):
    """
    Draw a set of color patches, from a list of linear rgb colors.
    """
    def draw_patch(x0, y0, color, name, patch_gap):
        """
        Draw a patch of color.
        """
        # patch relative vertices
        m = patch_gap
        poly_dx = [m, m, 1 - m, 1 - m]
        poly_dy = [m, 1 - m, 1 - m, m]
        # construct vertices
        poly_x = [x0 + dx_i for dx_i in poly_dx]
        poly_y = [y0 + dy_i for dy_i in poly_dy]
        plt.fill(poly_x, poly_y, color)
        if name is not None:
            dtext = 0.1
            plt.text(x0 + dtext, y0 + dtext, name, size=8.0)

    # make plot with each color with one patch
    plt.figure()
    for i, RGB in enumerate(RGBs):
        (iy, ix) = divmod(i, num_across)
        colorstring = RGB.to_IRGB().to_string()
        name = color_names[i] if color_names is not None else None
        draw_patch(float(ix), float(-iy), colorstring, name, patch_gap)
    plt.axis("off")
    plt.title(title)
    plt.show()


def xyz_patch_plot(XYZs, color_names, title, patch_gap=0.05, num_across=6):
    """
    Draw a set of color patches specified as xyz colors.
    """
    RGBs = [RGB_Color.from_XYZ(XYZ) for XYZ in XYZs]
    rgb_patch_plot(
        RGBs, color_names, title, patch_gap=patch_gap, num_across=num_across
    )


def spectrum_subplot(S):
    """
    Plot a spectrum, with x-axis the wavelength, and y-axis the intensity. The
    curve is colored at that wavelength by the (approximate) color of a pure
    spectral color at that wavelength, with intensity constant over wavelength.
    (This means that dark looking colors here mean that wavelength is poorly
    viewed by the eye.

    This is not a complete plotting function, e.g. no file is saved, etc.
    It is assumed that this function is being called by one that handles those things.
    """
    XYZs = (XYZ_from_wavelength(lam) for lam in S.wavelength)
    RGBs = [RGB_Color.from_XYZ(XYZ) for XYZ in XYZs]
    rgb_max = max(RGB._values.max() for RGB in RGBs)
    RGBs = [RGB / rgb_max for RGB in RGBs]
    # draw color patches (thin vertical lines matching the spectrum curve) in color
    iterables = (
        itertools.pairwise(S.wavelength.value),
        itertools.pairwise(S.flux.value),
        RGBs,
    )
    for (x0, x1), (y0, y1), RGB in zip(*iterables):
        color_string = RGB.to_IRGB().to_string()
        poly_x = [x0, x1, x1, x0]
        poly_y = [0.0, 0.0, y1, y0]
        plt.fill(poly_x, poly_y, color_string, edgecolor=color_string)
    plt.plot(S.wavelength, S.flux, color="k", linewidth=2.0)


def spectrum_plot(S, title, xlabel="Wavelength($nm$)", ylabel="Intensity ($W/m^2$)"):
    """
    Plot for a single spectrum -
    In a two part graph, plot:
    top: color of the spectrum, as a large patch.
    low: graph of spectrum intensity vs wavelength (x axis).
    The graph is colored by the (approximated) color of each wavelength.
    Each wavelength has equal physical intensity, so the variation in
    apparent intensity (e.g. 400, 800 nm are very dark, 550 nm is bright),
    is due to perceptual factors in the eye.  This helps show how much
    each wavelength contributes to the percieved color.

    spectrum - spectrum to plot
    title    - title for plot
    xlabel   - label for x axis
    ylabel   - label for y axis
    """
    # upper plot - solid patch of color that matches the spectrum color
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(title)
    color_string = S.to_IRGB().to_string()
    poly_x = [0.0, 1.0, 1.0, 0.0]
    poly_y = [0.0, 0.0, 1.0, 1.0]
    plt.fill(poly_x, poly_y, color_string)
    # draw a solid line around the patch to look nicer
    plt.plot(poly_x, poly_y, color="k", linewidth=2.0)
    plt.axis("off")
    # lower plot - spectrum vs wavelength, with colors of the associated spectral lines below
    plt.subplot(2, 1, 2)
    spectrum_subplot(S)
    plt.xlim(S.wavelength.value[(0, -1),])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def color_vs_param_plot(
    param_list,
    RGBs,
    title,
    tight=False,
    plotfunc=plt.plot,
    xlabel="param",
    ylabel="RGB Color",
):
    """
    Plot for a color that varies with a parameter -
    In a two part figure, draw:
    top: color as it varies with parameter (x axis)
    low: r,g,b values, as linear 0.0-1.0 values, of the attempted color.

    param_list - list of parameters (x axis)
    RGBs       - numpy array, one row for each param in param_list
    title      - title for plot
    plotfunc   - optional plot function to use (default plt.plot)
    xlabel     - label for x axis
    ylabel     - label for y axis (default 'RGB Color')
    """
    plt.figure()
    # draw color bars in upper plot
    plt.subplot(2, 1, 1)
    plt.title(title)
    # no xlabel, ylabel in upper plot
    for (x0, x1), RGB in zip(itertools.pairwise(param_list), RGBs):
        poly_x = [x0, x1, x1, x0]
        poly_y = [0, 0, 1, 1]
        color_string = RGB.to_IRGB().to_string()
        plt.fill(poly_x, poly_y, color_string, edgecolor=color_string)
    if tight:
        plt.xlim(min(param_list), max(param_list))
    # draw rgb curves in lower plot
    plt.subplot(2, 1, 2)
    # no title in lower plot
    R, G, B = [[getattr(RGB, c) for RGB in RGBs] for c in "rgb"]
    plotfunc(param_list, R, color="r", label="Red")
    plotfunc(param_list, G, color="g", label="Green")
    plotfunc(param_list, B, color="b", label="Blue")
    if tight:
        plt.xlim(min(param_list), max(param_list))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def visible_spectrum_plot(CMF='1931'):
    """
    Plot the visible spectrum, as a plot vs wavelength.
    """
    _wavelengths, _ = CMFs[CMF]
    XYZs = (XYZ_from_wavelength(lam, CMF=CMF) for lam in _wavelengths)
    RGBs = [RGB_Color.from_XYZ(XYZ) for XYZ in XYZs]
    # scale to make brightest rgb value = 1.0
    rgb_max = max(RGB._values.max() for RGB in RGBs)
    RGBs = [RGB / rgb_max for RGB in RGBs]
    # plot colors and rgb values vs wavelength
    color_vs_param_plot(
        _wavelengths.value,
        RGBs,
        "The Visible Spectrum",
        tight=True,
        xlabel=r"Wavelength (nm)",
        ylabel=r"RGB Color",
    )

def cie_matching_functions_plot(CMF='1931'):
    """
    Plot the CIE XYZ matching functions, as three spectral subplots.
    """
    _wavelengths, _xyz_table = CMFs[CMF]
    cie_x, cie_y, cie_z = [Spectrum(_wavelengths, cie) for cie in _xyz_table]
    
    plt.figure()
    # X
    plt.subplot(3, 1, 1)
    plt.title("1931 CIE XYZ Matching Functions")
    plt.ylabel("CIE $X$")
    spectrum_subplot(cie_x)
    plt.xlim(_wavelengths.value[(0,-1),])
    # Y
    plt.subplot(3, 1, 2)
    plt.ylabel("CIE $Y$")
    spectrum_subplot(cie_y)
    plt.xlim(_wavelengths.value[(0,-1),])
    # Z
    plt.subplot(3, 1, 3)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("CIE $Z$")
    spectrum_subplot(cie_z)
    plt.xlim(_wavelengths.value[(0,-1),])
    plt.show()


def scattered_visual_brightness(CMF='1931'):
    """
    Plot the perceptual brightness of Rayleigh scattered light.
    """
    _wavelengths, _xyz_table = CMFs[CMF]
    CIE_Y = _xyz_table[1]
    # get 'spectra' for y matching functions and multiply by 1/wl^4
    rayleigh = (550.0 / _wavelengths.value)**4
    S = Spectrum(_wavelengths, rayleigh*CIE_Y)

    plt.figure()
    spectrum_subplot(S)
    plt.title("Perceptual Brightness of Rayleigh Scattered Light")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(r"CIE $Y$ / $\lambda^4$")
    plt.xlim(S.wavelength.value[(0,-1),])
    plt.show()


def shark_fin_plot(CMF='1931', color_space=sRGB,):
    """
    Draw the 'shark fin' CIE chromaticity diagram of the pure spectral lines
    (plus purples) in xy space.
    """
    cs = color_space
    # get array of (approximate) colors for the boundary of the fin
    XYZs = get_normalized_spectral_line_colors(
        num_purples=200, dwl=2*u.AA, CMF=CMF
    )
    # get normalized colors
    xys = [XYZ.normalize() for XYZ in XYZs]

    def get_direc_to_white(xy, cs):
        """
        Get unit vector (xy plane) in direction of the white point.
        """
        direc = cs.W - xy
        mag = math.hypot(direc.x, direc.y)
        if mag == 0.0:
            raise ValueError
        return direc / mag

    plt.figure()
    # draw best attempt at pure spectral colors on inner edge of shark fin
    s = 0.025  # distance in xy plane towards white point
    for (xy0, xy1), XYZ in zip(itertools.pairwise(xys), XYZs):
        (x0, y0), (x1, y1) = (xy0.x, xy0.y), (xy1.x, xy1.y)
        # get unit vectors in direction of white point
        dir0 = get_direc_to_white(xy0, cs)
        dir1 = get_direc_to_white(xy1, cs)
        # polygon vertices
        poly_x = [x0, x1, x1 + s * dir1.x, x0 + s * dir0.x]
        poly_y = [y0, y1, y1 + s * dir1.y, y0 + s * dir0.y]
        # draw (using full color, not normalized value)
        RGB = RGB_Color.from_XYZ(XYZ)
        color_string = RGB.to_IRGB().to_string()
        plt.fill(poly_x, poly_y, color_string, edgecolor=color_string)

    # fill in the monitor gamut with true colors
    def fill_gamut_slice(v0, v1, v2):
        """
        Fill in a slice of the monitor gamut with the correct colors.
        """
        num_s, num_t = 50, 50
        dv10, dv21 = v1 - v0, v2 - v1
        for i_s, i_t in itertools.product(range(num_s), range(num_t)):
            s_a = i_s / num_s
            s_b = (i_s + 1) / num_s
            t_a = i_t / num_t
            t_b = (i_t + 1) / num_t
            # vertex coords
            v_aa = v0 + t_a * (dv10 + s_a * dv21)
            v_ab = v0 + t_b * (dv10 + s_a * dv21)
            v_ba = v0 + t_a * (dv10 + s_b * dv21)
            v_bb = v0 + t_b * (dv10 + s_b * dv21)
            # poly coords
            poly_x = [v_aa.x, v_ba.x, v_bb.x, v_ab.x]
            poly_y = [v_aa.y, v_ba.y, v_bb.y, v_ab.y]
            # average color
            XYZ_avg = 0.25 * (v_aa + v_ab + v_ba + v_bb)
            # convert to rgb and scale to maximum displayable brightness
            RGB = RGB_Color.from_XYZ(XYZ_avg, brightest=True)
            color_string = RGB.to_IRGB().to_string()
            plt.fill(poly_x, poly_y, color_string, edgecolor=color_string)

    fill_gamut_slice(cs.W, cs.B, cs.G)
    fill_gamut_slice(cs.W, cs.G, cs.R)
    fill_gamut_slice(cs.W, cs.R, cs.B)

    # draw the curve of the xy values of the spectral lines and purples
    plt.plot([xy.x for xy in xys], [xy.y for xy in xys], color="#808080", linewidth=3.0)
    # draw monitor gamut and white point
    tri_x = [cs.R.x, cs.G.x, cs.B.x, cs.R.x]
    tri_y = [cs.R.y, cs.G.y, cs.B.y, cs.R.y]
    plt.plot(tri_x, tri_y, "o-", color="k")
    plt.plot(cs.W.x, cs.W.y, "o", color="k")
    # label phosphors
    dx = dy = 0.01
    plt.text(cs.R.x + dx, cs.R.y, "Red", ha="left", va="center")
    plt.text(cs.G.x, cs.G.y + dy, "Green", ha="center", va="bottom")
    plt.text(cs.B.x - dx, cs.B.y, "Blue", ha="right", va="center")
    plt.text(cs.W.x, cs.W.y + dy, "White", ha="center", va="bottom")
    # titles etc
    plt.axis([0.0, 0.85, 0.0, 0.85])
    plt.xlabel(r"CIE $x$")
    plt.ylabel(r"CIE $y$")
    plt.title(r"CIE Chromaticity Diagram")
    plt.show()


def figures(CMF='1931'):
    """Draw specific figures not used anywhere else."""
    visible_spectrum_plot(CMF=CMF)
    cie_matching_functions_plot(CMF=CMF)
    shark_fin_plot(CMF=CMF)
    scattered_visual_brightness(CMF=CMF)
