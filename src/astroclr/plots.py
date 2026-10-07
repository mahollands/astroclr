"""
Functions to draw various types of plots for light spectra.
"""

from itertools import pairwise
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from astropy import units as u
from .colormodels import XYZ_Color, RGB_Color, ClipMethod
from .spectral import CMFs, Spectrum, XYZ_from_wavelength
from .rgb_color_spaces import sRGB


def color_vs_param_plot(
    param_list,
    RGBs,
    title,
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
    for (x0, x1), RGB in zip(pairwise(param_list), RGBs):
        poly_x = [x0, x1, x1, x0]
        poly_y = [0, 0, 1, 1]
        color_string = RGB.to_IRGB().to_string()
        plt.fill(poly_x, poly_y, color_string, edgecolor=color_string)
    plt.ylim(0, 1)
    plt.yticks(visible=False)
    # draw rgb curves in lower plot
    plt.subplot(2, 1, 2)
    # no title in lower plot
    plotfunc(param_list, RGBs.R, color="r", label="Red")
    plotfunc(param_list, RGBs.G, color="g", label="Green")
    plotfunc(param_list, RGBs.B, color="b", label="Blue")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def visible_spectrum_plot(CMF="1931"):
    """
    Plot the visible spectrum, as a plot vs wavelength.
    """
    wavelengths = CMFs[CMF].wavelength
    XYZs = XYZ_from_wavelength(wavelengths, CMFname=CMF)
    RGBs = RGB_Color.from_XYZ(XYZs)
    # add grey to avoid gamut clipping
    grey = RGBs.values.min()
    RGBs.values -= grey
    # scale to make brightest rgb value = 1.0
    RGBs /= RGBs.values.max()
    # plot colors and rgb values vs wavelength
    color_vs_param_plot(
        wavelengths.value,
        RGBs,
        "The Visible Spectrum",
        tight=True,
        xlabel=r"Wavelength (nm)",
        ylabel=r"RGB Color",
    )


def cie_matching_functions_plot(CMF="1931"):
    """
    Plot the CIE XYZ matching functions, as three spectral subplots.
    """
    wavelengths, xyz_table, _ = CMFs[CMF]
    cie_x, cie_y, cie_z = xyz_table

    plt.figure()
    # X
    plt.subplot(3, 1, 1)
    plt.title("1931 CIE XYZ Matching Functions")
    plt.ylabel("CIE $X$")
    plt.plot(wavelengths, cie_x, 'r-')
    plt.xlim(wavelengths.value[(0, -1),])
    # Y
    plt.subplot(3, 1, 2)
    plt.ylabel("CIE $Y$")
    plt.plot(wavelengths, cie_y, 'g-')
    plt.xlim(wavelengths.value[(0, -1),])
    # Z
    plt.subplot(3, 1, 3)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("CIE $Z$")
    plt.plot(wavelengths, cie_z, 'b-')
    plt.xlim(wavelengths.value[(0, -1),])


def fill_cie_space(
    xy_mono,
    color_space=sRGB,
    clip_method=ClipMethod.ADD_WHITE,
    brightest=True,
    N_pixels=500,
    N_AA=2,
):
    """
    Fill the CIE chromaticity diagram
    """
    grid_px = np.linspace(0, 1, N_AA*N_pixels)
    x_grid, y_grid = np.meshgrid(grid_px, grid_px)

    p = path.Path(np.array([xy_mono.X, xy_mono.Y]).T)
    inside = p.contains_points(np.array([x_grid.ravel(), y_grid.ravel()]).T)
    inside = inside.reshape(x_grid.shape)

    xyz_grid = XYZ_Color(x_grid, y_grid)
    rgb_grid = RGB_Color.from_XYZ(xyz_grid, color_space=color_space)
    if brightest:
        rgb_grid /= rgb_grid.values.max(axis=0)
    else:
        rgb_grid /= rgb_grid.values.sum(axis=0) / 1.5
    rgb_grid.values = [np.where(inside, rgb_grid_i, 1) for rgb_grid_i in rgb_grid.values]
    rgb_grid.values = np.reshape(
        rgb_grid.values, (3, N_pixels, N_AA, N_pixels, N_AA)).mean(axis=(2, 4)
    )
    irgb_grid = rgb_grid.to_IRGB(clip_method=clip_method)

    plt.imshow(irgb_grid.values_image, origin='lower', extent=(0, 1, 0, 1))


def RGB_triangle(color_space):
    primaries = list(color_space.primaries)
    primaries.append(primaries[0])
    primaries = XYZ_Color.from_list(primaries)
    plt.plot(primaries.X, primaries.Y, "k:")
    for primary, c in zip(primaries, ["#FF0000", "#00FF00", "#0000FF"]):
        plt.plot(primary.X, primary.Y, "s", color=c, mec="k")


def lam_ticks_labels(wavelengths, xy_mono, offsetx=0.03, offsety=0.01):
    dist = 0.005
    lams = wavelengths.to_value(u.nm).astype(np.int64)
    for j, (lam, xyj) in enumerate(zip(lams, xy_mono)):
        if not 440 <= lam <= 640:
            continue
        if lam % 5 == 0:
            ticklen = 0.01 if lam % 20 == 0 else 0.005
            xy_grad = xy_mono[j + 1] - xy_mono[j - 1]
            dx, dy = xy_grad.X, xy_grad.Y
            ds = np.hypot(dx, dy)
            dx, dy = dy * ticklen / ds, -dx * ticklen / ds
            plt.plot([xyj.X, xyj.X + dx], [xyj.Y, xyj.Y + dy], "k-", lw=1.5)
        if lam < 460:
            continue
        if lam % 20 == 0:
            dx *= -dist / ticklen
            dy *= -dist / ticklen
            dx += dx / dist * offsetx
            dy += dy / dist * offsety
            plt.annotate(
                str(lam),
                (xyj.X + dx, xyj.Y + dy),
                fontsize=10,
                va="center",
                ha="center",
            )


def BB_locus(CMF):
    T = np.geomspace(1000, 1e5, 100)
    wavelengths = CMFs[CMF][0]
    BBs = [Spectrum.BlackBody(Ti * u.K, wavelengths) for Ti in T]
    flux_inf = wavelengths.value**-4
    BBs.append(Spectrum(wavelengths, flux_inf / flux_inf[0]))  # Tinfinity
    T_XYZs = XYZ_Color.from_list([BB.to_XYZ(CMFname=CMF) for BB in BBs])
    T_xys = T_XYZs.normalize()
    plt.plot(T_xys.X, T_xys.Y, "k-")
    plt.plot(T_xys.X[-1], T_xys.Y[-1], "k.")
    ticklen = 0.005
    Tmin, Tmax, Tstep = 2000, 10000, 1000
    Tarr = np.arange(Tmin, Tmax + 1, Tstep)
    for Ttick in Tarr:
        j = np.argmin(abs(Ttick - T))
        T_xy, xy_grad = T_xys[j], T_xys[j + 1] - T_xys[j - 1]
        dx, dy = xy_grad.X, xy_grad.Y
        ds = np.hypot(dx, dy)
        dx, dy = dy * ticklen / ds, -dx * ticklen / ds
        plt.plot([T_xy.X - dx, T_xy.X + dx], [T_xy.Y - dy, T_xy.Y + dy], "k-")


def chromaticity_diagram(
    CMF="1931",
    color_space=sRGB,
    draw_cie_space=True,
    draw_bb_locus=True,
    draw_rgb_triangle=True,
    clip_method=ClipMethod.ADD_WHITE,
    brightest=True,
    N_pixels = 500,
    N_AA = 2
):
    plt.figure(figsize=(5, 4.71))
    xy_mono = XYZ_from_wavelength(CMFname=CMF).normalize()
    if draw_cie_space:
        fill_cie_space(xy_mono, color_space, clip_method, brightest, N_pixels, N_AA)
    plt.plot(xy_mono.X, xy_mono.Y, "k-", lw=2)
    if draw_bb_locus:
        BB_locus(CMF)
    lam_ticks_labels(CMFs[CMF][0], xy_mono)
    if draw_rgb_triangle:
        RGB_triangle(color_space)
    plt.xlim(-0.06, 0.81)
    plt.ylim(0, 0.87)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()


def figures(CMF="1931"):
    """Draw specific figures not used anywhere else."""
    #cie_matching_functions_plot(CMF=CMF)
    #plt.show()

    chromaticity_diagram(CMF=CMF, color_space=sRGB, brightest=True)
    plt.show()
