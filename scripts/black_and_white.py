"""
Demonstration of 'correctly' converting an image to black and white. Naively
one might think to simply average the R,G,B channels. That will produce
something that looks plausible correct, but the relative brightnesses of some
colours will not be preserved. For instance full green is perceptually much
brighter than full blue.
"""
import numpy as np
import matplotlib.pyplot as plt
from astroclr import RGB_Color, IRGB_Color 

#Load Test Card F with dimensions NxMx3, with data in range 0--225.
img_clr = plt.imread("data/clr_wheel.jpeg")

#The wrong way to make a black and white image by averaging RGB channels
img_bw_avg = img_clr.mean(axis=2, keepdims=True, dtype=np.int32)
img_bw_avg = np.repeat(img_bw_avg, 3, axis=2)

#The correct way to make a black and white image from the luminance
# Step 1: Convert image to diplay RGB object
# Step 2: Convert to linear RGB via EOTF
# Step 3: Transform to linear XYZ object
# Step 4: Create new RGB object using XYZ luminance for all channels (Y)
# Step 5: Convert back to display RGB
# Step 6: Extract as numpy array to use with plt.imshow
IRGB = IRGB_Color.from_numpy_image(img_clr)
RGB = IRGB.to_RGB()
XYZ = RGB.to_XYZ()
RGB_bw = RGB_Color.from_grey(XYZ.Y)
IRGB_bw = RGB_bw.to_IRGB()
img_bw_lum = IRGB_bw.values_image

imgs = [
    (img_clr, "RGB"),
    (img_bw_avg, "B&W from averaging"),
    (img_bw_lum, "B&W from luminance"),
]

plt.figure(figsize=(11, 3))
for panel, (img, title) in enumerate(imgs, 1):
    plt.subplot(1, 3, panel)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
plt.tight_layout()
plt.show()
