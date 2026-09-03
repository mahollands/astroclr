import numpy as np
import matplotlib.pyplot as plt
from astroclr import RGB_Color, IRGB_Color

xim, yim = np.meshgrid(
    np.linspace(-1.2, 1.2, 500),
    np.linspace(-1.2, 1.2, 500),
)

rim = np.hypot(xim, yim)

V = 1.0
S = rim
C = V*S

H = np.degrees(np.arctan2(yim, xim))
H_ = H / 60 % 6
X = C * (1 - np.abs(H_ % 2 - 1))

condlist = [
    (S <= 1) & (H_ < 1),
    (S <= 1) & (H_ < 2),
    (S <= 1) & (H_ < 3),
    (S <= 1) & (H_ < 4),
    (S <= 1) & (H_ < 5),
    (S <= 1) & (H_ < 6),
]

R1 = np.select(
    condlist = condlist,
    choicelist = [C, X, 0, 0, X, C],
    default = np.zeros_like(X),
)

G1 = np.select(
    condlist = condlist,
    choicelist = [X, C, C, X, 0, 0],
    default = np.zeros_like(X),
)

B1 = np.select(
    condlist = condlist,
    choicelist = [0, 0, X, C, C, X],
    default = np.zeros_like(X),
)

m = V - C
R, G, B = [(255*np.clip(c + m, 0, 1)).astype(int) for c in (R1, G1, B1)]
IRGB = IRGB_Color(R, G, B)

plt.figure(figsize=(5, 5))
plt.imshow(IRGB.values_image)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.show()
