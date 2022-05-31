import numpy as np
from skimage import exposure

def get_rgb(s2):
    s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
               "S2B12"]

    rgb_idx = [s2bands.index(b) for b in np.array(['S2B4', 'S2B3', 'S2B2'])]

    X = np.clip(s2, a_min=0, a_max=1)

    rgb = np.swapaxes(X[rgb_idx, :, :], 0, 2)
    # rgb = exposure.rescale_intensity(rgb)
    rgb = exposure.equalize_hist(rgb)
    # rgb = exposure.equalize_adapthist(rgb, clip_limit=0.1)
    # rgb = exposure.adjust_gamma(rgb, gamma=0.8, gain=1)

    rgb *= 255

    return rgb
