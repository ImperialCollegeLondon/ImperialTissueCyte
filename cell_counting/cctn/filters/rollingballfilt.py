"""
################################################################################
Rolling ball filter
Author: Gerald M

This function returns a rolling ball subtracted image.

Parameters
data : ndarray type uint8
    image data (assumed to be on a regular grid)
ball_radius : float
    the radius of the ball to roll
spacing : int or sequence
    the spacing of the image data
top : bool
    whether to roll the ball on the top or bottom of the data
kwargs : key word arguments
    these are passed to the ndimage morphological operations

Returns
data_nb : ndarray
    data with background subtracted as uint8
bg : ndarray
    background that was subtracted from the data

Updates
20.02.19 - Modified the code to use OpenCV for faster image processing.
################################################################################
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage._ni_support import _normalize_sequence
import cv2

def rolling_ball_filter(data, ball_radius, spacing=None, top=False, **kwargs):
    data = data.astype(np.int16)
    ndim = data.ndim
    if spacing is None:
        spacing = _normalize_sequence(1, ndim)
    else:
        spacing = _normalize_sequence(spacing, ndim)

    radius = np.asarray(_normalize_sequence(ball_radius, ndim))
    mesh = np.array(np.meshgrid(*[np.arange(-r, r + s, s) for r, s in zip(radius, spacing)], indexing="ij"))
    structure = np.uint8(2 * np.sqrt(2 - ((mesh / radius.reshape(-1, *((1,) * ndim)))**2).sum(0)))
    structure[~np.isfinite(structure)] = 0

    if not top:
        # ndi.white_tophat(y, structure=structure, output=background)
        background = cv2.erode(data, structure, **kwargs)
        background = cv2.dilate(background, structure, **kwargs)
    else:
        # ndi.black_tophat(y, structure=structure, output=background)
        background = cv2.dilate(data, structure, **kwargs)
        background = cv2.erode(background, structure, **kwargs)

    data_corr = data - background
    data_corr[data_corr<0] = 0

    return data_corr.astype(np.uint8), background.astype(np.uint8)
