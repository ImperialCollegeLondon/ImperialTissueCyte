#============================================================================================
# Circularity Based Thresholding Function
# Author: Gerald M
#
# This function returns the threshold at which the image has the highest mean circularity.
#============================================================================================

import scipy.ndimage
from scipy.optimize import curve_fit
import numpy as np
import math, csv
from multiprocessing import Pool, cpu_count, Array, Manager
from functools import partial
from skimage.measure import regionprops, label
from PIL import Image

def func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def parcircthresh(A,SIZE,THRESHLIM):
    B = 5
    thresh_int = 5
    thresh_all = np.arange(thresh_int,255,thresh_int)

    circfunc = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

    circ_all = []
    for thresh in thresh_all:
        A_thresh = (A>thresh).astype(int)
        A_thresh = scipy.ndimage.morphology.binary_fill_holes(A_thresh).astype(int)

        #Image.fromarray(A_thresh.astype(float)).save('/Users/gm515/Desktop/temp/'+str(thresh)+'.tif')

        A_thresh = label(A_thresh, connectivity=A_thresh.ndim)

        # Find circularity
        circfunc = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

        circ = [circfunc(region) for region in regionprops(A_thresh) if region.area>SIZE and region.area<SIZE*4 and region.perimeter>0]

        if len(circ)>0:
            circ_all.append(np.mean(np.array(circ)))
        else:
            circ_all.append(0.)

    # Fit a polynomial and find optimum threshold
    func = np.polyfit(thresh_all, np.array(circ_all), 2)
    yfit = np.poly1d(func)

    T = np.max(yfit)

    # If threshold from fit is less than minimum threshold limit, set optimum threshold to threshold limit
    if T<THRESHLIM:
        T = THRESHLIM

    return T
