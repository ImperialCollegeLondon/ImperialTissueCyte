"""
################################################################################
Circularity Based Thresholding Function
Author: Gerald M

This function returns the threshold at which the image has a mean circularity
above a chosen threshold value.

Updates
20.02.19 - Added CIRCLIM as the circularity limit and PAR for the parallel
           computation as input variables. The parallel code has been modified
           to terminate once the CIRCLIM condition has been met to prevent
           unneccessary calculation. circularity() now returns a tuple with the
           threshold to allow that condition to be checked. Speed increase from
           5.00775 to 0.6733 seconds.
################################################################################
"""

import scipy.ndimage
from scipy.optimize import curve_fit
import numpy as np
import math, csv
from multiprocessing import Pool, cpu_count, Array, Manager
from functools import partial
from skimage.measure import regionprops, label
from PIL import Image
import matplotlib.pyplot as plt
import time

def func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def circularity(thresh, A, SIZE):
    A_thresh = (A>thresh).astype(int)

    A_thresh = scipy.ndimage.morphology.binary_fill_holes(A_thresh).astype(int)

    #Image.fromarray(A_thresh.astype(float)).save('/Users/gm515/Desktop/temp/circ/'+str(thresh)+'.tif')
    A_label = label(A_thresh, connectivity=A_thresh.ndim)

    # Find circularity
    circfunc = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

    circ = [circfunc(region) for region in regionprops(A_label) if region.area>SIZE and region.area<SIZE*10 and region.perimeter>0]

    if len(circ)>0:
        return (thresh, np.mean(np.array(circ)))
    else:
        return (thresh, 0.)

def circthresh(A,SIZE,THRESHLIM,CIRCLIM,PAR=False):
    thresh_int = 2

    if PAR:
        thresh_all = np.arange(THRESHLIM,np.max(A),thresh_int)
        T = 0
        # Get mean circularity
        pool = Pool(cpu_count())
        # circ_all = np.squeeze(np.array([pool.map(partial(circularity, A=A, SIZE=SIZE), thresh_all)]), axis=0)
        circ_all = pool.imap(partial(circularity, A=A, SIZE=SIZE), thresh_all)
        pool.close()
        for th, circ in circ_all:
            if circ > CIRCLIM:  # or set other condition here
                pool.terminate()
                T = th
                break
        pool.join()

    else:
        thresh = THRESHLIM
        while (circularity(thresh, A, SIZE)[1] < CIRCLIM) & (thresh < np.max(A)):
            thresh += thresh_int
        T = thresh

    # If threshold from fit is less than minimum threshold limit, set optimum threshold to minimum
    if T<THRESHLIM:
        T = THRESHLIM

    return T
