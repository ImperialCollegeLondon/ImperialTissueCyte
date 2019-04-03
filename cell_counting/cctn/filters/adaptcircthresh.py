"""
################################################################################
Adaptive Local Circularity Thresholding Function
Author: Gerald M

This function returns an image which has been locally thresholded based on
circularity of objects.

Updates
20.02.19 - Added CIRCLIM as the circularity limit and PAR for the parallel
           computation as input variables. The parallel code has been modified
           to terminate once the CIRCLIM condition has been met to prevent
           unneccessary calculation. circularity() now returns a tuple with the
           threshold to allow that condition to be checked. Speed increase from
           5.00775 to 0.6733 seconds.
02.04.19 - Changed the circularity so at every iteration, those matching
           circularity condition are kept and added to new label image. This
           means that circualrity threshold is adaptive for the local background.
################################################################################
"""

import scipy.ndimage
import numpy as np
import math
from multiprocessing import Pool, cpu_count, Array, Manager
from functools import partial
from skimage.measure import regionprops, label
from PIL import Image

def func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def circularity(thresh, A, SIZE, CIRCLIM):
    A_thresh = (A>thresh).astype(int)

    A_thresh = scipy.ndimage.morphology.binary_fill_holes(A_thresh).astype(int)

    #Image.fromarray(A_thresh.astype(float)).save('/Users/gm515/Desktop/temp/circ/'+str(thresh)+'.tif')
    A_label = label(A_thresh, connectivity=A_thresh.ndim)

    # Find circularity
    circfunc = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

    circ = [circfunc(region) for region in regionprops(A_label) if region.perimeter>0]
    areas = [region.area for region in regionprops(A_label) if region.perimeter>0]
    labels = [region.label for region in regionprops(A_label) if region.perimeter>0]

    A_out = np.full(A.shape, False)

    if len(areas) > 0:
        for i, _ in enumerate(areas):
            if areas[i] > SIZE and areas[i] < SIZE*10 and circ[i] > CIRCLIM:
                # (row, col) centroid
                # So flip the order for (col, row) as (x, y)
                A_out += A_label==labels[i]

    return A_out

def adaptcircthresh(A,SIZE,THRESHLIM,CIRCLIM,PAR=False):
    thresh_int = 2
    A_out = np.full(A.shape, False)

    if PAR:
        thresh_all = np.arange(THRESHLIM,int(np.max(A)/4),thresh_int)
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
        for thresh in np.arange(THRESHLIM,int(np.max(A)/4),thresh_int):
            A_out += circularity(thresh, A, SIZE, CIRCLIM)

    return A_out
