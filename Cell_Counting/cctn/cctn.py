#!/usr/bin/python

# cell analysis
#
# Cell counting script to count cells across z-stack images

################################################################################
## Module import
################################################################################

import os
import time
import numpy
import math
import matplotlib.pyplot as plt
from filters.gaussmedfilt import gaussmedfilt
from filters.medfilt import medfilt
from filters.circthresh import circthresh
from PIL import Image

################################################################################
## User defined parameters
################################################################################

# Fill in the following details to choose the analysis parameters
mask = True
over_sample = True
xy_res = 12.5
z_res = 5
# Fill in structure_list using acronyms and separating structures with a ','
# E.g. 'LGd, LGv, IGL, RT'
structure_list = ''

# Cell descriptors
size = 900
radius = 25

# Directory of files to count
ch1_count_path = '/Users/gm515/Desktop/Cell Counting/tiff ch1'
ch2_count_path = '/Users/gm515/Desktop/Cell Counting/tiff ch2'
# Number of files [None,None] for all, or [start,end] for specific range
number_files = [1,1]

################################################################################
## Initialisation parameters
################################################################################

# List of files to count
count_files = []
count_files += [each for each in os.listdir(ch2_count_path) if each.endswith('.tif')]
count_files = sorted(count_files)
if number_files[0] != None:
    count_files = count_files[number_files[0]-1:number_files[1]]

small_count = 0
area_count = 0
overlapping_small_count = 0
overlapping_area_count = 0
prev_image = []
overlap_trigger = False

################################################################################
## Counting
################################################################################

new_cells = [None]*len(count_files)
for slice_number in range(1,len(count_files)+1):
    # Load image and convert to dtype=float
    ch1_image = numpy.array(Image.open(ch1_count_path+'/'+count_files[slice_number-1])).astype(numpy.float)/numpy.iinfo('uint16').max
    ch2_image = numpy.array(Image.open(ch2_count_path+'/'+count_files[slice_number-1])).astype(numpy.float)/numpy.iinfo('uint16').max
    image_copy = numpy.zeros(image.shape)

    # Sliding window
    windowSize = [60, 60]
    stepSize = int(windowSize[0]/4)

    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            image_filt = medfilt(image[y:y + windowSize[1], x:x + windowSize[0]],3)
            image_copy[y:y + windowSize[1], x:x + windowSize[0]] = numpy.max(image_filt)

    image_filt = gaussmedfilt(image,3,1)

    plt.ion()
    for thresh in xrange(0, int(numpy.max(image_filt)*1000), int(numpy.max(image_filt)*1000/20)):

        idx = (image_filt>thresh/1000.)
        image_thresh = image_filt
        image_thresh[~idx] = 0
        plt.imshow(image_filt/numpy.max(image_filt))
        plt.title(str(thresh/1000.))
        plt.pause(0.5)


    # Perform gaussmedfilt function to remove large structures
    #image_copy = gaussmedfilt(image,5,2.5)
    # Perform median filter function small kernel to smoothen remaining structure
    #image_copy = medfilt(image_copy,2)


    # Perform histogram partitioning
    #thresh_interval = 0.05
    #for thresh in numpy.arange(thresh_interval,1+thresh_interval,thresh_interval):
    #    image_copy[(image_copy>thresh-thresh_interval) & (image_copy<=thresh)] = thresh
