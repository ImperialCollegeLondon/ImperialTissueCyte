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
import nibabel as nib

################################################################################
## User defined parameters
################################################################################

# Fill in the following details to choose the analysis parameters
mask = True
over_sample = True
xy_res = 10
z_res = 5
# Fill in structure_list using acronyms and separating structures with a ','
# E.g. 'LGd, LGv, IGL, RT'
structure_list = ['LGd','LGv','RT']

# Cell descriptors
size = 300
radius = 12

# Directory of files to count
ch2_count_path = raw_input('Counting folder path (drag-and-drop): ').rstrip()
# Number of files [None,None] for all, or [start,end] for specific range
number_files = (1,1)

# Read CSV file containing structure information
structure_csv = numpy.genfromtxt('2017_annotation_structure_info.csv', delimiter=",", dtype=None)
structure_header = structure_csv[0,:]
structure_data = structure_csv[1:len(structure_csv),:]

################################################################################
## Initialisation parameters
################################################################################

# List of files to count
count_files = []
count_files += [each for each in os.listdir(ch2_count_path) if each.endswith('.tif')]
count_files = sorted(count_files, key=len)
number_files = eval(raw_input('Range of files to count, as tuple. Leave empty for default all (1,end): ').rstrip() or (0,len(count_files)))
if number_files != (0, len(count_files)):
    count_files = count_files[int(number_files[0])-1:int(number_files[1])]

if mask:
    seg_path = raw_input('Segmented NII file (drag-and-drop): ').rstrip()
    seg_file = nib.load(seg_path)
    seg_file = seg_file.get_data()

small_count = 0
area_count = 0
overlapping_small_count = 0
overlapping_area_count = 0
prev_image = []
overlap_trigger = False

################################################################################
## Finding structures
################################################################################

if mask:
    structure_index = numpy.zeros(len(structure_list))
    for structure in range(0,len(structure_list)):
        idx = numpy.where(structure_data[:,2] == structure_list[structure])
        if not idx[0]:
            raise UserWarning('Could not find structure')
        structure_index[structure] = idx[0]

################################################################################
## Counting
################################################################################

new_cells = [None]*len(count_files)
for slice_number in range(1,len(count_files)+1):
    # Load image and convert to dtype=float
    image = numpy.array(Image.open(ch2_count_path+'/'+count_files[slice_number-1])).astype(numpy.float)/numpy.iinfo('uint16').max

    # Apply gaussian donut filter
    image_filt = gaussmedfilt(image,5,2.5)
