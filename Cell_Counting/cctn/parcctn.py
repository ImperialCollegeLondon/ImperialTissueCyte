#============================================================================================
# Cell Counting in Target Nuclei Script
# Author: Gerald M
#
# This script performs automated cell counting in anatomical structures of interest, or a
# a stack of TIFFs. It works by first determining an ideal threshold based on the circularity
# of objects. Then by tracking cells/objects over multiple layers to account for oversampling.
# The output provides a list of coordinates for identified cells. This should then be fed
# into the image predictor to confirm whether objects are cells or not.
#
# Instructions:
# 1) Run the script in a Python IDE
#============================================================================================

################################################################################
## Module import
################################################################################

import os, time, numpy, math, json, warnings, csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.misc as sc
from filters.gaussmedfilt import gaussmedfilt
from filters.medfilt import medfilt
from filters.parcircthresh import parcircthresh
from skimage.measure import regionprops, label
from multiprocessing import Pool, cpu_count, Array, Manager
from functools import partial
from PIL import Image
from skimage import io

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

################################################################################
## Function definitions
################################################################################

def get_children(json_obj, ids):
    for obj in json_obj:
        if obj['children'] == []:
            ids.append(obj['id'])
        else:
            get_children(obj['children'], ids)
    return ids

def get_structure(json_obj, acronym):
    found = (False, None)
    for obj in json_obj:
        if obj['acronym'].lower() == acronym:
            #print obj['acronym'], obj['id']
            ids = get_children(obj['children'], [])
            #print ids
            if ids == []:
                ids = [obj['id']]
                return (True, ids)
            else:
                ids.append(obj['id'])
                return (True, ids)
        else:
            found = get_structure(obj['children'], acronym)
            if found:
                return found

def append_total_cells(slice_number, cells):
    total_cells.update({slice_number+1 : np.array(cells)})

def slice_count(slice_file, all_slice_files, temp, ymin, ymax, xmin, xmax, seg, structure, total_cells):
    # Load image and convert to dtype=float
    image = np.array(Image.open(slice_file))[ymin:ymax, xmin:xmax]
    image = np.multiply(np.divide(image,65535.), 255.)

    # Apply mask if required
    if seg:
        mask_image = np.array(Image.fromarray(seg[all_slice_files.index(slice_file)]).resize(temp.size, Image.NEAREST))[ymin:ymax, xmin:xmax]
        mask_image[mask_image!=structure] = 0
        image[mask_image==0] = 0

    # Perform gaussian donut median filter
    image = gaussmedfilt(image, 5, 2.5)

    if np.max(image) != 0.:
        image = np.multiply(np.divide(image, np.max(image)), 255.)

        # Perform circularity threshold
        image = image>parcircthresh(image,size,50)

        # Make label image to manipulate objects
        image_label = label(image, connectivity=image.ndim)

        circfunc = lambda r: (4 * math.pi * r.area) / ((r.perimeter * r.perimeter) + 0.00000001)

        circ = [circfunc(region) for region in regionprops(image_label)]
        areas = [region.area for region in regionprops(image_label)]
        labels = [region.label for region in regionprops(image_label)]
        centroids = [region.centroid for region in regionprops(image_label)]

        image = np.zeros_like(image_label)

        # Threshold the objects based on size and circularity and store centroids
        cells = []
        for i, _ in enumerate(areas):
            if areas[i] > size and areas[i] < size*4 and circ[i] > 0.65:
                # (row, col) centroid
                cells.append(centroids[i])
                image += image_label==labels[i]
    else:
        cells = []

    append_total_cells(all_slice_files.index(slice_file), cells)

    print str(all_slice_files.index(slice_file)+1)

if __name__ == '__main__':
    ################################################################################
    ## User defined parameters
    ################################################################################

    # Fill in the following details to choose the analysis parameters
    mask = False
    over_sample = True
    xy_res = 10
    z_res = 5

    # Fill in structure_list using acronyms and separating structures with a ','
    # E.g. 'LGd, LGv, IGL, RT'
    if mask:
        structure_list = ''

    # Cell descriptors
    size = 200
    radius = 25

    # Directory of files to count
    #count_path = raw_input('Image path (drag-and-drop): ').rstrip()
    count_path = '/Users/gm515/Documents/Cell Counting/full sized tiff/full sized tiff A'
    # Number of files [None,None] for all, or [start,end] for specific range
    number_files = [None,None]

    ################################################################################
    ## Initialisation parameters
    ################################################################################

    # List of files to count
    count_files = []
    count_files += [each for each in os.listdir(count_path) if each.endswith('.tif')]
    count_files = sorted(count_files)
    if number_files[0] != None:
        count_files = count_files[number_files[0]-1:number_files[1]]

    ################################################################################
    ## Retrieving structures IDs
    ################################################################################

    if mask:
        #path = raw_input('NII/TIFF file path (drag-and-drop): ').rstrip()
        path = '/Users/gm515/Documents/Registration/aMAP-0.0.1/atlas/2017/annotation_10.tif'
        file, extension = os.path.splitext(path)
        if extension == '.nii':
            seg = nib.load(path).get_data()
        else:
            seg = io.imread(path)

    ids = []
    if mask:
        anno_file = json.load(open('/Users/gm515/Documents/Registration/aMAP-0.0.1/atlas/2017/2017_annotation_structure_info.json'))
        structure_list = structure_list.lower().split(",")
        for elem in structure_list:
            ids.append(np.array(get_structure(anno_file['children'], elem)[1]))
    else:
        ids.append('None')

    ################################################################################
    ## Counting
    ################################################################################

    tstart = time.time()

    # Dictionary to store centroids - each key is a new slice
    manager = Manager()
    total_cells = manager.dict()

    temp = Image.open(count_path+'/'+count_files[0])
    if mask:
        scale = float(temp.size[1])/seg[1].shape[0]

    for structure in ids:
        ################################################################################
        ## Obtain crop information for structure if mask required
        ################################################################################
        if mask:
            index = np.array([[],[],[]])
            for elem in structure:
                if elem in seg:
                    index = np.concatenate((index, np.array(np.nonzero(elem == seg))), axis=1)
                else:
                    structure.remove(elem)

            zmin = int(index[0].min())
            zmax = int(index[0].max())
            ymin = int(index[1].min()*scale)
            ymax = int(index[1].max()*scale)
            xmin = int(index[2].min()*scale)
            xmax = int(index[2].max()*scale)
        else:
            zmin = 0
            zmax = len(count_files)
            ymin = 0
            ymax = temp.size[1]
            xmin = 0
            xmax = temp.size[0]

        ################################################################################
        ## Loop through slices based on cropped boundaries
        ################################################################################
        slice_files = []
        for slice_number in range(zmin,zmax):
            slice_files.append(count_path+'/'+count_files[slice_number])

        pool = Pool(cpu_count())
        if mask:
            results = [pool.map(partial(slice_count, all_slice_files=slice_files, temp=temp, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax, seg=seg, structure=structure, total_cells=total_cells), slice_files)]
        else:
            results = [pool.map(partial(slice_count, all_slice_files=slice_files, temp=temp, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax, seg=None, structure=structure, total_cells=total_cells), slice_files)]
        pool.close()
        pool.join

        # Peform correction for overlap
        if over_sample:
            for slice_number in range(len(total_cells)-1):
                if total_cells.values()[slice_number].size != 0 and total_cells.values()[slice_number+1].size != 0:
                    for cell in total_cells.values()[slice_number+1]:
                        total_cells[slice_number] = total_cells.values()[slice_number][np.invert(abs( (cell[0] - total_cells.values()[slice_number][:,0])**2 + (cell[1] - total_cells.values()[slice_number][:,1])**2 ) < radius**2)]

        num_cells = sum(map(len, total_cells.values()))

        print num_cells

        csv_file = structure+'_count.csv'
        with open(csv_file, 'w+') as f:
            for key in total_cells.keys():
                if total_cells.values()[key-1].size:
                    csv.writer(f, delimiter=',').writerows(np.round(np.concatenate((np.ones((total_cells.values()[key-1].shape[0], 1))*key, total_cells.values()[key-1]), axis=1)))

    print 'Fin'

    minutes, seconds = divmod(time.time()-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    print 'Counting completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)
