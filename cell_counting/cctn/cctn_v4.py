"""
################################################################################
Cell Counting in Target Nuclei Script
Author: Gerald M

This script performs automated cell counting in anatomical structures of
interest, or a stack of TIFFs. It works by first determining an ideal threshold
based on the circularity of objects. Then by tracking cells/objects over
multiple layers to account for oversampling. The output provides a list of
coordinates for identified cells. This should then be fed into the image
predictor to confirm whether objects are cells or not.

Version 4 - v4
This version incorporates parallelisation using a queuing system and adds new
circularity threshold.

Instructions:
1) Go to the user defined parameters from roughly line 140
2) Make changes to those parameters as neccessary
3) Execute the code in a Python IDE
################################################################################
"""

################################################################################
## Module import
################################################################################

import collections
import csv
import cv2
import json
import os
import psutil
import sys
import time
import warnings
import numpy as np
import nibabel as nib
from filters.gaussmedfilt import gaussmedfilt
from filters.adaptcircthresh import adaptcircthresh
from skimage.measure import regionprops, label
from PIL import Image
from skimage import io
from natsort import natsorted
from filters.rollingballfilt import rolling_ball_filter
from multiprocessing import Manager, Pool, Queue, current_process

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

################################################################################
## Function definitions
################################################################################

def distance(a, b):
    return (a[0] - b[0])**2  + (a[1] - b[1])**2

def get_children(json_obj, acr, ids):
    for obj in json_obj:
        if obj['children'] == []:
            acr.append(obj['acronym'])
            ids.append(obj['id'])
        else:
            acr.append(obj['acronym'])
            ids.append(obj['id'])
            get_children(obj['children'], acr, ids)
    return (acr, ids)

def get_structure(json_obj, acronym):
    found = (False, None)
    for obj in json_obj:
        if obj['acronym'].lower() == acronym:
            [acr, ids] = get_children(obj['children'], [], [])
            if ids == []:
                acr = [obj['acronym']]
                ids = [obj['id']]
                return (True, acr, ids)
            else:
                acr.append(obj['acronym'])
                ids.append(obj['id'])
                return (True, acr, ids)
        else:
            found = get_structure(obj['children'], acronym)
            if found:
                return found

def progressBar(sliceno, value, endvalue, bar_length=50):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '/'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rSlice {0} [{1}] {2}%".format(sliceno, arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

def cellcount(imagequeue, radius, size, circ_thresh, use_medfilt, res):
    while True:
        item = imagequeue.get()
        if item is None:
            break
        else:
            qnum, image, row_idx, col_idx = item
            centroids = []

            if image.shape[0]*image.shape[1] > (radius*2)**2 and np.max(image) != 0.:
                # Perform gaussian donut median filter
                if use_medfilt:
                    image = gaussmedfilt(image, 3, 1.5)
                else:
                    image = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=3)

                if np.max(image) != 0.:
                    # Perform rolling ball background subtraction to remove uneven background signal
                    image, background = rolling_ball_filter(np.uint8(image), 24)

                    if np.max(image) != 0.:
                        # Perform circularity threshold
                        image = adaptcircthresh(image,size,int(np.mean(image)),circ_thresh,False)

                        # Remove objects smaller than chosen size
                        image = label(image, connectivity=image.ndim)

                        # Get centroids list as (row, col) or (y, x)
                        centroids = [region.centroid for region in regionprops(image)]

                        # Convert coordinate of centroid to coordinate of whole image if mask was used
                        coordfunc = lambda celly, cellx : (row_idx[celly], col_idx[cellx])

                        # Centroids are currently (row, col) or (y, x)
                        # Flip order so (x, y)
                        centroids = [coordfunc(int(c[1]), int(c[0])) for c in centroids]

            # Append centroid information to shared dictionary
            res[qnum] = centroids
            print 'Finished processing queue position '+str(qnum)+' on worker '+str(current_process())

if __name__ == '__main__':
    ################################################################################
    ## User defined parameters - please fill in the parameters in this section only
    ################################################################################

    # Do you want to use a mask taken from a registered segmentation atlas
    mask = True

    # Do you want to perform over sampling correction?
    # Cells within a radius on successive images will be counted as one cell
    over_sample = True

    # If you are using a mask, input the mask path and the structures you want to count within
    # E.g. 'LGd, LGv, IGL, RT'
    if mask:
        mask_path = '/mnt/TissueCyte80TB/181218_Gerald_HET_2_Pt2/het2-Mosaic/het181218_segres_10um.tif'
        structure_list = 'LGd'

    # Input details for the cell morphology
    # Can be left as default values
    size = 200.
    radius = 12.

    # Input the directory path of the TIFF images for counting
    count_path = '/mnt/TissueCyte80TB/181218_Gerald_HET_2_Pt2/het2-Mosaic/Ch2_Stitched_Sections'

    # Of the images in the above directory, how many will be counted?
    # Number of files [None,None] for all, or [start,end] for specific range
    number_files = [1,500]

    # Do you want to use the donut median filter?
    use_medfilt = False

    # For the circularity threshold, what minimum background threshold should be set and what circularity value needs to be achieved
    # You can estimate this by loading an image in ImageJ, perform a gaussian filter radius 3, then perform a rolling ball background subtraction radius 8, and choose a threshold which limits remaining background signal
    circ_thresh = 0.8

    # Voxel size for volume calculation
    xyvox = 0.54
    zvox = 10.

    # Number of cpus to use for processing queue
    ncpu = 6

    ################################################################################
    ## Initialisation
    ################################################################################

    # Create directory to hold the counts in same folder as the images
    if not os.path.exists(count_path+'/counts'):
        os.makedirs(count_path+'/counts')

    # List of files to count
    count_files = []
    count_files += [each for each in os.listdir(count_path) if each.endswith('.tif')]
    count_files = natsorted(count_files)
    if number_files[0] != None:
        count_files = count_files[number_files[0]-1:number_files[1]]
    print 'Counting in files: '+count_files[0]+' to '+count_files[-1]

    ################################################################################
    ## Retrieving structures IDs
    ################################################################################

    if mask:
        #path = raw_input('NII/TIFF file path (drag-and-drop): ').rstrip()
        file, extension = os.path.splitext(mask_path)
        if extension == '.nii':
            seg = nib.load(mask_path).get_data()
        else:
            seg = io.imread(mask_path)
        print 'Loaded segmentation data'

    ids = []
    acr = []
    if mask:
        anno_file = json.load(open('2017_annotation_structure_info.json'))
        structure_list = [x.strip() for x in structure_list.lower().split(",")]
        for elem in structure_list:
            a, i = get_structure(anno_file['children'], elem)[1:]
            ids.extend(i)
            acr.extend(a)
    else:
        ids.extend('none')
        acr.extend('none')
    print 'Counting in structures: '+str(acr)

    ################################################################################
    ## Counting
    ################################################################################

    tstart = time.time()

    structure_index = 0

    for name, structure in zip(acr,ids):
        print 'Counting in '+str(name)
        proceed = True

        # Dictionary to store centroids - each key is a new slice number
        total_cells = dict()

        ################################################################################
        ## Obtain crop information for structure if mask required
        ################################################################################
        if mask:
            index = np.array([[],[],[]])
            if structure in seg:
                index = np.concatenate((index, np.array(np.nonzero(structure == seg))), axis=1)
            else:
                proceed = False

            if index.size > 0:
                zmin = int(index[0].min())
                zmax = int(index[0].max())
            else:
                proceed = False
        else:
            zmin = 0
            zmax = len(count_files)

        ################################################################################
        ## Check whether to proceed with the count
        ################################################################################
        if proceed:
            ################################################################################
            ## Loop through slices based on cropped boundaries and store into one array
            ################################################################################
            image_array = []
            row_idx_array = []
            col_idx_array = []
            pxvolume = 0

            # Create shared dictionary
            manager = Manager()
            res = manager.dict()

            # Create a Queue and push images to queue
            print 'Setting up Queue'
            imagequeue = Queue()

            # Start processing images
            print 'Creating threads to process Queue items'
            imageprocess = Pool(ncpu, cellcount, (imagequeue, radius, size, circ_thresh, use_medfilt, res))

            print 'Loading all images and storing into parallel array'
            for slice_number in range(zmin,zmax):
                # Load image and convert to dtype=float and scale to full 255 range
                image = Image.open(count_path+'/'+count_files[slice_number])
                temp_size = image.size
                image = np.array(image).astype(float)
                image = np.multiply(np.divide(image,np.max(image)), 255.)

                # Apply mask if required
                if mask:
                    # Resize mask
                    mask_image = np.array(Image.fromarray(seg[slice_number]).resize(tuple([int(x) for x in temp_size]), Image.NEAREST))
                    mask_image[mask_image!=structure] = 0

                    # Use mask to get global coordinates
                    idx = np.ix_(mask_image.any(1),mask_image.any(0))
                    row_idx = idx[0].flatten()
                    col_idx = idx[1].flatten()

                    # Apply crop to image and mask then apply mask
                    image = image[idx]
                    mask_image = mask_image[idx]

                    mask_image = cv2.medianBlur(np.array(mask_image).astype(np.uint8), 121) # Apply median filter to massively reduce box like boundary to upsized mask

                    image[mask_image==0] = 0
                    mask_image = None

                    # Keep track of pixel volume
                    pxvolume += mask_image.any(axis=-1).sum()

                # Add queue number, image, row and col idx to queue
                imagequeue.put((slice_number-zmin, image, row_idx, col_idx))
                print count_path.split(os.sep)[3]+' Added item to queue position '+str(slice_number-zmin)+' [Memory info] Percent: '+str(psutil.virtual_memory().percent)+' Used: '+str(int(psutil.virtual_memory().used*1e-6))+' MB'

                #progressBar(slice_number, slice_number-zmin, zmax-zmin)

            # Tell all pools to no longer wait
            for close in range(ncpu):
                imagequeue.put(None)

            imageprocess.close()
            imageprocess.join()

            print 'Finished queue processing'

            for slice_number, cells in zip(range(zmin,zmax+1),res.values()):
                total_cells.update({slice_number : cells})

            # Sort the dictionary so key are in order
            total_cells = collections.OrderedDict(sorted(total_cells.items()))

            # Peform correction for overlap
            if over_sample:
                print '\nCorrecting for oversampling'
                for index, (x, y) in enumerate(zip(total_cells.values()[1:], total_cells.values()[:-1])):
                    if len(x) * len(y) > 0:
                        total_cells[zmin+index] = [ycell for ycell in y if all(distance(xcell,ycell)>radius**2 for xcell in x)]

            # Count number of possible cells
            num_cells = sum(map(len, total_cells.values()))

            print str(name)+' '+str(num_cells)

            # Save cell centroids to csv
            csv_file = count_path+'/counts/'+str(name)+'_v2_count.csv'
            with open(csv_file, 'w+') as f:
                for key in sorted(total_cells.keys()):
                    if len(total_cells[key])>0:
                        csv.writer(f, delimiter=',').writerows(np.round(np.concatenate(([( (np.array(val))).tolist() for val in total_cells[key]], np.ones((len(total_cells[key]), 1))*(key+1)), axis=1)))

            # Save volume
            vol_file = count_path+'/counts/volumes.csv'
            with open(vol_file, 'a') as f:
                csv.writer(f, delimiter=',').writerows(np.array([[str(name), pxvolume*xyvox*zvox]]))

        structure_index += 1
        print ''

    print '~Fin~'

    minutes, seconds = divmod(time.time()-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    print 'Counting completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)
