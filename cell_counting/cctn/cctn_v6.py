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

Version 6 - v6 (Python 3)
This version incorporates parallelisation using a queuing system and adds new
circularity threshold from v4. All non-existant structures are removed prior to
counting. In addition, hemisphere atlas is used to classifier hemispheres.
Each image file is loaded in sequential order with ALL strucures counted, rather
than counting in structures in a sequential order (v5). This means the overhead
of image loading is restricted to once per image rather than individually per
structure, as before. A file lock and unlock is implemented to allow multiple
thread writing to the same files.

Instructions:
1) Run from command line with input parameters
################################################################################
"""

################################################################################
## Module import
################################################################################

import argparse
import collections
import copy
import csv
import cv2
import json
import fcntl
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

def slack_message(text, channel, username):
    from urllib import request, parse
    import json

    post = {"text": "{0}".format(text),
        "channel": "{0}".format(channel),
        "username": "{0}".format(username),
        "icon_url": "https://github.com/gm515/gm515.github.io/blob/master/Images/imperialstplogo.png?raw=true"}

    try:
        json_data = json.dumps(post)
        req = request.Request('https://hooks.slack.com/services/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',
            data=json_data.encode('ascii'),
            headers={'Content-Type': 'application/json'})
        resp = request.urlopen(req)
    except Exception as em:
        print("EXCEPTION: " + str(em))

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

def progressBar(sliceno, value, endvalue, statustext, bar_length=50):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)) + '/'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rSlice {0} [{1}] {2}%\n{3}".format(sliceno, arrow + spaces, int(round(percent * 100)), statustext))
        sys.stdout.flush()

def cellcount(imagequeue, radius, size, circ_thresh, use_medfilt):
    while True:
        item = imagequeue.get()
        if item is None:
            break
        else:
            slice_number, image, hemseg_image, row_idx, col_idx, count_path, name = item
            centroids = []

            if image.shape[0]*image.shape[1] > (radius*2)**2 and np.max(image) != 0.:
                # Perform gaussian donut median filter
                if use_medfilt:
                    image = gaussmedfilt(image, 3, 1.5)
                else:
                    # image = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=3)
                    image = cv2.medianBlur(np.uint8(image), 7)

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

                        if row_idx is not None:
                            # Convert coordinate of centroid to coordinate of whole image if mask was used
                            if hemseg_image is not None:
                                coordfunc = lambda celly, cellx : (col_idx[cellx], row_idx[celly], slice_number, int(hemseg_image[celly,cellx]))
                            else:
                                coordfunc = lambda celly, cellx : (col_idx[cellx], row_idx[celly], slice_number)
                        else:
                            coordfunc = lambda celly, cellx : (cellx, celly, slice_number)

                        # Centroids are currently (row, col) or (y, x)
                        # Flip order so (x, y) using coordfunc
                        centroids = [coordfunc(int(c[0]), int(c[1])) for c in centroids]

            # Write out results to file
            csv_file = count_path+'/counts_v6/'+str(name)+'_v6_count_INQUEUE.csv'

            while True:
                try:
                    with open(csv_file, 'a+') as f:
                        # Lock file during writing
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        csv.writer(f, delimiter=',').writerows(centroids)

                        # Unlock file and clsoe out
                        fcntl.flock(f, fcntl.LOCK_UN)
                    break
                except IOError as e:
                    # raise on unrelated IOErrors
                    if e.errno != errno.EAGAIN:
                        raise
                    else:
                        time.sleep(0.1)

            # print('Finished - Queue position: '+str(slice_number)+' Worker ID: '+str(current_process())+' Structure: '+str(name))

if __name__ == '__main__':
    ################################################################################
    ## User defined parameters via command line arguments
    ################################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument('imagepath', default=[], type=str, help='Image directory path for counting')
    parser.add_argument('-maskpath', default=[], type=str, dest='maskpath', help='Annotation file path for masking')
    parser.add_argument('-hempath', default=[], type=str, dest='hempath', help='Hemisphere file path for hemisphere classification')
    parser.add_argument('-structures', default=[], type=str, dest='structures', help='List of structures to count within')
    parser.add_argument('-oversample', action='store_false', default=True, dest='oversample', help='Oversample correction')
    parser.add_argument('-start', default=None, type=int, dest='start', help='Start image number if required')
    parser.add_argument('-end', default=None, type=int, dest='end', help='End image number if required')
    parser.add_argument('-medfilt', default=False, action='store_true', dest='medfilt', help='Use custom median donut filter')
    parser.add_argument('-circthresh', default=0.7, type=float, dest='circthresh', help='Circularity threshold value')
    parser.add_argument('-xyvox', default=0.54, type=float, dest='xyvox', help='XY voxel size')
    parser.add_argument('-zvox', default=10., type=float, dest='zvox', help='Z voxel size')
    parser.add_argument('-ncpu', default=6, type=int, dest='ncpu', help='Number of CPUs to use')
    parser.add_argument('-size', default=200., type=float, dest='size', help='Approximate radius of detected objects')
    parser.add_argument('-radius', default=6, type=float, dest='radius', help='Approximate radius of detected objects')

    args = parser.parse_args()

    count_path = args.imagepath
    mask_path = args.maskpath
    hem_path = args.hempath
    structure_list = args.structures
    over_sample = args.oversample
    number_files = [None, None]
    number_files[0] = args.start
    number_files[1] = args.end
    use_medfilt = args.medfilt
    circ_thresh = args.circthresh
    xyvox = args.xyvox
    zvox = args.zvox
    ncpu = args.ncpu
    size = args.size
    radius = args.radius

    if mask_path:
        mask = True
    else:
        mask = False
    if hem_path:
        hem = True
    else:
        hem = False

    print ('User defined parameters')
    print( "Image path: {} \nAnnotation path: {} \nHemisphere path: {} \nStructure list: {} \nOversample: {} \nStart: {} \nEnd: {} \nCustom median donut filter: {} \nCircularity threshold: {} \nXYvox: {} \nZvox: {} \nncpu: {} \nSize: {} \nRadius: {}".format(
            count_path,
            mask_path,
            hem_path,
            structure_list,
            over_sample,
            number_files[0],
            number_files[1],
            use_medfilt,
            circ_thresh,
            xyvox,
            zvox,
            ncpu,
            size,
            radius
            ))
    print ('')

    ################################################################################
    ## Initialisation
    ################################################################################

    # Create directory to hold the counts in same folder as the images
    if not os.path.exists(count_path+'/counts_v6'):
        os.makedirs(count_path+'/counts_v6')

    # List of files to count
    count_files = []
    count_files += [each for each in os.listdir(count_path) if each.endswith('.tif')]
    count_files = natsorted(count_files)
    if number_files[0] != None:
        count_files = count_files[number_files[0]-1:number_files[1]]
    print ('Counting in files: '+count_files[0]+' to '+count_files[-1])

    ################################################################################
    ## Retrieving structures IDs
    ################################################################################

    if mask:
        file, extension = os.path.splitext(mask_path)
        if extension == '.nii':
            seg = nib.load(mask_path).get_data()
        else:
            seg = io.imread(mask_path)
        print ('Loaded segmentation atlas')

    if hem:
        file, extension = os.path.splitext(hem_path)
        if extension == '.nii':
            hemseg = nib.load(hem_path).get_data()
        else:
            hemseg = io.imread(hem_path)
        print ('Loaded hemisphere atlas')

    ids = []
    acr = []
    index = np.array([[],[],[]])
    if mask:
        anno_file = json.load(open('2017_annotation_structure_info.json'))
        structure_list = [x.strip() for x in structure_list.lower().split(",")]
        for elem in structure_list:
            a, i = get_structure(anno_file['children'], elem)[1:]
            for name, structure in zip(a, i):
                if structure in seg:
                        index = np.concatenate((index, np.array(np.nonzero(structure == seg))), axis=1)
                        ids.append(structure)
                        acr.append(name)
                else:
                    print (name+' not found -> Removed')
    else:
        ids.extend(['None'])
        acr.extend(['None'])

    ################################################################################
    ## Counting
    ################################################################################
    print ('')

    tstart = time.time()

    structure_index = 0

    ################################################################################
    ## Loop through each slice and count in chosen structure
    ################################################################################
    proceed = True

    if mask:
        index = np.array([[],[],[]])
        index = np.concatenate((index, np.where(np.isin(seg,ids))), axis=1)

        if index.size > 0:
            zmin = int(index[0].min())
            zmax = int(index[0].max())
        else:
            proceed = False
    else:
        zmin = 0
        zmax = len(count_files)


    if proceed:
        # Create a Queue and push images to queue
        print ('Setting up Queue')
        imagequeue = Queue()

        # Start processing images
        print ('Creating threads to process Queue items')
        imageprocess = Pool(ncpu, cellcount, (imagequeue, radius, size, circ_thresh, use_medfilt))
        print ('')

        for slice_number in range(zmin,zmax):
            # Load image and convert to dtype=float and scale to full 255 range
            image = Image.open(count_path+'/'+count_files[slice_number])
            temp_size = image.size
            image = np.array(image).astype(float)
            image = np.multiply(np.divide(image,np.max(image)), 255.)

            if mask:
                # Get annotation image for slice
                mask_image = np.array(Image.fromarray(seg[slice_number]).resize(tuple([int(x) for x in temp_size]), Image.NEAREST))

            # Initiate empty lists
            row_idx = []
            col_idx = []

            ################################################################################
            ## Loop through slices based on cropped boundaries and store into one array
            ################################################################################
            row_idx_array = None
            col_idx_array = None
            # pxvolume = 0

            # Loop through structures available in each slice
            for name, structure in zip(acr,ids):
                # If masking is not required, submit to queue with redundat variables
                if not mask:
                    imagequeue.put((slice_number, image, [None], [None], [None], count_path, name))
                    print (count_path.split(os.sep)[3]+' Added slice: '+str(slice_number)+' Queue position: '+str(slice_number-zmin)+' Structure: '+str(name)+' [Memory info] Usage: '+str(psutil.virtual_memory().percent)+'% - '+str(int(psutil.virtual_memory().used*1e-6))+' MB')
                else:
                    if structure in mask_image:
                        # Resize mask
                        mask_image_per_structure = copy.deepcopy(mask_image)
                        mask_image_per_structure[mask_image_per_structure!=structure] = 0

                        # Use mask to get global coordinates
                        idx = np.ix_(mask_image_per_structure.any(1),mask_image_per_structure.any(0))
                        row_idx = idx[0].flatten()
                        col_idx = idx[1].flatten()

                        # Apply crop to image and mask then apply mask
                        image_per_structure = copy.deepcopy(image)
                        image_per_structure = image_per_structure[idx]
                        mask_image_per_structure = mask_image_per_structure[idx]

                        mask_image_per_structure = cv2.medianBlur(np.array(mask_image_per_structure).astype(np.uint8), 121) # Apply median filter to massively reduce box like boundary to upsized mask

                        image_per_structure[mask_image_per_structure==0] = 0

                        # Keep track of pixel volume
                        # pxvolume += mask_image_per_structure.any(axis=-1).sum()

                        mask_image_per_structure = None

                        if hem:
                            hemseg_image_per_structure = np.array(Image.fromarray(hemseg[slice_number]).resize(tuple([int(x) for x in temp_size]), Image.NEAREST))
                            hemseg_image_per_structure = hemseg_image_per_structure[idx]

                        # Add queue number, image, row and col idx to queue
                        imagequeue.put((slice_number, image_per_structure, hemseg_image_per_structure, row_idx, col_idx, count_path, name))

                        image_per_structure = None
                        hemseg_image_per_structure = None

                        statustext = count_path.split(os.sep)[3]+' Added slice: '+str(slice_number)+' Queue position: '+str(slice_number-zmin)+' Structure: '+str(name)+' [Memory info] Usage: '+str(psutil.virtual_memory().percent)+'% - '+str(int(psutil.virtual_memory().used*1e-6))+' MB'

                        progressBar(slice_number, slice_number-zmin, zmax-zmin, statustext)

        for close in range(ncpu):
            imagequeue.put(None)

        imageprocess.close()
        imageprocess.join()

        print ('Finished queue processing')
        print ('')
        print ('Performing oversampling correction...')

        for name in acr:
            with open(count_path+'/counts_v6/'+str(name)+'_v6_count_INQUEUE.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                centroids = []
                retainedcentroids = []

                for row in csvReader:
                    centroids.append(row)

            centroids = np.array(centroids).astype(int)

            for x, y in zip(np.unique(centroids[:,2])[:-1], np.unique(centroids[:,2])[1:]):
                # Check if they're in successive slices otherwise they can't be oversampled
                if y-x == 1:
                    xcells = centroids[centroids[:,2]==x]
                    ycells = centroids[centroids[:,2]==y]
                    retainedcentroids.extend([xcell.tolist() for xcell in xcells if all(distance(xcell,ycell)>radius**2 for ycell in ycells)])

            with open(count_path+'/counts_v6/'+str(name)+'_v6_count.csv', 'w+') as f:
                csv.writer(f, delimiter=',').writerows(retainedcentroids)

            # Save volume
            # vol_file = count_path+'/counts/volumes.csv'
            # with open(vol_file, 'a') as f:
            #     csv.writer(f, delimiter=',').writerows(np.array([[str(name), pxvolume*xyvox*zvox]]))

            os.remove(count_path+'/counts_v6/'+str(name)+'_v6_count_INQUEUE.csv')
            print (name+' oversampling done')

    print ('~Fin~')
    print (count_path)

    minutes, seconds = divmod(time.time()-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    text = 'Counting completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)

    print (text)
    slack_message(text, '#cctn', 'CCTN')
