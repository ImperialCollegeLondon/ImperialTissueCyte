'''
Cell Counting Predictor - Parallel version
Author: Gerald M
#
This script uses a convolution neural network classifier, trained on manually identified
cells, to confirm whether a potential cell/neuron is correctly identified.

This version uses serial computation.

Installation:
1) Navigate to the folder containing cc_predictor_par.py

Instructions:
1) Pass in directory containing count files as first argument. Optionally pass in image
directory as second argument if count directory is not nested into the image directory
'''

from __future__ import division
import os, sys, warnings, time
import numpy as np
import pandas as pd
from PIL import Image
from xml.dom import minidom
from multiprocessing import Pool, cpu_count, Array, Manager
from contextlib import closing
from functools import partial
import tqdm
import csv
from natsort import natsorted
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import glob

# Warning supression and allowing large images to be laoded
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

#=============================================================================================
# Define function for predictions
#=============================================================================================

# # Appends cell coordinates to manager list - manager allows access from a thread
# def append_cell(coord):
#     cell_markers.append(coord)
#
# # Appends no-cell coordinates to manager list - manager allows access from a thread
# def append_nocell(coord):
#     nocell_markers.append(coord)

# Function to predict/classify object as cell or no-cell
def cellpredict(cell, img, cell_markers, nocell_markers):

    # Predict [1,0] for cell or [0,1] for no cell
    prediction = model.predict(np.asarray(img))

    if prediction[0][0] == 1: # Cell
        cell_value = 1
        cell_markers.append(cell)
        # image.array_to_img(img[0,:,:,:]).save('/Users/gm515/Desktop/cell_par/'+str(cell)+'.tif')
    else: # No cell
        cell_value = 0
        nocell_markers.append(cell)
        # image.array_to_img(img[0,:,:,:]).save('/Users/gm515/Desktop/nocell_par/'+str(cell)+'.tif')

    result[cell] = cell_value

    return


# Main function
if __name__ == '__main__':
    # Import modules - required for each independant thread
    import keras
    from keras.preprocessing import image
    from keras.models import load_model, model_from_json

    #=============================================================================================
    # User definied parameters
    #=============================================================================================

    # CNN model paths
    model_weights_path = 'models/2019_01_29/cc_model_2019_01_29.h5'
    model_json_path = 'models/2019_01_29/cc_model_2019_01_29.json'

    if len(sys.argv) == 2:
        try:
            sys.argv[1]
        except NameError:
            count_path = '/Volumes/TissueCyte/181024_Gerald_HET/het-Mosaic/Ch2_Stitched_Sections_New/counts'
            image_path = '/Volumes/TissueCyte/181024_Gerald_HET/het-Mosaic/Ch2_Stitched_Sections_New'
        else:
            count_path = str(sys.argv[1])
            image_path = str(sys.argv[1])[:-7]
    if len(sys.argv) == 3:
        try:
            sys.argv[1]
        except NameError:
            count_path = '/Volumes/TissueCyte/181024_Gerald_HET/het-Mosaic/Ch2_Stitched_Sections_New/counts'
        else:
            count_path = sys.argv[1]

        try:
            sys.argv[2]
        except NameError:
            image_path = '/Volumes/TissueCyte/181024_Gerald_HET/het-Mosaic/Ch2_Stitched_Sections_New'
        else:
            image_path = sys.argv[2]

    print count_path
    print image_path

    # Load the classifier model
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights_path)

    #=============================================================================================
    # Loop through the coordinate files and predict cells
    #=============================================================================================

    # Create directory to hold the counts in same folder as the images
    if not os.path.exists(count_path+'_cnn'):
        os.makedirs(count_path+'_cnn')

    # Get list of files containing the coordinates in x, y, z
    #image_path = raw_input('Counting file path (drag-and-drop): ').strip('\'').rstrip()
    filename = natsorted([file for file in os.listdir(image_path) if file.endswith('.tif')])

    if os.path.isdir(count_path):
        all_marker_path = glob.glob(count_path+'/*_count.csv')
    else:
        all_marker_path = count_path

    # Create empty pands dataframe to store data
    df = pd.DataFrame(columns = ['ROI', 'Original', 'True', 'False'])

    for marker_path in all_marker_path:

        marker_filename, marker_file_extension = os.path.splitext(marker_path)

        if marker_file_extension == '.xml':
            xml_doc = minidom.parse(marker_path)

            marker_x = xml_doc.getElementsByTagName('MarkerX')
            marker_y = xml_doc.getElementsByTagName('MarkerY')
            marker_z = xml_doc.getElementsByTagName('MarkerZ')

            marker = np.empty((0,3), int)

            for elem in range (0, marker_x.length):
                marker = np.vstack((marker, [int(marker_x[elem].firstChild.data), int(marker_y[elem].firstChild.data), int(marker_z[elem].firstChild.data)]))
        if marker_file_extension == '.csv':
            marker = np.genfromtxt(marker_path, delimiter=',', dtype=np.float).astype(int)

        #=============================================================================================
        # Load images into RAM
        #=============================================================================================

        if marker.shape[0] > 0:

            print 'Loading all images to RAM...'

            all_img = np.empty((0,1,80,80,1))

            if marker.ndim == 1:
                for slice in np.unique(marker[2]):
                    img = Image.open(os.path.join(image_path, filename[slice-1]))
                    for cell in marker[marker[2] == slice]:
                        img_crop = img.crop((cell[0]-40, cell[1]-40, cell[0]+40, cell[1]+40))
                        img_crop = image.img_to_array(img_crop)
                        img_crop = np.expand_dims(img_crop, axis = 0)
                        img_crop = np.expand_dims(img_crop, axis = 0)
                        all_img = np.append(all_img, img_crop, axis = 0)
            else:
                for slice in np.unique(marker[:,2]):
                    img = Image.open(os.path.join(image_path, filename[slice-1]))
                    for cell in marker[marker[:,2] == slice]:
                        img_crop = img.crop((cell[0]-40, cell[1]-40, cell[0]+40, cell[1]+40))
                        img_crop = image.img_to_array(img_crop)
                        img_crop = np.expand_dims(img_crop, axis = 0)
                        img_crop = np.expand_dims(img_crop, axis = 0)
                        all_img = np.append(all_img, img_crop, axis = 0)

            print 'Done!'

            manager = Manager()
            result = Array('i', marker.shape[0])
            cell_markers = manager.list()
            nocell_markers = manager.list()

            print 'Classifiying '+marker_filename

            cell_index = range(marker.shape[0])

            tstart = time.time()

            #=============================================================================================
            # Classify images
            #=============================================================================================


            pbar = tqdm.tqdm(total=len(all_img))
            for cell, img in zip(range(len(all_img)), all_img):
                cellpredict(cell, img, cell_markers, nocell_markers)
                pbar.update(1)
            pbar.close()

            # Append to Pandas dataframe
            df = df.append({'ROI':marker_filename.split('/')[-1][:-9], 'Original': result[:].count(1)+result[:].count(0), 'True': result[:].count(1), 'False': result[:].count(0)}, ignore_index=True)

            correct_markers = marker[cell_markers[:],:]
            pd.DataFrame(correct_markers).to_csv(count_path+'_cnn/'+marker_filename.split('/')[-1][:-9]+'_corrected_markers.csv', header=None, index=None)

        # Create directory to hold the counts in same folder as the images
        if not os.path.exists(count_path+'_cnn'):
            os.makedirs(count_path+'_cnn')

        # Write dataframe to csv
        df.to_csv(count_path+'_cnn/counts_table.csv', index=False)

        print 'Done!'

print df
print '{0:.0f}:{1:.0f} (MM:SS)'.format(*divmod(time.time()-tstart,60))
