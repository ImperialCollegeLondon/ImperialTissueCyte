"""
################################################################################
Cell Counting Predictor - Serial version
Author: Gerald M

This script uses a convolution neural network classifier, trained on manually identified
cells, to confirm whether a potential cell/neuron is correctly identified.

This version uses serial computation.

Installation:
1) Navigate to the folder containing predictor.py

Instructions:
1) Pass in directory containing count files as first argument. Optionally pass in image
directory as second argument if count directory is not nested into the image directory
################################################################################
"""

from __future__ import division
import argparse
import glob
import os
import sys
import tifffile
import time
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from natsort import natsorted
from keras.models import model_from_json
from keras.optimizers import SGD

# Warning supression and allowing large images to be loaded
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

def progressBar(value, endvalue, bar_length=50):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '/'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\r[{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

if __name__ == '__main__':
    ################################################################################
    ## User defined parameters via command line arguments
    ################################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument('imagepath', default=[], type=str, help='Image directory')
    parser.add_argument('-countpath', default=[], type=str, dest='countpath', help='Count path if not in parent directory of imagepath')
    parser.add_argument('-modelpath', default='models/2019_03_29_GoogleInception/model_2019_03_29.json', type=str, dest='modelpath', help='Model path')
    parser.add_argument('-weightspath', default='models/2019_03_29_GoogleInception/weights_2019_03_29.h5', type=str, dest='weightspath', help='Weights path')

    args = parser.parse_args()

    image_path = args.imagepath
    count_path = args.countpath
    model_path = args.modelpath
    weights_path = args.weightspath

    if not count_path:
        count_path = glob.glob('/'+os.path.join(*image_path.split(os.sep)[0:-1])+'/counts*')[0]

    # Load the classifier model, initialise and compile
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)

    initial_lrate = 0.01
    sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
        loss_weights=[1, 0.3, 0.3],
        optimizer=sgd,
        metrics=['accuracy'])

    # Create directory to hold the counts in same folder as the images
    if not os.path.exists(count_path+'_cnn'):
        os.makedirs(count_path+'_cnn')

    print ('')
    print ('User defined parameters')
    print ('Image path: {} \nCount path: {} \nModel path: {} \nWeights path: {}'.format(
        image_path,
        count_path,
        model_path,
        weights_path))

    # Get list of files containing the coordinates in x, y, z
    filename = natsorted([file for file in os.listdir(image_path) if file.endswith('.tif')])

    # Get list of all count csv files
    all_marker_path = glob.glob(count_path+'/*_count.csv')

    # Create empty pands dataframe to store data
    df = pd.DataFrame(columns = ['ROI', 'Original', 'True', 'False'])

    # Loop through each csv count file
    for marker_path in all_marker_path:

        marker_filename, marker_file_extension = os.path.splitext(marker_path)

        marker = np.genfromtxt(marker_path, delimiter=',', dtype=np.float).astype(int)

        #=============================================================================================
        # Load images into RAM
        #=============================================================================================

        if marker.shape[0] > 0:

            print ('Loading all images to RAM...')

            all_img = []
            i = 0

            if marker.ndim == 1:
                for slice in np.unique(marker[2]):
                    # img = Image.open(os.path.join(image_path, filename[slice-1]), 'r')
                    # img = np.frombuffer(img.tobytes(), dtype=np.uint8, count=-1).reshape(img.size[::-1]) # RGB as 3 channel for Google Inception

                    img = tifffile.imread(os.path.join(image_path, filename[slice-1]), key=0)

                    for cell in marker[marker[2] == slice]:
                        img_crop = img[cell[1]-40:cell[1]+40, cell[0]-40:cell[0]+40]
                        img_crop = np.stack((img_crop,)*3, axis=-1)
                        img_crop = np.expand_dims(img_crop, axis = 0)
                        all_img.append(img_crop)
            else:
                for slice in np.unique(marker[:,2]):
                    # img = Image.open(os.path.join(image_path, filename[slice-1]), 'r')
                    # img = np.frombuffer(img.tobytes(), dtype=np.uint8, count=-1).reshape(img.size[::-1]) # RGB as 3 channel for Google Inception
                    img = tifffile.imread(os.path.join(image_path, filename[slice-1]), key=0)
                    for cell in marker[marker[:,2] == slice]:
                        img_crop = img[cell[1]-40:cell[1]+40, cell[0]-40:cell[0]+40]
                        img_crop = np.stack((img_crop,)*3, axis=-1)
                        img_crop = np.expand_dims(img_crop, axis = 0)
                        all_img.append(img_crop)
                        print (img_crop.shape)
                    i += 1
                    progressBar(i, len(np.unique(marker[:,2])))

            all_img = np.vstack(all_img)
            print ('')
            print ('Done!')

            print ('Classifiying: '+marker_filename)

            tstart = time.time()

            #=============================================================================================
            # Classify images
            #=============================================================================================
            classes = model.predict(all_img)
            cells = np.count_nonzero(np.argmax(classes[0], axis=1)==0)
            nocells = np.count_nonzero(np.argmax(classes[0], axis=1))
            cell_markers = marker[np.where(np.argmax(classes[0], axis=1)==0)]
            nocell_markers = marker[np.where(np.argmax(classes[0], axis=1))]
            leftcells = len(cell_markers[cell_markers[:,3]==0])
            rightcells = len(cell_markers)-leftcells

            # Append to Pandas dataframe
            df = df.append({'ROI':marker_filename.split('/')[-1][:-9], 'Original': cells+nocells, 'True': cells, 'L':leftcells, 'R':rightcells, 'False': nocells}, ignore_index=True)

            correct_markers = marker[np.flatnonzero(np.argmin(classes[0], axis=1)),:]

            pd.DataFrame(correct_markers).to_csv(count_path+'_cnn/'+marker_filename.split('/')[-1][:-9]+'_corrected_markers.csv', header=None, index=None)

        # Write dataframe to csv
        df.to_csv(count_path+'_cnn/counts_table.csv', index=False)

        print ('Done!')

print (df)
print ('{0:.0f}:{1:.0f} (MM:SS)'.format(*divmod(time.time()-tstart,60)))
