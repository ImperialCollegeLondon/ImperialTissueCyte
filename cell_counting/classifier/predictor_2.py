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
import pickle
import sys
import tifffile
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from multiprocessing import cpu_count
from natsort import natsorted
from keras.models import model_from_json
from keras.optimizers import SGD
from keras import backend as K

os.environ["OMP_NUM_THREADS"] = str(cpu_count())
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,noverbose,compact,1,0"

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Warning supression and allowing large images to be loaded
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

config = tf.ConfigProto(intra_op_parallelism_threads = cpu_count(),
                        inter_op_parallelism_threads = cpu_count(),
                        allow_soft_placement = True,
                        device_count = {'CPU': cpu_count() })

session = tf.Session(config=config)

K.set_session(session)

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
    parser.add_argument('-modelpath', default='models/2019_08_22_Inception/inception_model.json', type=str, dest='modelpath', help='Model path')
    parser.add_argument('-weightspath', default='models/2019_08_22_Inception/inception_weights_e47_va0.9962.h5', type=str, dest='weightspath', help='Weights path')
    parser.add_argument('-normpath', default='models/2019_08_14_Inception/feature_standardisation.npy', type=str, dest='normpath', help='Normlisation variable path')

    args = parser.parse_args()

    image_path = args.imagepath
    count_path = args.countpath
    model_path = args.modelpath
    weights_path = args.weightspath
    norm_path = args.normpath

    if not count_path:
        count_path = glob.glob('/'+os.path.join(*image_path.split(os.sep)[0:-1])+'/counts_v?')[0]

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
    print ('Image path: {} \nCount path: {} \nModel path: {} \nWeights path: {} \nNormalisation path: {}'.format(
        image_path,
        count_path,
        model_path,
        weights_path,
        norm_path))

    # Get list of files containing the coordinates in x, y, z
    filename = natsorted([file for file in os.listdir(image_path) if file.endswith('.tif')])

    # Get list of all count csv files
    all_marker_path = glob.glob(count_path+'/*_count.csv')

    # Create empty pands dataframe to store all markers
    all_marker_df = pd.DataFrame(columns = ['ROI', 'X', 'Y', 'Z', 'Hemisphere'])

    tstart = time.time()

    # Loop through each csv count file
    for marker_path in all_marker_path:

        marker_filename, marker_file_extension = os.path.splitext(marker_path)
        roi = marker_filename.split('/')[-1][:-9]

        marker = pd.read_csv(marker_path, delimiter=',', dtype=np.float, names=['X', 'Y', 'Z', 'Hemisphere']).astype(int)
        marker['ROI'] = roi

        all_marker_df = all_marker_df.append(marker, ignore_index=True, sort=False)

    all_marker_df = all_marker_df.sort_values(by=['Z'])

    print ('Loading all images to RAM...')

    all_img = []
    i = 0

    cells = 0
    nocells = 0
    leftcells = 0
    rightcells = 0

    # If there are detection to classify
    if all_marker_df.shape[0] > 0:
        for slice in np.unique(all_marker_df['Z']):
            # img = Image.open(os.path.join(image_path, filename[slice-1]), 'r')
            # img = np.frombuffer(img.tobytes(), dtype=np.uint8, count=-1).reshape(img.size[::-1]) # RGB as 3 channel for Google Inception

            img = tifffile.imread(os.path.join(image_path, filename[slice-1]), key=0)

            for index, cell in all_marker_df.loc[all_marker_df['Z'] == slice].iterrows():
                img_crop = img[cell['Y']-40:cell['Y']+40, cell['X']-40:cell['X']+40]
                img_crop = np.stack((img_crop,)*3, axis=-1)
                img_crop = np.expand_dims(img_crop, axis = 0)
                all_img.append(img_crop)

                i += 1
                progressBar(i, all_marker_df.shape[0])

        all_img = np.vstack(all_img).astype(np.float32)

        print ('')
        print ('Done!')

        print ('Normalising image range...')

        for idx, img in enumerate(all_img):
            if (np.max(img)-np.min(img)) == 0:
                print ('Min: {0:.2f} Max: {1:.2f}'.format(np.min(img), np.max(img)))
            all_img[idx] = (img-np.min(img))/(np.max(img)-np.min(img))

        print ('Done!')

        print ('Classifiying...')

        classes = model.predict(all_img)
        # cells = np.count_nonzero(np.argmax(classes[0], axis=1)==0)
        # nocells = np.count_nonzero(np.argmax(classes[0], axis=1))
        cell_markers = all_marker_df.iloc[np.where(np.argmax(classes[0], axis=1)==0)]
        nocell_markers = all_marker_df.iloc[np.where(np.argmax(classes[0], axis=1))]

        print ('Done!')

        print ('Splitting and writing out markers...')

        # Create empty pands dataframe to store data
        df = pd.DataFrame(columns = ['ROI', 'Original', 'True', 'False', 'L', 'R'])

        for roi in np.unique(all_marker_df['ROI']):
            roi_cell_df = cell_markers.loc[cell_markers['ROI'] == roi][['X', 'Y', 'Z', 'Hemisphere']]
            roi_nocell_df = nocell_markers.loc[nocell_markers['ROI'] == roi][['X', 'Y', 'Z', 'Hemisphere']]

            cells = len(roi_cell_df)
            nocells = len(roi_nocell_df)
            leftcells = len(roi_cell_df.loc[roi_cell_df['Hemisphere']==0])
            rightcells = len(roi_cell_df.loc[roi_cell_df['Hemisphere']==1])

            # Check if there are cells detected and save their markers if so
            if len(roi_cell_df) > 0:
                pd.DataFrame(roi_cell_df).to_csv(count_path+'_cnn/'+roi+'_corrected_markers.csv', header=None, index=None)

            # Append to Pandas dataframe
            df = df.append({'ROI':roi, 'Original':cells+nocells, 'True':cells,  'False':nocells, 'L':leftcells, 'R':rightcells}, ignore_index=True)

            # Write dataframe to csv
            df.to_csv(count_path+'_cnn/counts_table.csv', index=False)

        print ('Done!')

print (df)
print ('{0:.0f}:{1:.0f} (MM:SS)'.format(*divmod(time.time()-tstart,60)))
