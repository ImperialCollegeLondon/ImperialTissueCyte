'''
Cell Counting Predictor - Serial version
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
from natsort import natsorted
from keras.preprocessing import image
import glob
from keras.preprocessing import image
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

# Main function
if __name__ == '__main__':
    # CNN model paths
    model_weights_path = 'models/2019_03_29_GoogleInception/weights_2019_03_29.h5'
    model_json_path = 'models/2019_03_29_GoogleInception/model_2019_03_29.json'

    count_path = '/Users/gm515/Desktop/PFC/counts'
    image_path = '/Users/gm515/Desktop/PFC/'

    if len(sys.argv) == 2:
        try:
            sys.argv[1]
        except NameError:
            count_path = count_path
            image_path = image_path
        else:
            count_path = str(sys.argv[1])
            image_path = str(sys.argv[1])[:-7]
    if len(sys.argv) == 3:
        try:
            sys.argv[1]
        except NameError:
            count_path = count_path
            image_path = image_path
        else:
            count_path = sys.argv[1]

        try:
            sys.argv[2]
        except NameError:
            count_path = count_path
            image_path = image_path
        else:
            image_path = sys.argv[2]

    print count_path
    print image_path

    # Load the classifier model
    with open(model_json_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_path)

    initial_lrate = 0.01
    sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)

    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
        loss_weights=[1, 0.3, 0.3],
        optimizer=sgd,
        metrics=['accuracy'])

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

            all_img = []
            i = 0

            if marker.ndim == 1:
                for slice in np.unique(marker[2]):
                    img = Image.open(os.path.join(image_path, filename[slice-1])).convert(mode='RGB') # RGB as 3 channel for Google Inception
                    for cell in marker[marker[2] == slice]:
                        img_crop = img.crop((cell[0]-40, cell[1]-40, cell[0]+40, cell[1]+40))
                        img_crop = image.img_to_array(img_crop)
                        img_crop = np.expand_dims(img_crop, axis = 0)
                        all_img.append(img_crop)
            else:
                for slice in np.unique(marker[:,2]):
                    img = Image.open(os.path.join(image_path, filename[slice-1])).convert(mode='RGB') # RGB as 3 channel for Google Inception
                    for cell in marker[marker[:,2] == slice]:
                        img_crop = img.crop((cell[0]-40, cell[1]-40, cell[0]+40, cell[1]+40))
                        img_crop = image.img_to_array(img_crop)
                        img_crop = np.expand_dims(img_crop, axis = 0)
                        all_img.append(img_crop)
                    i += 1
                    progressBar(i, len(np.unique(marker[:,2])))

            all_img = np.vstack(all_img)
            print ''
            print 'Done!'

            print 'Classifiying: '+marker_filename

            tstart = time.time()

            #=============================================================================================
            # Classify images
            #=============================================================================================
            classes = model.predict(all_img)
            cells = np.count_nonzero(np.argmax(classes[0], axis=1)==0)
            nocells = np.count_nonzero(np.argmax(classes[0], axis=1))
            cell_markers = marker[np.where(np.argmax(classes[0], axis=1)==0)]
            nocell_markers = marker[np.where(np.argmax(classes[0], axis=1))]
            leftcells = np.nan#len(cell_markers[cell_markers[:,0]<10500])
            rightcells = np.nan#len(cell_markers)-leftcells

            # Append to Pandas dataframe
            df = df.append({'ROI':marker_filename.split('/')[-1][:-9], 'Original': cells+nocells, 'True': cells, 'L':leftcells, 'R':rightcells, 'False': nocells}, ignore_index=True)

            correct_markers = marker[np.flatnonzero(np.argmin(classes[0], axis=1)),:]

            pd.DataFrame(correct_markers).to_csv(count_path+'_cnn/'+marker_filename.split('/')[-1][:-9]+'_corrected_markers.csv', header=None, index=None)

        # Create directory to hold the counts in same folder as the images
        if not os.path.exists(count_path+'_cnn'):
            os.makedirs(count_path+'_cnn')

        # Write dataframe to csv
        df.to_csv(count_path+'_cnn/counts_table.csv', index=False)

        print 'Done!'

print df
print '{0:.0f}:{1:.0f} (MM:SS)'.format(*divmod(time.time()-tstart,60))
