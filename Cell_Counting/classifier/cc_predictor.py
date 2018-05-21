# Simple test to predict results

import os, sys
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image

model = load_model('cc_model.h5')

cell_predictions = 0
nocell_predictions = 0
cell_files = []
nocell_files = []

directory = '/Users/gm515/Documents/Python/Machine Learning/cell correction/8-bit/test_data/cell'
for filename in os.listdir(directory):
    if filename.endswith(".tif"):
        test_image = image.load_img(os.path.join(directory, filename), target_size = (80, 80))
        test_image = test_image.convert('I')
        #test_image = Image.open(os.path.join(directory, filename)) # also works if commenting out above two lines
        test_image = image.img_to_array(test_image)

        # test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis = 0)

        result = model.predict(np.asarray(test_image))

        #training_data.class_indices

        if result[0][0] == 0:
            prediction = 'cell'
            cell_predictions += 1
            cell_files.append(filename)
        else:
            prediction = 'no cell'
            nocell_predictions += 1
            nocell_files.append(filename)

        print prediction

print 'Cell preditions:', cell_predictions
print 'No cell predictions:', nocell_predictions
print 'Accuracy:', (float(cell_predictions)/(cell_predictions+nocell_predictions))

split_data = raw_input('Do you want to move temporarily split data [y/n]?: ')

if split_data == 'y':
    temp_directory = '/Users/gm515/Documents/Python/Machine Learning/cell correction/8-bit/test_data/'
    for filename in cell_files:
        os.rename(os.path.join(directory, filename), os.path.join(temp_directory, 'cell_temp/', filename))
    for filename in nocell_files:
        os.rename(os.path.join(directory, filename), os.path.join(temp_directory, 'nocell_temp/', filename))
