# Perform final correction of cell counts

import os, sys, warnings
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
from xml.dom import minidom

warnings.simplefilter('ignore', Image.DecompressionBombWarning)

#=============================================================================================
# Load CNN classifier model
#=============================================================================================

model_path = raw_input('Model file path (drag-and-drop): ').rstrip()
model = load_model(model_path)

#=============================================================================================
# Parameters
#=============================================================================================

xml_path = raw_input('XML file path (drag-and-drop): ').rstrip()
xml_doc = minidom.parse(xml_path)

marker_x = xml_doc.getElementsByTagName('MarkerX')
marker_y = xml_doc.getElementsByTagName('MarkerY')
marker_z = xml_doc.getElementsByTagName('MarkerZ')

marker = np.empty((0,3), int)

for elem in range (0, marker_x.length):
    marker = np.vstack((marker, [int(marker_x[elem].firstChild.data), int(marker_y[elem].firstChild.data), int(marker_z[elem].firstChild.data)]))

#=============================================================================================
# Load images and correct cell count by predicting
#=============================================================================================

all_result = [None] * marker_x.length
#cell_predictions = 0
#nocell_predictions = 0
cell_markers = np.empty((0,3), int)
nocell_markers = np.empty((0,3), int)

image_path = raw_input('Counting file path (drag-and-drop): ').rstrip()

filename = [file for file in os.listdir(image_path) if file.endswith('.tif')]

for cell in range(0, marker_x.length):
    prev_slice = 0
    if prev_slice < marker[cell, 2]:
        #img = image.load_img(os.path.join(image_path, filename[cell]), target_size = (80, 80))
        #img = img.convert('I')
        img = Image.open(os.path.join(image_path, filename[marker[cell, 2]]))
        img = image.img_to_array(img)
        img = np.lib.pad(img, pad_width = ((40, 40), (40, 40), (0, 0)), mode = 'constant', constant_values=0)

    # The additional 1230 is a correction from the cropping between the original data and the segmented set - remove as necessary
    cell_crop = img[marker[cell, 1]+1230 : marker[cell, 1]+1230 + 80, marker[cell, 0]+1230 : marker[cell, 0]+1230 + 80]
    #cell_crop = img[marker[cell, 1] : marker[cell, 1] + 80, marker[cell, 0] : marker[cell, 0] + 80]
    cell_crop = np.expand_dims(cell_crop, axis = 0)


    result = model.predict(np.asarray(cell_crop))

    if result[0][0] == 0:
        prediction = 'cell'
        all_result[cell] = 1
        #cell_predictions += 1
        cell_markers = np.vstack((cell_markers, marker[cell,:]))
        image.array_to_img(cell_crop[0,:,:,:]).save('/Users/gm515/Desktop/cells/'+str(cell)+'.tif')
    else:
        prediction = 'no cell'
        all_result[cell] = 0
        #nocell_predictions += 1
        nocell_markers = np.vstack((nocell_markers, marker[cell,:]))
        image.array_to_img(cell_crop[0,:,:,:]).save('/Users/gm515/Desktop/nocells/'+str(cell)+'.tif')

    #progress = int(float(cell)/marker_x.length*100)
    #sys.stdout.write('\r{0}%'.format(progress))
    #sys.stdout.flush()

    for _ in tqdm.tqdm(pool.imap_unordered(cellpredict, cell), total=marker_x.length):
        pass

print '\n'
print 'Correct cell preditions:', all_result.count(1)
print 'Potential false cell predictions:', all_result.count(0)
