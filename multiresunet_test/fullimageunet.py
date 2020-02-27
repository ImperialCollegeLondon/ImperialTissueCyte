"""
Full Image Classification
Author: Gerald M

Splits the image into blocks of 512x512 and classifies each block. For interest,
gives time to calculate 1000 images.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, time
import warnings
import numpy as np
from PIL import Image

# Modules for deep learning
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.python.platform import gfile

from tensorflow.keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

if __name__ == '__main__':
    img_path = '/Volumes/thefarm2/live/TissueCyte/ScanData/190221_Gerald_RabiesBL/rabiesbl-Mosaic/Ch2_Stitched_Sections/Stitched_Z473.tif'

    print ('Loading image...')
    orig_img = np.array(Image.open(img_path)).astype(np.float32)
    orig_img = (orig_img-np.min(orig_img))/(np.max(orig_img)-np.min(orig_img))
    print ('Done!')

    model_path = 'models/2020_02_25_ADAM_lr0.0001_MultiResUNet/multires_unet_model.json'
    weights_path = 'models/2020_02_25_ADAM_lr0.0001_MultiResUNet/multires_unet_weights.best.hdf5'

    # Load the classifier model, initialise and compile
    print ('Loading model...')
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)
    print ('Done!')

    tstart = time.time()

    print ('Running 512x512 window through image and predicting...')
    pred_img = np.zeros_like(orig_img)
    window_size = 512
    counter = 0
    for y in range(0,orig_img.shape[0],window_size):
        for x in range(0,orig_img.shape[1],window_size):
            img = np.zeros((window_size, window_size))
            img_crop = orig_img[y:y+window_size, x:x+window_size]

            if np.sum(img_crop>0) > 0.25*(512*512): # If image is <25% full, then skip as it's likely background
                img_crop = (img_crop-np.min(img_crop))/(np.max(img_crop)-np.min(img_crop))

                img[0:img_crop.shape[0], 0:img_crop.shape[1]] = img_crop

                img_array = []
                img_array.append(img)
                img_array = np.array(img_array)
                img_array = img_array[..., np.newaxis]

                pred = model.predict(img_array)

                # Set a prediction of >50% as class 1
                pred = np.squeeze(pred[0])

            else:
                pred = np.zeros_like(img)

            pred_img[y:y+window_size, x:x+window_size] = pred[0:img_crop.shape[0], 0:img_crop.shape[1]]

            counter+=1
            print (str(counter)+' of '+(str(np.ceil(orig_img.shape[0]/window_size)*np.ceil(orig_img.shape[1]/window_size))))

    telapsed = time.time()-tstart
    hours, rem = divmod(telapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print ("Single image {:0>2} hours {:0>2} minutes {:05.2f} seconds".format(int(hours),int(minutes),seconds))

    hours, rem = divmod(telapsed*1000, 3600)
    minutes, seconds = divmod(rem, 60)
    print ("1000 images  {:0>2} hours {:0>2} minutes {:05.2f} seconds".format(int(hours),int(minutes),seconds))

    Image.fromarray((pred_img*255).astype(np.uint8)).save('fullimageprediction.tif')
