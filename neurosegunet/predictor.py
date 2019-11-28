"""
################################################################################
Segmentation predictor
Author: Gerald M
################################################################################
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import glob
import os
import sys
import math
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Modules for deep learning
from tensorflow.keras.models import model_from_json

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

if __name__ == '__main__':
    model_path = 'models/2019_11_28_UNet/focal_unet_model.json'
    weights_path = 'models/2019_11_28_UNet/focal_unet_weights.best.hdf5'

    # Load the classifier model, initialise and compile
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)

    images_array = []

    img = np.array(Image.open('test0.tif')).astype(np.float32)
    img = (img-np.min(img))/(np.max(img)-np.min(img))
    img_copy = np.copy(img)

    orig_shape = img.shape

    newshape = tuple((int( 16 * math.ceil( i / 16. )) for i in orig_shape))
    img = np.pad(img, ((0,np.subtract(newshape,orig_shape)[0]),(0,np.subtract(newshape,orig_shape)[1])), 'constant')

    images_array.append(img)
    images_array = np.array(images_array)
    images_array = images_array[..., np.newaxis]

    pred = model.predict(images_array)

    pred = np.squeeze(pred[0])

    pred = pred[0:orig_shape[0],0:orig_shape[1]]

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img_copy)
    axarr[1].imshow(pred)
    plt.show(block=False)
    plt.tight_layout()
