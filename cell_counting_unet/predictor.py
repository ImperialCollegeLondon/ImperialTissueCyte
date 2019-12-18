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
    ################################################################################
    ## User defined parameters via command line arguments
    ################################################################################

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('imagepath', default=[], type=str, help='Image directory')
    # parser.add_argument('-modelpath', default='models/2019_09_02_UNet/focal_unet_do_0.1_activation_ReLU_model.json', type=str, dest='modelpath', help='Model path')
    # parser.add_argument('-weightspath', default='models/2019_09_02_UNet/focal_unet_do_0.1_activation_ReLU_weights.best.hdf5', type=str, dest='weightspath', help='Weights path')

    # args = parser.parse_args()

    # image_path = args.imagepath
    # model_path = args.modelpath
    # weights_path = args.weightspath


    model_path = 'models/2019_12_18_UNet/focal_unet_model.json'
    weights_path = 'models/2019_12_18_UNet/focal_unet_weights.best.hdf5'

    # Load the classifier model, initialise and compile
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)

    # sgd = SGD(lr=0.01, momentum=0.90, decay=1e-6)
    # input_size = (None, None, 1)
    # model = focal_tversky_unetmodel.unet(sgd, input_size, losses.focal_tversky)
    # model.load_weights(weights_path)

    images_array = []

    img = np.array(Image.open('testprediction/00007.tif')).astype(np.float32)
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

    # contours, hierarchy = cv2.findContours(np.uint8(pred>((np.max(pred)-np.min(pred))/2.)), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # overlay = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    import matplotlib.pyplot as plt

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img_copy)
    axarr[1].imshow(pred)
    plt.show(block=False)
    plt.tight_layout()
