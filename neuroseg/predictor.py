"""
################################################################################
Segmentation predictor
Author: Gerald M
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
import cv2

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

    model_path = '/Users/gm515/Desktop/2019_09_26_UNet/focal_unet_model.json'
    weights_path = '/Users/gm515/Desktop/2019_09_26_UNet/focal_unet_weights.best.hdf5'

    # Load the classifier model, initialise and compile
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)

    images_array = []

    img = np.array(Image.open('/Users/gm515/Desktop/unet-test/00002.tif')).astype(np.float32)
    img = (img-np.min(img))/(np.max(img)-np.min(img))

    images_array.append(img)
    images_array = np.array(images_array)
    images_array = images_array[..., np.newaxis]

    pred = model.predict(images_array)

    pred = np.squeeze(pred[0])

    pred = 255.*(pred - np.min(pred))/(np.max(pred)-np.min(pred))
    print (np.min(pred), np.max(pred))

    import matplotlib.pyplot as plt

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img)
    axarr[1].imshow(pred)
    plt.show(block=False)
    plt.tight_layout()
