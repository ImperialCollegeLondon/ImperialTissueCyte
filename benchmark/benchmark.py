"""
Benchmark
Author: Gerald M

Benchmark testing using a complete 2P coronal stitched section from an unseen
sample.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, time, glob
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import metrics

# Modules for deep learning
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.python.platform import gfile

from tensorflow.keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

if __name__ == '__main__':
    if len(sys.argv) > 1:
        modeldir = str(sys.argv[1])

    print ('Loading image (for prediction) and true result for benchmark testing...')
    image = np.array(Image.open('data/pvcresox14_GM.tif'))
    true = np.array(Image.open('data/mask_pvcresox14_GM.tif'))

    print ('Loading model for prediction...')
    modelpath = glob.glob(os.path.join(modeldir, '*.json'))[0]
    weightspath = glob.glob(os.path.join(modeldir, '*.hdf5'))[0]
    with open(modelpath, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weightspath)

    # Split true and image into 512x512 blocks
    imgarray = []
    ws = 512
    for y in range(0,image.shape[0], ws):
        for x in range(0,image.shape[1], ws):
            imagecrop = image[y:y+ws, x:x+ws]


            imagecroppad = np.zeros((ws, ws))
            if (np.max(imagecrop)-np.min(imagecrop))>0: # Ignore any empty data and zero divisions
                imagecroppad[:imagecrop.shape[0],:imagecrop.shape[1]] = (imagecrop-np.min(imagecrop))/(np.max(imagecrop)-np.min(imagecrop))
            imagecroppad = imagecroppad[..., np.newaxis]
            imgarray.append(imagecroppad)

    imgarray = np.array(imgarray)

    print ('Predicting...')
    predarray = model.predict(imgarray, batch_size=6)

    pred = np.zeros((int(np.ceil(image.shape[0]/ws)*ws), int(np.ceil(image.shape[1]/ws)*ws)))
    i = 0
    for y in range(0,image.shape[0], ws):
        for x in range(0,image.shape[1], ws):
            pred[y:y+ws, x:x+ws] = np.squeeze(predarray[i])
            i += 1

    pred = pred[:image.shape[0],:image.shape[1]]
    pred = (pred>0.5).astype(np.int)
    true = (true>0.5).astype(np.int)

    print ('Benchmarking...')
    jac = metrics.jaccard(true, pred)
    acc = metrics.accuracy(true, pred)
    pre = metrics.precision(true, pred)
    rec = metrics.recall(true, pred)
    coh = metrics.colocalisedhits(true, pred)

    modelname = os.path.basename(modeldir)

    print ('Saving...')
    pd.DataFrame({'Model':[modelname], 'Jaccard':[jac], 'Accuracy':[acc], 'Precision':[pre], 'Recall':[rec], 'Colocalised':[coh]}).to_csv('results/benchmarkresults.csv', mode='a', header=True, index=False)
