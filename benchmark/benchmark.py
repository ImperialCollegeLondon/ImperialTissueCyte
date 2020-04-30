"""
Benchmark Testing
Author: Gerald M

Benchmark testing using a complete 2P coronal stitched section from an unseen
sample - PVCre-Sox14 and Rabies tracing.

Benchmark tests are:
- IoU (Jaccard)
- Accuracy
- Precision
- Recall
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

if tf.test.is_gpu_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

if __name__ == '__main__':
    if len(sys.argv) > 1:
        modeldir = str(sys.argv[1]).rstrip('/')

    print ('Loading image (for prediction) and true result for benchmark testing...')

    imagepaths = ['data/pvcresox14_GM.tif', 'data/rabies_GM.tif']
    truepaths = ['data/mask_pvcresox14_GM.tif', 'data/mask_rabies_GM.tif']

    for imagepath, truepath in zip(imagepaths, truepaths):
        image = np.array(Image.open(imagepath))
        true = np.array(Image.open(truepath))

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
        tstart = time.time()
        predarray = model.predict(imgarray, batch_size=6)
        print (time.time()-tstart)
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
        if np.isnan(jac*pre*rec):
            coh = np.nan
        else:
            coh = metrics.colocalisedhits(true, pred)

        modelname = os.path.basename(modeldir)

        print ('Saving...')

        resultdf = pd.DataFrame({'Model':[modelname], 'Image':[os.path.basename(imagepath)], 'IoU':[jac], 'Accuracy':[acc], 'Precision':[pre], 'Recall':[rec], 'Colocalised':[coh]})
        if not os.path.isfile('results/benchmarkresults.csv'):
            resultdf.to_csv('results/benchmarkresults.csv', mode='a', header=True, index=False)
        else:
            olddf = pd.read_csv('results/benchmarkresults.csv', header=0)
            resultdf = pd.concat([olddf, resultdf], ignore_index=True).sort_values(by=['Model', 'Image'])
            resultdf.to_csv('results/benchmarkresults.csv', mode='w', header=True, index=False)

        Image.fromarray((pred*255).astype(np.uint8)).save('results/'+modelname+'_predicted_'+os.path.basename(imagepath)+'.tif')
