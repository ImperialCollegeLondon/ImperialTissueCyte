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

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

if __name__ == '__main__':
    if len(sys.argv) > 1:
        modeldir = str(sys.argv[1]).rstrip('/')

    print ('Loading image (for prediction) and true result for benchmark testing...')

    imagepaths = ['data/Aimee_pvcresox14_GM.tif', 'data/Aimee_rabies_GM.tif']
    truepaths = ['data/mask_pvcresox14_GM.tif', 'data/mask_rabies_GM.tif']

    for imagepath, truepath in zip(imagepaths, truepaths):
        pred = np.array(Image.open(imagepath))
        true = np.array(Image.open(truepath))

        pred = (pred-np.min(pred))/(np.max(pred)-np.min(pred))
        true = (true-np.min(true))/(np.max(true)-np.min(true))

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

        print ('Saving...')

        resultdf = pd.DataFrame({'Model':['Aimee'], 'Image':[os.path.basename(imagepath)], 'IoU':[jac], 'Accuracy':[acc], 'Precision':[pre], 'Recall':[rec], 'Colocalised':[coh]})
        if not os.path.isfile('results/humanbenchmarkresults.csv'):
            resultdf.to_csv('results/humanbenchmarkresults.csv', mode='a', header=True, index=False)
        else:
            olddf = pd.read_csv('results/humanbenchmarkresults.csv', header=0)
            resultdf = pd.concat([olddf, resultdf], ignore_index=True).sort_values(by=['Model', 'Image'])
            resultdf.to_csv('results/humanbenchmarkresults.csv', mode='w', header=True, index=False)

        # Image.fromarray((pred*255).astype(np.uint8)).save('results/'+modelname+'_predicted_'+os.path.basename(imagepath)+'.tif')
