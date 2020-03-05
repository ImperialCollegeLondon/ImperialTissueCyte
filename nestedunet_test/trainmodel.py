"""
Model Training
Author: Gerald M

Trains the model by using clean-up, pre-processing and augmentation modules.
Can be run from command line with following,

    ipython trainmodel.py -- Adam 1e-4

to dictate the loss function and learning rate. Model architecture, weights and
training history are saved into a dated directory under models/.
"""

import argparse
import cleanup
from glob import glob
import os
import sys
import datetime
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback, Callback
from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from numpy import random
import tensorflow as tf
from multiprocessing import cpu_count
from random import randint
import nestedunetmodel
import preprocessing
import losses

# Set the training precision to speed u training time..?
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Alpha call back for the changing weight loss functions
class AlphaCallback(Callback):
    def __init__(self, alpha):
        self.alpha = alpha
    def on_epoch_end(self, epoch, logs={}):
        K.set_value(self.alpha, np.clip(self.alpha - 0.01, 0.01, 1))

alpha = K.variable(0.5, dtype='float32')

if __name__ == '__main__':
    if len(sys.argv) > 0:
        opt_arg1 = str(sys.argv[1])
        lr_arg2 = float(sys.argv[2])
        loss_arg3 = str(sys.argv[3])

        if opt_arg1 == 'Adam': optimizer = Adam(lr=lr_arg2)
        if opt_arg1 == 'SGD': optimizer = SGD(lr=lr_arg2)

        if loss_arg3 == 'BCE': loss = 'binary_crossentropy'
        if loss_arg3 == 'FTL': loss = losses.focal_tversky

    # Get today's date for model saving
    strdate = datetime.datetime.today().strftime('%Y_%m_%d')

    savedirpath = os.path.join('models', strdate+'_'+opt_arg1+'_lr'+str(lr_arg2)+'_'+loss_arg3+'_MultiResUNet')
    if not os.path.exists(savedirpath):
        os.makedirs(savedirpath)

    modelname = "multires_unet_"

    train_x, train_y, val_x, val_y = preprocessing.preprocess()

    filepath = os.path.join(savedirpath, modelname+'weights.best.hdf5')

    batch = 4

    # Loss functions for training
    model = nestedunetmodel.nestedunet(inputsize=(None, None, 1), optfn=optimizer, lossfn=loss, deep_supervision=True)
    # model = unetmodel.get_unet(losses.binary_focal_loss)
    # model = unetmodel.get_unet(losses.focal_tversky)
    # model = unetmodel.get_unet(losses.bce_dice_loss)
    # model = unetmodel.get_unet(losses.focal_loss)
    # model = unetmodel.get_unet(losses.weighted_cross_entropy(0.75))
    # model = unetmodel.get_unet(losses.bce_focal_tversky_loss(alpha))
    # model = unetmodel.get_unet(losses.surface_loss)
    # model = unetmodel.get_unet(losses.dice_focal_tversky_loss(alpha))
    # model = unetmodel.get_unet(losses.dice_surface_loss)
    # model = unetmodel.get_unet(losses.bce_surface_loss)
    # model = unetmodel.get_unet(losses.balanced_cross_entropy(0.3))
    # model = unetmodel.get_unet(losses.iou)

    checkpoint = ModelCheckpoint(filepath, monitor='val_dice_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor='val_dice_loss', mode='min', patience=30, verbose=1)
    redonplat = ReduceLROnPlateau(monitor='val_dice_loss', mode='min', patience=20, verbose=1)
    newalpha = AlphaCallback(alpha)
    callbacks_list = [checkpoint, early, redonplat, newalpha]

    history = model.fit(train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=batch,
        epochs=150,
        shuffle=True,
        callbacks=callbacks_list)

    # Serialize model to JSON
    modeljson = model.to_json()
    with open(os.path.join(savedirpath, modelname+'model.json'), 'w') as jsonfile:
        jsonfile.write(modeljson)

    # Write out the training history to file
    pd.DataFrame(history.history).to_csv(os.path.join(savedirpath, 'trainhistory.csv'))

    cleanup.clean()

    # Plot out to see progress
    # import matplotlib.pyplot as plt
    # plt.plot(history.history['dice_loss'])
    # plt.plot(history.history['val_dice_loss'])
    # plt.title('Dice loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper right')
    # plt.show()
