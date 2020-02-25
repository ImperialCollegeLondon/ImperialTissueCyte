# part of this script was taken from https://github.com/jocicmarko/ultrasound-nerve-segmentation
import argparse
import cleanup
from glob import glob
import os
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
import multiresunetmodel
import preprocessing
import losses

# Alpha call back for the changing weight loss functions
class AlphaCallback(Callback):
    def __init__(self, alpha):
        self.alpha = alpha
    def on_epoch_end(self, epoch, logs={}):
        K.set_value(self.alpha, np.clip(self.alpha - 0.01, 0.01, 1))

alpha = K.variable(0.5, dtype='float32')

if __name__ == '__main__':
    # Get today's date for model saving
    strdate = datetime.datetime.today().strftime('%Y_%m_%d')

    savedirpath = os.path.join('models', strdate+'_MultiResUNet')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    modelname = "multires_unet_"

    train_x, train_y, val_x, val_y = preprocessing.preprocess()


    filepath = os.path.join(savedirpath, modelname+'weights.best.hdf5')

    batch = 4

    # Loss functions for training
    model = multiresunetmodel.multiresunet(input_size=(None, None, 1), loss_fn='binary_crossentropy')
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
    early = EarlyStopping(monitor='val_dice_loss', mode='min', patience=50, verbose=1)
    redonplat = ReduceLROnPlateau(monitor='val_dice_loss', mode='min', patience=20, verbose=1)
    newalpha = AlphaCallback(alpha)
    callbacks_list = [checkpoint, early, redonplat, newalpha]

    history = model.fit(train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=batch,
        epochs=100,
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
    import matplotlib.pyplot as plt
    plt.plot(history.history['dice_loss'])
    plt.plot(history.history['val_dice_loss'])
    plt.title('Dice loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()
