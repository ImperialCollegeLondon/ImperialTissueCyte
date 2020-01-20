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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from numpy import random
import tensorflow as tf
from multiprocessing import cpu_count
from random import randint
import unetmodel
import preprocessing
import losses

alpha = K.variable(1, dtype='float32')

def update_alpha(epoch,logs):
    #maybe use epoch+1, because it starts with 0
    K.set_value(alpha, np.clip(alpha - 0.01, 0.01, 1))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dropout", required=False,
                    help="dropout", type=float, default=0.1)
    ap.add_argument("-a", "--activation", required=False,
                    help="activation", default="ReLU")

    args = vars(ap.parse_args())

    activation = globals()[args['activation']]

    # Checkpoint to only save the best model, metric = val_acc
    strdate = datetime.datetime.today().strftime('%Y_%m_%d')

    if not os.path.exists('models/'+strdate+'_UNet'):
        os.makedirs('models/'+strdate+'_UNet')

    model_name = "focal_unet_"

    train_x, train_y, val_x, val_y = preprocessing.preprocess()

    file_path = 'models/'+strdate+'_UNet/'+model_name+'weights.best.hdf5'

    batch = 4

    # ORIG BATCH NORM LOSS
    # model = unetmodel.get_unet(losses.binary_focal_loss)
    # model = unetmodel.get_unet('binary_crossentropy')
    # model = unetmodel.get_unet(losses.focal_tversky)
    # model = unetmodel.get_unet(losses.bce_dice_loss)
    # model = unetmodel.get_unet(losses.focal_loss)
    # model = unetmodel.get_unet(losses.weighted_cross_entropy(0.1))
    # model = unetmodel.get_unet(losses.bce_focal_tversky_loss)
    # model = unetmodel.get_unet(losses.surface_loss)
    # model = unetmodel.get_unet(losses.dice_focal_tversky_loss(alpha))
    # model = unetmodel.get_unet(losses.dice_surface_loss)
    # model = unetmodel.get_unet(losses.bce_surface_loss)
    model = unetmodel.get_unet(losses.balanced_cross_entropy(0.1))

    checkpoint = ModelCheckpoint(file_path, monitor='val_dice_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=50, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=20, verbose=1)
    newalpha = LambdaCallback(on_epoch_end=update_alpha)
    callbacks_list = [checkpoint, early, redonplat, newalpha]

    history = model.fit(train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=batch,
        epochs=200,
        shuffle=True,
        callbacks=callbacks_list)

    # Serialize model to JSON
    model_json = model.to_json()
    with open('models/'+strdate+'_UNet/'+model_name+'model.json', 'w') as json_file:
        json_file.write(model_json)

    # Write out the training history to file
    pd.DataFrame(history.history).to_csv('models/'+strdate+'_UNet/trainHistoryDict.csv')

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
