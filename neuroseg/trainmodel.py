# part of this script was taken from https://github.com/jocicmarko/ultrasound-nerve-segmentation
import argparse
import cleanup
from glob import glob
import os
import datetime
import numpy as np
from PIL import Image
from keras import backend as K
from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, MaxPooling2D
from keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, ReLU
from keras.models import Model
from keras.optimizers import Adam, SGD
from numpy import random
import tensorflow as tf
from multiprocessing import cpu_count
from random import randint
import unetmodel
import preprocessing
import focal_tversky_unetmodel
import losses

config = tf.ConfigProto(intra_op_parallelism_threads = cpu_count(),
                        inter_op_parallelism_threads = cpu_count(),
                        allow_soft_placement = True,
                        device_count = {'CPU': cpu_count() })

session = tf.Session(config=config)

K.tensorflow_backend._get_available_gpus()
K.set_session(session)

os.environ["OMP_NUM_THREADS"] = str(cpu_count())
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,noverbose,compact,1,0"

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
    # model = unetmodel.get_unet()
    #
    # checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # early = EarlyStopping(monitor="val_loss", mode="min", patience=50, verbose=1)
    # redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=20, verbose=1)
    # callbacks_list = [checkpoint, early, redonplat]  # early
    #
    # history = model.fit(train_x, train_y,
    #     validation_data=(val_x, val_y),
    #     batch_size=batch,
    #     epochs=30,
    #     shuffle=True,
    #     callbacks=callbacks_list)

    # TVERSKY LOSS
    estop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, mode='auto')
    checkpoint = ModelCheckpoint(file_path, monitor='val_final_dsc',
                                 verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='max')

    sgd = SGD(lr=0.01, momentum=0.90, decay=1e-6)
    input_size = (512, 512, 1)
    model = focal_tversky_unetmodel.unet(sgd, input_size, losses.focal_tversky)

    hist = model.fit(train_x, [train_y[:,::8,::8,:], train_y[:,::4,::4,:], train_y[:,::2,::2,:], train_y],
                    validation_data=(val_x, [val_y[:,::8,::8,:], val_y[:,::4,::4,:], val_y[:,::2,::2,:], val_y]),
                    shuffle=True,
                    epochs=30,
                    batch_size=batch,
                    verbose=True,
                    callbacks=[checkpoint])

    # Serialize model to JSON
    model_json = model.to_json()
    with open('models/'+strdate+'_UNet/'+model_name+'model.json', 'w') as json_file:
        json_file.write(model_json)

    cleanup.clean()
