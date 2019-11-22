# part of this script was taken from https://github.com/jocicmarko/ultrasound-nerve-segmentation
import argparse
import cleanup
from glob import glob
import os
import datetime
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
import focal_tversky_unetmodel
import losses

# GPU = False
#
# if not GPU:
#     import os
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#
#     print ('Using CPU.')
# else:
#     print ('Using GPU.')
#
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.compat.v1.Session(config=config)
#
# import keras.backend.tensorflow_backend as K
# K.set_session(sess)

# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = cpu_count(),
#                         inter_op_parallelism_threads = cpu_count(),
#                         allow_soft_placement = True,
#                         device_count = {'CPU': cpu_count() })
#
# session = tf.compat.v1.Session(config=config)
#
# K.tensorflow_backend._get_available_gpus()
# K.set_session(session)
#
# os.environ["OMP_NUM_THREADS"] = str(cpu_count())
# os.environ["KMP_BLOCKTIME"] = "30"
# os.environ["KMP_SETTINGS"] = "1"
# os.environ["KMP_AFFINITY"]= "granularity=fine,noverbose,compact,1,0"
#
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

    # Determine weights for classes using training data
    combinedsize = train_x.shape[0]*train_x.shape[1]*train_x.shape[2]
    class1size = np.sum(train_x)
    class0size = combinedsize-np.sum(train_x)
    class0weights = class0size/combinedsize
    class1weights = 1-class0weights
    classweights = {0:class0weights, 1:class1weights}
    sampleweights = np.array(len(train_x) * [[class0weights, class1weights]])

    sampleweights = np.where(train_y==0, class0weights, train_y)
    sampleweights = np.where(sampleweights==1, class1weights, sampleweights)
    sampleweights = sampleweights.reshape((len(train_x),512, 512))

    file_path = 'models/'+strdate+'_UNet/'+model_name+'weights.best.hdf5'

    batch = 4

    # ORIG BATCH NORM LOSS
    model = unetmodel.get_unet()

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=50, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=20, verbose=1)
    callbacks_list = [checkpoint, early, redonplat]  # early

    history = model.fit(train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=batch,
        epochs=30,
        shuffle=True,
        callbacks=callbacks_list,
        sample_weight=sampleweights)

    # TVERSKY LOSS
    # estop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, mode='auto')
    # checkpoint = ModelCheckpoint(file_path, monitor='val_dsc',
    #                              verbose=1, save_best_only=True,
    #                              save_weights_only=True, mode='max')
    #
    # sgd = SGD(lr=0.01, momentum=0.90, decay=1e-6)
    # input_size = (None, None, 1)
    # model = focal_tversky_unetmodel.unet(sgd, input_size, losses.focal_tversky)
    #
    # hist = model.fit(train_x, train_y,
    #                 validation_data=(val_x, val_y),
    #                 shuffle=True,
    #                 epochs=30,
    #                 batch_size=batch,
    #                 verbose=True,
    #                 callbacks=[checkpoint])

    # Serialize model to JSON
    model_json = model.to_json()
    with open('models/'+strdate+'_UNet/'+model_name+'model.json', 'w') as json_file:
        json_file.write(model_json)

    cleanup.clean()
