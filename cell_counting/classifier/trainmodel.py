
# -*- coding: utf-8 -*-

"""
################################################################################
Classfier Google Inception Model Training
Author: Gerald M

This script runs preprocessing steps, data augmentation, google inception model
architecture build and then compiles and trains.
################################################################################
"""

import cleanup
import datetime
import glob
import googleinceptionmodel
import math
import os
import pickle
import preprocessing
import pandas as pd
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from multiprocessing import cpu_count
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

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

    raw_data_dir = '8-bit/raw_data'
    training_data_dir = '8-bit/training_data'
    test_data_dir = '8-bit/test_data'

    # Checkpoint to only save the best model, metric = val_acc
    strdate = datetime.datetime.today().strftime('%Y_%m_%d')

    if not os.path.exists('models/'+strdate+'_Inception'):
        os.makedirs('models/'+strdate+'_Inception')

    # Preprocess the data and retrieve training and testing data
    print ('################################################################################')
    print ('1. Pre-processing data...')
    print ('################################################################################')

    training_data_all, training_data_all_label, test_data_all, test_data_all_label = preprocessing.preprocess(normalise=2)
    end

    print ('################################################################################')
    print ('2. Building Inception model...')
    print ('################################################################################')

    model = googleinceptionmodel.create_model()

    print ('################################################################################')
    print ('3. Compiling model...')
    print ('################################################################################')

    steps_epoch = len([filename for filename in os.listdir(training_data_dir+'/cell') if filename.endswith(".tif")]) + len([filename for filename in os.listdir(training_data_dir+'/nocell') if filename.endswith(".tif")])//32
    steps_valid = len([filename for filename in os.listdir(test_data_dir+'/cell') if filename.endswith(".tif")]) + len([filename for filename in os.listdir(test_data_dir+'/nocell') if filename.endswith(".tif")])//32

    # Write model summary to file
    with open('models/'+strdate+'_Inception/model_summary.txt','w') as f:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    epochs = 50
    initial_lrate = 0.01

    def decay(epoch, steps=100):
        initial_lrate = 0.01
        drop = 0.96
        epochs_drop = 8
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)

    lr_sc = LearningRateScheduler(decay, verbose=1)

    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])

    filepath = 'models/'+strdate+'_Inception/inception_weights_e{epoch:02d}_va{val_output_acc:.4f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_output_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = checkpoint

    print ('Done')

    print ('################################################################################')
    print ('4. Training model...')
    print ('################################################################################')

    history = model.fit(
        training_data_all,
        [training_data_all_label, training_data_all_label, training_data_all_label],
        validation_data=(test_data_all, [test_data_all_label, test_data_all_label, test_data_all_label]),
        batch_size=128,
        epochs=epochs,
        callbacks=[lr_sc, callbacks_list])

    print ('################################################################################')
    print ('5. Saving model and clean up...')
    print ('################################################################################')

    pd.DataFrame(history.history).to_csv('models/'+strdate+'_Inception/TrainingHistoryDict.csv')

    # Serialize model to JSON
    model_json = model.to_json()
    with open('models/'+strdate+'_Inception/inception_model.json', 'w') as json_file:
        json_file.write(model_json)

    # Remove older weights
    weights_files = glob.glob('models/'+strdate+'_Inception/*.h5')
    weights_files.sort(key=os.path.getmtime)
    if len(weights_files) > 1:
        for file in weights_files[:-1]:
            os.remove(file)

    cleanup.clean()
