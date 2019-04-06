'''
Google Inception classifier model
Works well
'''


import keras
from keras.layers.core import Layer
import keras.backend as K
import tensorflow as tf

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D, AveragePooling2D, Flatten

import cv2
import numpy as np
from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils

import math
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

def cleanup():
    print 'Moving files back to training data directory...'

    for f in os.listdir('/home/gm515/Documents/GitHub/cell_counting/classifier/8-bit/test_data/cell/'):
        shutil.move('/home/gm515/Documents/GitHub/cell_counting/classifier/8-bit/test_data/cell/'+f,'/home/gm515/Documents/GitHub/cell_counting/classifier/8-bit/training_data/cell/'+f)

    for f in os.listdir('/home/gm515/Documents/GitHub/cell_counting/classifier/8-bit/test_data/nocell/'):
        shutil.move('/home/gm515/Documents/GitHub/cell_counting/classifier/8-bit/test_data/nocell/'+f,'/home/gm515/Documents/GitHub/cell_counting/classifier/8-bit/training_data/nocell/'+f)

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

input_layer = Input(shape=(80, 80, 3))

x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_3a')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=192,
                     filters_5x5_reduce=32,
                     filters_5x5=96,
                     filters_pool_proj=64,
                     name='inception_3b')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=192,
                     filters_3x3_reduce=96,
                     filters_3x3=208,
                     filters_5x5_reduce=16,
                     filters_5x5=48,
                     filters_pool_proj=64,
                     name='inception_4a')


x1 = AveragePooling2D((5, 5), strides=3)(x)
x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(2, activation='softmax', name='auxilliary_output_1')(x1)

x = inception_module(x,
                     filters_1x1=160,
                     filters_3x3_reduce=112,
                     filters_3x3=224,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4b')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4c')

x = inception_module(x,
                     filters_1x1=112,
                     filters_3x3_reduce=144,
                     filters_3x3=288,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4d')


x2 = AveragePooling2D((5, 5), strides=3)(x)
x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(2, activation='softmax', name='auxilliary_output_2')(x2)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_4e')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5a')

x = inception_module(x,
                     filters_1x1=384,
                     filters_3x3_reduce=192,
                     filters_3x3=384,
                     filters_5x5_reduce=48,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5b')

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

x = Dropout(0.4)(x)

x = Dense(2, activation='softmax', name='output')(x)

model = Model(input_layer, [x, x1, x2], name='inception_v1')

model.summary()

# ==============================================================================
# Get the data
# ==============================================================================

import os, glob, random, shutil
import numpy as np

training_data_dir = '/home/gm515/Documents/GitHub/cell_counting/classifier/8-bit/training_data'
test_data_dir = '/home/gm515/Documents/GitHub/cell_counting/classifier/8-bit/test_data'

if os.listdir(test_data_dir+'/cell/'):
    print 'Warning! Files detected in test data directory!'
    cleanup()

print "Splitting all training data into 70% training and 30% test data directories..."

test_cell_data = random.sample(os.listdir(training_data_dir+'/cell/'), int(0.3*len(os.listdir(training_data_dir+'/cell/'))))
test_nocell_data = random.sample(os.listdir(training_data_dir+'/nocell/'), int(0.3*len(os.listdir(training_data_dir+'/nocell/'))))

for f in test_cell_data:
    shutil.move(training_data_dir+'/cell/'+f,test_data_dir+'/cell/'+f)

for f in test_nocell_data:
    shutil.move(training_data_dir+'/nocell/'+f,test_data_dir+'/nocell/'+f)

from PIL import Image

training_data_cell = []
training_data_nocell = []
test_data_cell = []
test_data_nocell = []
training_label_cell = []
training_label_nocell = []
test_label_cell = []
test_label_nocell = []

for imagepath in glob.glob(training_data_dir+'/cell/*.tif'):
    image = Image.open(imagepath).convert(mode='RGB')
    training_data_cell.append(np.array(image))
    training_label_cell.append([1, 0])

for imagepath in glob.glob(training_data_dir+'/nocell/*.tif'):
    image = Image.open(imagepath).convert(mode='RGB')
    training_data_nocell.append(np.array(image))
    training_label_nocell.append([0, 1])

for imagepath in glob.glob(test_data_dir+'/cell/*.tif'):
    image = Image.open(imagepath).convert(mode='RGB')
    test_data_cell.append(np.array(image))
    test_label_cell.append([1, 0])

for imagepath in glob.glob(test_data_dir+'/nocell/*.tif'):
    image = Image.open(imagepath).convert(mode='RGB')
    test_data_nocell.append(np.array(image))
    test_label_nocell.append([0, 1])

training_data_cell = np.array(training_data_cell)
training_data_nocell = np.array(training_data_nocell)
test_data_cell = np.array(test_data_cell)
test_data_nocell = np.array(test_data_nocell)
training_label_cell = np.array(training_label_cell)
training_label_nocell = np.array(training_label_nocell)
test_label_cell = np.array(test_label_cell)
test_label_nocell = np.array(test_label_nocell)

training_data_all = np.concatenate((training_data_cell, training_data_nocell), axis=0)
test_data_all = np.concatenate((test_data_cell, test_data_nocell), axis=0)

training_data_all_label = np.concatenate((training_label_cell, training_label_nocell), axis=0)
test_data_all_label = np.concatenate((test_label_cell, test_label_nocell), axis=0)


# ==============================================================================
# Data augmentation
# ==============================================================================

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import Augmentor

training_datagen = Augmentor.Pipeline()

training_datagen.rotate_without_crop(probability=0.5, max_left_rotation=25, max_right_rotation=25, expand=False)
training_datagen.zoom(probability=0.5, min_factor=0.9, max_factor=1.1)
training_datagen.flip_left_right(probability=0.5)
training_datagen.flip_top_bottom(probability=0.5)
training_datagen.skew(probability=0.5, magnitude=0.3)
training_datagen.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
training_datagen.shear(probability=0.5,  max_shear_left=2, max_shear_right=2)
training_datagen.random_contrast(probability=0.5, min_factor=0.3, max_factor=1.)

training_data = training_datagen.keras_generator_from_array(training_data_all, training_data_all_label, batch_size=32, scaled=True)

# test data
test_datagen = Augmentor.Pipeline()

test_data = test_datagen.keras_generator_from_array(test_data_all, test_data_all_label, batch_size=32, scaled=True)

# ==============================================================================
# Model fit
# ==============================================================================

import datetime
from tensorflow.keras.callbacks import ModelCheckpoint

steps_epoch = len([filename for filename in os.listdir(training_data_dir+'/cell') if filename.endswith(".tif")]) + len([filename for filename in os.listdir(training_data_dir+'/nocell') if filename.endswith(".tif")])//32
steps_valid = len([filename for filename in os.listdir(test_data_dir+'/cell') if filename.endswith(".tif")]) + len([filename for filename in os.listdir(test_data_dir+'/nocell') if filename.endswith(".tif")])//32


# Checkpoint to only save the best model, metric = val_acc
strdate = datetime.datetime.today().strftime('%Y_%m_%d')

if not os.path.exists('models/'+strdate):
    os.makedirs('models/'+strdate)

filepath = 'models/'+strdate+'/cc_model_'+strdate+'.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = checkpoint

epochs = 25
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

history = model.fit(
    training_data_all,
    [training_data_all_label, training_data_all_label, training_data_all_label],
    validation_data=(test_data_all, [test_data_all_label, test_data_all_label, test_data_all_label]),
    batch_size=256,
    epochs=epochs,
    callbacks=[lr_sc, callbacks_list])

cleanup()
