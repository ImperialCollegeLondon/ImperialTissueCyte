# Importing Keras libraries

import os

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['openmp'] = 'True'

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

#=============================================================================================
# Construction of Convolution Neural Network
#=============================================================================================

print "Constructing Convolution Neural Network..."

# Create object of sequential class
classifier = Sequential()

# Convolution operation
# Number of filters
# Shape of each filter
# Input shape in first two, input type in third, 3 = RGB
# activation function, relu = rectifier function
classifier.add(Conv2D(32, (3, 3), input_shape = (80, 80, 1), activation = 'relu'))

# Pooling operation
# Pooling reduces size of images in order to reduce number of nodes
# 2x2 matrix
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), input_shape = (80, 80, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), input_shape = (80, 80, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening operation
# Convert pooled images into vectors
classifier.add(Flatten())

# Connect the set of nodes which input to connected layers
# Units is number of nodes
classifier.add(Dense(units = 128, activation = 'relu'))

# Prevent nodes from generating same classifiers
classifier.add(Dropout(0.5))

# Initialise output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile
# Optimizer is stochastic gradient descent
# Loss is loss function
# Metric is performance metric
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

print "Done!\n"

#=============================================================================================
# Preparing training and test data
#=============================================================================================

from keras.preprocessing.image import ImageDataGenerator

print "Preparing training and test data..."

# training data
train_datagen = ImageDataGenerator(rotation_range=180, rescale = 1./255, shear_range = 0.1, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, vertical_flip = True)
training_data = train_datagen.flow_from_directory('8-bit/training_data', target_size = (80, 80), batch_size = 32, class_mode = 'binary', color_mode = 'grayscale')
#training_data = train_datagen.flow_from_directory('8-bit/training_data', target_size = (80, 80), batch_size = 32, class_mode = 'binary', color_mode = 'grayscale', save_to_dir='preview', save_prefix='cell', save_format='jpeg')
# test data
test_datagen = ImageDataGenerator(rescale = 1./255)
test_data = test_datagen.flow_from_directory('8-bit/test_data', target_size = (80, 80), batch_size = 32, class_mode = 'binary', color_mode = 'grayscale')

print "Done!\n"

#=============================================================================================
# Fitting data to model
#=============================================================================================

print "Fitting data to model..."

# Find number of epoch and validation steps
steps_epoch = len([filename for filename in os.listdir('8-bit/training_data/cell') if filename.endswith(".tif")]) + len([filename for filename in os.listdir('8-bit/training_data/nocell') if filename.endswith(".tif")])
steps_valid = len([filename for filename in os.listdir('8-bit/test_data/cell') if filename.endswith(".tif")]) + len([filename for filename in os.listdir('8-bit/test_data/nocell') if filename.endswith(".tif")])

# Checkpoint to only save the best model, metric = val_acc
filepath = "cc_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# steps_per_epoch is number of images in training set
classifier.fit_generator(training_data, steps_per_epoch = steps_epoch, epochs = 50, callbacks=callbacks_list, validation_data = test_data, validation_steps = steps_valid, shuffle = True)

print "Done!\n"
