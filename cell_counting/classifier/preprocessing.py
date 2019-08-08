
# -*- coding: utf-8 -*-

"""
################################################################################
Classfier Data Preprocessing
Author: Gerald M

This script performs preprocesses the raw data stored in /raw_data by first
taking 70% of the raw data for training and 30% for testing. Data augmentation
is performed using Augmentor on the training data only.

ALL data is then standardised (x - x.mean()) / x.std() (featurewise rather than
sample wise) using the group mean() and std().

ALL standardised data is saved to the corresponding training and test directories.
################################################################################
"""

import Augmentor
import glob
import os
import random
import shutil
import cleanup
import augmentation
from PIL import Image
from tifffile import imsave
import numpy as np

def preprocess(standardise=True, normalise=False):
    raw_data_dir = '8-bit/raw_data'
    training_data_dir = '8-bit/training_data'
    test_data_dir = '8-bit/test_data'

    # Clean up any old directories and create new directories
    if os.path.exists(training_data_dir) and os.path.isdir(training_data_dir): cleanup.clean()

    if not os.path.exists(training_data_dir):
        os.makedirs(os.path.join(training_data_dir, 'cell'))
        os.makedirs(os.path.join(training_data_dir, 'nocell'))

    if not os.path.exists(test_data_dir):
        os.makedirs(os.path.join(test_data_dir, 'cell'))
        os.makedirs(os.path.join(test_data_dir, 'nocell'))

    # Randomly sample 30% of the raw_data set for test
    print ('Randomly selecting/moving 70% training and 30% test data...')
    raw_cell_data = os.listdir(raw_data_dir+'/cell/')
    raw_nocell_data = os.listdir(raw_data_dir+'/nocell/')
    random.shuffle(raw_cell_data)
    random.shuffle(raw_nocell_data)
    training_cell_data = raw_cell_data[:int(0.7*len(raw_cell_data))]
    training_nocell_data = raw_nocell_data[:int(0.7*len(raw_nocell_data))]
    test_cell_data  = raw_cell_data[int(0.7*len(raw_cell_data)):]
    test_nocell_data = raw_nocell_data[int(0.7*len(raw_nocell_data)):]

    for f in training_cell_data:
        shutil.copy(os.path.join(raw_data_dir,'cell',f), os.path.join(training_data_dir,'cell',f))

    for f in training_nocell_data:
        shutil.copy(os.path.join(raw_data_dir,'nocell',f), os.path.join(training_data_dir,'nocell',f))

    for f in test_cell_data:
        shutil.copy(os.path.join(raw_data_dir,'cell',f), os.path.join(test_data_dir,'cell',f))

    for f in test_nocell_data:
        shutil.copy(os.path.join(raw_data_dir,'nocell',f), os.path.join(test_data_dir,'nocell',f))

    print ('Done!')

    print ('Performing augmentation on training data...')

    n=1000
    augmentation.augment(n)

    print ('Augmented and saved with n='+str(n)+' samples!')

    print ('Loading images into array...')

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

    print ('Done!')

    if standardise:
        print ('Running featurewise (sample) standardisation...')

        feature_mean = np.mean(np.concatenate((training_data_all, test_data_all), axis=0))
        feature_std = np.std(np.concatenate((training_data_all, test_data_all), axis=0))

        training_data_all = (training_data_all-feature_mean)/feature_std
        test_data_all = (test_data_all-feature_mean)/feature_std

        print ('Done!')

    if normalise:
        print ('Running featurewise (sample) normalisation...')

        feature_max = np.max(np.concatenate((training_data_all, test_data_all), axis=0))

        training_data_all = training_data_all/feature_max
        test_data_all = test_data_all/feature_max

        print ('Done!')

    return (training_data_all, training_data_all_label, test_data_all, test_data_all_label)
