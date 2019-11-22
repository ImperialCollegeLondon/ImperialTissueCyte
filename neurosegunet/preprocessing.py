
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
import datetime
import glob
import os
import random
import shutil
import cleanup
import augmentation
import pickle
from PIL import Image
from tifffile import imsave
from natsort import natsorted
import numpy as np

def preprocess():
    raw_data_dir = 'data/raw-data'
    training_data_dir = 'data/training-data'
    test_data_dir = 'data/test-data'

    # Clean up any old directories and create new directories
    if os.path.exists(training_data_dir) and os.path.isdir(training_data_dir): cleanup.clean()

    if not os.path.exists(training_data_dir):
        os.makedirs(os.path.join(training_data_dir, 'images'))
        os.makedirs(os.path.join(training_data_dir, 'masks'))

    if not os.path.exists(test_data_dir):
        os.makedirs(os.path.join(test_data_dir, 'images'))
        os.makedirs(os.path.join(test_data_dir, 'masks'))

    print ('Randomly selecting/moving 70% training and 30% test data...')
    raw_images_data = os.listdir(raw_data_dir+'/images/')
    raw_masks_data = os.listdir(raw_data_dir+'/masks/')
    random.shuffle(raw_images_data)
    training_images_data = raw_images_data[:int(0.7*len(raw_images_data))]
    training_masks_data = [f.replace('image', 'mask') for f in training_images_data]
    test_images_data  = raw_images_data[int(0.7*len(raw_images_data)):]
    test_masks_data = [f.replace('image', 'mask') for f in test_images_data]

    for f in training_images_data:
        shutil.copy(os.path.join(raw_data_dir,'images',f), os.path.join(training_data_dir,'images',f))

    for f in training_masks_data:
        shutil.copy(os.path.join(raw_data_dir,'masks',f), os.path.join(training_data_dir,'masks',f))

    for f in test_images_data:
        shutil.copy(os.path.join(raw_data_dir,'images',f), os.path.join(test_data_dir,'images',f))

    for f in test_masks_data:
        shutil.copy(os.path.join(raw_data_dir,'masks',f), os.path.join(test_data_dir,'masks',f))

    print ('Done!')

    print ('Performing augmentation on training data...')

    n=4*len(training_images_data)
    augmentation.augment('data/training-data',n)

    aug_images = glob.glob('data/training-data/images/images_original*')
    aug_masks = glob.glob('data/training-data/images/_groundtruth*')
    aug_images.sort(key=lambda x:x[-40:])
    aug_masks.sort(key=lambda x:x[-40:])

    for i, (image_file, mask_file) in enumerate(zip(aug_images, aug_masks)):
        shutil.move(image_file, os.path.dirname(image_file)+'/aug_image_'+str(i)+'.tif')
        shutil.move(mask_file, os.path.dirname(mask_file).replace('/images','/masks')+'/aug_mask_'+str(i)+'.tif')

    print ('Augmented and saved with n='+str(n)+' samples!')

    print ('Performing augmentation on test data...')

    n=4*len(test_images_data)
    augmentation.augment('data/test-data',n)

    aug_images = glob.glob('data/test-data/images/images_original*')
    aug_masks = glob.glob('data/test-data/images/_groundtruth*')
    aug_images.sort(key=lambda x:x[-40:])
    aug_masks.sort(key=lambda x:x[-40:])

    for i, (image_file, mask_file) in enumerate(zip(aug_images, aug_masks)):
        shutil.move(image_file, os.path.dirname(image_file)+'/aug_image_'+str(i)+'.tif')
        shutil.move(mask_file, os.path.dirname(mask_file).replace('/images','/masks')+'/aug_mask_'+str(i)+'.tif')

    print ('Augmented and saved with n='+str(n)+' samples!')

    training_data_images = []
    training_data_masks = []
    test_data_images = []
    test_data_masks = []

    for imagepath in natsorted(glob.glob(training_data_dir+'/images/*')):
        image = Image.open(imagepath).resize((512, 512))
        training_data_images.append(np.array(image))

    for imagepath in natsorted(glob.glob(training_data_dir+'/masks/*')):
        image = Image.open(imagepath).resize((512, 512))
        training_data_masks.append(np.array(image))

    for imagepath in natsorted(glob.glob(test_data_dir+'/images/*')):
        image = Image.open(imagepath).resize((512, 512))
        test_data_images.append(np.array(image))

    for imagepath in natsorted(glob.glob(test_data_dir+'/masks/*')):
        image = Image.open(imagepath).resize((512, 512))
        test_data_masks.append(np.array(image))

    training_data_images = np.array(training_data_images).astype(np.float32)
    training_data_masks = np.array(training_data_masks).astype(np.float32)
    test_data_images = np.array(test_data_images).astype(np.float32)
    test_data_masks = np.array(test_data_masks).astype(np.float32)

    print ('Done!')

    print ('Running normalisation...')

    for idx, img in enumerate(training_data_images):
        training_data_images[idx] = (img-np.min(img))/(np.max(img)-np.min(img))

    for idx, img in enumerate(training_data_masks):
        img[img < (np.min(img)+np.max(img))/2] = 0.
        img[img >= (np.min(img)+np.max(img))/2] = 1.
        training_data_masks[idx] = img

    for idx, img in enumerate(test_data_images):
        test_data_images[idx] = (img-np.min(img))/(np.max(img)-np.min(img))

    for idx, img in enumerate(test_data_masks):
        img[img < (np.min(img)+np.max(img))/2] = 0.
        img[img >= (np.min(img)+np.max(img))/2] = 1.
        test_data_masks[idx] = img

    print ('Done!')

    print ('Removing any data with nan...')

    training_images_idx = []
    for idx, img in enumerate(training_data_images):
        if np.isnan(np.min(img) * np.max(img)) or (np.min(img)-np.max(img) == 0):
            training_images_idx.append(idx)

    training_masks_idx = []
    for idx, img in enumerate(training_data_masks):
        if np.isnan(np.min(img) * np.max(img)) or (np.min(img)-np.max(img) == 0):
            training_masks_idx.append(idx)

    test_images_idx = []
    for idx, img in enumerate(test_data_images):
        if np.isnan(np.min(img) * np.max(img)) or (np.min(img)-np.max(img) == 0):
            test_images_idx.append(idx)

    test_masks_idx = []
    for idx, img in enumerate(test_data_masks):
        if np.isnan(np.min(img) * np.max(img)) or (np.min(img)-np.max(img) == 0):
            test_masks_idx.append(idx)

    if len(training_images_idx) > 0:
        training_data_images = np.delete(training_data_images, np.array(training_images_idx), axis=0)

    if len(training_masks_idx) > 0:
        training_data_masks = np.delete(training_data_masks, np.array(training_masks_idx), axis=0)

    if len(test_images_idx) > 0:
        test_data_images = np.delete(test_data_images, np.array(test_images_idx), axis=0)

    if len(test_masks_idx) > 0:
        test_data_masks = np.delete(test_data_masks, np.array(test_masks_idx), axis=0)

    print ('Done!')

    training_data_images = training_data_images[..., np.newaxis]
    training_data_masks = training_data_masks[..., np.newaxis]
    test_data_images = test_data_images[..., np.newaxis]
    test_data_masks = test_data_masks[..., np.newaxis]

    return (training_data_images, training_data_masks, test_data_images, test_data_masks)
