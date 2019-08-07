
# -*- coding: utf-8 -*-

"""
################################################################################
Classfier Data Augmentation
Author: Gerald M

Augments the training data and saves to directory
################################################################################
"""

import os
import Augmentor

def augment(n = 1000):
    for category in ['cell', 'nocell']:
        training_datagen = Augmentor.Pipeline(source_directory=os.path.join('8-bit/training_data',category), output_directory='.', save_format='tif')

        training_datagen.rotate_without_crop(probability=0.5, max_left_rotation=25, max_right_rotation=25, expand=False)
        training_datagen.zoom(probability=0.5, min_factor=0.9, max_factor=1.1)
        training_datagen.flip_left_right(probability=0.5)
        training_datagen.flip_top_bottom(probability=0.5)
        training_datagen.skew(probability=0.5, magnitude=0.3)
        training_datagen.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
        training_datagen.shear(probability=0.5,  max_shear_left=2, max_shear_right=2)
        training_datagen.random_contrast(probability=0.5, min_factor=0.3, max_factor=1.)

        training_datagen.sample(n)

        print ('')
