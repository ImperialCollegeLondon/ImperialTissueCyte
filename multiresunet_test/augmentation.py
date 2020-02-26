
# -*- coding: utf-8 -*-

"""
################################################################################
Classfier Data Augmentation
Author: Gerald M

Augments the training data and saves to directory. Adds an operation to perform
poisson noise addition to images in pipeline.
################################################################################
"""

import os
import Augmentor

from Augmentor.Operations import Operation
from skimage.util import noise
from PIL import Image
import numpy as np

class PoissonNoise(Operation):
    """
    Custom class to add poisson noise to input images as part of Augmentor
    pipeline.

    Params
    ------
    probability : float
        Probability value in range 0 to 1 to execute the operation.
    """
    def __init__(self, probability):
        Operation.__init__(self, probability)

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        def do(image):
            image = noise.random_noise(np.array(image), mode='poisson')
            image.clip(0, 1, image)
            image = np.uint8(np.round(image * 255.))

            return Image.fromarray(image)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images

def augment(dir, n):
    training_datagen = Augmentor.Pipeline(source_directory=os.path.join(dir,'images'), output_directory='.', save_format='tif')

    poisson_noise = PoissonNoise(probability=0.75)

    training_datagen.ground_truth(os.path.join(dir,'masks'))

    training_datagen.add_operation(poisson_noise)
    training_datagen.rotate_without_crop(probability=0.5, max_left_rotation=25, max_right_rotation=25, expand=False)
    training_datagen.zoom(probability=0.5, min_factor=0.9, max_factor=1.1)
    training_datagen.flip_left_right(probability=0.5)
    training_datagen.flip_top_bottom(probability=0.5)
    training_datagen.skew(probability=0.5, magnitude=0.3)
    training_datagen.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
    training_datagen.shear(probability=0.5,  max_shear_left=2, max_shear_right=2)
    training_datagen.random_contrast(probability=0.5, min_factor=0.3, max_factor=0.8)

    training_datagen.sample(n)

    print ('')
