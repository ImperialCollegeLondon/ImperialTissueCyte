"""
Model Training Data Clean-Up
Author: Gerald M

Deletes the training data directory.
"""

import os
import shutil

def clean():
    training_data_dir = 'input/training_data'
    test_data_dir = 'input/test_data'
    raw_data_copy_dir = 'input/raw_data_copy'

    print ('Cleaning up directories...')

    if os.path.exists(training_data_dir) and os.path.isdir(training_data_dir): shutil.rmtree(training_data_dir)
    if os.path.exists(test_data_dir) and os.path.isdir(test_data_dir): shutil.rmtree(test_data_dir)
    if os.path.exists(raw_data_copy_dir) and os.path.isdir(raw_data_copy_dir): shutil.rmtree(raw_data_copy_dir)

    print ('Done!')
