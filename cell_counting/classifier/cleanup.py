
# -*- coding: utf-8 -*-

"""
################################################################################
Classfier Data Clean-up
Author: Gerald M

Cleans up the directories and keeps raw data back into /raw_data.
################################################################################
"""

import os
import shutil

def clean():
    training_data_dir = '8-bit/training_data'
    test_data_dir = '8-bit/test_data'

    print ('Cleaning up directories...')

    if os.path.exists(training_data_dir) and os.path.isdir(training_data_dir): shutil.rmtree(training_data_dir)
    if os.path.exists(test_data_dir) and os.path.isdir(test_data_dir): shutil.rmtree(test_data_dir)

    print ('Done!')
