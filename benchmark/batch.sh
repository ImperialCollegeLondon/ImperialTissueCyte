#!/bin/sh

ipython benchmark.py -- ../multiresunet_test/2020_02_25_Adam_lr0.0001_MultiResUNet
ipython benchmark.py -- ../multiresunet_test/2020_02_26_SGD_lr0.0001_MultiResUNet
ipython benchmark.py -- ../multiresunet_test/2020_02_26_SGD_lr0.1_MultiResUNet
ipython benchmark.py -- ../multiresunet_test/2020_02_27_SGD_lr0.5_MultiResUNet
ipython benchmark.py -- ../multiresunet_test/2020_02_26_Adam_lr0.0001_MultiResUNet
ipython benchmark.py -- ../multiresunet_test/2020_02_26_SGD_lr0.01_MultiResUNet
ipython benchmark.py -- ../multiresunet_test/2020_02_27_Adam_lr0.0001_MultiResUNet
