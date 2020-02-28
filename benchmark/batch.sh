#!/bin/sh

# ipython benchmark.py -- ../multiresunet_test/models/2020_02_25_Adam_lr0.0001_MultiResUNet
# ipython benchmark.py -- ../multiresunet_test/models/2020_02_26_SGD_lr0.0001_MultiResUNet
# ipython benchmark.py -- ../multiresunet_test/models/2020_02_26_SGD_lr0.1_MultiResUNet
# ipython benchmark.py -- ../multiresunet_test/models/2020_02_27_SGD_lr0.5_MultiResUNet
# ipython benchmark.py -- ../multiresunet_test/models/2020_02_26_Adam_lr0.0001_MultiResUNet
# ipython benchmark.py -- ../multiresunet_test/models/2020_02_26_SGD_lr0.01_MultiResUNet
# ipython benchmark.py -- ../multiresunet_test/models/2020_02_27_Adam_lr0.0001_MultiResUNet
#
# ipython benchmark.py -- ../unet_test/models/2020_01_20_UNet
# ipython benchmark.py -- ../unet_test/models/2020_01_21_UNet
# ipython benchmark.py -- ../unet_test/models/2020_01_22_UNet
# ipython benchmark.py -- ../unet_test/models/2020_02_21_UNet_BCE
# ipython benchmark.py -- ../unet_test/models/2020_02_24_UNet
# ipython benchmark.py -- ../unet_test/models/2020_02_24_UNet_BCE
# ipython benchmark.py -- ../unet_test/models/2020_01_20_UNet

for dir in ../cell_counting_unet/models/*/
do
    dir=${dir%*/}      # remove the trailing "/"
    echo ${dir}    # print everything after the final "/"
done

for dir in ../unet_test/models/*/
do
    dir=${dir%*/}      # remove the trailing "/"
    echo ${dir##*/}    # print everything after the final "/"
done

for dir in ../multiresunet_test/models/*/
do
    dir=${dir%*/}      # remove the trailing "/"
    echo ${dir##*/}    # print everything after the final "/"
done
