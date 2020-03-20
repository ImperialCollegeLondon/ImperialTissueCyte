#!/bin/sh

# ipython benchmark.py -- ../multiresunet_test/models/2020_02_25_Adam_lr0.0001_MultiResUNet

# for dir in ../cell_counting_unet/models/*/
# do
#     dir=${dir%*/}      # remove the trailing "/"
#     ipython benchmark.py -- ${dir}    # print everything after the final "/"
# done

# for dir in ../unet_test/models/*/
# do
#     dir=${dir%*/}      # remove the trailing "/"
#     ipython benchmark.py -- ${dir}    # print everything after the final "/"
# done

# for dir in ../multiresunet_test/models/*/
# do
#     dir=${dir%*/}      # remove the trailing "/"
#     ipython benchmark.py -- ${dir}    # print everything after the final "/"
# done

for dir in ../../machinelearning/models/*/
do
    dir=${dir%*/}      # remove the trailing "/"
    ipython benchmark.py -- ${dir}    # print everything after the final "/"
done
