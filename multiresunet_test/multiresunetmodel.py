import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Conv2D, Dropout, UpSampling2D, MaxPool2D, Add
from tensorflow.keras.optimizers import Adam, SGD

import losses

# Multi-resolution Inception style filters
def _multiresblock(inputs, filter_size1, filter_size2, filter_size3, filter_size4):
    """
    Multi-resolution block in the style of Inception module. This concatenates
    the features from different convolution filters as an appoximation for a
    3x3, 5x5 and 7x7 filter size convolution.
    """
    cnn1 = Conv2D(filter_size1, (3,3), padding='same', activation='relu')(inputs)
    cnn2 = Conv2D(filter_size2, (3,3), padding='same', activation='relu')(cnn1)
    cnn3 = Conv2D(filter_size3, (3,3), padding='same', activation='relu')(cnn2)

    cnn = Conv2D(filter_size4, (1,1), padding='same', activation='relu')(inputs)

    concat = Concatenate()([cnn1, cnn2, cnn3])
    add = Add()([concat, cnn])

    return add

# Concatenation path from encoder to decoder. This performs convolutions on the
# encoder segment to match higher level feature set seen in decoder segment
def _residualpath(inputs, filter_size, path_number):
    """
    Residual block which performs convolution on the encoder side before
    concatenating with the decoder side.
    """
    def block(x, fl):
        cnn1 = Conv2D(filter_size, (3,3), padding='same', activation='relu')(inputs)
        cnn2 = Conv2D(filter_size, (1,1), padding='same', activation='relu')(inputs)

        add = Add()([cnn1, cnn2])

        return add

    cnn = block(inputs, filter_size)
    if path_number <= 3:
        cnn = block(cnn, filter_size)
        if path_number <= 2:
            cnn = block(cnn, filter_size)
            if path_number <= 1:
                cnn = block(cnn, filter_size)

    return cnn

# Main multi-resolution UNet network
def multiresunet(loss_fn, input_size=(None, None, 1)):
    inputs = Input(input_size)

    multires1 = _multiresblock(inputs,8,17,26,51)
    pool1 = MaxPool2D()(multires1)

    multires2 = _multiresblock(pool1,17,35,53,105)
    pool2 = MaxPool2D()(multires2)

    multires3 = _multiresblock(pool2,31,72,106,209)
    pool3 = MaxPool2D()(multires3)

    multires4 = _multiresblock(pool3,71,142,213,426)
    drop4 = Dropout(0.5)(multires4) # Added dropout to last two layers
    pool4 = MaxPool2D()(multires4)

    multires5 = _multiresblock(pool4,142,284,427,853)
    drop5 = Dropout(0.5)(multires5) # Added dropout to last two layers
    upsample = UpSampling2D()(multires5)

    residual4 = _residualpath(multires4,256,4)
    concat = Concatenate()([upsample,residual4])

    multires6 = _multiresblock(concat,71,142,213,426)
    upsample = UpSampling2D()(multires6)

    residual3 = _residualpath(multires3,128,3)
    concat = Concatenate()([upsample,residual3])

    multires7 = _multiresblock(concat,31,72,106,209)
    upsample = UpSampling2D()(multires7)

    residual2 = _residualpath(multires2,64,2)
    concat = Concatenate()([upsample,residual2])

    multires8 = _multiresblock(concat,17,35,53,105)
    upsample = UpSampling2D()(multires8)

    residual1 = _residualpath(multires1,32,1)
    concat = Concatenate()([upsample,residual1])

    multires9 = _multiresblock(concat,8,17,26,51)
    sigmoid = Conv2D(1, (1,1), padding='same', activation='sigmoid')(multires9)

    model = Model(inputs, sigmoid)
    model.compile(optimizer=SGD(lr=1e-1), loss=[loss_fn], metrics=[losses.dice_loss, loss_fn])

    return model
