
import os

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot


# Helper to build a conv -> BN -> relu block
def conv_bn_relu(input, nb_filter, nb_row, nb_col, W_regularizer, subsample=(1, 1)):
    conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                         init="he_normal", border_mode="same",W_regularizer=W_regularizer)(input)
    norm = BatchNormalization(mode=0, axis=1)(conv)
    return Activation("relu")(norm)

# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def bn_relu_conv(input, nb_filter, nb_row, nb_col, W_regularizer, subsample=(1, 1)):
    norm = BatchNormalization(mode=0, axis=1)(input)
    activation = Activation("relu")(norm)
    return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                         init="he_normal", border_mode="same", W_regularizer=W_regularizer)(activation)


# Builds a residual block with repeating bottleneck blocks.
def residual_block(input, block_function, nb_filters, repetations, is_first_layer=False, subsample=False):
    for i in range(repetations):
        init_subsample = (1, 1)
        if i == 0 and (is_first_layer or subsample):
            init_subsample = (2, 2)
        input = block_function(input, nb_filters=nb_filters, init_subsample=init_subsample)
    return input
