#from keras.models import Sequential
#from keras.layers.core import Flatten, Dense, Dropout
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from keras.optimizers import SGD
#import cv2, numpy as np
import keras
from keras.layers import Dense, Activation, Dropout, Flatten
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_normal
from keras.utils import np_utils
#from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras import backend as K
from keras.models import Model
from keras.layers import concatenate, \
    Input, Conv2D, Conv3D, SeparableConv2D, Activation, BatchNormalization, Subtract, MaxPooling2D, Dropout, \
    UpSampling2D

def lenet(in_shape=(50,50,2), n_classes=40):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='relu',
                     input_shape=in_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    return model
def alexnet(in_shape=(50, 50, 2), n_classes=40, l2_reg=0.0, weights=None, fc1=4096,fc2=4096):
    """
    AlexNet model
    :param input_shape: input shape
    :param num_classes: the number of classes
    :param l2_reg:
    :param weights:
    :return: model
    """
    input_layer = Input(shape=in_shape)

    # Layer 1
    # In order to get the same size of the paper mentioned, add padding layer first
    x = ZeroPadding2D(padding=(2, 2))(input_layer)
    x = conv_block(x, filters=96, kernel_size=(5, 5),
                   strides=(2, 2), padding="valid", l2_reg=l2_reg, name='Conv_1_96_11x11_4')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="maxpool_1_3x3_2")(x)

    # Layer 2
    x = conv_block(x, filters=256, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_2_256_5x5_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="maxpool_2_3x3_2")(x)

    # Layer 3
    x = conv_block(x, filters=384, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_3_384_3x3_1")

    # Layer 4
    x = conv_block(x, filters=384, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_4_384_3x3_1")

    # Layer 5
    x = conv_block(x, filters=256, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_5_256_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="maxpool_3_3x3_2")(x)

    # Layer 6
    x = Flatten()(x)
    x = Dense(units=fc1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 7
    x = Dense(units=fc2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 8
    x = Dense(units=n_classes)(x)
    x = BatchNormalization()(x)
    x = Activation("softmax")(x)

    if weights is not None:
        x.load_weights(weights)
    model = Model(input_layer, x, name="AlexNet")
    return model


def conv_block(layer, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', l2_reg=0.0, name=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_regularizer=l2(l2_reg),
               kernel_initializer="he_normal",
               name=name)(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def vgg16(in_shape=(50,50,2), n_classes=40, fc1=1024, fc2=512):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape, name='block1_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv2'))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))


    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Flatten())

    model.add(Dense(fc1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(fc2, name='fc2'))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    num=n_classes
    model.add(Dense(num))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    return model


