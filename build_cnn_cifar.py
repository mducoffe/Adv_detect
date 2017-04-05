#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon Mar 27 10:52:49 2017

@author: mducoffe

ICLR CIFAR10
"""

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, normalization
from keras import backend as K
import keras.utils.np_utils as kutils
import numpy as np
import os
from adversarial import Adversarial_Goodfellow, Adversarial_DeepFool, Adversarial_Szegedy
import matplotlib.pyplot as plt
from fisher_layer import Fisher
from contextlib import closing
import pickle as pkl


def build_model_CIFAR10(num_classes=10, img_rows=32, img_cols=32, n_channels=3):

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if K.image_dim_ordering() == "th":
        x_train = x_train.reshape(x_train.shape[0], n_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], n_channels, img_rows, img_cols)
        input_shape = (n_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
        input_shape = (img_rows, img_cols, n_channels)
        
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = kutils.to_categorical(y_train)
    y_test = kutils.to_categorical(y_test)
    
    mean = np.mean(x_train, axis=0)
    x_train -= mean
    x_test -=mean
    

    model = Sequential()
    #model.add(Reshape(input_shape, input_shape=(np.prod(input_shape),)))
    model.add(Conv2D(32, 3, 3,
                     activation='relu',
                     input_shape=input_shape))
    model.add(normalization.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, 3,
                     activation='relu'))
    model.add(normalization.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, 3,
                     activation='relu'))
    model.add(normalization.BatchNormalization())
    #model.add(Dropout(0.25))
    model.add(Flatten())
    #model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    return model, ((x_train, y_train), (x_test, y_test))

def train_model():
    model, dataset = build_model_CIFAR10()
    batch_size = 128
    epochs = 100
    (x_train, y_train), (x_test, y_test) = dataset
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs,
                  verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test error', score[1])

if __name__=="__main__":
    train_model()