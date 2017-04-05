#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:20:31 2017

@author: ubuntu
"""

#### ensemble training for cifar 10 ######
import sys
sys.path.append('./models')
sys.path.append('./snapshot')
import json
import numpy as np
import sklearn.metrics as metrics
import argparse

import keras.utils.np_utils as kutils
from keras.datasets import cifar100
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from snapshot import SnapshotCallbackBuilder
from models import wide_residual_net as WRN, dense_net as DN

from fisher_layer import Fisher
import os
from contextlib import closing


def build_model(repo='', filename=''):
    parser = argparse.ArgumentParser(description='CIFAR 100 Ensemble Prediction')
    
    parser.add_argument('--optimize', type=int, default=0, help='Optimization flag. Set to 1 to perform a randomized '
                                                                'search to maximise classification accuracy. \n'
                                                                'Set to -1 to get non weighted classification accuracy')
    
    parser.add_argument('--num_tests', type=int, default=20, help='Number of tests to perform when optimizing the '
                                                                  'ensemble weights for maximizing classification accuracy')
    
    parser.add_argument('--model', type=str, default='wrn', help='Type of model to train')
    
    # Wide ResNet Parameters
    parser.add_argument('--wrn_N', type=int, default=2, help='Number of WRN blocks. Computed as N = (n - 4) / 6.')
    parser.add_argument('--wrn_k', type=int, default=4, help='Width factor of WRN')
    
    # DenseNet Parameters
    parser.add_argument('--dn_depth', type=int, default=40, help='Depth of DenseNet')
    parser.add_argument('--dn_growth_rate', type=int, default=12, help='Growth rate of DenseNet')
    
    args = parser.parse_args()

    
    (trainX, trainY), (testX, testY) = cifar100.load_data()
    nb_classes = len(np.unique(testY))
    
    trainX = trainX.astype('float32')
    trainX /= 255.0
    testX = testX.astype('float32')
    testX /= 255.0
    
    trainY = kutils.to_categorical(trainY)
    testY_cat = kutils.to_categorical(testY)
    
    if K.image_dim_ordering() == "th":
        init = (3, 32, 32)
    else:
        init = (32, 32, 3)
    
    model = WRN.create_wide_residual_network(init, nb_classes=100, N=args.wrn_N, k=args.wrn_k, dropout=0.00)
    model_prefix = 'WRN-CIFAR100-%d-%d' % (args.wrn_N * 6 + 4, args.wrn_k)
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc"])
    import os
    filename = os.path.join(repo, filename)
    if os.path.exists(filename):
        model.load_weights(filename)
    print("Finished compiling")
    return model, ((trainX, trainY), (testX, testY))

def evaluate(model, data):

    (X_train,Y_train), (X_test, Y_test) = data
    yPreds = model.predict(X_test)
    yPred = np.argmax(yPreds, axis=1)
    yTrue = Y_test
    
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)

def build_Fisher(model, dataset):
    (trainX, trainY), (testX, testY) = dataset
    fisher = Fisher(model)
    dico = fisher.fisher_information(trainX, trainY)
    print('suceed')
    dico = fisher.fisher_information(trainX, trainY)
    fisher.save('.', 'fisher_WRN_CIFAR100')
    for key in dico:
        assert not(np.isnan(dico[key]).any()), 'Nan values !'
        print(key)
        print(dico[key].shape)
    print('test WRN ok !')

if __name__=="__main__":
    filename='WRN-CIFAR100-16-4-Best.h5'
    model, dataset = build_model(repo='./weights', filename=filename)
    #evaluate(model, dataset)
    build_Fisher(model, dataset)
