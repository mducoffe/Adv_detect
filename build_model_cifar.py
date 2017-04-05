#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:59:37 2017

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
from keras.datasets import cifar10
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from snapshot import SnapshotCallbackBuilder
from models import wide_residual_net as WRN, dense_net as DN


import os
from contextlib import closing


def build_model(repo='', filename=''):
    parser = argparse.ArgumentParser(description='CIFAR 10 Ensemble Prediction')
    
    parser.add_argument('--M', type=int, default=5, help='Number of snapshots')
    parser.add_argument('--nb_epoch', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--alpha_zero', type=float, default=0.1, help='Initial learning rate')
    
    parser.add_argument('--model', type=str, default='wrn', help='Type of model to train')
    
    # Wide ResNet Parameters
    parser.add_argument('--wrn_N', type=int, default=2, help='Number of WRN blocks. Computed as N = (n - 4) / 6.')
    parser.add_argument('--wrn_k', type=int, default=4, help='Width factor of WRN')
    
    # DenseNet Parameters
    parser.add_argument('--dn_depth', type=int, default=40, help='Depth of DenseNet')
    parser.add_argument('--dn_growth_rate', type=int, default=12, help='Growth rate of DenseNet')
    
    args = parser.parse_args()
    
    ''' Snapshot major parameters '''
    M = args.M # number of snapshots
    nb_epoch = T = args.nb_epoch # number of epochs
    alpha_zero = args.alpha_zero # initial learning rate
    
    model_type = str(args.model).lower()
    assert model_type in ['wrn', 'dn'], 'Model type must be one of "wrn" for Wide ResNets or "dn" for DenseNets'
    
    snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)
    
    batch_size = 128 if model_type == "wrn" else 64
    img_rows, img_cols = 32, 32
    
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    
    trainX = trainX.astype('float32')
    trainX /= 255.0
    testX = testX.astype('float32')
    testX /= 255.0
    
    trainY_cat = kutils.to_categorical(trainY)
    testY_cat = kutils.to_categorical(testY)
    
    if K.image_dim_ordering() == "th":
        init = (3, img_rows, img_cols)
    else:
        init = (img_rows, img_cols, 3)
    
    model = WRN.create_wide_residual_network(init, nb_classes=10, N=args.wrn_N, k=args.wrn_k, dropout=0.00)
    
    model_prefix = 'WRN-CIFAR10-%d-%d' % (args.wrn_N * 6 + 4, args.wrn_k)
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc"])
    import os
    filename = os.path.join(repo, filename)
    if os.path.exists(filename):
        model.load_weights(filename)
    print("Finished compiling")
    return model, ((trainX, trainY_cat), (testX, testY))

def evaluate(model, data):

    (X_train,Y_train), (X_test, Y_test) = data
    yPreds = model.predict(X_test)
    yPred = np.argmax(yPreds, axis=1)
    yTrue = Y_test
    
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)
    return accuracy



if __name__=="__main__":
    filename='WRN-CIFAR10-16-4-Best.h5'
    model, dataset = build_model(repo='./weights', filename=filename)
    evaluate(model, dataset)
