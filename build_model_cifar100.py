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
from keras.layers import Activation


def build_model(repo='', filename=''):
    
    """
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
    """
    N = 2; k = 4
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
    
    model = WRN.create_wide_residual_network(init, nb_classes=100, N=N, k=k, dropout=0.00)
    model_prefix = 'WRN-CIFAR100-%d-%d' % (N * 6 + 4, k)
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
    return accuracy

def build_Fisher():
    filename='WRN-CIFAR100-16-4-Best_v2.h5'
    model, dataset = build_model(repo='./weights', filename=filename)
    (trainX, trainY), (testX, testY) = dataset
    fisher = Fisher(model)
    dico = fisher.fisher_information(testX, testY)
    print('suceed')
    dico = fisher.fisher_information(trainX, trainY)
    fisher.save('.', 'fisher_WRN_CIFAR100_test')
    for key in dico:
        assert not(np.isnan(dico[key]).any()), 'Nan values !'
        #print(key)
        #print(dico[key].shape)
    print('test WRN ok !')
    
def kl(ensemble, network, data):
    kl=[]
    (X_train, Y_train), (X_test, Y_test) = data
    X_train = X_test;
    
    p_network = network.predict(X_train) # (1000, 10)
    p_model = np.mean([model.predict(X_train, batch_size=128) for model in ensemble], axis=0) # (1000, 10)
    n = len(X_train)
    for i in range(n):
        kl_n = 0
        for j in range(10):
            kl_n += p_network[i,j]*np.log(p_network[i,j]/p_model[i,j])
        kl.append(kl_n)
    return np.mean(kl)

def correlation():
    filename='WRN-CIFAR10-16-4-Best.h5'
    model, data = build_model(repo='./weights', filename=filename)
    #filename='WRN-CIFAR10-16-4-1.h5'
    #network, _ = build_model(repo='./weights', filename=filename)
    fisher = Fisher(model)
    saving_filename = 'fisher_WRN_CIFAR10'
    fisher.load('.', saving_filename)
    network = fisher.fisher_sample()
    evaluate(network, data)

def ensemble():
    
    filename='WRN-CIFAR100-16-4-Best_v2.h5'
    #saving_filename = 'fisher_WRN_CIFAR100_v2_v2'
    saving_filename = 'fisher_WRN_CIFAR100_test'
    model, dataset = build_model(repo='./weights', filename=filename)

    fisher = Fisher(model)
    fisher.load('.', saving_filename)
    """
    for i in range(1):
        print(i)
        network = fisher.fisher_sample()
        evaluate(network, dataset)
        # correlation
        print(kl([model], network, dataset))
        
    return
    """
    ensemble_networks = [model] +  [fisher.fisher_sample() for i in range(5)]
    """
    kl_divergence = []
    for i in range(4):
        networks = [fisher.fisher_sample() for i in range(4)]
        kl_s = [kl(ensemble_networks, network, dataset) for network in networks]
        index = np.argmax(kl_s)
        kl_divergence.append(kl_s[index])
        ensemble_networks.append(networks[index])
    print('compute weights')
    print(kl_divergence)
    """
    from opt_weights import compute_opt_weights
    compute_opt_weights(ensemble_networks, dataset)
    return
        



   
    """
    (trainX, trainY), (testX, testY) = dataset
    predictions = np.concatenate([network.predict(testX)[None, :, :] for network in networks], axis=0)
    probabilities = np.concatenate([np.max(network.predict(testX), axis=1)[None, :] for network in networks], axis=0)
    (trainX, trainY), (testX, testY) = dataset

    for i in range(100):
        
        yPred = predictions[:, i]
        proba = probabilities[:, i]
        print(yPred, proba, testY[i])

    return
    accuracy = metrics.accuracy_score(testY, yPreds) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)
    return
    """
    from opt_weights import compute_opt_weights
    _, dataset = build_model(repo='./weights')
    compute_opt_weights(networks, dataset)
    
def proportionality():
    filename='WRN-CIFAR100-16-4-Best.h5'
    model, dataset = build_model(repo='./weights', filename=filename)
    (trainX, trainY), (testX, testY) = dataset
    print('BEFORE')
    #evaluate(model, dataset)
    
    
    
    layers = model.layers
    
    batch_norm_name = 'batchnormalization_1'
    conv_name = 'convolution2d_1'
    coeff=2.
    output_layer = None

    for layer in layers:
        print(layer.name)
        if layer.name == conv_name:
            W = layer.W
            b = layer.b
            W.set_value(coeff*W.get_value())
            b.set_value(coeff*b.get_value())

    """
    network = Model(model.input, output_layer.output)
    """
    y0 = model.predict(testX[0:1])[0]
    layer = layers[-1]
    W = layer.W
    b = layer.b
    #W.set_value(coeff*W.get_value())
    b.set_value(coeff+b.get_value())
    """
    for layer in layers:
        #print(layer.name)

        if layer.name == batch_norm_name:

            gamma = layer.gamma
            gamma.set_value(coeff*gamma.get_value())
            beta = layer.beta
            beta.set_value(coeff*beta.get_value())

        
        if layer.name == conv_name:
            W = layer.W
            b = layer.b
            W.set_value(coeff*W.get_value())
            b.set_value(coeff*b.get_value())
    """
    y1 = model.predict(testX[0:1])[0]
    print(np.max(np.abs(y0 - y1)))
    #evaluate(model, dataset)

if __name__=="__main__":
    ensemble()