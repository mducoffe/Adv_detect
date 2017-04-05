#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon Mar 20 10:06:26 2017

@author: mducoffe
CNN MNIST
"""

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras.utils.np_utils as kutils
import numpy as np
import os
from adversarial import Adversarial_Goodfellow, Adversarial_Szegedy, Adversarial_DeepFool
import matplotlib.pyplot as plt
import argparse
from fisher_layer import Fisher


def build_model_MNIST(num_classes=10, img_rows=28, img_cols=28):

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_dim_ordering() == "th":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        
        
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

    model = Sequential()
    #model.add(Reshape(input_shape, input_shape=(np.prod(input_shape),)))
    model.add(Conv2D(32, 3, 3,
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(10, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    return model, ((x_train, y_train), (x_test, y_test))


def train_ensemble(nb_ensemble=5):

    repo="./weights/ensemble/CNN/MNIST"
    for i in range(nb_ensemble):
        batch_size = 128
        epochs = 15
        model, dataset = build_model_MNIST()
        ((x_train, y_train), (x_test, y_test)) = dataset
        permut = np.random.permutation(len(y_train))
        x_train = x_train[permut]
        y_train = y_train[permut]
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs,
                  verbose=0, validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        model.save_weights(os.path.join(repo, 'ensemble_MNIST'+str(i)))
        print('Test accuracy:', score[1])


        
def test_ensemble_robustness_adv(eps=1.):
    
    ensemble=[]
    repo="./weights/ensemble/CNN/MNIST"
    filenames = os.listdir(repo)
    model, dataset = build_model_MNIST()
    (X_train, Y_train), (X_test, Y_test) = dataset
    model.load_weights(os.path.join(repo, filenames[0]))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', score[1])
    # build Fisher
    fisher = Fisher(model)
    
    X_train = X_train[:500]
    Y_train = Y_train[:500]
    #fisher.fisher_information(X_train, Y_train)
    #fisher.save('./weights/ensemble/CNN', 'fisher_mnist')
    fisher.load('./weights/ensemble/CNN', 'fisher_mnist')
    print('fisher downloaded')
    filenames = filenames[1:]

    for filename in filenames:
        #net, dataset = build_model_MNIST()
        net = fisher.fisher_sample()
        #ensemble.append(net)
        #return
        #score = net.evaluate(X_test, Y_test, verbose=0)
        #print(score[1])
        #return
        #net.load_weights(os.path.join(repo, filename))
        ensemble.append(net)
        
    # Test accuracy: 0.9814 -> 0.9829


    adv_object = Adversarial_Goodfellow(epsilon=eps, model=model, dataset=dataset,
                                        n_channels=1, img_nrows=28, img_ncols=28)

    #x_adv, y_adv = adv_object.generate()

    """
    for i in range(len(y_adv)):
        if not(y_adv[i][0]==np.argmax(model.predict(x_adv[i:i+1]))):
            print('PROBLEM')
    return
    """

    #adv_object.save('./adv', 'mnist_v0')
    #x_adv, y_adv = adv_object.load('./adv', 'mnist_v0')

    prediction = model.predict(x_adv)
    predict_ensemble = np.mean([net.predict(x_adv) for net in ensemble], axis=0)
    
    distance = prediction - predict_ensemble
    print('ADVERSARIAL')
    for i in range(len(distance)):
        print(np.linalg.norm(distance[i]), np.max(prediction[i]), np.max(predict_ensemble[i]))
        
    _, (x_test, y_test) = dataset
    n = len(y_test)
    x_true = x_test[np.random.permutation(n)][:len(x_adv)]
    
    prediction = model.predict(x_true)
    predict_ensemble = np.mean([net.predict(x_true) for net in ensemble[1:]], axis=0)
    
    distance = prediction - predict_ensemble
    print('GROUNDTRUTH')
    for i in range(len(distance)):
        print(np.linalg.norm(distance[i]), np.max(prediction[i]), np.max(predict_ensemble[i]))
    
    
    """
    index = np.random.randint(len(y_adv))
    for index in range(len(y_adv)):
        
        image = x_adv[index:index+1]
        tmp = adv_object.predict(image)
        if tmp != y_adv[index]:
            for i in range(10):
                print(tmp)
                tmp = adv_object.predict(image)
            import pdb; pdb.set_trace()
    """
    
def test_classification(adv_option, fisher_ensemble=False):

    ensemble=[]
    repo="./weights/ensemble/CNN/MNIST/ensemble"
    filenames = os.listdir(repo)
    model, dataset = build_model_MNIST()
    (X_train, Y_train), (X_test, Y_test) = dataset
    model.load_weights(os.path.join(repo, filenames[0]))
    
    fisher = None
    if fisher_ensemble:
        fisher = Fisher(model)
        fisher.load('./weights/ensemble/CNN', 'fisher_mnist')

    filenames = filenames[1:]
    for filename in filenames:
        if fisher_ensemble:
            net = fisher.fisher_sample()
        else:
            net, _ = build_model_MNIST()
            net.load_weights(os.path.join(repo, filename))
        
        ensemble.append(net)
        
    if not(adv_option in ['szegedy', 'goodfellow', 'deepfool']):
        raise NotImplementedError()
        
    if adv_option=='szegedy':
        adv_object =  Adversarial_Szegedy( model=model, dataset=dataset,
                                        n_channels=1, img_nrows=28, img_ncols=28, use_train=True)

        true_x, adv_x, adv_y = adv_object.load('./adv/MNIST/train', 'szegedy')
    if adv_option=='goodfellow':
        adv_object = Adversarial_Goodfellow(epsilon=1., model=model, dataset=dataset,
                                        n_channels=1, img_nrows=28, img_ncols=28, use_train=False)

        true_x, adv_x, adv_y = adv_object.load('./adv/MNIST/test', 'goodfellow')
    if adv_option=='deepfool':
        adv_object =  Adversarial_DeepFool(model=model, dataset=dataset,
                                        n_channels=1, img_nrows=28, img_ncols=28, use_train=False)

        true_x, adv_x, adv_y = adv_object.load('./adv/MNIST/test', 'deepfool')
        
    prediction = model.predict(adv_x)

    prediction_ensemble = [net.predict(adv_x) for net in ensemble[1:]]
    predict_ensemble = np.mean(prediction_ensemble, axis=0)
    
    prediction_true_ensemble = [net.predict(true_x) for net in ensemble[1:]]
    predict_true = np.mean(prediction_true_ensemble, axis=0)
    
    i=3
    print(np.linalg.norm(predict_ensemble[i] - predict_true[i]))
    return
    
    
    distance = prediction - predict_ensemble
    dico = {}
    dico['ADVERSARIAL']=[]
    print('ADVERSARIAL')
    for i in range(len(distance)):
        tmp = np.linalg.norm(distance[i])
        print( tmp, np.min(distance[i]), np.max(distance[i]), np.max(prediction[i]), np.max(predict_ensemble[i]), 0)
        dico['ADVERSARIAL'].append(tmp)
        
    _, (x_test, y_test) = dataset
    n = len(y_test)
    x_true = x_test[np.random.permutation(n)][:len(adv_x)]
    
    prediction = model.predict(x_true)
    prediction_ensemble = [net.predict(x_true) for net in ensemble[1:]]
    predict_ensemble = np.mean(prediction_ensemble, axis=0)
    
    distance = prediction - predict_ensemble
    dico['GROUNDTRUTH']=[]
    print('GROUNDTRUTH')
    for i in range(len(distance)):
        tmp = np.linalg.norm(distance[i])
        print( tmp, np.min(distance[i]), np.max(distance[i]), np.max(prediction[i]), np.max(predict_ensemble[i]), 1)
        dico['GROUNDTRUTH'].append(tmp)

    import pickle as pkl
    from contextlib import closing
    if fisher_ensemble:
        filename = os.path.join('./adv/MNIST/test', adv_option+'_density_adv')
    else:

        filename = os.path.join('./adv/MNIST/test', adv_option+'_density')
    with closing(open(filename, 'wb')) as f:
        pkl.dump(dico, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    
    


        

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='MNIST CNN adversarial examples')

    parser.add_argument('--option', type=str, default='szegedy', help='type of adversarial noise considered')
    parser.add_argument('--fisher', type=int, default=0, help='use Fisher ensemble')
    
    args = parser.parse_args()
    
    adv_option = args.option
    
    """
    if args.fisher==0:
        use_fisher=False
    else:
        use_fisher=True
    """
    use_fisher=False

    test_classification(adv_option, fisher_ensemble=use_fisher)
    
    #train_ensemble()






