#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Sat Mar 25 21:50:01 2017

@author: mducoffe
build confusion matrix
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras.utils.np_utils as kutils
import numpy as np
import os
from adversarial import Adversarial_Goodfellow, Adversarial_DeepFool, Adversarial_Szegedy
import matplotlib.pyplot as plt
from fisher_layer import Fisher
from contextlib import closing
import pickle as pkl


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


def predict(model, X, injectiv_table):
    prediction = np.argmax(model.predict(X), axis=1)
    for i in range(len(prediction)):
        prediction[i] = injectiv_table[prediction[i]]
    return prediction


def build_confusion_matrix(option_adv):

    repo="./weights/ensemble/CNN/MNIST"
    nb_class=10
    filenames = os.listdir(repo)
    model, dataset = build_model_MNIST()
    (X_train, Y_train), (X_test, Y_test) = dataset
    model.load_weights(os.path.join(repo, filenames[0]))

    if option_adv == 'goodfellow':
        adv_object = Adversarial_Goodfellow(epsilon=1., model=model, dataset=dataset,
                                        n_channels=1, img_nrows=28, img_ncols=28, use_train=True)
    

        z, x, y = adv_object.load('./adv/MNIST/train', 'goodfellow')
    
    if option_adv=='szegedy':
        adv_object = Adversarial_Szegedy(model=model, dataset=dataset,
                                        n_channels=1, img_nrows=28, img_ncols=28, use_train=True)


        z, x, y = adv_object.load('./adv/MNIST/train', 'szegedy')
    confusion_matrix = np.zeros((nb_class, nb_class))
    prediction = np.argmax(model.predict(z), axis=1)
    for i in range(len(y)):
        confusion_matrix[prediction[i], y[i]] +=1.
                        
        
                      
    dico_class = dict([(i,[]) for i in range(nb_class)])
    for i in range(nb_class):
        confusion_matrix[i] /= sum(confusion_matrix[i])
        label_sort = np.argsort(confusion_matrix[i])[::-1]
        confidence_sort = np.sort(confusion_matrix[i])[::-1]
        cumsum_confidence = np.cumsum(confidence_sort)
        tmp = cumsum_confidence - 0.8*np.ones((nb_class,))
        index = np.argmin((tmp)**2)
        if tmp[index]<0:
            index+=1
        dico_class[i] = np.sort(label_sort[:index+1])

    return dico_class
    
def generate_filenames(dico, option_adv):
    # step 1, check for redundancy
    dico_name = {}
    nb_class = 10
    filename = option_adv+'_committee_'
    index=0
    previous_subset = {}
    for key in dico.keys():
        c = list(dico[key])
        u = range(nb_class)
        for c_i in c:
            u.remove(c_i)
        if str(c) in previous_subset.keys():
            key_prev = previous_subset[str(c)]
            dico_name[key] = dico_name[key_prev]
        else:
            previous_subset[str(c)] = key
            dico_name[key] = [filename+'C_'+str(index), c, filename+'U_'+str(index), u]
            index+=1
        
    return  dico_name

def committee_member(repo, filename, injectiv_table):
    if os.path.exists(os.path.join(repo, filename)):
        return
    n = len(injectiv_table)
    dico_injective = dict([(j,i) for j,i in zip(injectiv_table, range(n))])

    model, dataset = build_model_MNIST(num_classes=n)
    ((x_train, y_train), (x_test, y_test)) = dataset
    permut = np.random.permutation(len(y_train))
    x_train = x_train[permut]
    y_train = y_train[permut]
    x_train_filter=[]
    y_train_filter = []
    for i in range(len(y_train)):
        label_i = np.argmax(y_train[i])
        if label_i in injectiv_table:
            tmp_i = np.zeros((n,))
            tmp_i[dico_injective[label_i]]=1.
            x_train_filter.append(x_train[i])
            y_train_filter.append(tmp_i)
       
    x_train_filter = np.array(x_train_filter)
    y_train_filter = np.array(y_train_filter)
    
    x_test_filter=[]
    y_test_filter = []
    for i in range(len(y_test)):
        label_i = np.argmax(y_test[i])
        if label_i in injectiv_table:
            tmp_i = np.zeros((n,))
            tmp_i[dico_injective[label_i]]=1.
            x_test_filter.append(x_test[i])
            y_test_filter.append(tmp_i)
       
    x_test_filter = np.array(x_test_filter)
    y_test_filter = np.array(y_test_filter)

    batch_size = 32
    epochs = 15
    model.fit(x_train_filter, y_train_filter, batch_size=batch_size, nb_epoch=epochs,
                  verbose=0, validation_data=(x_test_filter, y_test_filter))
    score = model.evaluate(x_test_filter, y_test_filter, verbose=1)
    print('Test error', score[1])
    model.save_weights(os.path.join(repo, filename))


def build_committee(repo, option_adv):
    dico_C_U = build_confusion_matrix(option_adv)
    dico_name = generate_filenames(dico_C_U, option_adv)
    
    for key in dico_name:
        committee_member(repo, dico_name[key][0], dico_name[key][1])
        committee_member(repo, dico_name[key][2], dico_name[key][3])
        
def test_committee(repo, option_adv):
    dico_C_U = build_confusion_matrix(option_adv)
    dico_name = generate_filenames(dico_C_U, option_adv)
    with closing(open('./MNIST_dico_ICLR_'+option_adv, 'rb')) as f:
        dico_name = pkl.load(f)
        #pkl.dump(dico_name, f, protocol=pkl.HIGHEST_PROTOCOL)
    #return
    nb_class = 10
    filename = 'ensemble_MNIST0'
    ensemble =[]
    label_table=[]
    model, dataset = model, dataset = build_model_MNIST()
    model.load_weights(os.path.join(repo, filename))
    ensemble.append(model)
    label_table.append(range(nb_class))
    for key in dico_name.keys():
        f_0, c, f_1, u = dico_name[key]
        net_c, _ = build_model_MNIST(num_classes=len(c))
        net_u, _ = build_model_MNIST(num_classes=len(u))
        net_c.load_weights(os.path.join(repo, f_0))
        net_u.load_weights(os.path.join(repo, f_1))
        label_table.append(c); label_table.append(u)
        ensemble.append(net_c); ensemble.append(net_u)
        
    # step 1 load adv examples
    adv_object = Adversarial_Szegedy(model=model, dataset=dataset,
                                        n_channels=1, img_nrows=28, img_ncols=28, use_train=True)

    #adv_object.save('./adv/MNIST/test', 'goodfellow')
    print('ADVERSARIAL')
    adv_x, _ = adv_object.load('./adv/MNIST/test', 'szegedy')  
    dico = dict([('ADVERSARIAL',[]), ('GROUNDTRUTH', [])])
    
    # step 1
    predict_ensemble = np.array([ predict(net, adv_x, table) for net, table in zip(ensemble, label_table)])
    confidency_ensemble = np.array([ np.max(net.predict(adv_x), axis=1) for net in ensemble ])
    
    for i in range(len(adv_x)):
        occurences = [len(np.where(predict_ensemble[:,i]==j)) for j in range(nb_class)]
        winning=max(occurences)==nb_class+1
        if winning :
            print('WINNING')
            # pick only the classifier with the right label
            label = np.argmax(occurences)
            
            networks_index = np.where(predict_ensemble[:,i]==label)
        else:
            networks_index = range(nb_class+1)
            
        result = np.max(confidency_ensemble[networks_index, i])
        dico['ADVERSARIAL'].append(result)
        print(result,0)

        
    print('GROUNDTRUTH')
    _, (x_test, y_test) = dataset
    n = len(y_test)
    x_true = x_test[np.random.permutation(n)][:len(adv_x)]
    
    # step 1
    predict_ensemble = np.array([ predict(net, x_true, table) for net, table in zip(ensemble, label_table)])
    confidency_ensemble = np.array([ np.max(net.predict(x_true), axis=1) for net in ensemble ])
    
    for i in range(len(x_true)):
        occurences = [len(np.where(predict_ensemble[:,i]==j)) for j in range(nb_class)]
        winning=max(occurences)==nb_class+1
        if winning :
            print('WINNING')
            # pick only the classifier with the right label
            label = np.argmax(occurences)
            networks_index = np.where(predict_ensemble[:,i]==label)
        else:
            networks_index = range(nb_class+1)
            
        result = np.max(confidency_ensemble[networks_index, i])
        dico['GROUNDTRUTH'].append(result)
        print(result,1)
            
        
    #import pdb; pdb.set_trace()
    filename = os.path.join('./adv/MNIST/test', option_adv+'_density_committee')
    with closing(open(filename, 'wb')) as f:
        pkl.dump(dico, f, protocol=pkl.HIGHEST_PROTOCOL)
    

        
    
        
        


if __name__=="__main__":
    repo="./weights/ensemble/CNN/MNIST"
    option_adv = 'goodfellow'
    test_committee(repo, option_adv)
    
