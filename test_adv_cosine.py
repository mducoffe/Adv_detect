#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:52:31 2017

@author: ubuntu
"""

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
from adversarial import Adversarial_Goodfellow
import matplotlib.pyplot as plt
import argparse
from fisher_layer import Fisher
import keras.backend as K
from scipy import spatial

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


def grad_func(model, input_shape, nb_class=10):

    x = K.placeholder(shape=(1,input_shape[0], input_shape[1], input_shape[2]))
    prediction = model.call(x)
    target = K.zeros((1, nb_class))
    label= K.argmax(prediction, axis=1)[0]
    target = K.theano.tensor.inc_subtensor(target[0,label], 1)
    loss_classif = K.mean(K.categorical_crossentropy(prediction, target))
    grad = K.gradients(loss_classif, [x])
    function = K.function([K.learning_phase(), x], grad[0].flatten())
    def func(x):
        return function([1,x])
    return func

def cosine_similarity(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)

def test_ensemble_robustness_adv(eps=1.):
    
    ensemble=[]
    repo="./weights/ensemble/CNN/MNIST"
    filenames = os.listdir(repo)
    model, dataset = build_model_MNIST()
    (X_train, Y_train), (X_test, Y_test) = dataset
    input_shape = X_train[0].shape
    model.load_weights(os.path.join(repo, filenames[0]))
    
    f_model = grad_func(model, input_shape, nb_class=10)
    # build Fisher
    fisher = Fisher(model)
    fisher.load('./weights/ensemble/CNN', 'fisher_mnist')
    print('fisher downloaded')
    filenames = filenames[1:]

    for filename in filenames:
        net, dataset = build_model_MNIST()
        #net = fisher.fisher_sample()
        ensemble.append(net)
        #return
        #score = net.evaluate(X_test, Y_test, verbose=0)
        #print(score[1])
        #return
        #net.load_weights(os.path.join(repo, filename))
        ensemble.append(net)
        
    f_ensemble = [grad_func(net, input_shape, nb_class=10) for net in ensemble]

    adv_object = Adversarial_Goodfellow(epsilon=eps, model=model, dataset=dataset,
                                        n_channels=1, img_nrows=28, img_ncols=28)

    x_adv, y_adv = adv_object.generate()
    
    
    # generate gradient function

    """
    for i in range(len(y_adv)):
        if not(y_adv[i][0]==np.argmax(model.predict(x_adv[i:i+1]))):
            print('PROBLEM')
    return
    """

    #adv_object.save('./adv', 'mnist_v0')
    #x_adv, y_adv = adv_object.load('./adv', 'mnist_v0')
    N = len(x_adv)
    prediction = model.predict(x_adv)
    predict_ensemble = np.mean([net.predict(x_adv) for net in ensemble], axis=0)
    distance = prediction - predict_ensemble
    grad_model = [f_model(x_adv[i:i+1]) for i in range(N)]
    grad_ensemble = [ [f_net(x_adv[i:i+1]) for f_net in f_ensemble] for i in range(N)]
    print('ADVERSARIAL')
    for i in range(N):
        print(np.linalg.norm(distance[i]), np.mean([ cosine_similarity(grad_model[i], grad_ensemble[i][j]) for j in range(len(ensemble))]))
        
    _, (x_test, y_test) = dataset
    n = len(y_test)
    x_true = x_test[np.random.permutation(n)][:len(x_adv)]
    prediction = model.predict(x_true)
    predict_ensemble = np.mean([net.predict(x_true) for net in ensemble[1:]], axis=0)
    
    distance = prediction - predict_ensemble
    grad_model = [f_model(x_true[i:i+1]) for i in range(N)]
    grad_ensemble = [ [f_net(x_true[i:i+1]) for f_net in f_ensemble] for i in range(N)]
    print('GROUNDTRUTH')
    for i in range(N):
        print(np.linalg.norm(distance[i]), np.mean([ cosine_similarity(grad_model[i], grad_ensemble[i][j]) for j in range(len(ensemble))]))
    
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


        

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='MNIST CNN adversarial examples')

    parser.add_argument('--epsilon', type=float, default=0.3, help='Noise')
    
    args = parser.parse_args()
    
    eps = args.epsilon

    test_ensemble_robustness_adv(eps)
    #train_ensemble()






