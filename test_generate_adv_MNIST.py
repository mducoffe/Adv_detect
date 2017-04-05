#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 20:48:45 2017

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
from adversarial import Adversarial_Goodfellow, Adversarial_DeepFool, Adversarial_Szegedy
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


        
def test_ensemble_robustness_adv():

    repo="./weights/ensemble/CNN/MNIST"
    filenames = os.listdir(repo)
    model, dataset = build_model_MNIST()
    (X_train, Y_train), (X_test, Y_test) = dataset

    model.load_weights(os.path.join(repo, filenames[0]))

    
    
    adv_object = Adversarial_Goodfellow(epsilon=0.03, model=model, dataset=dataset,
                                        n_channels=1, img_nrows=28, img_ncols=28, use_train=True)

    #adv_object.save('./adv/MNIST/train', 'goodfellow')
    #z, x, y = adv_object.load('./adv/MNIST/train', 'goodfellow')
    z,x, y = adv_object.generate()
    
    im_00 = z[0].reshape((28,28))
    im_01 = x[0].reshape((28,28))
    
    
    adv_object =  Adversarial_Szegedy( 0.5, model=model, dataset=dataset,
                                        n_channels=1, img_nrows=28, img_ncols=28, use_train=True)

    #adv_object.save('./adv/MNIST/train', 'szegedy')

    #z, x,y = adv_object.load('./adv/MNIST/train', 'szegedy')
    z,x,y = adv_object.generate()
    im_10 = z[0].reshape((28,28))
    im_11 = x[0].reshape((28,28))
    
    """
    adv_object =  Adversarial_DeepFool(model=model, dataset=dataset,
                                        n_channels=1, img_nrows=28, img_ncols=28, use_train=True)
    adv_object.generate()
    #adv_object.save('./adv/MNIST/test', 'deepfool')
    
    #z, x, y = adv_object.load('./adv/MNIST/test', 'deepfool')
    print('deepfool')
    #print(len(x))
    #import pdb; pdb.set_trace()
    """
    
    import matplotlib.pyplot as plt
    
    plt.subplot(2,2,1)
    plt.imshow(im_00)
    plt.subplot(2,2,2)
    plt.imshow(im_01)
    plt.subplot(2,2,3)
    plt.imshow(im_10)
    plt.subplot(2,2,4)
    plt.imshow(im_11)
    
    conf_0 = np.max(model.predict(im_10.reshape((1,28,28,1))))
    conf_1 = np.max(model.predict(im_11.reshape((1,28,28,1))))
    
    print(conf_0, conf_1)



    



        

if __name__=="__main__":

    test_ensemble_robustness_adv()
    #train_ensemble()






