#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:42:25 2017

@author: mducoffe

Adversarial example on cifar 10
"""
import numpy as np
from build_model_cifar import build_model
import keras.backend as K
import scipy
from contextlib import closing
import pickle as pkl
import os

class Adversarial_example(object):
    
    def __init__(self, repo='./weights', filename='WRN-CIFAR10-16-4-Best.h5', 
                                 n_channels=3, img_nrows=32, img_ncols=32, nb_class=10):
        model, dataset = build_model(repo, filename=filename)
        if K.image_dim_ordering() == 'th':
            img_shape = (1, n_channels, img_nrows, img_ncols)
            adversarial_image = K.placeholder((1, n_channels, img_nrows, img_ncols))
            adversarial_target = K.placeholder((1, nb_class))
        else:
            img_shape = (1,img_nrows, img_ncols, n_channels)
            adversarial_image = K.placeholder((1, img_nrows, img_ncols, 3))
            adversarial_target = K.placeholder((1, nb_class))
            
        self.model = model
        self.dataset = dataset
        self.adversarial_image= adversarial_image
        self.adversarial_target = adversarial_target
        self.img_shape = img_shape
        self.nb_class = nb_class


    def generate():
        raise NotImplementedError()
        
    def save(self, repo, filename):
        adv_samples = self.generate()
        with closing(open(os.path.join(repo, filename), 'wb')) as f:
            pkl.dump(adv_samples, f, protocol=pkl.HIGHEST_PROTOCOL)
            
class Adversarial_Goodfellow(Adversarial_example):
    
    def __init__(self,epsilon=3., **kwargs):
        super(Adversarial_Goodfellow, self).__init__(**kwargs)
        self.epsilon = epsilon
        loss_classif = K.mean(K.categorical_crossentropy(self.model.call(self.adversarial_image), self.adversarial_target))
        grad_adversarial = K.sign(K.gradients(loss_classif, self.adversarial_image))
        #f_loss = K.function([K.learning_phase(), self.adversarial_image, self.adversarial_target], loss_classif)
        self.f_grad = K.function([K.learning_phase(), self.adversarial_image, self.adversarial_target], grad_adversarial)
        
    def generate(self):
        N = len(self.dataset[1][1])
        x_adv = []
        y_adv = []
        
        for i in range(N):
            print(i)
            succeed, adv_im, adv_label = self.generate_sample(i)
            if succeed:
                x_adv.append(adv_im); y_adv.append(adv_label)
        x_adv = np.concatenate(x_adv, axis=0)
        y_adv = np.array(y_adv)[:,None]
        return (x_adv, y_adv)
    
    def generate_sample(self, index):
        true_image = self.dataset[1][0][index:index+1]
        true_label = self.dataset[1][1][index]
        other_labels = range(self.nb_class)
        other_labels.remove(true_label)
        adv_class = other_labels[np.random.randint(self.nb_class - 1)]
        adv_label = np.array([0.]*(self.nb_class))
        adv_label[adv_class] = 1.
        adv_label = adv_label[None, :]
        image_adv = true_image + self.epsilon*self.f_grad([1, true_image, adv_label])
        prediction = self.model.predict(image_adv)
        label_pred = np.argmax(prediction)
        
        if label_pred == true_label:
            print('SUCCEED')
            return False, true_image, true_label
        else:
            print('FAIL')
            return True, image_adv, adv_label
        
class Adversarial_Szegedy(Adversarial_example):
    
    def __init__(self,confidency=0.8, **kwargs):
        super(Adversarial_Szegedy, self).__init__(**kwargs)
        loss_classif = K.mean(K.categorical_crossentropy(self.model.call(self.adversarial_image), self.adversarial_target))
        grad_adversarial = K.gradients(loss_classif, self.adversarial_image)
        f_loss = K.function([K.learning_phase(), self.adversarial_image, self.adversarial_target], loss_classif)
        f_grad = K.function([K.learning_phase(), self.adversarial_image, self.adversarial_target], grad_adversarial)
        def eval_loss(adv_label):
            
            def function(true_image):
                x = true_image.astype('float32')
                x = x.reshape(self.img_shape)
                y = np.array([0.]*(self.nb_class))
                # pick random choice
                y[adv_label] = 1.
                y = y[None,:]
                return f_loss([1, x, y]).astype('float64')
            return function
        self.eval_loss = eval_loss
        
        def eval_grad(adv_label):
            
            def function(true_image):
                x = true_image.astype('float32')
                x = x.reshape(self.img_shape)
                y = np.array([0.]*(self.nb_class))
                # pick random choice
                y[adv_label] = 1.
                y = y[None,:]
                return f_grad([1, x, y]).flatten().astype('float64')
            return function
        
        self.eval_grad = eval_grad
        self.confidency = confidency
        
    def generate_sample(self, index):
        true_image = self.dataset[1][0][index:index+1]
        true_label = self.dataset[1][1][index]
        other_labels = range(self.nb_class)
        other_labels.remove(true_label)
        adv_label = other_labels[np.random.randint(self.nb_class - 1)]
        eval_loss = self.eval_loss(adv_label)
        eval_grad = self.eval_grad(adv_label)
        image_adv = true_image.flatten()
        
        for i in range(10):
            results = scipy.optimize.fmin_l_bfgs_b(eval_loss, image_adv, fprime=eval_grad, maxfun=20)
            image_adv = results[0].astype('float32')        
            result_adv = image_adv.reshape(self.img_shape)
            prediction = self.model.predict(result_adv)
            label_pred = np.argmax(prediction)
            label_confidency = np.max(prediction)
            
            if label_confidency >= self.confidency and label_pred != true_label:
                #print((label_confidency, label_pred, true_label))
                #print('SUCCEED')
                return True, result_adv, label_pred
            
        #print('FAIL')
        return False, true_image, true_label

    def generate(self):
        N = len(self.dataset[1][1])
        x_adv = []
        y_adv = []
        
        for i in range(N):
            print(i)
            succeed, adv_im, adv_label = self.generate_sample(i)
            if succeed:
                x_adv.append(adv_im); y_adv.append(adv_label)
        x_adv = np.concatenate(x_adv, axis=0)
        y_adv = np.array(y_adv)[:,None]
        return (x_adv, y_adv)
        
    


