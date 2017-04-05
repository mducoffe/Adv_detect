#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:42:25 2017

@author: mducoffe

Adversarial example on cifar 10
"""

import numpy as np
from build_model_cifar import build_model as build_cifar10
from build_model_cifar import evaluate
from build_model_cifar100 import build_model as build_cifar100
import keras.backend as K
import scipy
from contextlib import closing
import pickle as pkl
import os
from keras.models import Model

class Adversarial_example(object):
    
    def __init__(self, repo='./weights', filename='WRN-CIFAR10-16-4-Best.h5', 
                                 n_channels=3, img_nrows=32, img_ncols=32, nb_class=10, data_name='cifar10', model=None, dataset=None, use_train=False):
        if model is None:
            if data_name=='cifar10':
                model, dataset = build_cifar10(repo, filename=filename)
            else:
                model, dataset = build_cifar100(repo, filename=filename)
        if K.image_dim_ordering() == 'th':
            img_shape = (1, n_channels, img_nrows, img_ncols)
            adversarial_image = K.placeholder((1, n_channels, img_nrows, img_ncols))
            adversarial_target = K.placeholder((1, nb_class))
            adv_noise = K.placeholder((1, n_channels, img_nrows, img_ncols))
        else:
            img_shape = (1,img_nrows, img_ncols, n_channels)
            adversarial_image = K.placeholder((1, img_nrows, img_ncols, n_channels))
            adversarial_target = K.placeholder((1, nb_class))
            adv_noise = K.placeholder((1, img_nrows, img_ncols, n_channels))
            
        self.model = model
        self.model.trainable=False
        for layer in self.model.layers:
            layer.trainable=False
        self.dataset = dataset
        self.adversarial_image= adversarial_image
        self.adversarial_target = adversarial_target
        self.adv_noise = adv_noise
        self.img_shape = img_shape
        self.nb_class = nb_class
        self.use_train = use_train
        
        self.repo = repo
        self.filename = filename
        
        self.mean = np.mean(self.dataset[0][0])
        self.std = np.std(self.dataset[0][0])
        
        prediction = self.model.call(self.adversarial_image)
        self.predict_ = K.function([K.learning_phase(), self.adversarial_image], K.argmax(prediction, axis=1))

        
    def generate():
        raise NotImplementedError()
        
    def predict(self,image):
        return self.predict_([0, image])
        
    def generate_sample(self, index):
        raise NotImplementedError()
        
    def save(self, repo, filename):
        adv_samples = self.generate()
        with closing(open(os.path.join(repo, filename), 'wb')) as f:
            pkl.dump(adv_samples, f, protocol=pkl.HIGHEST_PROTOCOL)
            
    def load(self, repo, filename):
        data=None
        with closing(open(os.path.join(repo, filename), 'rb')) as f:
            data=pkl.load(f)
        return data
            
class Adversarial_DeepFool(Adversarial_example):
    
    def __init__(self,  **kwargs):
        super(Adversarial_DeepFool, self).__init__(**kwargs)
        
        # the network is evaluated without the softmax
        # you need to retrieve the last layer (Activation('softmax'))
        last_dense = self.model.layers[-2].output
        second_model = Model(self.model.input, last_dense)
        loss_classif = K.mean(second_model.call(self.adversarial_image)[0, K.argmax(self.adversarial_target)])
        grad_adversarial = K.gradients(loss_classif, self.adversarial_image)
        self.f_loss = K.function([K.learning_phase(), self.adversarial_image, self.adversarial_target], loss_classif)
        self.f_grad = K.function([K.learning_phase(), self.adversarial_image, self.adversarial_target], grad_adversarial)
        
        def eval_loss(x,y):
            y_vec = np.zeros((1, self.nb_class))
            y_vec[:,y] +=1
            return self.f_loss([0., x, y_vec])
        
        def eval_grad(x,y):
            y_vec = np.zeros((1, self.nb_class))
            y_vec[:,y] +=1
            return self.f_grad([0., x, y_vec]) 
        
        self.eval_loss = eval_loss
        self.eval_grad = eval_grad
        
    
    def generate(self):
        if self.use_train:
            N = len(self.dataset[0][1])
        else:
            N = len(self.dataset[1][1])
        x_adv = []
        y_adv = []
        true_images = []
        for i in range(N):
            succeed, true_im, adv_im, adv_label = self.generate_sample(i)
            if succeed:
                true_images.append(true_im)
                x_adv.append(adv_im); y_adv.append(adv_label)
                break
        true_images = np.concatenate(true_images, axis=0)
        x_adv = np.concatenate(x_adv, axis=0)
        y_adv = np.array(y_adv)[:,None]
        return (true_images, x_adv, y_adv)
    

    def generate_sample(self, index):
        if self.use_train:
            true_image = self.dataset[0][0][index:index+1]
            true_label_ = np.argmax(self.dataset[0][1][index])
        else:
            true_image = self.dataset[1][0][index:index+1]
            true_label_ = np.argmax(self.dataset[1][1][index])

        true_label = self.predict(true_image)
        
        if true_label_ != true_label_:
            return False, true_image, true_image, true_label
        x_i = np.copy(true_image); i=0
        while self.predict(x_i) == true_label and i<50:
            other_labels = range(self.nb_class)
            other_labels.remove(true_label)
            w_labels=[]; f_labels=[]
            for k in other_labels:
                w_k = (self.eval_grad(x_i,k).flatten() - self.eval_grad(x_i, true_label).flatten())
                f_k = np.abs(self.eval_loss(x_i, k).flatten() - self.eval_loss(x_i, true_label).flatten())
                w_labels.append(w_k); f_labels.append(f_k)
            result = [f_k/(np.linalg.norm(w_k)) for f_k, w_k in zip(f_labels, w_labels)]
            label_adv = np.argmin(result)
            
            r_i = (f_labels[label_adv]/(np.linalg.norm(w_labels[label_adv])**2))*w_labels[label_adv]
            #print(self.predict(x_i), f_labels[label_adv], np.mean(x_i), np.mean(r_i))
            if np.max(np.isnan(r_i))==True:
                return False, true_image, true_image, true_label
            x_i += r_i.reshape(true_image.shape)
            x_i = np.clip(x_i, self.mean - self.std, self.mean+self.std)
            i+=1
        adv_image = x_i
        adv_label = self.predict(adv_image)
        if adv_label == true_label:
            return False, true_image, true_image, true_label
        else:
            return True, true_image, adv_image, adv_label
                
class Adversarial_Goodfellow(Adversarial_example):
    
    def __init__(self,epsilon=0.1,**kwargs):
        super(Adversarial_Goodfellow, self).__init__(**kwargs)
        """
        self.models = []
        for i in range(3):
            net, _ = build_cifar10('./adv/ensemble/CIFAR', 'ensembleCIFAR10_'+str(i)+'.h5')
            self.models.append(net)
        """
        self.epsilon = epsilon
        #prediction = (self.models[0].call(self.adversarial_image) + self.models[1].call(self.adversarial_image) +self.models[2].call(self.adversarial_image) )/3.
        prediction = self.model.call(self.adversarial_image)
        loss_classif = K.mean(K.categorical_crossentropy(prediction, self.adversarial_target))
        grad_adversarial = K.sign(K.gradients(loss_classif, self.adversarial_image))
        #f_loss = K.function([K.learning_phase(), self.adversarial_image, self.adversarial_target], loss_classif)
        self.f_grad = K.function([K.learning_phase(), self.adversarial_image, self.adversarial_target], grad_adversarial)
                
    
    def generate(self):
        if self.use_train:
            N = len(self.dataset[0][1])
        else:
            N = len(self.dataset[1][1])
        true_images = []
        x_adv = []
        y_adv = []
        
        for i in range(N):
            #print(i)
            succeed, true_im, adv_im, adv_label = self.generate_sample(i)
            if succeed:
                    true_images.append(true_im)
                    x_adv.append(adv_im); y_adv.append(adv_label)
                    break
        true_images = np.concatenate(true_images, axis=0)
        x_adv = np.concatenate(x_adv, axis=0)
        y_adv = np.array(y_adv)[:,None]
        return (true_images, x_adv, y_adv)
    
    def generate_sample(self, index):
        if self.use_train:
            true_image = self.dataset[0][0][index:index+1]
            #import pdb; pdb.set_trace()
            true_label = np.argmax(self.dataset[0][1][index])
        else:
            true_image = self.dataset[1][0][index:index+1]
            #import pdb; pdb.set_trace()
            true_label = np.argmax(self.dataset[1][1][index])
        # remove if error
        true_label_pred = self.predict(true_image)
        if true_label_pred != true_label:
            return False, true_image, true_image, true_label
        other_labels = range(self.nb_class)
        other_labels.remove(true_label)
        adv_class = other_labels[np.random.randint(self.nb_class - 1)]
        adv_label = np.array([0.]*(self.nb_class))
        adv_label[adv_class] = 1.
        adv_label = adv_label[None, :]
        image_adv = np.clip(true_image + self.epsilon*self.f_grad([0, true_image, adv_label]), self.mean-self.std, self.mean+self.std)
        prediction = self.predict(image_adv)
        label_pred = prediction
        
        if label_pred == true_label:
            #print('FAIL')
            return False, true_image, true_image, true_label
        else:
            #print('SUCEED')
            return True, true_image, image_adv, label_pred
        
        
        
class Adversarial_Szegedy(Adversarial_example):
    
    def __init__(self,confidency=0.99, **kwargs):
        super(Adversarial_Szegedy, self).__init__(**kwargs)
        self.confidency_threshold = confidency
        loss_classif = K.mean(K.categorical_crossentropy(self.model.call(self.adversarial_image + self.adv_noise), self.adversarial_target)) +\
                        0.001*K.sum(K.abs(self.adv_noise))
        grad_adversarial = K.gradients(loss_classif, self.adv_noise)
        f_loss = K.function([K.learning_phase(), self.adv_noise, self.adversarial_image, self.adversarial_target], loss_classif)
        f_grad = K.function([K.learning_phase(), self.adv_noise, self.adversarial_image, self.adversarial_target], grad_adversarial)
        def eval_loss(adv_label, true_image):
            
            def function(noise):
                r = noise.astype('float32')
                r = r.reshape(self.img_shape)
                x = true_image.astype('float32')
                x = x.reshape(self.img_shape)
                y = np.array([0.]*(self.nb_class))
                # pick random choice
                y[adv_label] = 1.
                y = y[None,:]
                return f_loss([0, r, x, y]).astype('float64')
            return function
        self.eval_loss = eval_loss
        
        def eval_grad(adv_label, true_image):
            
            def function(noise):
                r = noise.astype('float32')
                r = r.reshape(self.img_shape)
                x = true_image.astype('float32')
                x = x.reshape(self.img_shape)
                y = np.array([0.]*(self.nb_class))
                # pick random choice
                y[adv_label] = 1.
                y = y[None,:]
                return f_grad([0, r, x, y]).flatten().astype('float64')
            return function
        
        self.eval_grad = eval_grad
        prediction = self.model.call(self.adversarial_image)
        self.confidency_ = K.function([K.learning_phase(), self.adv_noise, self.adversarial_image], prediction)


    def confidency(self,image):
        return self.confidency_([0, np.zeros_like(image), image])
        
    def generate_sample(self, index):
        if self.use_train:
            true_image = self.dataset[0][0][index:index+1]
            true_label = np.argmax(self.dataset[0][1][index])
            true_label_ = self.predict(true_image)
        else:
            true_image = self.dataset[1][0][index:index+1]
            true_label = np.argmax(self.dataset[1][1][index])
            true_label_ = self.predict(true_image)
        if true_label_!= true_label:
            return False, true_image, true_image, true_label
        
        #noise = np.zeros_like(true_image)
        
        # pick label with the second prediction
        """
        other_labels = range(self.nb_class)
        other_labels.remove(true_label)
        adv_label = other_labels[np.random.randint(self.nb_class - 1)]
        """
        adv_label = np.argsort(self.confidency(true_image))[0,-2]
        
        image_adv = np.copy(true_image.flatten())
        #eval_loss = self.eval_loss(adv_label, image_adv)
        #eval_grad = self.eval_grad(adv_label, image_adv)
        
        for i in range(100):
            print(i)
            eval_loss = self.eval_loss(adv_label, image_adv)
            eval_grad = self.eval_grad(adv_label, image_adv)
            noise = np.zeros_like(true_image).flatten()
            results = scipy.optimize.fmin_l_bfgs_b(eval_loss, noise,
                                                   fprime=eval_grad, 
                                                   maxfun=100)
            noise = results[0].astype('float32')
            image_adv = results[0].astype('float32') + true_image.flatten()     
            result_adv = np.clip(image_adv.reshape(self.img_shape), self.mean - self.std, self.mean+self.std)
            label_pred = self.predict(result_adv)
            
            label_confidency = np.max(self.confidency(result_adv))
            if label_confidency >= self.confidency_threshold and label_pred != true_label:
                #print((label_confidency, label_pred, true_label))
                return True, true_image, result_adv, label_pred
            
        #print('FAIL')
        return False, true_image, true_image, true_label

    def generate(self):
        if self.use_train:
            N = len(self.dataset[0][1])
        else:
            N = len(self.dataset[1][1])
        x_adv = []
        y_adv = []
        true_images = []
        for i in range(N):
            #print(i)
            succeed, true_im, adv_im, adv_label = self.generate_sample(i)
            if succeed:
                x_adv.append(adv_im); y_adv.append(adv_label)
                true_images.append(true_im)
                break
        true_images = np.concatenate(true_images, axis=0)
        x_adv = np.concatenate(x_adv, axis=0)
        y_adv = np.array(y_adv)[:,None]
        return (true_images, x_adv, y_adv)
        
    


