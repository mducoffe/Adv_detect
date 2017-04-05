#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:31:52 2017

@author: ubuntu
"""

import re
import numpy as np
COEFF_NORMALIZATION = 225.

def modifying_covariance(sigma):
    # normalize sigma ?
    # sigma is covariance function
    sigma += 1e-2*np.diag([1.]*sigma.shape[0])
    U, C, V = np.linalg.svd(sigma)
    U_1 = np.transpose(np.linalg.inv(U))
    C = np.sqrt(C)
    C_1 = []
    for c in C:
        if c==0:
            print('singular value !')
            c=1e-5
        """
        if c<5*1e-3:
            c = 5*1e-3
        """
        C_1.append(np.sqrt(1./c))

    C_1_vect = np.array(C_1)
    C_1 =np.diag(C_1_vect)
    #import pdb; pdb.set_trace()

    #n = sigma.shape[0]
    #return np.random.uniform(high=1./n,size=(n,n))
    
    # erreur inverse ?
    A = np.dot(U_1, C_1)
    #import pdb; pdb.set_trace()
    #print((np.mean(A), np.std(A)))
    return A/COEFF_NORMALIZATION

def reconstruct(sigma):
    inv_fisher = np.dot(sigma, sigma.T) + 1e-8*np.diag([1]*sigma.shape[0])
    return np.linalg.inv(inv_fisher)

def product_kronecker(A, B, X):
    # A and B are square matrices
    # X is a vector
    n = A.shape[0]
    m = B.shape[0]
    X = X.reshape((n, m)).T
    Y = np.dot(B, X) # shape (m, n)
    Z = np.dot(Y, A.T) # shape(m, n)
    return Z.T.reshape((m*n,))

class Sampling(object):
    
    def __init__(self, mean, var):
        self.mean = mean
        
        
        #self.var = {}

        self.var = dict([(key, modifying_covariance(var[key])) for key in var.keys()])
        #self.var = var
        #self.fisher = dict([(key, reconstruct(var[key])) for key in var.keys()])
    
    def prob(self, weights):
        def prob_priv(key):
            
            mean = self.mean[key]
            return mean + np.random.uniform(-1e-5, 1e-5, size=mean.shape)
            if re.match('conv_(.*)_bias', key):
                mean = self.mean[key]
                var = [self.fisher[key]]
            elif re.match('batchnormalization_(.*)', key):
                mean = self.mean[key]
                var = [self.fisher[key]]
            elif re.match('dense_(.*)', key):
                mean = self.mean[key]
                var = [self.fisher[key+'_input'], self.fisher[key+'_output']]
            elif re.match('conv_(.*)', key):
                mean = self.mean[key]
                var = [self.fisher[key+'_input'], self.fisher[key+'_output']]
            x = weights[key].flatten() - mean
            if len(var)==1:
                result= np.dot(x, np.dot(var[0], x))
            else:
                A = var[0]; B=var[1]
                
                result= np.dot(x, product_kronecker(A, B, x))
            return result
        probabilities=[prob_priv(key) for key in self.mean.keys()]
        return -0.5*sum(probabilities)
     
    def sample(self):

        def sample_priv(key):
            print('prob '+key)
            if re.match('conv_(.*)_bias', key):
                mean = self.mean[key]
                var = [self.var[key]]
            elif re.match('batchnormalization_(.*)', key):
                mean = self.mean[key]
                var = [self.var[key]]
            elif re.match('dense_(.*)', key):
                mean = self.mean[key]
                var = [self.var[key+'_input'], self.var[key+'_output']]
            elif re.match('conv_(.*)', key):
                mean = self.mean[key]
                var = [self.var[key+'_input'], self.var[key+'_output']]
            
            x = np.random.randn(mean.shape[0])
            if len(var)==1:
                weight= np.dot(var[0], x) + mean
                value = 1./(var[0].shape[0])
            else:
                # kronecker product
                A = var[0]; B=var[1]
                #print((key, np.std(A)*np.std(B)))
                weight= product_kronecker(A, B, x) + mean
            #weight = np.random.uniform(high=value, size=mean.shape)
            #print((value, np.mean(weight), np.std(weight)))
            return weight          

        # sample just one layer
        """
        nb_layer = len(self.mean.keys())
        indices = np.random.permutation(nb_layer)
        sampled_layers = [self.mean.keys()[indices[i]] for i in range(nb_layer)]
        #print sampled_layers
        dico_sample = self.mean
        for key in sampled_layers:
            dico_sample[key] = sample_priv(key)
        """
        dico_sample=dict([(key, sample_priv(key)) for key in self.mean.keys()])
        return dico_sample