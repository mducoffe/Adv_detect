#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:24:21 2017

@author: mducoffe
Markov Chain for sampling on a kronecker block multivariate gaussian
"""
import numpy as np
import re
from sampling import Sampling, product_kronecker

def approx_kro(psi, tau, alpha):
    n_psi = psi.shape[0]
    n_tau = tau.shape[0]
    id_psi = np.diag([1.]*n_psi)
    id_tau = np.diag([1.]*n_tau)
    coeff = np.sqrt(np.abs(alpha))
    if alpha >=0:
        return (psi + coeff*id_psi), (tau + coeff*id_tau)
    if alpha<0:
        return (psi - coeff*id_psi), (tau + coeff*id_tau)
    
def build_C_kronecker(psi, tau):
    n_psi = psi.shape[0]
    n_tau = tau.shape[0]
    id_psi = np.diag([1.]*n_psi)
    id_tau = np.diag([1.]*n_tau)
    return (id_psi, id_tau)


def positive_definite(M):
    U, C, V = np.linalg.svd(M)
    assert np.min(C)>0, 'the matrix is not positive definite, change the value of epislon'

def covariance_injected_noise(epsilon, psi, tau):
    alpha = 4./epsilon
    A = approx_kro(psi, tau, alpha)
    # check if A is positive definite
    # svd decomposition
    positive_definite(A[0])
    positive_definite(A[1])
    coeff = epsilon**2/4. 
    return coeff*A[0], A[1]

def inverse(M):
    U, C, V = np.linalg.svd(M)
    V_1 = np.linalg.inv(V)
    U_1 = np.linalg.inv(U)
    C_1 = np.diag([1./c for c in C])
    return np.dot(np.dot(V_1, C_1), U_1)

def modifying_covariance(sigma):
    # normalize sigma ?
    # sigma is covariance function
    U, C, V = np.linalg.svd(sigma)
    C = np.sqrt(C)
    C_1 = []
    for c in C:
        C_1.append(np.sqrt(c))

    C_1_vect = np.array(C_1)
    C_1 =np.diag(C_1_vect)
    A = np.dot(U, C_1)
    return A

class Sampling_MC(Sampling):

    def __init__(self, mean, var, epsilon=1.e-10):
        self.mean = mean # theata0
        self.theta = self.mean
        self.var = var 
        self.epsilon = epsilon
        self.kronecker_var = {} # fisher
        for key in self.mean.keys():
            if re.match('conv_(.*)_bias', key):
                psi = np.array([1])[:,None]
                tau = self.var[key]
            elif re.match('batchnormalization_(.*)', key):
                psi = np.array([1])[:,None]
                tau = self.var[key]
            elif re.match('dense_(.*)', key):
                psi = self.var[key+'_input']
                tau = self.var[key+'_output']
            elif re.match('conv_(.*)', key):
                psi = self.var[key+'_input']
                tau = self.var[key+'_output']
            self.kronecker_var[key] = [psi, tau]
        
        self.C = {}
        for key in self.kronecker_var:
            psi, tau = self.kronecker_var[key]
            psi_C = np.diag([1.]*psi.shape[0])
            tau_C = np.diag([1.]*tau.shape[0])
            self.C[key] = [psi_C, tau_C]

        self.injected_noise = dict([(key, covariance_injected_noise(epsilon,
                                                                    self.kronecker_var[key][0],
                                                                    self.kronecker_var[key][1]))
                                    for key in self.kronecker_var.keys()])
    
       
        # sample from the injected noise
        self.sampling_injected_noise = {}
        for key in self.injected_noise.keys():
            psi, tau = self.injected_noise[key]
            if psi.shape[0]==1:
                self.sampling_injected_noise[key] = [psi, modifying_covariance(tau)]
            else:
                self.sampling_injected_noise[key] = [modifying_covariance(psi), modifying_covariance(tau)]
            

    def reinit():
        self.theta = self.mean
        

    def sample(self):

        def noise_priv(key):
            mean = self.mean[key]
            psi_noise, tau_noise = self.sampling_injected_noise[key]
            x = np.random.randn(mean.shape[0])
            return product_kronecker(psi_noise, tau_noise, x)
        
        def second_term(key):
            x = self.theta[key] - self.mean[key]
            psi, tau = self.kronecker_var[key]
            return self.epsilon/2.*product_kronecker(psi, tau, x)
        
        theta= dict([(key, self.theta[key] + second_term(key) + noise_priv(key))
                for key in self.mean.keys()])
        self.theta = theta
        return self.theta


    
    
    
    

