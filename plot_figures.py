#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:28:01 2017

@author: mducoffe

plot function with matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from contextlib import closing
import pickle as pkl
import os


def plot_density(dico=None):
    if dico is None:
        print('No data to plot')
        #return
    
    # find maximum distance  (value ar strictly positive)
    key_0 = dico.keys()[0]
    data_0 = dico[key_0]
    max_val_0 = np.max(data_0)
    density_0 = gaussian_kde(data_0)
    x_0 = np.linspace(0,max_val_0,200)
    density_0.covariance_factor = lambda : .25
    density_0._compute_covariance()
    
    if len(dico.keys())>1:
        key_1 = dico.keys()[1]
        data_1 = dico[key_1]
        max_val_1 = np.max(data_1)
        density_1 = gaussian_kde(data_1)
        x_1 = np.linspace(0,max_val_1,200)
        density_1.covariance_factor = lambda : .25
        density_1._compute_covariance()
        
    if len(dico.keys())>2:
        key_2 = dico.keys()[2]
        data_2 = dico[key_2]
        max_val_2 = np.max(data_2)
        density_2 = gaussian_kde(data_2)
        x_2 = np.linspace(0,max_val_2,200)
        density_2.covariance_factor = lambda : .25
        density_2._compute_covariance()
        
    if len(dico.keys())>3:
        key_3 = dico.keys()[3]
        data_3 = dico[key_3]
        max_val_3 = np.max(data_3)
        density_3 = gaussian_kde(data_3)
        x_3 = np.linspace(0,max_val_3,200)
        density_3.covariance_factor = lambda : .25
        density_3._compute_covariance()
        
    if len(dico.keys())==1:
        plt.plot(x_0,density_0(x_0))
    if len(dico.keys())==2:
        plt.plot(x_0,density_0(x_0), x_1,density_1(x_1))
    if len(dico.keys())==3:
        plt.plot(x_0,density_0(x_0), x_1,density_1(x_1), x_2,density_2(x_2))
    if len(dico.keys())==4:
        plt.plot(x_0,density_0(x_0), x_1,density_1(x_1), x_2,density_2(x_2), x_3,density_3(x_3))
        
    plt.legend(dico.keys())
    plt.show()
    
def load_data(repo='./adv/MNIST/test', filenames=['goodfellow_density', 'szegedy_density']):
    dico_final = {}
    
    with closing(open(os.path.join(repo, 'goodfellow_density'), 'rb')) as f:
        dico = pkl.load(f)
        dico_final['goodfellow'] = dico['ADVERSARIAL']
        dico_final['groundtruth'] = dico['GROUNDTRUTH']
        
    with closing(open(os.path.join(repo, 'szegedy_density'), 'rb')) as f:
        dico = pkl.load(f)
        dico_final['szegedy'] = dico['ADVERSARIAL']

    with closing(open(os.path.join(repo, 'deepfool_density'), 'rb')) as f:
        dico = pkl.load(f)
        dico_final['deepfool'] = dico['ADVERSARIAL']

        
    return dico_final

if __name__=="__main__":
    dico = load_data()
    plot_density(dico)
    
    # add several line with colors and legend

