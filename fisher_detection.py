#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:41:29 2017

@author: mducoffe

# test Fisher matrix
"""
from build_model_cifar import build_model, evaluate
from fisher_layer import Fisher
import os
import pickle as pkl
from contextlib import closing
import numpy as np

def vote_entropy(sample, nb_model, nb_class=10):
    #sample shape (nb_model, nb_class)
    occ = np.zeros((nb_class))
    labels = np.argmax(sample, axis=1)
    for label in labels:
        occ[label]+=1.
    for i in range(nb_class):
        occ[i] = occ[i]/nb_model*np.log(occ[i]/nb_model+1)
    return sum(occ)
        

def build_Fisher(repo='./weights', filename='WRN-CIFAR10-16-4-Best.h5', 
                 saving_filename='fisher_WRN_CIFAR10', adv_repo='./adv/Goodfellow', adv_filename='cifar10_v0'):
    model, dataset = build_model(repo, filename)
    
    
    
    fisher = Fisher(model)
    fisher.load('.', saving_filename)
    
    with closing(open(os.path.join(adv_repo, adv_filename), 'rb')) as f:
        advX, advY = pkl.load(f)

    #networks_0 =  [fisher.fisher_sample() for i in range(3)]
    #load networks
    networks_0 = []
    for i in range(3):
        net, _ = build_model('./adv/ensemble/CIFAR', 'ensembleCIFAR10_'+str(i)+'.h5')
        networks_0.append(net)
        
    #networks_0 = [networks_0[1], networks_0[2]]
    
    print('ADVERSARIAL')
    model_prediction = model.predict(advX)
    network_prediction_0 = np.mean(np.array([network.predict(advX) for network in networks_0]), axis=0)
    #network_prediction_1 = np.mean(np.array([network.predict(advX) for network in networks_1]), axis=0)
    #network_prediction_2 = np.mean(np.array([network.predict(advX) for network in networks_2]), axis=0)
    tmp_0 = np.mean((model_prediction - network_prediction_0)**2, axis=1)
    #tmp_1 = np.mean((model_prediction - network_prediction_1)**2, axis=1)
    #tmp_2 = np.mean((model_prediction - network_prediction_2)**2, axis=1)
    for i in range(10):
        print(tmp_0[i])

    print('GROUNDTRUTH')
    _, (testX, testY) = dataset
    testX = testX[np.random.permutation(len(testX))[:len(advX)]]
    model_prediction = model.predict(testX)
    network_prediction_0 = np.mean(np.array([network.predict(testX) for network in networks_0]), axis=0)
    #network_prediction_1 = np.mean(np.array([network.predict(testX) for network in networks_1]), axis=0)
    #network_prediction_2 = np.mean(np.array([network.predict(testX) for network in networks_2]), axis=0)
    tmp_0 = np.mean((model_prediction - network_prediction_0)**2, axis=1)
    #tmp_1 = np.mean((model_prediction - network_prediction_1)**2, axis=1)
    #tmp_2 = np.mean((model_prediction - network_prediction_2)**2, axis=1)
    for i in range(10):
        print(tmp_0[i])
    return
    
    """
    import pdb; pdb.set_trace()
    
    print('ADVERSARIAL')
    for i in range(len(advY)):
        print vote_entropy(predictions[:,i,:], len(networks))
    
    
    
    # real data
    _, (testX, testY) = dataset
    print('GROUNDTRUTH')
    testX = testX[np.random.permutation(len(testX))[:len(advX)]]
    predictions = np.array([network.predict(advX) for network in networks])
    
    for i in range(len(advY)):
        print vote_entropy(predictions[:,i,:], len(networks))
    
    """

def test_adv_robustness(repo='./weights', filename='WRN-CIFAR10-16-4-Best.h5', adv_repo='./adv/Goodfellow', adv_filename='cifar10_v0'):
    
    model, dataset = build_model(repo, filename)
    with closing(open(os.path.join(adv_repo, adv_filename), 'rb')) as f:
        advX, advY = pkl.load(f)

    #networks_0 =  [fisher.fisher_sample() for i in range(3)]
    #load networks
    networks_0 = []
    for i in range(3):
        net, _ = build_model('./adv/ensemble/CIFAR', 'ensembleCIFAR10_'+str(i)+'.h5')
        networks_0.append(net)
        
    #model = networks_0[0]
    networks_0 = [networks_0[0]]
    same_label = 0.
    for adv_im in advX:
        labels = [np.argmax(net.predict(adv_im[None,:,:,:])) for net in networks_0]
        if min(labels)==max(labels):
            same_label+=1.
    print same_label/len(advX)
    print(len(advX))
    
    

if __name__=="__main__":
    #test_adv_robustness()
    build_Fisher()

