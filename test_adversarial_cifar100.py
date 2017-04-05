#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 11:22:22 2017

@author: mducoffe

ADVERSARIAL examples CIFAR100
"""
import os

from build_model_cifar100 import build_model, evaluate
from fisher_layer import Fisher
from adversarial import Adversarial_Szegedy, Adversarial_Goodfellow

# STEP 1 : pick the best model
def test_model_CIFAR100(repo, filename):
    print(filename)
    model, dataset = build_model(repo, filename)
    evaluate(model, dataset)
    
def test_models_CIFAR100(repo):
    filenames = os.listdir(repo)
    for filename in filenames:
        test_model_CIFAR100(repo, filename)

def build_fisher(repo, filename, repo_saving, filename_saving):
    model, dataset = build_model(repo=repo, filename=filename)
    (trainX, trainY), (testX, testY) = dataset
    fisher = Fisher(model)
    dico = fisher.fisher_information(testX, testY)
    print('suceed')
    dico = fisher.fisher_information(trainX, trainY)
    fisher.save(repo_saving, filename_saving)
    print('test WRN ok !')    


def test_Adversarial(option, repo, filename):
    if option=='Adversarial_Szegedy':
        adv_object = Adversarial_Szegedy(repo=repo, filename=filename, nb_class=100, data_name='cifar100')
        #adv_object.generate()
        adv_object.save('./adv/Szegedy', 'cifar100_v0')
    if option=='Adversarial_Goodfellow':
        adv_object = Adversarial_Goodfellow(repo=repo, filename=filename, nb_class=100, data_name='cifar100')
        adv_object.generate() 
        adv_object.save('./adv/Goodfellow', 'cifar100_v0')

    
if __name__=="__main__":
    option = 'Adversarial_Szegedy'
    model_name = 'WRN-CIFAR100-8-16-4-Best'
    repo='./weights/CIFAR_100'
    test_Adversarial(option, repo, model_name)
    
    
    'fisher_WRN_CIFAR100_test'
    
    
