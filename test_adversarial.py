#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:45:47 2017

@author: test adversarial generation
"""
from adversarial import Adversarial_Szegedy, Adversarial_Goodfellow, Adversarial_DeepFool

def test_Adversarial(option):
    if option=='Adversarial_Szegedy':
        adv_object = Adversarial_Szegedy()
        #adv_object.generate()
        adv_object.save('./adv/Szegedy', 'cifar10_v0')
    if option=='Adversarial_Goodfellow':
        adv_object = Adversarial_Goodfellow()
        #adv_object.generate()
        adv_object.save('./adv/Goodfellow', 'cifar10_v0')
        
    if option=='Adversarial_DeepFool':
        adv_object = Adversarial_DeepFool()
        adv_object.generate()
        
        
if __name__=="__main__":
    option = 'Adversarial_Goodfellow'
    test_Adversarial(option)

