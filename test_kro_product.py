#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:16:12 2017

@author: mducoffe

generate a random symmetric positive matrix for the Markov chain
and approximate the injected noise by a kronecker product
"""
import numpy as np

# generate a dense symmetric positive definite matrix
def generateSPDMatrix(n):
    
    A = np.random.randn(n,n)
    A = 0.5*(A + np.transpose(A, (1,0)))
    A = A + np.diag([n]*n)
    return A

def generate_C(m, n):
    return [generateSPDMatrix(m), generateSPDMatrix(n)]

def rearangement(epsilon, C, F, m_1, n_1):
    raise NotImplementedError()
    return np.zeros_like((m_1, n_1))

def checkInvertibility(epsilon, C, F):
    psi = np.dot(np.dot(C[0], F[0]), C[0])
    _, eigenvalues_psi, _ = np.linalg.svd(psi)
    
    tau = np.dot(np.dot(C[1], F[1]), C[1])
    _, eigenvalues_tau, _ = np.linalg.svd(tau)
    
    eigenvalues = []
    for lambda_psi in eigenvalues_psi:
        for lambda_tau in eigenvalues_tau:
            eigenvalues.append(lambda_psi, lambda_tau)
    check_value = 4./epsilon
    if check_value in eigenvalues:
        epsilon = 4./(max(eigenvalues) + 1.)
        print(('new value for epislon', epsilon))
    # change the value of epsilon if 4/epsilon matches an eigenvalue
    return epsilon

def priv_injected_noise(epsilon, C, F):
    m = F[0].shape[0]
    n = F[1].shape[0]
    A = rearangement(epsilon, C, F, m, n)
    U, sigma, V = np.linalg.svd(A)
    sigma_1 = np.max(sigma)
    B = sigma_1*np.zeros_like(C[0])
    D = np.zeros_like(C[1])
    
    return [epsilon*np.dot(C[0], B), epsilon*np.dot(C[1], D)]

def injected_noise(epsilon, F):
    # unique epislon for everyone
    C = {}
    for key in F.keys():
        fisher = F[key]
        m = fisher[0].shape[0]
        n = fisher[1].shape[0]
        C_key =generate_C(m,n)
        C[key] = C_key
    epsilons = [checkInvertibility(epsilon, C[key], F[key]) for key in F.keys()]
    epsilon = min(epsilons)
    injected_noise = {}
    for key in F.keys():
        noise_key = priv_injected_noise(epsilon, C[key], F[key])
        injected_noise[key] = noise_key
                      
    return epsilon, C, injected_noise


if __name__=="__main__":
    print('kikou')
    C = generateSPDMatrix(10)
    U, sigma, V = np.linalg.svd(C)
    Sigma = np.diag(sigma)
    print np.mean( Sigma - np.dot(np.dot(V, C), U))

