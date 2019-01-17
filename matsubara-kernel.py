import argparse
from time import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import sys
import os
from glob import glob

SAMPELING_MIN=-1
SAMPELING_MAX=1
SAMPELING_N=512
SAMPELING_SPACE=np.linspace( SAMPELING_MIN, SAMPELING_MAX, SAMPELING_N)


TAU_MIN = 1e-4
TAU_MAX = 1e1
TAU_N   = 1024
TAU_SPACE=np.linspace(TAU_MIN,TAU_MAX,TAU_N)

def get_one_gaussian(weight, mean, sigma):
    return np.exp(-(SAMPELING_SPACE-mean)**2/(2*sigma))/(np.sqrt(2*np.pi*sigma))*weight

def get_one_XY():
    mean=np.random.rand()*(SAMPELING_MAX/2-SAMPELING_MIN/2)+SAMPELING_MIN/2
    sigma=np.random.rand()*0.1+0.01
    weight=np.random.rand()*2+0.5
    return get_one_gaussian(weight, mean, sigma)

def get_matsubara_kernel_matrix(t):
    return np.array([[-np.exp(-tau*omega)/(1+np.exp(-omega/t))*(SAMPELING_MAX-SAMPELING_MIN)/SAMPELING_N for omega in SAMPELING_SPACE] for tau in TAU_SPACE])
def get_matsubara_kernel_matrix_alternative(t, tau_min=TAU_MIN, tau_max=TAU_MAX, tau_n=TAU_N):
    TAU_SPACE=np.linspace(tau_min,tau_max,tau_n)
    return TAU_SPACE,np.array([[-np.exp(-tau*omega)/(1+np.exp(-omega/t))*(SAMPELING_MAX-SAMPELING_MIN)/SAMPELING_N for omega in SAMPELING_SPACE] for tau in TAU_SPACE])
def get_matsubara_kernel_matrix_alternative2(t, tau_n=TAU_N):
    tau_min=0.
    tau_max=1.
    sampeling_min=-1
    sampeling_max=1
    SAMPELING_SPACE=np.linspace( sampeling_min, sampeling_max, tau_n)
    TAU_SPACE=np.linspace(tau_min,tau_max,tau_n)
    return np.array([[-np.exp(-tau*omega)/(1+np.exp(-omega/t))*(SAMPELING_MAX-SAMPELING_MIN)/SAMPELING_N for omega in SAMPELING_SPACE] for tau in TAU_SPACE])
def apply_matsubara_kernel(input_configuration, t):
    tmp=get_matsubara_kernel_matrix(t)
    return np.dot(tmp,input_configuration)
def apply_matsubara_kernel_alternative(input_configuration, t, tau_min=TAU_MIN, tau_max=TAU_MAX, tau_n=TAU_N):
    space,kernel=get_matsubara_kernel_matrix_alternative(t, tau_min, tau_max, tau_n)
    return space, np.dot(kernel,input_configuration)

X=[]
EVS=[]
for N in [2**i for i in range(4, 9)]:
    plt.figure()
    X.extend([N]*N) 
    k=get_matsubara_kernel_matrix_alternative2(0.001, N)
    EVS.extend(np.linalg.eigvals(k))
    plt.hist(np.linalg.eigvals(k), bins=50)
    print(len(EVS),len(X))
    plt.show(block=True)

#
#N=512
#T=0.001
#fig, (ax1,ax2)=plt.subplots(2,1)
#sigma=0.01
##for mean in [-.5,-.1,0.,.1]:
##    X=get_one_gaussian(1., mean,sigma)
##    plt.sca(ax1)
##    plt.plot(SAMPELING_SPACE, X)
##    plt.xlabel(r'$\omega$')
##    plt.ylabel(r'$A( \omega )$')
##
##    plt.sca(ax2)
##    s,t=apply_matsubara_kernel_alternative(X, T, tau_max=1./T,tau_n=N)
##    plt.semilogy(s, abs(t), label=str(mean))
##plt.legend(loc='best')
##plt.tight_layout()
##plt.show(block=True)
##
