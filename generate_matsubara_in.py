# Import stuff
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

# Output ptions
np.set_printoptions(precision=2)
logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

# Precompute useful stuff and define the necessary evils
SAMPELING_MIN=-1
SAMPELING_MAX= 1
SAMPELING_N=1024
SAMPLING_SPACE=np.linspace(SAMPELING_MIN,SAMPELING_MAX,SAMPELING_N)

def one_gaussian(weight, mean, sigma):
    return np.exp(-(SAMPLING_SPACE-mean)**2/(2*sigma))/np.sqrt(2*np.pi*sigma)
def get_sigma(m):
    sigma_min=0.001
    sigma_min_gradient=0.01
    sigma_max_gradient=2*sigma_min_gradient
    smin=sigma_min+abs(m)*sigma_min_gradient
    smax=sigma_min+abs(m)*sigma_max_gradient
    return np.random.rand()*(smax-smin)+smin

def get_weight(m):
    max_weight=1.
    min_weight=.1
    weight_gradient=-1.1
    tmp=max_weight+abs(m)*weight_gradient
    tmp=max(tmp,min_weight)
    return tmp*np.random.rand()+min_weight

def get_X_myself(a,b,nmin=8,nmax=33):
    while True:
        R=np.random.randint(nmin,nmax)
        R1=np.random.randint(R)
        R2=R-R1
        means=np.hstack((  np.random.randn(R1)*(-a)/4+a/4 ,np.random.randn(R2)*b/4+b/4 ))
        assert(len(means)>=nmin)
        sigmas=[get_sigma(m) for m in means]
        weights=[get_weight(m) for m in means]
        y=np.zeros(len(SAMPLING_SPACE))
        for w,m,s, in zip(weights,means,sigmas):
            y+=one_gaussian(w,m,s)
        y/=sum(y)/SAMPELING_N*(b-a)
        if y[0]>1e-4 or y[-1]>1e-4:
            continue
        break
    return y

def get_X_arsenault(a,b,nmin=8,nmax=33):
    assert(np.isclose(a,-1))
    assert(np.isclose(b, 1))
    while True:
        n_gaussians=np.random.randint(nmin,nmax)
        means=[0]*32
        while len([m for m in means if m>-.2 and m<.2])>3:
            means=(b-a)*np.random.rand(n_gaussians)+a
            means=means/2.
        means=sorted(means)
        sigmas=[]
        sigma_large_max=.01
        sigma_large_min=.001 
        sigma_small_max=.001 
        sigma_small_min=.0001
        for m in means:
            if m>0.2 or m<-.2:
                s=(sigma_large_max-sigma_large_min)*np.random.rand()+sigma_large_min
            else:
                s=(sigma_small_max-sigma_small_min)*np.random.rand()+sigma_small_min
            sigmas.append(s)
        weights=np.random.rand(len(means))
        y=np.zeros(len(SAMPLING_SPACE))
        for w,m,s, in zip(weights,means,sigmas):
            y+=one_gaussian(w,m,s)
        y/=sum(y)/SAMPELING_N*(b-a)
        if y[-1]>1e-4 or y[0]>1e-4: #one of the weird Arsenault checks
            continue
        Amax_low_w=max([(0 if x>.2 or x<-.2 else yy) for x,yy in zip(SAMPLING_SPACE, y)])
        Amax_highw=max([(0 if x<.2 and x>-.2 else yy) for x,yy in zip(SAMPLING_SPACE, y)])
        if Amax_low_w/Amax_highw>1+1e-6: # Arsenaults ratio check
            continue
        ytmp=np.array([yy for x,yy in zip(SAMPLING_SPACE, y) if (x<.2 and x>-.2)])
        second_deriv =(0.4/len(ytmp))*sum(abs( (ytmp[2:]+ytmp[:-2]-2*ytmp[1:-1])/(SAMPLING_SPACE[1]-SAMPLING_SPACE[0])**2))
        if second_deriv >10.9:
            continue
        break
    return y
for i in range(30):
    X=get_X_myself(-1,1,8,33)
    #X=get_X_arsenault(-1,1,8,33)
    plt.plot(SAMPLING_SPACE, X)
    plt.show(block=True)
sys.exit()

