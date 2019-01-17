import numpy as np

def one_gaussian(mean, sigma):
    return np.exp(-(SAMPLING_SPACE-mean)**2/(2*sigma))/np.sqrt(2*np.pi*sigma)
def one_random_gaussian(a,b):
    mean=np.random.rand()*(b-a)+a
    sigma=maxwidth*np.random.rand()*(b-a)
    return one_gaussian(mean,sigma)
def n_random_gaussians(a,b,n):
    y=np.zeros(len(SAMPLING_SPACE))
    for i in range(n):
        y+=one_random_gaussian(a,b)
    return y/n
def random_gaussians(a,b):
    n=np.random.randint(max_num_gaussians)+1
    return n_random_gaussians(a,b,n)
def get_heat_kernel(t):
    return np.exp(-SAMPLING_SPACE**2/(4*t))/np.sqrt(4*np.pi*t)
def apply_heat_kernel(input_configuration, t):
    return np.convolve(input_configuration,get_heat_kernel(t), 'same')*(SAMPELING_MAX-SAMPELING_MIN)/SAMPELING_N

#return X,y each a numpy array of the right dimension, without further preprocessing (as this is done in get_XY)
def get_one_XY(MAX_TIME_TO_EVOLVE, MIN_TIME_TO_EVOLVE, GAUSSIAN_MIN, GAUSSIAN_MAX, MAX_NUM_GAUSSIANS):
    TIME_TO_EVOLVE=np.random.rand()*(MAX_TIME_TO_EVOLVE-MIN_TIME_TO_EVOLVE)+MIN_TIME_TO_EVOLVE
    X=n_random_gaussians(GAUSSIAN_MIN,GAUSSIAN_MAX,MAX_NUM_GAUSSIANS)
    y=apply_heat_kernel(X,TIME_TO_EVOLVE)
    return TIME_TO_EVOLVE,X,y
def get_XY(SAMPLE_SIZE, MAX_TIME_TO_EVOLVE, MIN_TIME_TO_EVOLVE, GAUSSIAN_MIN, GAUSSIAN_MAX, MAX_NUM_GAUSSIANS ,LEARN_INVERSE=False):
    X=[]
    y=[]
    t=[]
    for i in range(SAMPLE_SIZE):
        tmpt,tmpx,tmpy=get_one_XY(MAX_TIME_TO_EVOLVE, MIN_TIME_TO_EVOLVE, GAUSSIAN_MIN, GAUSSIAN_MAX, MAX_NUM_GAUSSIANS)
        t.append(tmpt)
        X.append(tmpx)
        y.append(tmpy)
    t=np.array(t)
    X=np.array(X)
    y=np.array(y)
    if LEARN_INVERSE:
        return t,y,X
    return t,X,y
