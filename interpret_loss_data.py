import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import tensorboard.backend.event_processing.event_accumulator as ea
from glob import glob
from scipy.optimize import curve_fit
import re

MODEL_DIR='/models/'
EVAL_DIR ='/eval/'


# Output ptions
tf.logging.set_verbosity(tf.logging.WARN)
np.set_printoptions(precision=2)
logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
logging.info('Tensorflow %s' % tf.__version__) # 1.4.1

# Construct parser
parser = argparse.ArgumentParser(description="A python script facilitating the generation as well as the training on the (inverse) heat kernel problem")
parser.add_argument('--root-directory'             , type=str             , required=True   , help ='Where all the data is getting stored')
OPTIONS=vars(parser.parse_args())

# Print options
for k in OPTIONS:
    logging.info('\tOption {:<30}: {:>10}'.format(k, OPTIONS[k]))

# Set options
DATA_ROOT_PATH             =     OPTIONS ['root_directory']
def interpret_layers(s):
    tmp=[]
    while s.find('d')!=-1 or s.find('c')!=-1:
        t=s[0]
        next_d=s.find('d', 1)
        next_c=s.find('c', 1)
        if next_d==-1 and next_c==-1:
            tmp.append([t,int(s[1:])])
            s=''
            continue
        if next_d==-1 or next_c==-1:
            tmp.append([t,int(s[1:max(next_d, next_c)])])
            s= s[max(next_d, next_c):]
            continue
        else:
            tmp.append([t,int(s[1:min(next_d, next_c)])])
            s= s[min(next_d, next_c):]
            continue
    return tmp

def extract_model_information(filename):
    dic={}
    m=re.search('models/(.*)/eval',filename).groups()[0]
    m=re.search('(.*)_(.*)_(.*)_K(\d)_hu(.*)',m) 
    if m:
        dic['update fn']              = m.groups()[0]
        dic['activation fn']          = m.groups()[1]
        dic['activate conv']          = m.groups()[2]
        dic['kernel size']            = int(m.groups()[3])
        dic['hidden units']           = interpret_layers(m.groups()[4])
    else:
        print('Something went wrong')
    return dic

cmap=plt.matplotlib.cm.get_cmap('viridis')

model_names=glob(DATA_ROOT_PATH+MODEL_DIR+'*'+'/eval/events.out.tfevents*')
decay_const=[]
background=[]
params=[]

for m in model_names:
    acc=ea.EventAccumulator(m, size_guidance={ea.SCALARS: 0})
    acc.Reload()
    parm=extract_model_information(m)
    params.append(parm)
    loss        = [x.value for x in acc.Scalars('loss')]
    global_step = [x.step  for x in acc.Scalars('loss')]
    #plt_X=np.linspace(min(global_step),max(global_step))
    exp_decay = lambda x,A,B,l: A+B*np.exp(-l*x)
    try:
        popt,pconv = curve_fit(exp_decay, global_step, loss, p0=(1e-4, 1e0, 1./50000))
    except TypeError:
        logging.warning("Skip "+m)
        continue
    except RuntimeError:
        logging.warning("Optimal parameters not found for "+m+", ignore this one")
        continue
    decay_const.append(popt[-1])
    background.append(popt[1])
    #plt.semilogy(global_step, loss, 'C0' if parm['activation fn']=='elu' else 'C1', label=parm['activation fn'])
    plt.semilogy(global_step, loss, color=cmap(parm['kernel size']/5.)[0:3], label=parm['kernel size'])
plt.legend(loc='best')

ind=np.argsort(background) 
#ind=np.argsort(decay_const) 

for i in ind:
    print(
            '{update_fn:>12}, {activation_fn:>12}, K{ks:>1}: tau={dc:>10.5}, background = {bck:<8.5}'
            .format(
                update_fn=params[i]['update fn'],
                activation_fn=params[i]['activation fn'],
                ks=params[i]['kernel size'],
                bck=background[i],
                dc=1./decay_const[i]
                )
            )


plt.show(block=True)
