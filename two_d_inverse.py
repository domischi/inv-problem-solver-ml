#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Tensorflow DNNRegressor in Python
# CC-BY-2.0 Paul Balzer
# see: http://www.cbcity.de/deep-learning-tensorflow-dnnregressor-einfach-erklaert
#
TRAINING = False
#WITHPLOT = False
WITHPLOT = not TRAINING
PRINT_VARIABLES=False
LEARN_INVERSE=True

# Import Stuff
import tensorflow.contrib.learn as skflow
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import pandas as pd
from time import time
import numpy as np
np.set_printoptions(precision=2)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)
logging.info('Tensorflow %s' % tf.__version__) # 1.4.1

# This is the magic function which the Deep Neural Network
# has to 'learn' (see http://neuralnetworksanddeeplearning.com/chap4.html)
f = lambda x: np.array([np.cos(x),np.sin(x)])
# Generate the 'features'
x_min=-np.pi
x_max= np.pi


FEATURES=['X']
LABEL=['y']


def get_XY(SAMPLE_SIZE,LEARN_INVERSE=False):
    X=np.random.rand(SAMPLE_SIZE)*(x_max-x_min)-x_min
    y=f(X).transpose()
    if LEARN_INVERSE:
        return y,X
    return X,y
def get_input_fn(X,y, num_epochs=1, shuffle=False):
    assert(num_epochs==1)
    if shuffle:
        new_indices=list(range(len(X)))
        np.random.shuffle(new_indices)
        X=X[a]
        y=y[a]
    f=lambda :({FEATURES[0]: tf.convert_to_tensor(X, name='X')}, tf.convert_to_tensor(y, name='y'))
    return f

def get_shapes(gt_XY, LEARN_INVERSE=False):
    tmpx,tmpy=gt_XY(2,LEARN_INVERSE=LEARN_INVERSE)
    if len(tmpx.shape)==1: # 1D input data
        INPUT_SHAPE=(1,)
    elif len(tmpx.shape)==2: # vector input data
        INPUT_SHAPE=(tmpx.shape[-1],)
    else:
        logging.error('Try to input matrix (or higher tensor) data, not yet implemented')
        sys.exit()
    if len(tmpy.shape)==1: # 1D input data
        OUTPUT_SHAPE=1
    elif len(tmpy.shape)==2: # vector input data
        OUTPUT_SHAPE=tmpy.shape[-1]
    else:
        logging.error('Try to learn matrix (or higher tensor) data, not yet implemented')
        sys.exit()
    return INPUT_SHAPE,OUTPUT_SHAPE

INPUT_SHAPE,OUTPUT_SHAPE=get_shapes(get_XY, LEARN_INVERSE=LEARN_INVERSE)

# Network Design
# --------------
feature_columns = [tf.feature_column.numeric_column('X', shape=INPUT_SHAPE)]

STEPS_PER_EPOCH = 10000
EPOCHS = 200
BATCH_SIZE = 1000
EVAL_RATIO = .1

dropout=0.
list_of_hl=[
            [16, 16+16, 16+16],
            [16,16,16,16,16],
            [16+16, 16+16, 16],
            [16+16+16,16+16],
            [8,8]
        ]
for hidden_layers in list_of_hl:
    MODEL_PATH='../trained_networks/two_d_inverse/'
    for hl in hidden_layers:
        MODEL_PATH += '%s_' % hl
    MODEL_PATH += 'D0%s' % (int(dropout*100))
    logging.info('Saving to %s' % MODEL_PATH)

    # Validation and Test Configuration
    #validation_metrics = {"MSE": tf.contrib.metrics.streaming_mean_squared_error}
    test_config = tf.estimator.RunConfig(save_checkpoints_steps=None,
                                    save_checkpoints_secs=300)

    # Building the Network
    regressor = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                    label_dimension=OUTPUT_SHAPE,
                                    hidden_units=hidden_layers,
                                    model_dir=MODEL_PATH,
                                    dropout=dropout,
                                    config=test_config)

    # Train it
    if TRAINING:
        logging.info('Train the DNN Regressor...\n')
        MSEs = []	# for plotting
        STEPS = []	# for plotting

        for epoch in range(EPOCHS+1):
            regressor.train(input_fn=get_input_fn(*get_XY(BATCH_SIZE,LEARN_INVERSE=LEARN_INVERSE)), steps=STEPS_PER_EPOCH) 
            
            # Thats it ----------------------------- # Start Tensorboard in Terminal:
            # 	tensorboard --logdir='./DNNRegressors/'
            # Now open Browser and visit localhost:6006\

            
            # This is just for fun and educational purpose:
            # Evaluate the DNNRegressor every 10th epoch
            if epoch%10==0:
                eval_dict = regressor.evaluate(input_fn=get_input_fn(*get_XY(int(EVAL_RATIO*BATCH_SIZE),LEARN_INVERSE=LEARN_INVERSE), shuffle=False), steps=1)
                print('Epoch %i: %.5f MSE' % (epoch+1, eval_dict['average_loss']))
        # Now it's trained. We can try to predict some values.
    else:
        logging.info('No training today, just prediction')
        #try:
        # Get trained values out of the Network
        if PRINT_VARIABLES:
            for variable_name in regressor.get_variable_names():
                if str(variable_name).startswith('dnn/hiddenlayer') and \
                    (str(variable_name).endswith('weights') or \
                    str(variable_name).endswith('biases')):
                    print('\n%s:' % variable_name)
                    weights = regressor.get_variable_value(variable_name)
                    print(weights)
                    print('size: %i' % weights.size)

        # Final Plot
        if WITHPLOT:
            X_plot, y_plot=get_XY(30000, LEARN_INVERSE=LEARN_INVERSE)
            #y_dnn=regressor.predict(input_fn=get_input_fn(X_plot, y_plot, num_epochs=1, shuffle=False), steps=1)
            y_dnn=regressor.predict(input_fn=get_input_fn(X_plot,[np.nan]*len(X_plot), shuffle=False))
            tmp=[]
            for i,q  in enumerate(y_dnn):
                if i>=len(y_plot):
                    break
                tmp.append(q['predictions'])
            y_dnn=np.array(tmp)
            fig,(ax1,ax2)=plt.subplots(2,1,sharex=True)
            plt.sca(ax1)
            t=X_plot[:,0]
            plt.plot(t, y_plot,'.', label='function to predict')
            plt.plot(t, y_dnn,'.',
                            label='DNNRegressor prediction')
            plt.xlabel('cos(x)')
            plt.ylabel(r'$\theta`$')
            plt.legend(loc='best')
            plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
            plt.tight_layout()
            plt.sca(ax2)
            y_dnn=np.array(y_dnn)[:,0]
            tmp1=min([max([max(abs(y_dnn-y_plot)),1e-12]),0.01])
            plt.plot(t, y_dnn-y_plot , '.',label='Delta')
            ax2.set_ylim([-tmp1*1.1,tmp1*1.1])
            plt.savefig(MODEL_PATH + '.png', dpi=72)
            plt.close()
        #except:
        #    logging.error('Prediction failed! Maybe first train a model?')
