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
f = lambda x: np.sin(x)

# Generate the 'features'
#X = np.linspace(0, 1, 1000001).reshape((-1, 1))
INPUT_SHAPE=(1,)
x_min=-np.pi/2
x_max= np.pi/2
X = np.linspace(x_min,x_max,1001)

X_plot = np.linspace(-1,1,1001)
# Generate the 'labels'
y = f(X)

# Swap to train inverse

X,y=y,X

# Train/Test Dataset for Validation
X_train,X_test,y_train,y_test = train_test_split(X, y,
						test_size=0.1,
						random_state=2)

TRAIN_SET=pd.DataFrame({'X': X_train, 'y': y_train})
TEST_SET=pd.DataFrame({'X': X_test, 'y': y_test})
PREDICT_SET=pd.DataFrame({'X': X_plot, 'y':np.array([np.NaN]*len(X_plot))})

FEATURES=['X']
LABEL=['y']


def get_input_fn(data_set, num_epochs=None, shuffle=False):
    return tf.estimator.inputs.pandas_input_fn(
            x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
            y=pd.Series(data_set[LABEL].values[:,0]),
            num_epochs=num_epochs,
            shuffle=shuffle)

# Network Design
# --------------
feature_columns = [tf.feature_column.numeric_column('X', shape=INPUT_SHAPE)]

STEPS_PER_EPOCH = 1000
EPOCHS = 100
BATCH_SIZE = 1000

list_of_hl=[
            [16, 16+16, 16+16],
            [16,16,16,16,16],
            [16+16, 16+16, 16],
            [16+16+16,16+16],
            [8,8]
        ]
for dropout in [0.,.05,.1]:
    for hidden_layers in list_of_hl:
        MODEL_PATH='./arcsin_dnn/'
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
                                        label_dimension=1,
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

                # Fit the DNNRegressor (This is where the magic happens!!!)
                regressor.train(input_fn=get_input_fn(TRAIN_SET), steps=STEPS_PER_EPOCH)
                # Thats it -----------------------------
                # Start Tensorboard in Terminal:
                # 	tensorboard --logdir='./DNNRegressors/'
                # Now open Browser and visit localhost:6006\

                
                # This is just for fun and educational purpose:
                # Evaluate the DNNRegressor every 10th epoch
                if epoch%10==0:
                    eval_dict = regressor.evaluate(input_fn=get_input_fn(TEST_SET, num_epochs=1, shuffle=False))
                    print('Epoch %i: %.5f MSE' % (epoch+1, eval_dict['average_loss']))
            # Now it's trained. We can try to predict some values.
        else:
            logging.info('No training today, just prediction')
            #try:
            # Prediction
            y_pred = np.arcsin(X_plot)
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
                y_dnn=regressor.predict(input_fn=get_input_fn(PREDICT_SET, num_epochs=1, shuffle=False))
                y_dnn=list( p['predictions'] for p in y_dnn)
                fig,(ax1,ax2)=plt.subplots(2,1,sharex=True)
                plt.sca(ax1)
                plt.plot(X_plot, y_pred, label='function to predict')
                plt.plot(X_plot, y_dnn,
                                label='DNNRegressor prediction')
                plt.legend(loc='best')
                plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
                plt.tight_layout()
                plt.sca(ax2)
                y_dnn=np.array(y_dnn)[:,0]
                tmp1=max([max(abs(y_dnn-y_pred)),1e-12])
                plt.plot(X_plot, y_dnn-y_pred , label='Delta')
                ax2.set_ylim([-tmp1*1.1,tmp1*1.1])
                plt.savefig(MODEL_PATH + '.png', dpi=72)
                plt.close()
            #except:
            #    logging.error('Prediction failed! Maybe first train a model?')
