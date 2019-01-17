#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Tensorflow DNNRegressor in Python
# CC-BY-2.0 Paul Balzer
# see: http://www.cbcity.de/deep-learning-tensorflow-dnnregressor-einfach-erklaert
#
TRAINING = True
#WITHPLOT = False
WITHPLOT = not TRAINING
PRINT_VARIABLES=False
LEARN_INVERSE=False
OVERWRITE=False
DATA_ROOT_PATH='../trained_networks/heat_kernel_forward_small/'

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
import sys
import os
from glob import glob

# Generate the 'features'
FEATURES=['X', 't']
LABEL=['y']
NUM_SAMPLES_PER_FILE=512
NUM_TRAIN_SAMPLES=256
NUM_EVAL_SAMPLES=256
GAUSSIAN_MIN=-1
GAUSSIAN_MAX= 1
maxwidth=0.1
MAX_NUM_GAUSSIANS=1
SAMPELING_MIN=-4
SAMPELING_MAX= 4
SAMPELING_N=512
SAMPLING_SPACE=np.linspace(SAMPELING_MIN,SAMPELING_MAX,SAMPELING_N)
MAX_TIME_TO_EVOLVE=0.05

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
def get_one_XY():
    TIME_TO_EVOLVE=np.random.rand()*MAX_TIME_TO_EVOLVE
    X=n_random_gaussians(GAUSSIAN_MIN,GAUSSIAN_MAX,MAX_NUM_GAUSSIANS)
    y=apply_heat_kernel(X,TIME_TO_EVOLVE)
    return TIME_TO_EVOLVE,X,y 

def get_XY(SAMPLE_SIZE,LEARN_INVERSE=False):
    X=[]
    y=[]
    t=[]
    for i in range(SAMPLE_SIZE):
        tmpt,tmpx,tmpy=get_one_XY()
        t.append(tmpt)
        X.append(tmpx)
        y.append(tmpy)
    t=np.array(t)
    X=np.array(X)
    y=np.array(y)
    if LEARN_INVERSE:
        return t,y,X
    return t,X,y

# From https://gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f
def np_to_tfrecords(T, X, Y, file_path_prefix):
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecord'
    with tf.python_io.TFRecordWriter(result_tf_file) as writer:
        # iterate over each sample,
        # and serialize it as ProtoBuf.
        for idx in range(X.shape[0]):
            t = T[idx]
            x = X[idx]
            y = Y[idx]
            
            d_feature = {}
            d_feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=x))
            d_feature['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=y))
            d_feature['t'] = tf.train.Feature(float_list=tf.train.FloatList(value=[t]))
            
            features = tf.train.Features(feature=d_feature)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)


def write_tf_records(name, NUM_SAMPLES):
    for i in range(NUM_SAMPLES):
        t,X,y=get_XY(NUM_SAMPLES_PER_FILE)
        if LEARN_INVERSE:
            y,X=X,y
        filename=DATA_ROOT_PATH+'data/'+name+'_'+str(i).zfill(4)
        np_to_tfrecords(t,X,y,filename)
def generate_data(name, NUM_SAMPLES, OVERWRITE=False):
    filelist=glob(DATA_ROOT_PATH+'data/'+str(name)+'*.tfrecord')
    if len(filelist)<NUM_SAMPLES or OVERWRITE:
        logging.info('Generate new '+str(name)+' data')
        write_tf_records(name,NUM_TRAIN_SAMPLES)
        logging.info('Generation of '+name+' data finished')

os.makedirs(DATA_ROOT_PATH+'data/', exist_ok=True)
generate_data('train', NUM_TRAIN_SAMPLES,OVERWRITE=OVERWRITE)
generate_data('eval', NUM_EVAL_SAMPLES,OVERWRITE=OVERWRITE)
generate_data('validate', 1,OVERWRITE=OVERWRITE)

def dataset_input_fn(filenames,NUM_EPOCHS=1,NUM_BATCHES=32):
    dataset = tf.data.TFRecordDataset(filenames)
    
    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            't': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True ,default_value=tf.zeros([], dtype=tf.float32)),
            'X': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True ,default_value=tf.zeros([], dtype=tf.float32)),
            'y': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True ,default_value=tf.zeros([], dtype=tf.float32))
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        t=tf.cast(parsed['t'],tf.float32)
        X=tf.cast(parsed['X'],tf.float32)
        y=tf.cast(parsed['y'],tf.float32)
        return {"t": t, "X": X}, y

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(NUM_BATCHES)
    dataset = dataset.repeat(NUM_EPOCHS)
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return features, labels

def get_input_fn(filenames_start='train',NUM_EPOCHS=1,NUM_BATCHES=32):
    lst=glob(DATA_ROOT_PATH+'data/'+filenames_start+'*')
    np.random.shuffle(lst)
    print(filenames_start,DATA_ROOT_PATH+'data/'+filenames_start+'*',len(lst))
    assert(len(lst)>5)
    lst=lst[0:4]
    return lambda: dataset_input_fn(lst,NUM_EPOCHS,NUM_BATCHES)

def get_shapes(gt_XY, LEARN_INVERSE=False):
    _, tmpx,tmpy=gt_XY(2,LEARN_INVERSE=LEARN_INVERSE)
    if len(tmpx.shape)==1: # 1D input data
        INPUT_SHAPE=(1,)
    elif len(tmpx.shape)==2: # vector input data
        INPUT_SHAPE=(tmpx.shape[-1],)
    else:
        logging.error('Try to input matrix (or higher tensor) data, not yet implemented')
    if len(tmpy.shape)==1: # 1D input data
        OUTPUT_SHAPE=1
    elif len(tmpy.shape)==2: # vector input data
        OUTPUT_SHAPE=tmpy.shape[-1]
    else:
        logging.error('Try to learn matrix (or higher tensor) data, not yet implemented')
        sys.exit()
    return INPUT_SHAPE,OUTPUT_SHAPE

INPUT_SHAPE,OUTPUT_SHAPE=get_shapes(get_XY, LEARN_INVERSE=LEARN_INVERSE)

# --------------
feature_columns = [tf.feature_column.numeric_column('t', shape=(1,)),tf.feature_column.numeric_column('X', shape=INPUT_SHAPE)]

STEPS_PER_EPOCH =1000
EPOCHS = 200
BATCH_SIZE = 128
EVAL_RATIO = .1

dropout=0.
list_of_hl=[
            [16, 16+16, 16+16],
            [16,16,16,16,16],
            [16+16, 16+16, 16],
            [16+16+16,16+16],
            [8,8],
            [256],
            [256,256]
        ]
for hidden_layers in list_of_hl:
    MODEL_PATH=DATA_ROOT_PATH+'models/'
    for hl in hidden_layers:
        MODEL_PATH += '%s_' % hl
    MODEL_PATH += 'D0%s' % (int(dropout*100))
    logging.info('Saving to %s' % MODEL_PATH)

    # Validation and Test Configuration
    #validation_metrics = {"MSE": tf.contrib.metrics.streaming_mean_squared_error}
    test_config = tf.estimator.RunConfig(save_checkpoints_steps=None,
                                    save_checkpoints_secs=3600)

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
            regressor.train(input_fn=get_input_fn('train'), steps=STEPS_PER_EPOCH) 
            
            # Thats it ----------------------------- # Start Tensorboard in Terminal:
            # 	tensorboard --logdir='./DNNRegressors/'
            # Now open Browser and visit localhost:6006\

            
            # This is just for fun and educational purpose:
            # Evaluate the DNNRegressor every 10th epoch
            if epoch%10==0:
                eval_dict = regressor.evaluate(input_fn=get_input_fn('eval'), steps=1)
                print('Epoch %i: %.5f MSE' % (epoch+1, eval_dict['average_loss']))
        # Now it's trained. We can try to predict some values.
        os.remove(glob(MODEL_PATH+'/events*')[0]) # prevent saving huge log files for training, eval is enough
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
