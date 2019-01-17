#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Tensorflow DNNRegressor in Python
# CC-BY-2.0 Paul Balzer
# see: http://www.cbcity.de/deep-learning-tensorflow-dnnregressor-einfach-erklaert
#

# this is next to impossible to set up nicely with argparse, therefore move it at least to the top of the file
#list_of_hl=[
#            [16, 16+16, 16+16],
#            [16,16,16,16,16],
#            [16+16, 16+16, 16],
#            [16+16+16,16+16],
#            [8,8],
#            [256],
#            [256,256],
#            #[512],
#            #[512,512]
#        ]

# Import stuff
import argparse
from time import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import sys
import os
from glob import glob

# Output ptions
tf.logging.set_verbosity(tf.logging.WARN)
np.set_printoptions(precision=9)
logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
logging.info('Tensorflow %s' % tf.__version__) # 1.4.1

# Construct parser
parser = argparse.ArgumentParser(description="A python script facilitating the generation as well as the training on the (inverse) heat kernel problem")
parser.add_argument('--root-directory'             , type=str             , required=True   , help ='Where all the data is getting stored')
parser.add_argument('--validation'                 , action='store_true'  ,                   help ='Should the model be validated?')
parser.add_argument('--training'                   , action='store_true'  ,                   help ='Should the model be trained?')
parser.add_argument('--learn-inverse'              , action='store_true'  ,                   help ='Learn the inverse?')
parser.add_argument('--samples-per-file'           , type=int             , default=512     , help ='How many samples should be stored in the files')
parser.add_argument('--num-train-files'            , type=int             , default=256     , help ='How many training files should be generated?')
parser.add_argument('--num-eval-files'             , type=int             , default=32      , help ='How many evaluation files should be generated?')
parser.add_argument('--max-number-of-gaussians'    , type=int             , default=3       , help ='Number of Gaussians to be generated')
parser.add_argument('--gaussian-range'             , type=float           , default=1       , help ='The range for the centers of the gaussians is between +- this value')
parser.add_argument('--sigma-gaussian'             , type=float           , default=0.02    , help ='The width of the gaussians')
parser.add_argument('--simple-gaussians'           , action='store_true'                    , help ='Only give nice gaussians to learn from, well separated and having not a too narrow width')
parser.add_argument('--sampling-range'             , type=float           , default=3       , help ='The sampeling range is between +- this value')
parser.add_argument('--num-sampeling-points'       , type=int             , default=512     , help ='How many sampeling points should be given?')
parser.add_argument('--max-time-to-evolve'         , type=float           , default=0.01    , help ='What is the maximum time the network should revert?')
parser.add_argument('--min-time-to-evolve'         , type=float           , default=0.001 , help ='What is the minimum time the network should revert?')
parser.add_argument('--steps-per-epoch'            , type=int             , default=512     , help ='Steps per epoch')
parser.add_argument('--num-epochs'                 , type=int             , default=200     , help ='Number of epochs')
parser.add_argument('--batch-size'                 , type=int             , default=16384   , help ='Batch size')
parser.add_argument('--dropout'                    , type=float           , default=0.      , help ='Dropout (probably always stick to 0)')
parser.add_argument('--dont-delete-detailed-statistics' , action='store_true' ,                   help ='Delete detailed statistics, as they bloat up the directories')
parser.add_argument('--overwrite'                  , action='store_true'  ,                   help ='Overwrite the generated files')
OPTIONS=vars(parser.parse_args())

# Print options
for k in OPTIONS:
    logging.info('\tOption {:<30}: {:>10}'.format(k, OPTIONS[k]))

# Set options
TRAINING                   =     OPTIONS ['training']
VALIDATION                 =     OPTIONS ['validation']
LEARN_INVERSE              =     OPTIONS ['learn_inverse']
DATA_ROOT_PATH             =     OPTIONS ['root_directory']
NUM_SAMPLES_PER_FILE       =     OPTIONS ['samples_per_file']
NUM_TRAIN_SAMPLES          =     OPTIONS ['num_train_files']
NUM_EVAL_SAMPLES           =     OPTIONS ['num_eval_files']
MAX_NUM_GAUSSIANS          =     OPTIONS ['max_number_of_gaussians']
GAUSSIAN_MIN               =  -  OPTIONS ['gaussian_range']
GAUSSIAN_MAX               =     OPTIONS ['gaussian_range']
maxwidth                   =     OPTIONS ['sigma_gaussian']
SAMPELING_MIN              =  -  OPTIONS ['sampling_range']
SAMPELING_MAX              =     OPTIONS ['sampling_range']
SAMPELING_N                =     OPTIONS ['num_sampeling_points']
MAX_TIME_TO_EVOLVE         =     OPTIONS ['max_time_to_evolve']
MIN_TIME_TO_EVOLVE         =     OPTIONS ['min_time_to_evolve']
STEPS_PER_EPOCH            =     OPTIONS ['steps_per_epoch']
EPOCHS                     =     OPTIONS ['num_epochs']
BATCH_SIZE                 =     OPTIONS ['batch_size']
dropout                    =     OPTIONS ['dropout']
DELETE_DETAILED_STATISTICS = not OPTIONS ['dont_delete_detailed_statistics']
OVERWRITE                  =     OPTIONS ['overwrite']
SIMPLE_GAUSSIANS           =     OPTIONS ['simple_gaussians']

# Precompute useful stuff and define the necessary evils
SAMPLING_SPACE=np.linspace(SAMPELING_MIN,SAMPELING_MAX,SAMPELING_N)
FEATURES=['X', 't']
LABEL=['y']
DATA_PREFIX='/data/'
DATA_SUFFIX='.tfrecord'
TRAIN_PREFIX='train'
EVAL_PREFIX='eval'
VALIDATE_PREFIX='validate'
ERROR_VS_TIME_PREFIX='error_vs_time'

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
    result_tf_file = file_path_prefix + DATA_SUFFIX
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
        #if LEARN_INVERSE:
        #    y,X=X,y
        filename=DATA_ROOT_PATH+DATA_PREFIX+name+str(i)
        np_to_tfrecords(t,X,y,filename)
def generate_data(name, NUM_SAMPLES, OVERWRITE=False):
    filelist=glob(DATA_ROOT_PATH+DATA_PREFIX+str(name)+'*'+DATA_SUFFIX)
    if len(filelist)<NUM_SAMPLES or OVERWRITE:
        logging.info('Generate new '+str(name)+' data')
        write_tf_records(name,NUM_SAMPLES)
        logging.info('Generation of '+name+' data finished')

def dataset_input_fn(filenames,NUM_EPOCHS=1,NUM_BATCHES=32):
    dataset = tf.data.TFRecordDataset(filenames)
    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            #'t': tf.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
            't': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True ,default_value=tf.zeros([], dtype=tf.float32)),
            'X': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True ,default_value=tf.zeros([], dtype=tf.float32)),
            'y': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True ,default_value=tf.zeros([], dtype=tf.float32)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        #t=tf.cast(parsed['t'],tf.float32)
        #X=tf.cast(parsed['X'],tf.float32)
        #y=tf.cast(parsed['y'],tf.float32)
        t=parsed['t']
        X=parsed['X']
        y=parsed['y']
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
def get_input_fn(filenames_start,NUM_EPOCHS=1,NUM_BATCHES=32):
    lst=glob(DATA_ROOT_PATH+DATA_PREFIX+filenames_start+'*')
    assert(len(lst)>0)
    np.random.shuffle(lst)
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
        sys.exit()
    if len(tmpy.shape)==1: # 1D input data
        OUTPUT_SHAPE=1
    elif len(tmpy.shape)==2: # vector input data
        OUTPUT_SHAPE=tmpy.shape[-1]
    else:
        logging.error('Try to learn matrix (or higher tensor) data, not yet implemented')
        sys.exit()
    return INPUT_SHAPE,OUTPUT_SHAPE

def model_name(parm):
    s=''
    #s+=parm['optimizer']+'_'
    s+=parm['activation']+'_K'
    #s+=str(parm['conv_also_activate'])+'_'
    s+=str( parm['kernel_size'] )+'_t'
    hl_list=parm['hidden_units_t']
    for hl in hl_list:
        s += '{}{}'.format(hl[0][0], hl[1]) 
    s+="_X"
    hl_list=parm['hidden_units_X']
    for hl in hl_list:
        s += '{}{}'.format(hl[0][0], hl[1]) 
    s+="_"
    hl_list=parm['hidden_units_combined']
    for hl in hl_list:
        s += '{}{}'.format(hl[0][0], hl[1]) 
    return s
def read_tffile(PATH):
    record_iterator = tf.python_io.tf_record_iterator(path=PATH)
    X=[]
    Y=[]
    T=[]
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        X.append(np.array(  example.features.feature['X'].float_list.value ))
        Y.append(np.array(  example.features.feature['y'].float_list.value ))
        T.append(np.array(  example.features.feature['t'].float_list.value ))
    T=np.array(T)
    X=np.array(X)
    Y=np.array(Y)
    return T,X,Y
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
def n_random_gaussians_simplified(a,b,n):
    y=np.zeros(len(SAMPLING_SPACE))
    means=None
    while True:
        means=np.sort( np.random.rand(np.random.randint(n)+1)*(b-a)+a )
        if len(means)== 1 or min(abs( means[1:]-means[:-1] ))> (b-a)/(2*len(means)):
            break
    minwidth=maxwidth/3
    sigmas=(maxwidth-minwidth)*np.random.rand(len(means))+minwidth
    for m,s in zip(means,sigmas):
        y+=one_gaussian(m,s)
    return y/n
def n_random_gaussians_simplified(a,b,n):
    y=np.zeros(len(SAMPLING_SPACE))
    means=None
    while True:
        means=np.sort( np.random.rand(np.random.randint(n)+1)*(b-a)+a )
        if len(means)==1 or min(abs( means[1:]-means[:-1] ))> (b-a)/(2*len(means)):
            break
    minwidth=maxwidth/3
    sigmas=(maxwidth-minwidth)*np.random.rand(len(means))+minwidth
    for m,s in zip(means,sigmas):
        y+=one_gaussian(m,s)
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
    TIME_TO_EVOLVE=np.random.rand()*(MAX_TIME_TO_EVOLVE-MIN_TIME_TO_EVOLVE)+MIN_TIME_TO_EVOLVE
    if SIMPLE_GAUSSIANS:
        X=n_random_gaussians_simplified(GAUSSIAN_MIN,GAUSSIAN_MAX,MAX_NUM_GAUSSIANS)
    else:
        X=n_random_gaussians(GAUSSIAN_MIN,GAUSSIAN_MAX,MAX_NUM_GAUSSIANS)
    y=apply_heat_kernel(X,TIME_TO_EVOLVE)
    return TIME_TO_EVOLVE,X,y

os.makedirs(DATA_ROOT_PATH+DATA_PREFIX, exist_ok=True)
#generate_data(TRAIN_PREFIX    , NUM_TRAIN_SAMPLES, OVERWRITE=OVERWRITE )
#generate_data(EVAL_PREFIX     , NUM_EVAL_SAMPLES , OVERWRITE=OVERWRITE )
#generate_data(VALIDATE_PREFIX , 1                , OVERWRITE=OVERWRITE )

def prod(iterable):
    s=1
    for i in iterable:
        s*=i
    return s

def mixed_model(features, labels, mode, params):
    """Create a network for the inverse heat kernel problem
        params: dictionary of the relevant values
            params['hidden_units']: list of (description_string, num_neurons)
                    where description_string can be 'dense' or 'conv'
            params['activation']: 'relu','leaky_relu', 'elu', 'relu6', 'selu'
            params['conv_also_activate']: boolean
            params['optimizer']: 'adam', 'adagrad', 'adadelta', 'rmsprop', 'ftrl'
    """
    ACTIVATION_FN=None
    if 'activation' not in params.keys():
        logging.error('Did not supply a activation parameter, aborting...')
        sys.exit()
    if 'optimizer' not in params.keys():
        logging.error('Did not supply an optimizer parameter, aborting...')
        sys.exit()
    if 'conv_also_activate' not in params.keys():
        logging.error('Did not supply a conv_also_activate parameter, aborting...')
        sys.exit()

    if params['activation']== 'relu':
        ACTIVATION_FN=tf.nn.relu
    elif params['activation']== 'leaky_relu' :
        ACTIVATION_FN=tf.nn.leaky_relu
    elif params['activation']== 'elu'        :
        ACTIVATION_FN=tf.nn.elu
    elif params['activation']== 'relu6'      :
        ACTIVATION_FN=tf.nn.relu6
    elif params['activation']== 'selu'       :
        ACTIVATION_FN=tf.nn.selu
    else:
        logging.error('Unrecognized activation function {}, aborting'.format(params['activation']))
        sys.exit()

    X_net=tf.feature_column.input_layer(features, [q for q in params['feature_columns'] if q.key=='X'] )
    t_net=tf.feature_column.input_layer(features, [q for q in params['feature_columns'] if q.key=='t'] )
    for units in params['hidden_units_t']:
        if units[0]=='dense':
            t_net = tf.layers.dense(t_net, units=units[1], activation=ACTIVATION_FN)
        else:
            logging.error('Unrecognized layer type {}, aborting'.format(units[0]))
            sys.exit()
    for units in params['hidden_units_X']:
        if units[0]=='dense':
            X_net = tf.layers.dense(X_net, units=units[1], activation=ACTIVATION_FN)
        elif units[0]=='conv':
            if len(X_net.get_shape().as_list())==2:
                X_net=tf.expand_dims(X_net, axis=-1)
            if params['conv_also_activate']:
                X_net = tf.layers.conv1d(X_net, filters=units[1], kernel_size=params['kernel_size'], activation=ACTIVATION_FN)
            else:
                X_net = tf.layers.conv1d(X_net, filters=units[1], kernel_size=params['kernel_size'], activation=None)
        else:
            logging.error('Unrecognized layer type {}, aborting'.format(units[0]))
            sys.exit()
    X_net=tf.reshape(X_net,[-1,prod(X_net.get_shape().as_list()[1:])])
    net= tf.concat([t_net, X_net], -1)
    for units in params['hidden_units_combined']:
        if units[0]=='dense':
            net = tf.layers.dense(net, units=units[1], activation=ACTIVATION_FN)
        else:
            logging.error('Unrecognized layer type {}, aborting'.format(units[0]))
            sys.exit()

    net = tf.layers.dense(net, units=SAMPELING_N, activation=None)

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predictions': net,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.mean_squared_error(labels=labels, predictions=net)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer=None
    if params['optimizer']=='adam':
        optimizer = tf.train.AdamOptimizer()
    elif params['optimizer']== 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    elif params['optimizer']== 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer()
    elif params['optimizer']== 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    elif params['optimizer']== 'ftrl':
        optimizer = tf.train.FtrlOptimizer(0.05)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    

INPUT_SHAPE,OUTPUT_SHAPE=get_shapes(get_XY, LEARN_INVERSE=LEARN_INVERSE)

# --------------
feature_columns = [
            tf.feature_column.numeric_column('t', shape=(1,)),
            tf.feature_column.numeric_column('X', shape=INPUT_SHAPE)
        ]

activation_list=['leaky_relu','elu']
layer_type_list=['conv','dense']
layer_width_list_t=[2**i for i in range(4,9)]
layer_width_list_X=[16,32]
layer_width_list_combined=[128,256]
def random_choice(iterable):
    return iterable[np.random.randint(len(iterable))]
def monte_carlo_the_model():
    params={ 'feature_columns': feature_columns }
    params['optimizer']='rmsprop'
    params['activation']=random_choice(activation_list)
    params['conv_also_activate']=True
    #params['conv_also_activate']=bool(np.random.randint(2))
    params['hidden_units_t']=[]
    params['hidden_units_X']=[]
    params['hidden_units_combined']=[]
    params['kernel_size']=np.random.randint(5)+1
    for i in range( np.random.randint(2) +1):
        tmp=['dense', random_choice(layer_width_list_t)]
        params['hidden_units_t'].append(tmp)
    for i in range( np.random.randint(2) ):
        tmp=['conv', random_choice(layer_width_list_X)]
        params['hidden_units_X'].append(tmp)
    for i in range( 1 ):
        tmp=['dense', random_choice(layer_width_list_combined)]
        params['hidden_units_combined'].append(tmp)
    return params

def get_one_X():
    if SIMPLE_GAUSSIANS:
        X=n_random_gaussians_simplified(GAUSSIAN_MIN,GAUSSIAN_MAX,MAX_NUM_GAUSSIANS)
    else:
        X=n_random_gaussians(GAUSSIAN_MIN,GAUSSIAN_MAX,MAX_NUM_GAUSSIANS)
    return X
def get_one_XY_fixed_t(TIME_TO_EVOLVE):
    if SIMPLE_GAUSSIANS:
        X=n_random_gaussians_simplified(GAUSSIAN_MIN,GAUSSIAN_MAX,MAX_NUM_GAUSSIANS)
    else:
        X=n_random_gaussians(GAUSSIAN_MIN,GAUSSIAN_MAX,MAX_NUM_GAUSSIANS)
    y=apply_heat_kernel(X,TIME_TO_EVOLVE)
    return TIME_TO_EVOLVE,X,y

def get_XY_fixed_t(NUM_SAMPLES_PER_FILE, fixed_t=None):
    if fixed_t==None:
        return get_XY(NUM_SAMPLES_PER_FILE)
    else:
        X=[]
        y=[]
        t=[]
        for i in range(NUM_SAMPLES_PER_FILE):
            tmpx=get_one_X()
            for T in np.linspace(MIN_TIME_TO_EVOLVE, MAX_TIME_TO_EVOLVE, fixed_t):
                tmpy=apply_heat_kernel(tmpx,T)
                t.append(T)
                X.append(tmpx)
                y.append(tmpy)
        t=np.array(t)
        X=np.array(X)
        y=np.array(y)
        if LEARN_INVERSE:
            return t,y,X
        return t,X,y

def generate_error_data(name, NUM_SAMPLES, NUM_TIMES, OVERWRITE=False):
    filelist=glob(DATA_ROOT_PATH+DATA_PREFIX+str(name)+'*'+DATA_SUFFIX)
    if len(filelist)<NUM_SAMPLES or OVERWRITE:
        logging.info('Generate new '+str(name)+' data')
        t,X,y=get_XY_fixed_t(NUM_SAMPLES, NUM_TIMES)
        if LEARN_INVERSE:
            y,X=X,y
        filename=DATA_ROOT_PATH+DATA_PREFIX+name+str(0)
        np_to_tfrecords(t,X,y,filename)
        logging.info('Generation of '+name+' data finished')

already_done=[]
counter=0
PLOT_ALL=False
NUM_TIMES=20
NUM_SAMPLES_FOR_ERR_VS_TIME=128
generate_error_data(ERROR_VS_TIME_PREFIX,NUM_SAMPLES_FOR_ERR_VS_TIME,NUM_TIMES,OVERWRITE=True)
while True:
    parm=monte_carlo_the_model()
    MODEL_PATH=DATA_ROOT_PATH+'/models/'
    MODEL_PATH += model_name(parm)
    if not os.path.exists(MODEL_PATH):
        continue
    logging.info('Loading %s' % MODEL_PATH)
    if MODEL_PATH in already_done:
        counter+=1
        if counter == 20:
            print('Did not find no new models, aborting...')
            break
        continue
    counter=0
    already_done.append(MODEL_PATH)
    # Validation and Test Configuration
    test_config = tf.estimator.RunConfig(save_checkpoints_steps=None,
                                    save_checkpoints_secs=600,
                                    keep_checkpoint_max=1,
                                    model_dir=MODEL_PATH)

    # Building the Network
    regressor = tf.estimator.Estimator(
                        model_fn=mixed_model,
                        config=test_config,
                        params=parm
                    )
    plt.ioff()
    logging.info('Only do the validation')
    VALIDATION_PATH=DATA_ROOT_PATH+'/validation/'+model_name(parm)+'/'
    #T,X,Y=read_tffile(glob( DATA_ROOT_PATH+'/data/'+ERROR_VS_TIME_PREFIX+'*' )[0])
    T,Y,X=read_tffile(glob( DATA_ROOT_PATH+'/data/'+ERROR_VS_TIME_PREFIX+'*' )[0])
    y_dnn=regressor.predict(input_fn=lambda :{'X': tf.convert_to_tensor(X, name='X'), 't': tf.convert_to_tensor(T, name = 't')})
    logging.info('\tStart validation for '+model_name(parm))

    L2_t0=dict()
    L2_t1=dict()
    for i,y in enumerate(y_dnn):
        if i >= len(X) or i >=NUM_TIMES*NUM_SAMPLES_FOR_ERR_VS_TIME:
            break
        if LEARN_INVERSE:
            y_pred_for_plot=np.array(y['predictions'])
            tim = T[i][0]
            if PLOT_ALL:
                fig, (ax1,ax2,ax3)=plt.subplots(3,1,figsize=(10,6), sharex=True)
                plt.sca(ax1)
                plt.title('t={:5f}'.format(float(0)))
                plt.plot(SAMPLING_SPACE , Y[i]            , label='Real')
                plt.plot(SAMPLING_SPACE , y_pred_for_plot , label='Network')
                plt.ylabel('Heat profile')
                plt.legend(loc='best')

                plt.sca(ax2)
                plt.title('t={:5f}'.format(float(tim )))
                plt.plot(SAMPLING_SPACE , X[i]            , label='Real' )
                plt.plot(SAMPLING_SPACE , apply_heat_kernel( y_pred_for_plot, tim ) , label='Network')
                plt.ylabel('Heat profile')
                plt.legend(loc='best')
                plt.sca(ax3)
                plt.plot(SAMPLING_SPACE , y_pred_for_plot-Y[i] , label=r'$\Delta(t=0)$')
                plt.plot(SAMPLING_SPACE , apply_heat_kernel( y_pred_for_plot, tim )-X[i] , label=r'$\Delta(t=$'+str(tim)+'$)$')
                plt.ylabel('Difference')
                plt.legend(loc='best')
                plt.show(block=True)
            os.makedirs(VALIDATION_PATH, exist_ok=True)
            if LEARN_INVERSE:
                assert(max(X[i])<max(Y[i]))
            e_t0=np.linalg.norm(y_pred_for_plot-Y[i]) /len(Y[i])
            e_t1=np.linalg.norm(apply_heat_kernel( y_pred_for_plot, tim )-X[i])/len(X[i])
            #e_t1=np.linalg.norm(y_pred_for_plot-Y[i]) /len(Y[i])
            #e_t0=np.linalg.norm(apply_heat_kernel( y_pred_for_plot, tim )-X[i])/len(X[i])
            if tim in L2_t0.keys():
                L2_t0[tim].append(e_t0)
                L2_t1[tim].append(e_t1)
            else:
                L2_t0[tim]=[e_t0]
                L2_t1[tim]=[e_t1]
        else:
            assert(False)
    del y_dnn
    Ts=np.array(sorted( L2_t0.keys() ))
    print('debug', len(L2_t1[Ts[0]]))
    L2_t0_mean=[np.mean(L2_t0[t]) for t in Ts]
    L2_t0_err =[np.std(L2_t0[t]) for t in Ts]
    L2_t1_mean=[np.mean(L2_t1[t]) for t in Ts]
    L2_t1_err =[np.std(L2_t1[t]) for t in Ts]
    eps=4e-5
    plt.errorbar(Ts+eps, L2_t0_mean, yerr=L2_t0_mean, label='Reconstruced')
    plt.errorbar(Ts-eps, L2_t1_mean, yerr=L2_t1_mean, label='Forwarded')
    plt.legend(loc='best')
    logging.info('\tEnded validation for '+model_name(parm))
    plt.show(block=True)
    sys.exit()
