'''
 (c) Copyright 2021
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 @author: Yasser Abduallah
'''
from __future__ import division
import warnings 
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import csv 
from datetime import datetime
import argparse
import pickle 
import time 

# import tsinet_utils as si_utils
from tsinet_utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('turning logging of is not available')

normalize_data = True 
model_verbose = 1
epochs =20
batch_size = 16
max_irradiance = 0
data_dir = 'train_data/'
result_data_dir = 'results/'
data_train_validate_file_name = 'train_data' + os.sep + 'SORCE_TSI.csv'
num_value = 1
num_units=400
verbose = False
attention_layer=True

def train_tsinet():
    starting_time = int(round(time.time() * 1000))
    log('data_train_validate_file_name:', data_train_validate_file_name)

    dataset1 = si_utils.load_data(data_train_validate_file_name,index_col=[], remove_zero_cols=['irradiance'])
    date_data = dataset1['date']
    
    irradiance_data_orig = dataset1['irradiance']
    
    dataset = si_utils.load_data(data_train_validate_file_name, remove_zero_cols=['irradiance'])

    dataset = si_utils.drop_column_from_dataset(dataset,'julian_date')
    dataset = si_utils.drop_column_from_dataset(dataset,'sunspot')
    cols = dataset.columns
    for c in cols:
        if c.strip().lower() != 'irradiance':
            dataset = si_utils.drop_column_from_dataset(dataset, c)
    log('columns:', dataset.columns) 
    n_features = len(dataset.columns) - 1
    
    log(dataset.columns)
    irradiance_col_index = 0
    for i in range(0 , len(dataset.columns)):
        c = dataset.columns[i]
        if c =='irradiance':
            irradiance_col_index = i
    irradiance_data = dataset['irradiance']
    median_irradiance = np.median( irradiance_data)
    max_irradiance  = irradiance_data.max()
    min_irradiance = irradiance_data.min()
    si_utils.save_model_irradiance_stats(max_irradiance,min_irradiance)
    subtract_irradiance_value = min_irradiance
    log('median_irradiance:', median_irradiance)
    log('max_irradiance:', max_irradiance)
    log('min_irradiance:' , min_irradiance)
    log('subtract_irradiance_value:', subtract_irradiance_value)
    if normalize_data:
        subtract_irradiance_value = min_irradiance
        log("Normalizing data...")
        irradiance_data = irradiance_data - subtract_irradiance_value
        irradiance_data = irradiance_data / (max_irradiance - min_irradiance)
        irradiance_data = irradiance_data * num_value 
        
        dataset['irradiance'] = irradiance_data
  
    dataset_x, dataset_y = create_time_frames(dataset, input_col='irradiance')
    log('dataset_x size:', len(dataset_x)) 
    log('dataset_y size:', len(dataset_y))
    dataset = dataset.values
    count = 0
    train_x, test_x = split_train_test_dataset(dataset_x, test_size=0)
    log('train_x size from split:', len(train_x)) 
    log('test_x size from split:', len(test_x))
    train_y, test_y = split_train_test_dataset(dataset_y,test_size=0)
    log('train_y size from split:', len(train_y)) 
    log('test_y size from split:', len(test_y))
    log('type of train_y:', type(train_y))
    
    log('train_x.shape',train_x.shape)
    log('train_y.shape', train_y.shape)
    log('test_x', test_x.shape)
    log('test_y', test_y.shape)
    log('train_y[0:1]', train_y[0:1])
    test_date_starting_index = len(train_x)
    log('orig date size: ', len(date_data))
    
    load_data_time = int(round(time.time() * 1000))
    log('Time taken to load and prepare data:', int((load_data_time - starting_time)/1000) , 'second(s)')
    starting_time = int(round(time.time() * 1000))  

    model = build_model(train_x, train_y,num_units=num_units,epochs=epochs, attention=attention_layer, model_verbose=model_verbose) 
    log('saving the TSInet model')
    if not verbose:
        print('Saving the TSInet model')
    save_model(model, model_type='tsinet',model_name='tsinet')
    save_model_data(train_x, model_type='tsinet', data_name ='train_x')
    save_model_data(train_y,  model_type='tsinet', data_name ='train_y')
    print('The model is saved in the models directory')
    log('Finished TSInet training...')
    if not verbose:
        print('Finished TSInet training...')


def process_args(args):
    global verbose
    global num_units 
    global epochs 
    global attention_layer 
    global normalize_data
    global model_verbose
    global data_train_validate_file_name
    
    if 'verbose' in args:
        verbose = boolean(args['verbose'])
    set_verbose(verbose)
    if 'num_units' in args:
        num_units = int( args['num_units'])
    if 'epochs' in args:
        epochs = args['epochs']
    if 'attention_layer' in args:
        attention_layer = boolean(args['attention_layer'])
        
    if 'normalize_data' in args:
        normalize_data = boolean(args['normalize_data'])
    if 'model_verbose' in args:
        model_verbose=int(args['model_verbose'])
    if 'training_file' in args:
        data_train_validate_file_name = args['training_file']
        
    if not os.path.exists(data_train_validate_file_name):
        print('\nTraining dataset does not exist:', data_train_validate_file_name,'. Please check the ReadMe file on how to download the artifacts.')
        sys.exit()

if __name__ == "__main__":
    '''
    Command line parameters parser
    '''
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-e", "--epochs", type=int, default=epochs,
        help="Number of epochs to train the network.")
    
    ap.add_argument("-a", "--attention_layer", required = False, default=True,
        help="Add the additional layer to generate more focused input and output. Default is True")
    
    ap.add_argument("-u", "--num_units", type = int, required = False, default= 400,
        help="The number of LSTM units to use during learning. Default is 400")
    
    ap.add_argument("-n", "--normalize_data", type = bool, required = False, default= True,
        help="Normalize the TSI data before processing, default is True.")
    
    ap.add_argument("-l", "--verbose", type = bool, required = False, default= False,
        help="Verbose level to print processing logs, default is False.")
    
    ap.add_argument("-v", "--model_verbose", type = int, required = False, default= 2,
        help="Verbose level to print processing logs, default is 2, one line per epoch. 1 is a progress bar for each epoch, and 0 is silent.")
    
    
    args = vars(ap.parse_args())
    
    parse_args(args)
    
    train_tsinet()