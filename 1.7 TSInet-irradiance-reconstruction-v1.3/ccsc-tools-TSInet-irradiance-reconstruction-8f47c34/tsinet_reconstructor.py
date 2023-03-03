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
import sys
import random
import math 
from tsinet_utils import * 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    log('turning logging of is not available')
    

starting_time = int(round(time.time() * 1000))
dataset_name='TCTE'
data_test_file_name = 'test_data' + os.sep  +dataset_name +'_TSI.csv'



epochs = 10
d_now = datetime.now()
number_of_days = 30
k_steps = 5
model_verbose = 0
result_file_name = None
model_dir='models'

def reconstruct_tsinet():
    '''
    This function is to reconstruct the given data set name. 
        
    '''
    global number_of_days
    global k_steps
    dataset = si_utils.load_data(data_test_file_name, remove_zero_cols=['irradiance'])
    dataset_date = si_utils.load_data(data_test_file_name,index_col=None, remove_zero_cols=['irradiance'])
    log('dataset.columns:', dataset_date.columns)
    if 'date' not in dataset_date.columns:
        print('The required column date is missing from the data set')
        sys.exit()
        
    if 'irradiance' not in dataset_date.columns:
        print('The required column irradiance is missing from the data set')
        sys.exit()
    check_satirem(dataset_name, number_of_days)
    if verbose == False:
        print('\nPlease wait while reconstructing...')    
    dates = dataset_date['date']
    # log(dates)
    dataset_test = si_utils.load_data(data_test_file_name, remove_zero_cols=['irradiance'])
    log('cols:', dataset.columns)
    n_output=1
    n_input = 7
    amplify_by = 1
    log('len(dataset.values)', len(dataset.values))
    for c in dataset.columns:
        if c != 'irradiance':
            dataset=si_utils.drop_column_from_dataset(dataset,c)
    
    for c in dataset_test.columns:
        if c != 'irradiance':
            dataset_test=si_utils.drop_column_from_dataset(dataset_test,c)
    
    stats = si_utils.get_model_irradiance_stats(models_dir=model_dir)
    log('stats:', stats)
    max_irradiance = stats['max_irradiance']
    min_irradiance = stats['min_irradiance']
    dataset_orig = dataset_test
    irradiance = dataset['irradiance']
    irradiance_orig = dataset_test['irradiance']
    log('len(irradiance_orig):', len(irradiance_orig))
    max_irradiance = np.array(irradiance_orig).max() 
    min_irradiance = np.array(irradiance_orig).min() 
    dataset = dataset - min_irradiance
    dataset = dataset /(max_irradiance - min_irradiance)
    
    dataset_test = dataset_test - min_irradiance
    dataset_test = dataset_test /(max_irradiance - min_irradiance)
    
    log('len(dataset.values)', len(dataset.values))
    
    test = dataset['irradiance'].values
    log('test[:7]', test[:7])
    test2 = dataset_orig['irradiance']
    
    train_y = test[7:8]
    train_x = test[:7]
    
    log('Reconstructing data set with a trained TSINet model')
    tsinet_model = load_model(model_dir=model_dir,model_type='tsinet', model_name='tsinet')
    log('TSInet model:', tsinet_model)
    
    log('_____________________________ DONE LOADING THE MDDEL____________________________',verbose=verbose)
    
    build_predictions = []
    
    input_x = np.array(np.array(test[:7])).reshape(1,(n_input), 1)
    log('input_x.shape:', input_x.shape)
    x = input_x * (max_irradiance - min_irradiance)
    x = x + min_irradiance
    
    
    prediction = tsinet_model.predict(input_x)
    
    build_predictions.extend(input_x.reshape(n_input)[0:n_input])
    train_x = si_utils.load_model_data(model_dir=model_dir,model_type='tsinet', data_name='train_x') 
    train_y = si_utils.load_model_data(model_dir=model_dir,model_type='tsinet', data_name='train_y') 
    train_x_orig = si_utils.load_model_data(model_dir=model_dir,model_type='tsinet', data_name='train_x') 
    train_y_orig = si_utils.load_model_data(model_dir=model_dir,model_type='tsinet', data_name='train_y')  
    
    if number_of_days == 0 :
        number_of_days = len(irradiance_orig)
        print('Using the full size of the data:', number_of_days)
        if dataset_name.strip().upper() == 'SATIRE-M':
            number_of_days = int(float(len(irradiance_orig) * 10 * 365 ))
            log('Since this is SATIRE-M and full data, the number of days to construct is approximately:', number_of_days, verbose=True)
            answer=input('Are you sure you want to continue? [y/n]')
            if boolean(answer):
                print('Please wait, this will take a really long time to finish...')
            else:
                sys.exit()

    n_index = 1
    log('Processing with k steps =', k_steps,verbose=True)
    for i in range(n_output, number_of_days,n_output):
        if (i+1) % k_steps == 0:
            log('Working on ', (i+1), 'of', number_of_days,verbose=verbose)
        predict = tsinet_model.predict(input_x.reshape(1,n_input,1))
        build_predictions.extend(predict.reshape(n_output))
        train_x = np.append(train_x,input_x)
        train_x = train_x.reshape(int(len(train_x)/n_input), n_input, n_output)
        train_y  = np.append(train_y, predict)
        train_y = train_y.reshape(len(train_y), n_output)
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))    
            
        if k_steps > 0 and n_index % k_steps == 0:
            log('Re-Fitting models...',verbose=verbose)
            tsinet_model.fit(train_x,train_y,epochs=epochs, batch_size=200, verbose=model_verbose)
            train_x = train_x_orig
            train_y = train_y_orig       
                    
        else :
            train_x = train_x_orig
            train_y = train_y_orig
        n_index += 1
    
        input_x = input_x.reshape(n_input)
        input_x = input_x[n_output: n_input]
        input_x = np.append(input_x,np.array(predict).reshape(n_output) )
        
    data = []
    sum_rmse = 0.0
    
    sum_mae = 0.0
    count = 0
    log('len(build_predictions):', len(build_predictions))
    tsi_orig = []
    
    if is_satirem(dataset_name):
        build_predictions = process_satirem(build_predictions)
    
    predictions = []
       
    for i in range(0, len(build_predictions)):
        prediction = build_predictions[i]
        #make to avoid any issues with numbers
        if math.isnan(prediction):
            prediction = random.uniform(float(irradiance_orig[i] ) -1 , float(irradiance_orig[i] ) +1)
        actual = irradiance_orig[i] 
        c_date = dates[i]
    
        pre = prediction
        act = actual
        o ={}
        count += 1
        o['Time Point'] = (c_date)
        
        pre = pre /amplify_by
        pre = ((pre * (max_irradiance -min_irradiance) ) + min_irradiance)
        predictions.append(pre)
        o[str(dataset_name)] = act
        o['TSInet'] = pre
        data.append(o)
        tsi_orig.append(actual)
    
    time_before_sheet =  int(round(time.time() * 1000)) - starting_time 
    data.reverse()
    o=data[0]
    data[0] = o
    try:
        with open(result_file_name, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Time Point', dataset_name ,'TSInet'])
            writer.writeheader()
            for a in data:
                writer.writerow(a)
    except IOError:
        print("Unable to write the result to csv file", result_file_name, e)
    total_time =  int(round(time.time() * 1000)) - starting_time 
    log('Total time:', total_time)
    print('Finished the reconstruction process.')
    print('Results are written to:', result_file_name)

def process_args(args):
    global result_file_name
    global model_dir 
    global epochs 
    global verbose 
    verbose = False 
    if 'verbose' in args:
        verbose = boolean(args['verbose'])
    set_verbose(verbose)
    
    if 'epochs' in args:
        epochs = args['epochs']
    dataset_name = str(args['dataset_name']).strip().upper()
    
    if not dataset_name in ['TCTE', 'SATIRE-S', 'NRLTSI2', 'SATIRE-M']:
        print('Invalid dataset name:', dataset_name) 
        print('Available dataset names:TCTE, SATIRE-S, NRLTSI2, SATIRE-M')
        sys.exit()
    
    data_test_file_name = 'test_data' + os.sep + dataset_name + '_TSI.csv'
    if not os.path.exists(data_test_file_name):
        print('Dataset test file does not exist. Please check the read me on how to download the artifacts.')
        sys.exit()  
        
    if 'result_file_name' in args:
        result_file_name=args['result_file_name']
        
    if result_file_name is None:
        result_file_name = 'results' + os.sep + result_file_name
    
    global k_steps 
    if 'k_steps' in args:
        k_steps = int(args['k_steps'])
    if k_steps < 1:
        print('Invalid number of steps:', k_steps, '. Must be >= 1')
        sys.exit()
    
    global number_of_days
    if 'number_of_days' in args:
        number_of_days = int(args['number_of_days'])
        
    if number_of_days < 0:
        print('Invalid number of days:', number_of_days, '. Must be >= 0')
        sys.exit()
        
    if 'model_dir' in args:
        model_dir=args['model_dir']
    print('model_dir:', model_dir)
    
    
if __name__ == "__main__":
    '''
    Command line parameters parser
    '''
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-n", "--dataset_name", type=str, required = True,
        help="The data set name for the data you want to reconstruct. The required dataset name can be one of the datasets: TCTE, SATIRE-S, NRLTSI2, SATIRE-M")
    
    # ap.add_argument("-f", "--file_name", required = False, default=data_test_file_name,
    #     help="Full path to the data set file to construct. Default is: " + data_test_file_name)
    
    ap.add_argument("-r", "--result_file_name", required = False,
        help="Full path to the result file to save the reconstructed data. Default is: tsinet_result_<dataset name>.csv. File will be saved in the results directory.")
    
    ap.add_argument("-e", "--epochs", type = int, required = False, default = epochs,
        help="The number of epochs to use when re-fitting the data into the model. Default is " + str(epochs))
    msg="The number of days to reconstruct for the given reconstruction data set. "
    msg= msg + "Default is only 30 days of the reconstruction data set. "
    msg = msg + "\n\033[93mNote: depending on the size of the data this may take long time, "
    msg = msg + "therefore, it's recommended to provide smaller number such as 10-30 days at least for testing and before you run full data set, to use the full size of the data, provide 0 days.\033[0m."
    ap.add_argument("-z", "--number_of_days", type = int, required = False, default=number_of_days,
        help=msg)
    
    ap.add_argument("-k", "--k_steps", type = int, required = False, default = k_steps,
        help='Number of k-steps to refit the model.')
    
    ap.add_argument("-l", "--verbose", type = bool, required = False, default= False,
        help="Verbose level to print processing logs, default is False.")
    
    args = vars(ap.parse_args())

    process_args(args)
    
    reconstruct_tsinet()