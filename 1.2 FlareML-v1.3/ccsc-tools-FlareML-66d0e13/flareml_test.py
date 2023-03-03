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

import numpy as np
import os
import csv 
from datetime import datetime
import argparse
import time 
from sklearn.metrics import confusion_matrix

from flareml_utils import *

TEST_INPUT = 'data/test_data/flaringar_simple_random_40.csv'
normalize_data = False 

def test_model(args):
    pm = {}
    if not 'algorithm' in args.keys():
        args['algorithm'] = 'ENS'
    algorithm = args['algorithm']
    if not algorithm.strip().upper() in algorithms:
        print('Invalid algorithm:', algorithm, '\nAlgorithm must one of: ', algorithms)
        sys.exit()
    TEST_INPUT = args['test_data_file']
    if TEST_INPUT.strip() == '':
        print('Testing data file can not be empty')
        sys.exit() 
    if not os.path.exists(TEST_INPUT):
        print('Testing data file does not exist:', TEST_INPUT)
        sys.exit()
    if not os.path.isfile(TEST_INPUT):
        print('Testing data is not a file:', TEST_INPUT)
        sys.exit()
    modelid = args['modelid']
    if modelid.strip() == '':
        print('Model id can not be empty')
        sys.exit()
    verbose = False;
    if 'verbose' in args:
        verbose = boolean(args['verbose'])
    
    set_log_to_terminal(verbose)

    log('Your provided arguments as: ', args)

    models_dir = custom_models_dir
    alg = algorithm.strip().upper()
    printOutput = False
    
    default_id_message = ' or use the default model id.'
    if modelid == 'default_model' :
        models_dir = default_models_dir
        default_id_message =   '.'
    exists = are_model_files_exist(models_dir , modelid, alg=alg)
    log('model exists:', exists, 'for:', modelid, 'and', alg)
    log('partial_ens_trained:', get_partial_ens_trained())
    
    if not exists:
        if not  get_partial_ens_trained():
            log("\nModel id", modelid," does not exist for algorithm " + alg + "." + '\nPlease make sure to run training task with this id first' + default_id_message, logToTerminal=True, no_time=True)
        sys.exit()
            
    log("=============================== Logging Stared using algorithm: " + algorithm +" ==============================")
    log("Execution time started: " + timestr)
    log("Log files used in this run: " + logFile)
    log("train data set: " + TEST_INPUT)
    log("Creating a model with id: " + modelid)
    print("Starting testing with a model with id:",  modelid, 'testing data file:', TEST_INPUT)
    print('Loading data set...')
    dataset = load_dataset_csv(TEST_INPUT)
    log("orig cols: " + dataset.columns)
    for c in dataset.columns:
        if not c in req_columns:
            dataset = removeDataColumn(c,dataset)
    log("after removal cols: " , dataset.columns)
    print('Done loading data...')
    cols = list(dataset.columns)
    if not flares_col_name in cols:
        print('The required flares class column:', flares_col_name, ' is not included in the data file')
        sys.exit()
    print('Formatting and mapping the flares classes..')
    dataset['flarecn'] = [convert_class_to_num(c) for c in dataset[flares_col_name]]
    log('all columns: ', dataset.columns) 
    log('\n', dataset.head())
    flares_names = list (dataset[flares_col_name].values)
    dataset = removeDataColumn(flares_col_name, dataset)
    log("after removal cols: " + dataset.columns)
    cols = list(dataset.columns)
    if normalize_data:
        log('Normalizing and scaling the data...')
        for c in cols:
            if not c =='flarecn':
                dataset[c] = normalize_scale_data(dataset[c].values)
                    
    
    test_y = dataset['flarecn']
    test_x = removeDataColumn('flarecn',dataset)

    print('Prediction is in progress, please wait until it is done...')
    true_y = [mapping[y] for y in test_y]
    if alg in ['RF','ENS']:
        rf_result = model_prediction_wrapper('RF',None, test_x, test_y, model_id = modelid)
    
    if alg in ['MLP','ENS']:
        mlp_result = model_prediction_wrapper('MLP',None, test_x, test_y, model_id = modelid)

    if alg in ['ELM','ENS']:
        elm_result = model_prediction_wrapper('ELM',None, test_x, test_y, model_id = modelid)
    
    if alg == 'ENS':
        result = compute_ens_result(rf_result, mlp_result, elm_result)
        pm ['ENS'] = log_cv_report(true_y,result)

        rf_result = map_prediction(rf_result)
        pm['RF'] = log_cv_report(true_y,rf_result)
        
        mlp_result = map_prediction(mlp_result)
        pm['MLP'] = log_cv_report(true_y,mlp_result)
        
        elm_result = map_prediction(elm_result)
        pm['ELM'] = log_cv_report(true_y,elm_result)
        pm = check_pm_precision('ENS','RF','B',pm)
        pm = check_pm_precision('ENS','RF','C',pm)
        
    elif alg == 'RF':
        result = map_prediction(rf_result)
        pm['RF'] = log_cv_report(true_y,result)
    elif alg == 'MLP':
        result = map_prediction(mlp_result)
        pm['MLP'] = log_cv_report(true_y,result)
    else:
        result = map_prediction(elm_result)
        pm['ELM'] = log_cv_report(true_y,result)
    log_cv_report(true_y,result)


    print('Finished the prediction task..')
    # return pm
    res = {}
    res[alg] = pm[alg]
    res["alg"] = alg
    res['result']  = pm
    return  res
                    
'''
Command line parameters parser
'''
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test_data_file',default=TEST_INPUT, help='full path to a file includes test data to test/predict using a trained model, must be in csv with comma separator')
parser.add_argument('-l', '--logfile', default=logFile,  help='full path to a file to write logging information about current execution.')
parser.add_argument('-v', '--verbose', default=False,  help='True/False value to include logging information in result json object, note that result will contain a lot of information')
parser.add_argument('-a',  '--algorithm', default='ENS',  help='Algorithm to use for training. Available algorithms: ENS, RF, MLP, and ELM. \nENS \tthe Ensemble algorithm is the default, RF Random Forest algorithm, \nMLP\tMultilayer Perceptron algorithm, \nELM\tExtreme Learning Machine.')
parser.add_argument('-m', '--modelid', default='default_model', help='model id to save or load it as a file name. This is to identity each trained model.')
parser.add_argument('-n', '--normalize_data', default=normalize_data, help='Normalize and scale data.')

args, unknown = parser.parse_known_args()
args = vars(args)

if __name__ == "__main__":
    pm=test_model(args)
    plot_result(pm)


