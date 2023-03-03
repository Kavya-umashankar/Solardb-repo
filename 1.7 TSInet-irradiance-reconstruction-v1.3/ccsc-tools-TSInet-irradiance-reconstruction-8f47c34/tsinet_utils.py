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
import os 
import sys 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
py_version_info = sys.version_info
py_vers = str(py_version_info[0]) + '.' + str(py_version_info[1]) +'.'+ str(py_version_info[2])
py_vers_2 = str(py_version_info[0]) + '.' + str(py_version_info[1])
print('Python version:', py_vers)
tf_version = tf.__version__
print('Tensorflow backend version:',tf_version )

def boolean(b):
    if b == None:
        return False 
    b = str(b).strip().lower()
    if b in ['y','yes','ye','1','t','tr','tru','true']:
        return True 
    return False


from math import sqrt
from numpy import split
from numpy import array
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import MaxPooling2D
from keras_self_attention import SeqSelfAttention
from tensorflow.keras import Sequential

from pathlib import Path
from os import listdir
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import csv 
from datetime import datetime
import argparse
import pickle 
import re 
from scipy.stats import wilcoxon
import time
from contextlib import contextmanager
import sys
from os import path
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md
import matplotlib.ticker as mticker

import tsinet_utils as si_utils

verbose = True
frame_size = 7

model_verbose = 2
batch_size = 16
cnn_type = 'conv2d'
factor = 0.4
model_id = None

save_stdout = sys.stdout

@contextmanager
def stdout_redirected(new_stdout):
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout

@contextmanager
def stdout_default():
    sys.stdout = save_stdout



def log(*message, end=' ', verbose=False):
    if verbose:
        for m in message:
            print(m,end=end)
        print('')
    logFile='logs/tsinet.log'
    with open(logFile,"a+") as logFileHandler :
        with stdout_redirected(logFileHandler) :
            print ('[' + str(datetime.now().replace(microsecond=0))  +'] ',end=end)
            for msg in message:
                print (msg,end=end)  
            print('')
                    
def set_verbose(ver):
    global verbose
    verbose = ver
    
def save_model(trained_model, model_dir='models', model_type='tsinet', model_name='tsinet'):
    if model_name == None or  model_name == '':
        model_name = 'tsinet'
    if str(model_name).startswith('_'):
        model_name = model_name[1:]
    file_ext = '.sav'
    if int(tf_version[0]) > 1 :
        file_ext = ''
        
    model_file = model_dir + os.sep +  model_name + "_model" + file_ext
    
    if model_type == 'tsinet' :
        log("saving model with save function to file: " , model_file)
        trained_model.save(model_file)
    else :    
        log("saving model with pickle to file: " , model_file)
        pickle.dump(trained_model, open(model_file, 'wb'))
    
    return model_file
def load_model( model_dir='models', model_type='tsinet', model_name='tsinet'):
    if model_name == None or  model_name == '':
        model_name = 'tsinet'
    file_ext = '.sav'
    if int(tf_version[0]) > 1 :
        file_ext = ''
        
    model_file = model_dir + os.sep +  model_name + "_model" + file_ext
    default_model_file = 'default_model' + os.sep +  model_name + "_model" + file_ext      
    log("Loading model file: " + model_file,verbose=True)
    loading_file_name = model_file
    if is_file_exists(model_file) :
        loading_file_name = model_file
    elif is_file_exists(default_model_file):
        log('Model was not found, trying the default model',verbose=True)
        loading_file_name = default_model_file
    else:
        print('\033[93m','\n\t\tERROR: No model found to reconstruct the data set, please train a model first using tsinet_train','\033[0m')
        sys.exit()
    log('Reconstruction will be performed using the model file:', loading_file_name, verbose=True)
    model = keras.models.load_model(loading_file_name, custom_objects=SeqSelfAttention.get_custom_objects())
   
    log("Returning loaded model from file:", loading_file_name)
    return model  
def save_model_data(trained_model_data, model_dir='models', model_type='tsinet', data_name='train_x'):
    log('model_type:', model_type, 'data_name:', data_name)
    model_file = model_dir + os.sep+  model_type + "_" + data_name +".sav"
    log("saving model training data with pickle", model_id ,  " to file: " , model_file)
    pickle.dump(trained_model_data, open(model_file, 'wb'))
        
    return model_file

def save_model_objects(trained_model_data, model_dir='models', model_type='tsinet', data_name='train_x'):
    log('model_type:', model_type, 'data_name:', data_name)
    model_file = model_dir + os.sep+  model_type + "_" + data_name +".sav"
    log("saving model training data with pickle", model_id ,  " to file: " , model_file)
    pickle.dump(trained_model_data, open(model_file, 'wb'))
    
def load_model_data(model_dir='models', model_type='tsinet', data_name='train_x'):
    model_file = model_dir + os.sep+  model_type + "_" + data_name +".sav"
    log('Check model data file:', model_file)
    if not isfile(model_file):
        log('Model and or its objects are not found, trying the default model..', verbose=False)
        model_file = 'default_model/' + model_type + "_" + data_name +".sav"
        if not os.path.exists(model_file):
            log('Mode and or its objects are not found in default_model directory. Please download the default model from our github', verbose=True)
            sys.exit()
    log("loading model objects with pickle", model_id ,  " from file: " , model_file)
    return pickle.load(open(model_file, 'rb'))
        
def split_dataset(data, train_split=0.9, n_output=7, split_data=True, as_is=False):
    log('data to split length:', len(data))
    n_shifts = len(data) % n_output 
    if n_shifts > 0: 
        data = data[:-n_shifts]
        log('data to split length after shift:', len(data))
    # split into standard weeks
    if as_is:
        train = array(split(data, len(data)/n_output))
        return train, None
    split_size = int(len(data) * 0.9) 
    log('1- split_size:', split_size)    
    split_size = split_size - (split_size % n_output)
    log('2- split_size:', split_size)
    train, test = data[0:split_size], data[split_size:len(data)]
    log('train size:', len(train)) 
    log('test  size:', len(test))
    if split_data:
        log('len(data) 1:', len(train))
        train = array(split(train, len(train)/n_output))
        log('len(data) 2:', len(train))
        test = array(split(test, len(test)/n_output))
        log('-------------------')
    return train, test

def to_supervised(train, n_input, n_out=7):
    # flatten data
    log('train.shape:', train.shape, '\nn_input:', n_input,'\nn_out:',n_out)
    log('(train.shape[0]*train.shape[1], train.shape[2])', (train.shape[0]*train.shape[1], train.shape[2]))
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    log('data.shape:', data.shape)
    X, y = list(), list()
    in_start = 0
    log('len(data):', len(data))
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
#         log('in_end:', in_end, 'out_end:', out_end)
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))

            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        in_start += 1
    t_X, t_Y = array(X), array(y)

    return t_X, t_Y



def is_file_exists(file):
    path = Path(file)
    log("Check if file exists: " + file + " : " + str(path.exists()))
    return path.exists()

def get_max_model_id(models_dir='models'):
    files = listdir(models_dir)
    nums = []
    for f in files:
        f = f.strip()
        f = re.sub('[^0-9]','', f)
        f = int(f) 
        nums.append(f)
    log('files:', files)
    log('nums:', nums)
    nums = np.array(nums) 
    if len(nums) > 0:
        return nums.max()
    return 0
def get_model_irradiance_stats(file_prefix='irradiance_max_min', models_dir='models'):
    file_name= models_dir + '/'  + file_prefix  +'.txt'
    if not os.path.exists(file_name):
        log('required file does not exist:', file_name, ', using default model file: default_model/'  + file_prefix  +'.txt')
        file_name = file_name=  'default_model/'  + file_prefix  +'.txt'
        if not os.path.exists(file_name):
            print('required file does not exist. Please make sure to download the required files from our github or retrain the model.')
            sys.exit()
    handler = open(file_name,'r')
    o={}
    for l in handler:
        l = l.strip()
        tokens = l.split(':')
        o[tokens[0].strip()] = float( tokens[1].strip())
    log('o', o)
    
    handler.close()
    return o

def save_model_irradiance_stats(max_irradiance, min_irradiance,  file_prefix='irradiance_max_min', models_dir='models'):
    file_name= models_dir + '/' + file_prefix  +'.txt'
    handler = open(file_name, 'w') 
    handler.write('max_irradiance:' + str(max_irradiance) + '\nmin_irradiance:' + str(min_irradiance) + '\n')
    handler.flush()
    handler.close()


def save_prediction_result(predictions, file_name=None, result_data_dir='results/', result_file_name_option=''):
    index  = 0
    d_now = datetime.now()
    if file_name is None:
        file_name = 'result' +   str(result_file_name_option)  + '_'   + str(d_now.day) + '' + str(d_now.strftime('%b')) + ''+str(d_now.year) + ".xlsx"
    chart_excel_file = result_data_dir + file_name
    writer = pd.ExcelWriter(chart_excel_file, engine='xlsxwriter', datetime_format='d-mmm-yyyy',
                        date_format='d-mmm-yyyy')
    sum = 0.0
    data=[]
    if normalize_data:
        log('adding normalization back to array...')
        test_y = test_y / num_value   
        test_y = test_y * (max_irradiance - min_irradiance)
        test_y  = test_y + min_irradiance

    for prediction in predictions:
        pre = float(prediction[0][0])
        if normalize_data:
            pre = pre / num_value
            pre = pre * (max_irradiance - min_irradiance)
            pre  = pre + min_irradiance 
        
        o = {}
        o['Date'] = date_data[test_date_starting_index]
        test_date_starting_index=test_date_starting_index+1  
        o['TSInet'] = pre                    
        ac = float(test_y[index])
        o['Actual'] = ac 
         
                                    
        dif = float(abs(pre-ac))        
        perc = float((dif / ac) * 100)        
        po = pow(dif,2)                      
            
        sum = sum + po
        index = index + 1
        data.append(o)
    o = data[0]
    d = sum / float(len(test_y)) 
    sq_r = sqrt(d)
    o['MSE'] = sq_r 
    data[0] = o

    sheet='MSE_Result'
    
    writer = write_sheet(writer, sheet, data)
    log('Saving result to excel file:', chart_excel_file)
    writer.save()
    
def write_sheet(writer, sheet, data, add_linear_reg=False, dataset_name='TCTE'):
    df = pd.DataFrame.from_records(data)
    df.to_excel(writer, sheet_name=sheet, index=False,columns=data[0].keys())
    workbook  = writer.book
    worksheet = writer.sheets[sheet]
    
    fmt = workbook.add_format({'align':'center'})
    
    worksheet.set_column(0, len((data[0].keys())), 18,fmt)
    workbook.add_format({'align': 'center'})
    chart = workbook.add_chart({'type': 'line'})
    
    # # Configure the series of the chart from the dataframe data.
    chart.add_series({
    'name' : dataset_name,
    'categories': '=\'' +sheet+'\'!$A2:$A'+str(len(data) + 1),
    'values':     '=\''+sheet +'\'!$B$2:$B$'+str(len(data) + 1),
    })

    chart.add_series({
    'name' : 'TSInet',
    'categories': '=\'' +sheet+'\'!$A2:$A'+str(len(data) + 1),
    'values':     '=\''+sheet +'\'!$C$2:$C$'+str(len(data) + 1),
    })            
    chart.set_x_axis({'name': 'Time Point'})
    chart.set_y_axis({'name': 'TSI (W/m^2)'})
    
    chart.set_legend({'position': 'top'})
    # # Insert the chart into the worksheet.
#     worksheet.insert_chart('A4', chart,{'x_offset': 4,'x_scale': 2.5, 'y_scale': 1})
    return writer

def update_zero_data(dataset,  size,start_index = 0 ):
    log('Updating data with zeros with the mean value of each column')
    for i in range(start_index, size):
        c_data = dataset[:,i]
        mean = c_data[c_data > 0].mean()
        c_data[c_data == 0] = mean
        dataset[:,i] = c_data
    return dataset 

  
def load_data(dataset_file, header=0,parse_dates=['date'],
              index_col = ['date'] ,
              sort_col='date', 
              inplace=True, 
              ascending=False,
              remove_zero_cols=['sunspot'],
              date_parser = lambda x: pd.datetime.strptime(x, "%m-%d-%Y")):
    
    dataset = pd.read_csv(dataset_file, 
                          header=0, 
                          infer_datetime_format=True, 
                          parse_dates=parse_dates, 
                          index_col=index_col)
    dataset.sort_values(by='date',inplace=inplace, ascending=ascending)
    if remove_zero_cols != None and len(remove_zero_cols) > 0:
        for c in remove_zero_cols:
            dataset = dataset[dataset[c] > 0]
    dataset = dataset.reset_index(drop=True)
    
    return dataset 

def remove_zero_data(dataset, col):
    if col in dataset.columns:
        dataset= dataset[dataset[col] != 0]
        dataset = dataset.reset_index(drop=True)
    return dataset 

def drop_column_from_dataset(dataset, col):
    if col in dataset.columns:
        log('Removing', col)
        dataset = dataset.drop(col, axis=1)
    return dataset 


def split_train_test_dataset(data, test_size=0):
    global test_date_starting_index
    '''
    Split the data into train and test data sets. The test is test_size % and the train is 100-test_size %
    The default test_size is 10% so that it's train:test = 90:10%
    '''
    if test_size == None or test_size == 0:
        test_size = 0
    data_size = len(data) 
    log('data_size', data_size)
    test_data_size = int((data_size * test_size))
    
    log('test data size:',test_data_size )
    
    train_data_size = data_size - test_data_size 
    train_data_size = train_data_size + (data_size - (train_data_size + test_data_size))
    log('train data size:', train_data_size)
    
    train_index = train_data_size
    test_index = train_data_size
    test_date_starting_index = test_index
    log('test_data starting index:', test_index)
    
    train, test = data[:train_index], data[train_index-1:]
    # restructure into windows of weekly data
    train = array(train)
    test = array(test)
    return train, test    

def create_time_frames(dataset, input_col='irradiance', output_col='irradiance', frame_size=7, n_output=1):  
    if frame_size <= 0:
        raise Exception('Invalid frame_size, should be >= 1. The value of frame_size was: {}'.format(frame_size))
    log('dataset size:', len(dataset), 'frame_size:', frame_size)
    input_data = dataset[input_col]
    output_data = dataset[output_col]
    X, y = list(), list()
    in_start = 0
    out_index = frame_size
    log('len(data):', len(dataset))
    for _ in range(len(dataset) - frame_size):
        in_end = in_start + frame_size
        out_end = in_start
        x_input = []
        if out_end <= len(dataset) - (frame_size+n_output):
            for i in range(in_start, in_end):
                x_a = []
                x_a.append(output_data[i])
                x_input.append(x_a)
            x_input = np.array(x_input)
            x_input = x_input.reshape((len(x_input), 1))
            
            X.append(x_input)
            '''
            keeping the same shape
            '''
            y_a = np.zeros(n_output)
            y_a = np.array(output_data[out_index:out_index + n_output])
            y.append(y_a)
        out_index = out_index + 1
        in_start += 1
    train_X, train_Y = array(X), array(y)
    
    return train_X, train_Y 

def build_model(train_x, train_y,attention=True, num_units=400,epochs=10, cnn=True, save_data=False, model_verbose=model_verbose):
    log('train_x.shape', train_x.shape)
    log('train_y.shape', train_y.shape)
    filters = 128
    k_size=2
    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    log('n_timesteps, n_features, n_outputs', n_timesteps, n_features, n_outputs)          
    model = Sequential()
    lstm_units = 10
    if cnn:
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        log('train_x.shape inside if cnn', train_x.shape)
        log('train_y.shape inside if cnn', train_y.shape)
        log('input_shape=(n_timesteps,n_features):', (n_timesteps,n_features))
        if save_data: 
            save_model_data(train_x, model_id, model_type='tsinet', data_name ='train_x')
            save_model_data(train_y, model_id, model_type='tsinet', data_name ='train_y')  
                
        model.add(Conv1D(filters=filters, kernel_size=k_size, activation='relu', 
                         input_shape=(n_timesteps,n_features)))
        model.add(Conv1D(filters=filters*2, kernel_size=k_size, activation='relu'))
        model.add(Conv1D(filters=filters*4, kernel_size=k_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=2, strides=1))
             
        model.add(Flatten())
        model.add(RepeatVector(n_outputs))
         
        model.add(LSTM(lstm_units, activation='relu', return_sequences=True))
    else :
        model.add(LSTM(lstm_units, activation='relu', input_shape=input_shape))
    if attention:
        log('Adding the attention layer...')
        model.add(SeqSelfAttention(attention_activation='relu'))
    model.add(TimeDistributed(Dense(num_units, activation='relu')))
    model.add(TimeDistributed(Dense(num_units, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
         
    opt = Adam(lr=1e-3, decay=1e-3 / 400)
    model.compile(loss='mse', optimizer=opt)
    log('model.summary')
    if verbose:
        model.summary()
    log('model verbose:', model_verbose)
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=model_verbose)
    return model

    
def fit_model(model, train_x, train_y, num_units=400,epochs=10, model_verbose=model_verbose):
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=model_verbose)
    return model;
 
def rmse(y_true, y_pred):
    a = np.array(y_pred) - np.array(y_true)
    a = np.square(a)
    a = a.sum()
    a  = a / len(y_true)
    a = np.sqrt(a)
    return a

def stats_test(y_true,y_pred):
    w, p = wilcoxon(y_true, y_pred)
    return f"{p:.9f}"

def normalize_data(d):
    max = np.array(d).max()
    min = np.array(d).min()
    d= (d - min)/(max - min)
    return d
def denormalize_data(d,max,min):
    return (((d * (max -min) ) + min))

def save_result_to_excel(data,file_name,dataset_name):    
    writer = pd.ExcelWriter(chart_excel_file, engine='xlsxwriter', datetime_format='d-mmm-yyyy',
                        date_format='d-mmm-yyyy')
    writer = write_sheet(writer, sheet, data, dataset_name=dataset_name)
    log('Saving result to excel file:', file_name)
    writer.save()

def is_satirem(dataset_name):
    return dataset_name.strip().lower() in ['satirem','satire-m','m-satire']

def check_satirem(dataset_name, number_of_days):
    if is_satirem(dataset_name):
        if number_of_days == 0:
            print('\033[93mYou are reconstructing SATIRE-M data set with a lot of data to reconstruct which may take extremely long time.'+
                  '\nIt is recommended to use a smaller number of days unless you are running the program in a powerful GPU machine. \nAre you sure you want to continue?[y|n]\033[0m')
            answer = input()
            if not boolean(str(answer)):
                sys.exit() 
            return             
        if number_of_days < 365 * 10:
            print('\033[91mYou are reconstructing SATIRE-M data set which requires large number of days for each entry in SATIRE-M, it requires at least:', (365*10),' number of days reconstruction.\nYou must provide larger number of days or 0 for full dataset size from the file.\033[0m')
            sys.exit()
        else:
            if number_of_days % 365 * 10 != 0:
                print('\033[93mYou are reconstructing SATIRE-M data which requires multiples of 3650 days (10 years), the value you entered is not valid:', number_of_days, '\033[0m')
                sys.exit()
            print('\033[93mYou are reconstructing SATIRE-M data set with a lot of data to reconstruct which may take extremely long time. \nAre you sure you want to continue?[y|n]\033[0m')
            answer = input()
            if not boolean(str(answer)):
                sys.exit() 
            

def plot_figures(result_file):
    data = pd.read_csv(result_file)
    dates = data[data.columns[0]]
    tsinet = data[data.columns[2]]
    dataset = data[data.columns[1]]
    dataset_name = data.columns[1]
    figsize=(8, 3)
    fig, ax = plt.subplots(figsize=figsize)
    dates = [parse_date(d) for d in dates]
    ax.plot(dates, dataset,color='blue')
    ax.plot(dates, tsinet,
             color='orange',   
             linewidth=1.0,  
             linestyle='solid' 
            )
    plt.ylabel('TSI')
    plt.xlabel('Time')
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    nums = len(dates)//4
    if dataset_name == 'TCTE':
        dates = [str(d).replace(' 00:00:00','').strip() for d in dates]
        yticks = [1358,1359,1360,1361,1362,1363,1364]
        ax.yaxis.set_ticks(yticks)
    
    xfmt = md.DateFormatter('%m/%d/%y')
    ax.xaxis.set_major_formatter(xfmt)
    
    ax.set_xticklabels(('',dates[0],dates[nums], dates[nums*2], dates[nums*3],dates[len(dates)-1]))
    print('\n\n*NOTE: the figure scale is large compared to the difference between the observed and predicted values, therefore you may see a big difference, but in actuality, the absolute difference between the values are very small.')
    plt.show()

def parse_date(d):
    d=str(d)
    return datetime.strptime(d, "%Y-%m-%d %H:%M:%S")   
def process_satirem(predictions):
    print('number of predictions points:', len(predictions))
    a = np.array_split(np.array(predictions), int(float(len(predictions)/(365*10))))
    s_predictions = []
    for d in a:
        s_predictions.append(np.average(d))
    
    return s_predictions

def create_default_dirs():
    for d in ['default_model', 'models', 'logs', 'test_data','train_data', 'results','reconstructed_tsinet']:
        if not os.path.exists(d) :
            os.mkdir(d)

create_default_dirs()  