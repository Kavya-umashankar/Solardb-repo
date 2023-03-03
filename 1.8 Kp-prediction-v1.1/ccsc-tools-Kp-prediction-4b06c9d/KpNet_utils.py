from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
import tensorflow
import os 
import numpy as np 
import pandas as pd 
import sys
import csv
import pandas as pd 
import sys
import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.ticker as mticker
import numpy as np 
import sys 
import os 
import matplotlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime, timedelta
import argparse
import time
from time import sleep
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error
import math
from math import sqrt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import random
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from tensorflow.keras import layers, models
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
verbose = False 
tfd = tfp.distributions

columns_names = ['ScalarB', 'BX_GSE_GSM', 'BY_GSE', 'BZ_GSE', 'BY_GSM', 'BZ_GSM', 'SW_Plasma_Temperature','SW_Proton_Density', 'SW_Plasma_Speed', 'Flow_pressure']
kp_col = 'Kp_index'
fill_values = [999.9, 999.9, 999.9, 999.9, 999.9, 999.9, 999.9, 9999999, 999.9, 9999, 999.9, 999.9, 99.99, 999.99]
fill_values = [999.9, 999.9, 999.9, 999.9, 999.9, 999.9, 9999999, 999.9, 9999, 999.9, 999.9, 99.99, 999.99]

c_date = datetime.now()

t_window = ''
d_type = ''
data_dir = 'data'
file_prefix = 'omniweb_kp_data_'
log_handler = None
interval_type = 'hourly'
interval_type = 'daily'


def create_log_file(dir_name='logs'):
    global log_handler
    try:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, True)
        log_file = dir_name + '/run_' + str(c_date.month) + '-' + str(c_date.day) + '-' + str(c_date.year) + '.log'
    except Exception as e:
        print('creating default logging file..')
        log_file = 'logs/run_' + str(c_date.month) + '-' + str(c_date.day) + '-' + str(c_date.year) + '.log'
    log_handler = open(log_file, 'a')
    sys.stdout = Logger(log_handler)  
    # log('')
    # log('********************************  Executing Python program:', sys.argv[0].split(os.sep)[-1], '  ********************************')         


def boolean(b):
    b = str(b).lower() 
    if b[0] in ['t','1','y']:
        return True 
    return False 
def set_logging(dir_name='logs'):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, True)    
    log_file = dir_name + os.sep + 'kp_run.log'
    global log_handler
    if os.path.exists(log_file):
        l_stats = os.stat(log_file)
        # print(l_stats)
        l_size = l_stats.st_size
        # print('l_size:', l_size)
            
        if l_size >= 1024 * 1024 * 50:
            files_list = os.listdir('logs')
            files = []
            for f in files_list:
                if 'solarmonitor_html_parser_' in f:
                    files.append(int(f.replace('logs', '').replace('/', '').replace('kp_run_', '').replace('.log', '')))
            files.sort()
            # print(files)
            if len(files) == 0:
                files.append(0)
            os.rename(log_file, log_file.replace('.log', '_' + str(files[len(files) - 1] + 1) + '.log'))
            log_handler = open(log_file, 'w')
        else:
            log_handler = open(log_file, 'a')
    else:
        log_handler = open(log_file, 'w')
    # print('log_handler:', log_handler)

    
class Logger(object):

    def __init__(self, logger):
        self.terminal = sys.stdout
        self.log = logger

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass  


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

  
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis,:],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)    


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis,:]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def get_d_str(t):
    y = str(t.year)
    m = str(t.month)
    if len(m) == 1:
        m = '0' + m
    d = str(t.day)
    if len(d) == 1:
        d = '0' + d
    return str(t.year) + '-' + m + '-' + d 


def truncate_float(number, digits=4) -> float:
    try:
        if math.isnan(number):
            return 0.0
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper
    except Exception as e:
        return number


def set_verbose(b):
    global verbose
    verbose = b

    
def log(*message, verbose=False, end=' '):
    global log_handler
    if True:
        if verbose:
            print('[' + str(datetime.now().replace(microsecond=0)) + '] ', end='')
        log_handler.write('[' + str(datetime.now().replace(microsecond=0)) + '] ')
        for m in message:
            if verbose:
                print(m, end=end)
            log_handler.write(str(m) + ' ')
        if verbose:
            print('')
        log_handler.write('\n')
        
    log_handler.flush()

    
def clean_line(l):
    r = []
    for s in l:
        s = str(s).strip() 
        if s.endswith('.'):
            s = s[:-1]
        r.append(s) 
    return r


def get_clean_data(file_name, save_to_file=True, file_to_save='data' + os.sep + 'omniweb_kp_data.csv'):
    print('loading from file:', file_name)
    h = open(file_name, 'r')
    data = [ clean_line(str(l).strip().split()) for l in h if str(l).strip() != '']
    
    cols = data[0]
    matched_data = []
    print(cols)
    for d in data:
        if len(d) != len(cols):
            print(False, d)
        else:
            matched_data.append(d)
        
    print('sizes:', len(data), len(matched_data))
    if save_to_file:
        np.savetxt(file_to_save,
           matched_data,
           delimiter=",",
           fmt='% s')
    return matched_data


def convert_year_day_hour_to_date(year, day, hour=None, debug=False):
    day_num = str(day)
    if debug:
        log('The year:', year , 'day number :', str(day_num), 'hour:', hour)
      
    day_num.rjust(3 + len(day_num), '0')
      
    # Initialize year
    year = str(year)
      
    # converting to date
    if hour is not None:
        if debug:
            log("the hour number:", hour)
        res = datetime.strptime(year + "-" + day_num, "%Y-%j") + timedelta(hours=int(hour))
        if debug:
            log("type(res):" , type(res))
    else:
        res = datetime.strptime(year + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")
      
    # printing result
    if debug:
        log("Resolved date : " + str(res), 'from:', year, ' ', day, ' ', hour)
    return res


def preprocess_data(data,
                    d_type='h',
                    file_prefix ='omniweb_kp_data_full_timestamp',
                    training_years=[y for y in range(1969, 2021)],
                    test_years=[2021]):
    # columns_names =['ScalarB','BZ_GSM', 'SW_Plasma_Temperature','SW_Proton_Density', 'SW_Plasma_Speed', 'Flow_pressure','Elecrtric_field']
    year_data = list(data['YEAR'].values)
    day_data = list(data['DOY'].values)
    hr_data = list(data['HR'].values)
    dates = []
    dates_str = []
    for i in range(len(day_data)):
        y = year_data[i]
        m = day_data[i]
        h = hr_data[i]
        y_to_data = convert_year_day_hour_to_date(y, m, h)
        dates.append(y_to_data)
        dates_str.append(str(y_to_data.year) + '-' + str(y_to_data.month) + '-' + str(y_to_data.day) + '-' + str(y_to_data.hour))
    data.insert(0, 'Timestamp', dates_str)
    prev_date = dates[0]
    dates_diff = []
    for i in range(1, len(dates) - 1):
        cur_date = dates[i]
        dif = cur_date - prev_date 
        dif = dif.total_seconds() / (60 * 60)
        # print('first:', prev_date, cur_date, dif)
        if dif > 1:
            dates_diff.append([i, str(prev_date), str(cur_date), dif])
        prev_date = cur_date
    for d in dates_diff:
        print(d)
    if 'index' in data.columns:
       data = data.drop('index', axis=0)
    print('Number of missing entries:', len(dates_diff))
    # training_orig = data.loc[data['YEAR'].isin(training_years) ]
    # testing_orig = data.loc[data['YEAR'].isin(test_years) ]
    # training1 = training[:]
    # kp_vals = list(training1[kp_col].values)
    # print('kp_vals before: ' , len(kp_vals))
    # kp_vals = kp_vals[n_periods:]
    # print('kp_vals after:', len(kp_vals))
    # training1.drop(training1.tail(n_periods).index,
    #     inplace = True)
    # print('training 1 len:' ,len(training1))
    # training1[kp_col] = kp_vals 
    # training1.to_csv('data/training1.csv')
    data = data.reset_index()
    log('data to stor has columns:', data.columns)
    columns = ['Timestamp']
    columns.extend(['YEAR', 'DOY', 'HR'])
    columns.extend(columns_names)
    columns.append(kp_col)
    
    log('Saving to file:', 'data' + os.sep + file_prefix + '_' + str(d_type) + '.csv')
    data.to_csv('data' + os.sep + file_prefix + '_' + str(d_type) + '.csv', index=False, columns=columns)
    # for i in range(1, 10):
    #     training = training_orig[:]
    #     testing = testing_orig
    #     training[kp_col] = training[kp_col].shift(-1 * i)
    #     training = training.dropna()
    #
    #     testing[kp_col] = testing[kp_col].shift(-1 * i)
    #     testing = testing.dropna()
    #
    #     # data.to_csv('data/test.csv')
    #     training = training.reset_index()
    #     testing = testing.reset_index()
    #     if 'index' in training.columns:
    #         drop_columns(data, 'index')
    #     if 'index' in testing.columns:
    #         drop_columns(data, 'index')
    #     # training.to_csv('data/omniweb_kp_data_training_' + str(i) +str(d_type) +'.csv', index=False, columns=columns) 
    #     # testing.to_csv('data/omniweb_kp_data_testing_' + str(i) +str(d_type) +'.csv', index=False, columns=columns) 
    # return [training, testing]


def get_data (t, dataset_name='training', d_type='hr'):
    file = data_dir + os.sep + file_prefix + str(dataset_name) + '_' + str(t) + str(d_type) + '.csv'
    log('Loading:', dataset_name , 'from:', file)
    data = pd.read_csv(file) 
    return data


def clean_filled_values(data):
    for i in range(len(columns_names)):
        c = columns_names[i]
        fill_val = fill_values[i]
        data = data.loc[~data[c].isin([str( fill_val)])]
    return data


def drop_columns(data, cols=[]):
    for c in cols:
        if c in data.columns:
            log('dropping column:', c)
            data = data.drop(c, axis=1)
    return data


def group_data_series_len(X_train, y_train, series_len):
    X_train_series = []
    y_train_series = []
    print(len(X_train))
    for k in range(len(X_train) - series_len):
        group_data = []
        kp_data = None
        for g in range(series_len):
            group_data.append(X_train[k + g])
            kp_data = int(float((y_train[k + g])))
        X_train_series.append(group_data) 
        y_train_series.append(kp_data)
    # print(len(X_train_series), len(y_train_series))
    # print(X_train_series[0], y_train_series[0])
    X_train_series = np.array(X_train_series)
    print('X_train_series.shape:', X_train_series.shape)
    X_train_series = X_train_series.reshape(X_train_series.shape[0], X_train_series.shape[1], X_train_series.shape[2])
    print('X_train_series.shape:', X_train_series.shape)
    return [np.array(X_train_series), np.array(y_train_series)]

def get_good_kp_data_new_from_file(num_hours, interval_type, scale_down=False):
    s = interval_type[0]
    train_file_name = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_train.csv'
    test_file_name = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_test.csv'
    data_file_full = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_all.csv'    
    num_hours = str(num_hours) 
    # print('Working on', interval_type[0],'#:', num_hours)
    day_dir = interval_type[0]

    tr_file = data_dir + os.sep + 'solar-wind-data' + '-new' + os.sep + num_hours + day_dir + os.sep + train_file_name
    ts_file = data_dir + os.sep + 'solar-wind-data' + '-new' + os.sep + num_hours + day_dir + os.sep + test_file_name
    # print('Loading from file:', tr_file)
    all_data = pd.read_csv(tr_file)
    # all_data[kp_col] = [k//10 for k in all_data[kp_col].values]
    # print('all_data len before:', len(all_data)) 
    # print('all_data.columns:', all_data.columns)

    # all_data = all_data.loc[all_data['YEAR'] != 2022].reset_index()
    # all_data = all_data.sample(2000)
    all_data = all_data[:]
    if 'index' in all_data.columns:
        all_data = all_data.drop('index', axis=1) 
        
    test_data_all = pd.read_csv(ts_file, dtype=None)
    if scale_down:
        test_data_all[kp_col] = [float(v)/10 for v in test_data_all[kp_col]]
        all_data[kp_col] = [float(v)/10 for v in all_data[kp_col]]
    # test_data_all[kp_col] = [k//10 for k in test_data_all[kp_col].values]
    # print('total size:'   , (len(test_data_all) + len(all_data)))
    # print('test_data_all[Timestamp][0]', test_data_all['Timestamp'][0])
    # print('test_data_all[Timestamp][last]', test_data_all['Timestamp'][len(test_data_all)-1])

    # test_data = test_data_all.loc[test_data_all['Timestamp'].str.contains('|'.join(['2021-7','2021-8','2021-9']))].reset_index()
    test_filter = ['2021-10-' + str(i) + '-' for i in range(1, 32)]
    test_filter.extend(['2021-11-' + str(i) + '-' for i in range(1, 31)])
    test_filter =['2022-']
    # test_data = test_data_all.loc[test_data_all['Timestamp'].str.contains('|'.join(test_filter))].reset_index()
    test_data = test_data_all
    if verbose:
        log('test_data.max:', np.array(test_data[kp_col].values).max())
        log('test_data.min:', np.array(test_data[kp_col].values).min())
        log('1 test_data[Timestamp][0]', test_data['Timestamp'][0])
        log('1 test_data[Timestamp][last]', test_data['Timestamp'][len(test_data) - 1])    
    orig_y_test = test_data[kp_col].values
    # data_2021 = test_data_all.loc[test_data_all['Timestamp'].str.contains('|'.join(['2021-' + str(i) for i in range(1,9)]))]
    # print('data_2021[Timestamp][0]', data_2021['Timestamp'][0])
    # print('data_2021[Timestamp][last]', data_2021['Timestamp'][len(data_2021)-1])
    if verbose:
        log('all_data.columns:', all_data.columns)
    # data_2021 = test_data_all.loc[~test_data_all['Timestamp'].isin(test_filter)]
    # all_data = pd.concat([all_data, data_2021])
    # all_data.sort_values(by=['Timestamp'])
    # print('all_data.size:', len(data_2021))
    # all_data  = all_data.reset_index()
        print('all_data[Timestamp][0]', all_data['Timestamp'][0])
    # print('all_data[Timestamp][last]', all_data['Timestamp'][len(all_data)-1])  
        
    cols = all_data.columns 
    # features = ['B_IMF', 'B_GSE', 'B_GSM', 'SW_Temp', 'SW_Speed', 'P_Pressure', 'E_Field']
    # columns_names  =['Scalar_B',  'BZ_GSE', 'SW_Plasma_Temperature',  'SW_Proton_Density','SW_Plasma_Speed', 'Flow_pressure', 'E_elecrtric_field']

    features = columns_names
    f_index = kp_col
    # print(features, kp_col)    
    
    norm_data = all_data[f_index]
    fig_optional_name = ''
    
    train_percent = int(float(80. / 100. * len(all_data))) 
    test_val_precent = int((len(all_data) - train_percent) / 2) - 50
    # print('train_precent:', train_percent, 'validate:', test_val_precent, 'test:', test_val_precent)
    
    train_data = all_data[:]
    
    train_data = clean_filled_values(train_data)
    test_data = clean_filled_values(test_data)
        
#     print(train_data)
    valid_data = all_data[train_percent:-test_val_precent]
    
    # print('size of the test_data_all:', len(test_data_all))
    # print('len(train_data):', len(train_data), 'len(valid_data):', len(valid_data), 'len(test_data):', 
          # len(test_data),
          # 'len(orig_y_test):', len(orig_y_test))

    X_train = train_data[features].values
    X_train = reshape_x_data(X_train)
    # print('X_train.shape:', X_train.shape)
    y_train = reshape_y_data(norm_data[:])
    
    X_valid = valid_data[features].values
    X_valid = reshape_x_data(X_valid)
    y_valid = reshape_y_data(norm_data[train_percent:-test_val_precent])
    
    X_test = test_data[features].values
    X_test = reshape_x_data(X_test)

    # print('X_test size:', len(X_test))
#     y_test = reshape_y_data(norm_data[train_percent + test_val_precent:])
    y_test = reshape_y_data(test_data[f_index])
    # print('y_test len:', len(y_test))
    orig_y_test = reshape_y_data(orig_y_test)
    y = test_data['YEAR'][0]
    d = test_data['DOY'][0]
    h = test_data['HR'][0]
                
    y1 = test_data['YEAR'][len(test_data) - 1]
    d1 = test_data['DOY'][len(test_data) - 1]
    h1 = test_data['HR'][len(test_data) - 1]
    d = get_date_from_days_year_split(d, y)
    # print('d:', d)
    x_dates = []        
    # x_dates = list(test_data_all['YEAR'].values)
    for i in range (len(test_data)):
        x_dates.append(get_date_from_days_year_split(test_data['DOY'][i], test_data['YEAR'][i]))
    # x_dates=list(set(x_dates))
    # print('x_dates:', x_dates)

    return [ X_train, y_train, X_test, y_test, X_valid, y_valid, x_dates]



def load_training_and_testing_data(num_hours, scale_down=False, scale_up=True,scale_value=10,use_all_file=False):
    interval_type='hourly'
    kp_col ='Kp_German'
    # kp_col ='Kp_index'
    s = interval_type[0]
    train_file_name = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_train.csv'
    test_file_name = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_test.csv'
    data_file_full = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_all.csv'    
    num_hours = str(num_hours) 
    # print('Working on', interval_type[0],'#:', num_hours)
    day_dir = interval_type[0]

    tr_file = data_dir + os.sep + 'solar-wind-data' + os.sep + num_hours + day_dir + os.sep + train_file_name
    ts_file = data_dir + os.sep + 'solar-wind-data' + os.sep + num_hours + day_dir + os.sep + test_file_name
    
    if use_all_file:
        tr_file = data_dir + os.sep + 'solar-wind-data' + os.sep + num_hours + day_dir + os.sep + data_file_full
        print('Loading from train file:', tr_file)
    log('Loading from train file:', tr_file)
    log('Loading from test  file:', ts_file)
    all_data = pd.read_csv(tr_file)

    all_data = all_data[:]
    if 'index' in all_data.columns:
        all_data = all_data.drop('index', axis=1) 
        
    test_data_all = pd.read_csv(ts_file, dtype=None)
    if scale_up:
        s_val = 10.0
        if scale_value is not None:
            s_val = scale_value 
        test_data_all[kp_col] = [int(float(v)*s_val) for v in test_data_all[kp_col]]
        all_data[kp_col] = [int(float(v)*s_val) for v in all_data[kp_col]]
    else:
        if scale_down:
            test_data_all[kp_col] = [float(v)*0.1 for v in test_data_all[kp_col]]
            all_data[kp_col] = [float(v)*0.1 for v in all_data[kp_col]]            
            
    test_data = test_data_all
    if verbose:
        log('test_data.max:', np.array(test_data[kp_col].values).max())
        log('test_data.min:', np.array(test_data[kp_col].values).min())
        log('1 test_data[Timestamp][0]', test_data['Timestamp'][0])
        log('1 test_data[Timestamp][last]', test_data['Timestamp'][len(test_data) - 1])    
    orig_y_test = test_data[kp_col].values
    log('all_data.columns:', all_data.columns)
        
    cols = all_data.columns 
    log('data columns:', cols)
   
    log('columns_names:', columns_names)
    features = columns_names
    f_index = kp_col
    # print(features, kp_col)    
    
    norm_data = all_data[f_index]
    fig_optional_name = ''
    
    train_percent = int(float(80. / 100. * len(all_data))) 
    test_val_precent = int((len(all_data) - train_percent) / 2) - 50
    
    train_data = all_data[:]
    
    train_data = clean_filled_values(train_data)
    test_data = clean_filled_values(test_data)
        
    valid_data = all_data[train_percent:-test_val_precent]

    X_train = train_data[features].values
    X_train = reshape_x_data(X_train)
    y_train = reshape_y_data(norm_data[:])
    
    X_valid = valid_data[features].values
    X_valid = reshape_x_data(X_valid)
    y_valid = reshape_y_data(norm_data[train_percent:-test_val_precent])
    
    X_test = test_data[features].values
    X_test = reshape_x_data(X_test)

    y_test = reshape_y_data(test_data[f_index])
    orig_y_test = reshape_y_data(orig_y_test)
    y = test_data['YEAR'][0]
    d = test_data['DOY'][0]
    h = test_data['HR'][0]
                
    y1 = test_data['YEAR'][len(test_data) - 1]
    d1 = test_data['DOY'][len(test_data) - 1]
    h1 = test_data['HR'][len(test_data) - 1]
    d = get_date_from_days_year_split(d, y)
    x_dates = []        
    for i in range (len(test_data)):
        x_dates.append(get_date_from_days_year_split(test_data['DOY'][i], test_data['YEAR'][i]))

    return [ X_train, y_train, X_test, y_test, X_valid, y_valid, x_dates]


def load_training_and_testing_data_old(num_hours, interval_type='hourly'):
    kp_col ='Kp_German'
    s = interval_type[0]
    train_file_name = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_train.csv'
    test_file_name = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_test.csv'
    data_file_full = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_all.csv'    
    num_hours = str(num_hours) 
    day_dir = interval_type[0]

    tr_file = data_dir + os.sep + 'solar-wind-data'  + os.sep + num_hours + day_dir + os.sep + train_file_name
    ts_file = data_dir + os.sep + 'solar-wind-data'  + os.sep + num_hours + day_dir + os.sep + test_file_name
    all_data = pd.read_csv(tr_file)

    all_data = all_data.loc[all_data['YEAR'] != 2022].reset_index()

    all_data = all_data[:]
    if 'index' in all_data.columns:
        all_data = all_data.drop('index', axis=1) 

    test_data_all = pd.read_csv(ts_file, dtype=None)
    if verbose:
        log('total size:'   , (len(test_data_all) + len(all_data)))
    test_data = test_data_all
    if verbose:
        log('test_data.max:', np.array(test_data[kp_col].values).max())
        log('test_data.min:', np.array(test_data[kp_col].values).min())
        log('test_data[Timestamp][0]', test_data['Timestamp'][0])
        log('test_data[Timestamp][last]', test_data['Timestamp'][len(test_data) - 1])    
    orig_y_test = test_data[kp_col].values
    if verbose:
        log('all_data.columns:', all_data.columns)
        log('all_data[Timestamp][0]', all_data['Timestamp'][0])
        
    cols = all_data.columns 
    features = columns_names
    f_index = kp_col
    
    norm_data = all_data[f_index]
    fig_optional_name = ''
    
    train_percent = int(float(80. / 100. * len(all_data))) 
    test_val_precent = int((len(all_data) - train_percent) / 2) - 50
    
    train_data = all_data[:]
    
    train_data = clean_filled_values(train_data)
    test_data_all = clean_filled_values(test_data_all)
        
    valid_data = all_data[train_percent:-test_val_precent]
    

    X_train = train_data[features].values
    X_train = reshape_x_data(X_train)
    y_train = reshape_y_data(norm_data[:])
    
    X_valid = valid_data[features].values
    X_valid = reshape_x_data(X_valid)
    y_valid = reshape_y_data(norm_data[train_percent:-test_val_precent])
    
    X_test = test_data[features].values
    X_test = reshape_x_data(X_test)

    y_test = reshape_y_data(test_data[f_index])
    orig_y_test = reshape_y_data(orig_y_test)
    y = test_data['YEAR'][0]
    d = test_data['DOY'][0]
    h = test_data['HR'][0]
                
    y1 = test_data['YEAR'][len(test_data) - 1]
    d1 = test_data['DOY'][len(test_data) - 1]
    h1 = test_data['HR'][len(test_data) - 1]
    d = get_date_from_days_year_split(d, y)
    x_dates = []        
    for i in range (len(test_data)):
        x_dates.append(get_date_from_days_year_split(test_data['DOY'][i], test_data['YEAR'][i]))

    return [ X_train, y_train, X_test, y_test, X_valid, y_valid, x_dates]


def get_good_kp_data(num_hours, interval_type):
    train_file_name = 'solar_wind_parameters_data_' + interval_type + '_train.csv'
    test_file_name = 'solar_wind_parameters_data_' + interval_type + '_test.csv'
    data_file_full = 'solar_wind_parameters_data_' + interval_type + '_all.csv'    
    num_hours = str(num_hours) 
    # print('Working on', interval_type[0],'#:', num_hours)
    day_dir = interval_type[0]

    tr_file = data_dir + os.sep + 'solar-wind-data' + os.sep + num_hours + day_dir + os.sep + train_file_name
    ts_file = data_dir + os.sep + 'solar-wind-data' + os.sep + num_hours + day_dir + os.sep + test_file_name
    # print('Loading from file:', tr_file)
    all_data = pd.read_csv(tr_file)
    # print('all_data len before:', len(all_data)) 
    # print('all_data.columns:', all_data.columns)

    all_data = all_data.loc[all_data['YEAR'] != 2022].reset_index()
    # print('all_data len after:', len(all_data)) 
    # print(all_data.tail(10))

    # all_data = all_data.sample(2000)
    # all_data.to_csv('data/omniweb_good_' + str(num_hours) + 'h_all.csv')
    if 'index' in all_data.columns:
        all_data = all_data.drop('index', axis=1) 
    # print('loading test data from file:', ts_file)
    
    if int(num_hours) < 10:
        skip = int(num_hours)
        test_data_all = pd.read_csv(ts_file, dtype=None, nrows=150, skiprows=range(1, skip + 1))
    else:
        skip = 0
        # test_data_all = pd.read_csv(ts_file,dtype=None, nrows=150)
        # test_data_all = pd.read_csv(ts_file,dtype=None, nrows=150)
        test_data_all = pd.read_csv(ts_file, dtype=None)
  
    cols = all_data.columns 
    features = ['B_IMF', 'B_GSE', 'B_GSM', 'SW_Temp', 'SW_Speed', 'P_Pressure', 'E_Field']
    f_index = 'Dst_Index'
    # print(features, kp_col)    
    
    f_data_orig = all_data[f_index].values
    max_val = np.array(all_data[f_index]).max() 
    min_val = np.array(all_data[f_index]).min() 
    # print('max_val:', max_val, 'min_val:', min_val)
    
    max_val_t = np.array(test_data_all[f_index]).max() 
    min_val_t = np.array(test_data_all[f_index]).min()
    # print('max_val_t:',  max_val_t, 'max_val:', max_val)
    # print('min_val_t:',  min_val_t, 'min_val:', min_val)
    if max_val_t > max_val:
        # print('Using test data max:', max_val_t, max_val)
        max_val = max_val_t 
    if min_val_t < min_val:
        # print('Using test data min:', min_val_t, min_val)
        min_val = min_val_t
    # print('Final max and min values...')
    # print('max_val_t:',  max_val_t, 'max_val:', max_val)
    # print('min_val_t:',  min_val_t, 'min_val:', min_val)        
    norm_data = all_data[f_index]
    fig_optional_name = ''
    
    train_percent = int(float(80. / 100. * len(all_data))) 
    test_val_precent = int((len(all_data) - train_percent) / 2) - 50
    # print('train_precent:', train_percent, 'validate:', test_val_precent, 'test:', test_val_precent)
    
    train_data = all_data[:train_percent]
#     print(train_data)
    valid_data = all_data[train_percent:-test_val_precent]
    
    # print('size of the test_data_all:', len(test_data_all))

    test_data = test_data_all
    orig_y_test = test_data_all[f_index].values
    # print('len(train_data):', len(train_data), 'len(valid_data):', len(valid_data), 'len(test_data):', 
          # len(test_data),
          # 'len(orig_y_test):', len(orig_y_test))
    X_train = train_data[features].values
    X_train = reshape_x_data(X_train)
    # print('X_train.shape:', X_train.shape)
    y_train = reshape_y_data(norm_data[:train_percent])
    
    X_valid = valid_data[features].values
    X_valid = reshape_x_data(X_valid)
    y_valid = reshape_y_data(norm_data[train_percent:-test_val_precent])
    
    X_test = test_data[features].values
    X_test = reshape_x_data(X_test)

    # print('X_test size:', len(X_test))
#     y_test = reshape_y_data(norm_data[train_percent + test_val_precent:])
    y_test = reshape_y_data(test_data_all[f_index])
    # print('y_test len:', len(y_test))
    orig_y_test = reshape_y_data(orig_y_test)
    y = test_data_all['YEAR'][0]
    d = test_data_all['DOY'][0]
    h = test_data_all['HR'][0]
                
    y1 = test_data_all['YEAR'][len(test_data_all) - 1]
    d1 = test_data_all['DOY'][len(test_data_all) - 1]
    h1 = test_data_all['HR'][len(test_data_all) - 1]
    d = get_date_from_days_year_split(d, y)
    # print('d:', d)
    x_dates = []        
    # x_dates = list(test_data_all['YEAR'].values)
    for i in range (len(test_data_all)):
        x_dates.append(get_date_from_days_year_split(test_data_all['DOY'][i], test_data_all['YEAR'][i]))
    # x_dates=list(set(x_dates))
    # print('x_dates:', x_dates)
    return [ X_train, y_train, X_test, y_test, X_valid, y_valid, x_dates]


def get_good_kp_from_data(num_hours, interval_type, training_data, testing_data):
    # print('training_data after:', len(testing_data))
    
    training_data[kp_col] = training_data[kp_col].shift(-1 * (num_hours))
    training_data = training_data.dropna()
    # print('training_data after:', len(testing_data))
    
    testing_data = testing_data.iloc[7 - num_hours:,:].reset_index()
    testing_data[kp_col] = testing_data[kp_col].shift(-1 * num_hours)
    testing_data = testing_data.dropna()   
    
    training_data = clean_filled_values(training_data)
    testing_data = clean_filled_values(testing_data)
    # testing_data = testing_data.loc[testing_data[kp_col]  >= 0].reset_index()
    # training_data = training_data.loc[training_data[kp_col]  >= 0].reset_index()
    test_data_all = testing_data
    # test_data_all = testing_data[len(testing_data) -150:]
    all_data = training_data 
    test_data_all = test_data_all 
    num_hours = str(num_hours) 
    # print('Working on', interval_type[0],'#:', num_hours)
    day_dir = interval_type[0]

    # print('all_data len before:', len(all_data)) 
    # print('all_data.columns:', all_data.columns)

    # all_data = all_data.loc[all_data['YEAR'] != 2020].reset_index()
    # print('all_data len after:', len(all_data)) 
    # print(all_data.tail(10))

    # all_data = all_data.sample(3000)
    # all_data = pd.concat([ training_data, testing_data[:len(testing_data) -150]])
    # all_data = all_data.sample(2000)
    # all_data.to_csv('data/omniweb_good_' + str(num_hours) + 'h_all.csv')
    if 'index' in all_data.columns:
        all_data = all_data.drop('index', axis=1) 
  
    cols = all_data.columns 
    # features = ['B_IMF', 'B_GSE', 'B_GSM', 'SW_Temp', 'SW_Speed', 'P_Pressure', 'E_Field']
    features = features1
    f_index = kp_col 
    # print(features , kp_col)    
    
    f_data_orig = all_data[f_index].values
    max_val = np.array(all_data[f_index]).max() 
    min_val = np.array(all_data[f_index]).min() 
    # print('max_val:', max_val, 'min_val:', min_val)
    
    max_val_t = np.array(test_data_all[f_index]).max() 
    min_val_t = np.array(test_data_all[f_index]).min()
    # print('max_val_t:',  max_val_t, 'max_val:', max_val)
    # print('min_val_t:',  min_val_t, 'min_val:', min_val)
    if max_val_t > max_val:
        # print('Using test data max:', max_val_t, max_val)
        max_val = max_val_t 
    if min_val_t < min_val:
        # print('Using test data min:', min_val_t, min_val)
        min_val = min_val_t
    # print('Final max and min values...')
    # print('max_val_t:',  max_val_t, 'max_val:', max_val)
    # print('min_val_t:',  min_val_t, 'min_val:', min_val)        
    norm_data = all_data[f_index]
    fig_optional_name = ''
    
    train_percent = int(float(80. / 100. * len(all_data))) 
    test_val_precent = int((len(all_data) - train_percent) / 2) - 50
    # print('train_precent:', train_percent, 'validate:', test_val_precent, 'test:', test_val_precent)
    
    train_data = all_data[:train_percent]
#     print(train_data)
    valid_data = all_data[train_percent:-test_val_precent]
    
    # print('size of the test_data_all:', len(test_data_all))

    test_data = test_data_all
    orig_y_test = test_data_all[f_index].values
    # print('len(train_data):', len(train_data), 'len(valid_data):', len(valid_data), 'len(test_data):', 
    #       len(test_data),
    #       'len(orig_y_test):', len(orig_y_test))
    X_train = train_data[features].values
    X_train = reshape_x_data(X_train)
    # print('X_train.shape:', X_train.shape)
    y_train = reshape_y_data(norm_data[:train_percent])
    
    X_valid = valid_data[features].values
    X_valid = reshape_x_data(X_valid)
    y_valid = reshape_y_data(norm_data[train_percent:-test_val_precent])
    
    X_test = test_data[features].values
    X_test = reshape_x_data(X_test)

    # print('X_test size:', len(X_test))
#     y_test = reshape_y_data(norm_data[train_percent + test_val_precent:])
    y_test = reshape_y_data(test_data_all[f_index])
    # print('y_test len:', len(y_test))
    orig_y_test = reshape_y_data(orig_y_test)
    x_dates = list(test_data_all['Timestamp'].values)
    x_dates = [ d.split('-') for d in x_dates]
    # print('x_dates:', x_dates)
    return [ X_train, y_train, X_test, y_test, X_valid, y_valid, x_dates]


def get_date_from_days_year(d, y):
    return datetime.strptime('{} {}'.format(d, y), '%j %Y')


def get_date_from_days_year_split(d, y):
    date = get_date_from_days_year(d, y)
    return [date.year, date.month, date.day]

    
def reshape_x_data(data):
    data = [ np.array(c).reshape(len(c), 1) for c in data]
    data = np.array(data)
    data = data.reshape(data.shape[0], 1, data.shape[1])
    return data


def reshape_y_data(data):
    data = [ np.array(c) for c in data]
    data = np.array(data)
    data = data.reshape(data.shape[0], 1)
    return data


def custom_loss_function(y_true, y_pred):
   squared_difference = tensorflow.square(y_true - y_pred)
   return tensorflow.reduce_mean(squared_difference, axis=-1)

def plot_figure(x, y_test, y_preds_mean, y_preds_var, num_hours, label='Kp index',
                file_name=None, block=True,
                show_fig=False,
                return_fig=False,
                figsize=(5,2.5),
                interval='d', denormalize=False, norm_max=1, norm_min=0, boxing=False, wider_size=False,
                observation_color=None,prediction_color=None, uncertainty_label='Epistemic Uncertainty',
                fill_graph=False,
                uncertainty_margin=1,
                uncertainty_color='#aabbcc',
                x_labels=None,
                x_label='Time',
                scale_down=False,
                ylimit_min=-10,
                ylimit_max=90,
                verbose=True,
                fill_file=None):
    linewidth = 1.3
    markersize = 1.7
    marker = 'o'
    linestyle = 'solid'

    fig, ax = plt.subplots(figsize=figsize)

    
    ax.plot(x, y_preds_mean ,
            label='Prediction',
            linewidth=linewidth,
            markersize=markersize,
            marker=marker,
            linestyle=linestyle,
            color=prediction_color)
    ax.plot(x, y_test,
            label='Observation',
            linewidth=linewidth,
            markersize=markersize,
            marker=marker,
            linestyle=linestyle,
            color=observation_color
            )

    lower =(y_preds_mean - y_preds_var * uncertainty_margin)
    upper = (y_preds_mean + y_preds_var * uncertainty_margin)

    if fill_graph:
        plt.fill_between(x, (y_preds_mean - y_preds_var * uncertainty_margin),
                             (y_preds_mean + y_preds_var * uncertainty_margin),
                             color=uncertainty_color, alpha=0.5, label=uncertainty_label)
        
    ylim_mx = np.array(y_test).max()
    ylim_min = np.array(y_test).min()
    ax.set_ylim(-165, 65)
    ax.set_ylim(ylim_min,ylim_mx)
    ax.set_ylim(-1, 9)
    if scale_down:
        ax.set_ylim(-5, 10)

    plt.xlabel(x_label)
    
    label_y = label
    if label_y.startswith('F'):
        label_y = 'F10.7'
    plt.ylabel(label_y)
    plt.title(str(num_hours) + '' + interval + ' forecasting', fontsize=13, fontweight='bold')

    if not  boxing:
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction='in')
    if len(x) <= 6:
        ax.xaxis.set_ticks(x)
    if x_labels is not None:
        ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    xfmt = md.DateFormatter('%m/%d/%y')
    ax.xaxis.set_major_formatter(xfmt)
    if file_name is not None:
        if verbose:
            log('Saving figure to file:', file_name)
        plt.savefig(file_name, bbox_inches='tight')
    if return_fig:
        return plt
    if show_fig:
        plt.show(block=block)


def copyModel2Model(model_source, model_target, certain_layer=""): 
    for l_tg, l_sr in zip(model_target.layers, model_source.layers):
        wk0 = l_sr.get_weights()
        l_tg.set_weights(wk0)
        if l_tg.name == certain_layer:
            break
    return model_target


def predict_finetune(model, val, r=50):
    # predict stochastic dropout model T times
    p_hat = []
    for t in range(r):
        p_hat.append(model.predict(val))
    p_hat = np.array(p_hat)

    # mean prediction
    prediction = np.mean(p_hat, axis=0)

    # estimate uncertainties
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2

    return np.squeeze(prediction), np.squeeze(aleatoric), np.squeeze(epistemic)


# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[...,:n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])

  
  # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])


def process_val(t, preds,scale_down=False):
    return t
    r = np.array(preds).mean()
    negative = -15
    positive = 15
    if scale_down:
        negative = -1
        postivie = 1
    # print(t, r, t-r)
    if t < 0:
        r = r if  ((int(round(np.array(preds).mean())) - t) <= negative) else int(t - (int(round(np.array(preds).mean())) - t) / 2)
    else:
        r = r if  (abs(int(round(np.array(preds).mean())) - t) <= positive) else  int(t - (int(round(np.array(preds).mean())) - t) / 2)
    return r 


def select_random_k(l, k):
    random.shuffle(l)
    result = []
    for i in range(0, k):
        result.append(l[i])
    return result
 
def create_dirs():
    dirs = ['models', 'data','logs','results','figures']
    for d in dirs:
        os.makedirs(d,  exist_ok=True)


create_dirs()
create_log_file()

