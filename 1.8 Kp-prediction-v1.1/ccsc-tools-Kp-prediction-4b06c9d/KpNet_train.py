'''
 (c) Copyright 2022
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
from tensorflow import keras
from pprint import pprint
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

from KpNet_utils import *
from KpNet_model import KpNetModel


num_hours = 1
interval_type = 'hourly'
epochs = 100
prev_weights = None
w_dir=None

def train_model(start_hour, end_hour,epochs=epochs):
    for k in range(start_hour,end_hour):
        print('Running training for h =', k, 'hour ahead')
        num_hours = k
        X_train,y_train, X_test, y_test, X_valid, y_valid,x_dates = load_training_and_testing_data(num_hours)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = KpNetModel()
        
        model.build_base_model(input_shape, dropout=0.6, kl_weight=1/X_train.shape[0] )
        model.models()
        model.compile()
        model.fit(X_train, y_train, epochs=epochs)
        model.save_weights(num_hours,w_dir=w_dir)

if __name__ == '__main__':
    starting_hour = 1
    ending_hour = 10
    if len(sys.argv) >= 2 :
        if str(sys.argv[1]).strip().isnumeric():
            starting_hour = int(float(sys.argv[1]))
        else:
            print('Error: the number of hour must be number between 1 and 9. You entered:', sys.argv[1])
            exit()
        ending_hour = starting_hour + 1

    
    if starting_hour < 1 or starting_hour -1> 9:
        print('Invalid starting hour:', starting_hour,'\nHours must be between 1 and 9.')
        exit()
    print('Starting hour:', starting_hour, 'ending hour:', ending_hour-1)
    train_model(starting_hour, ending_hour)